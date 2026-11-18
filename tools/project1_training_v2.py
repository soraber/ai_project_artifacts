"""CLI-only, assistant-loss QLoRA using Transformers Trainer and PEFT.

No holdout feedback, remote Python code, pickle weights, or overwrite of prior runs.
Sequence packing is deliberately disabled for this new reference implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    revision: str
    train_json: str
    output_dir: str
    cutoff: int = 2048
    batch_size: int = 2
    accumulation_steps: int = 8
    learning_rate: float = 2e-4
    epochs: float = 1.0
    rank: int = 8
    alpha: int = 16
    seed: int = 42
    max_samples: int = 50000
    smoke: bool = True

    def validate(self):
        if not re.fullmatch(r'[a-fA-F0-9]{40}', self.revision):
            raise ValueError('An immutable model commit SHA is required')
        if self.cutoff < 2 or min(self.batch_size, self.accumulation_steps, self.rank, self.max_samples) < 1:
            raise ValueError('Invalid training sizes')
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError('Learning rate and epochs must be positive')


def token_ids(value):
    if hasattr(value, 'ids'):
        value = value.ids
    elif isinstance(value, dict) or hasattr(value, 'keys'):
        value = value['input_ids']
    elif hasattr(value, 'input_ids'):
        value = value.input_ids
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError('Expected one encoded example')
        value = value[0]
    return [int(item) for item in value]


def encode_example(row, tokenizer, cutoff):
    if cutoff < 2 or not row.get('instruction', '').strip() or not row.get('output', '').strip():
        raise ValueError('Each row requires a prompt, answer, and usable cutoff')
    content = row['instruction'].strip()
    if row.get('input', '').strip():
        content += '\n\n' + row['input'].strip()
    messages = [{'role': 'user', 'content': content}]
    prefix = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    full = token_ids(tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row['output'].strip()}],
                                                   tokenize=True, add_generation_prompt=False))
    if not prefix or full[:len(prefix)] != prefix:
        raise ValueError('Chat template does not expose a stable assistant boundary')
    answer = full[len(prefix):]
    if not answer:
        raise ValueError('No assistant tokens')
    # Preserve the historical proportional prompt/response cutoff allocation.
    if len(answer)*2 < cutoff:
        target = len(answer)
    elif len(prefix)*2 < cutoff:
        target = cutoff-len(prefix)
    else:
        target = int(cutoff*len(answer)/(len(prefix)+len(answer)))
    target = max(1, min(target, len(answer)))
    source = max(1, min(cutoff-target, len(prefix)))
    answer, prefix = answer[:target], prefix[:source]
    return {'input_ids': prefix+answer, 'attention_mask': [1]*(source+target),
            'labels': [-100]*source+answer}


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        width = max(len(row['input_ids']) for row in features)
        return {key: torch.tensor([row[key]+[pad]*(width-len(row[key])) for row in features], dtype=torch.long)
                for key, pad in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]}


def train(config: TrainingConfig):
    config.validate()
    from training_environment import validate
    validate(Path(__file__).with_name('requirements-training-v2.hashes.txt'))
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('The QLoRA runner requires CUDA; CPU tests use a separate tiny model')
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'version': 'upgrade-v2', 'status': 'preparing', 'purpose': 'smoke' if config.smoke else 'full',
                'model': config.model, 'model_revision': config.revision, 'packing': False,
                'loss_scope': 'assistant tokens only; right-padding labels=-100',
                'holdout_used_for_training_or_selection': False,
                'settings': {k:v for k,v in asdict(config).items() if k not in {'train_json', 'output_dir'}},
                'gpu': torch.cuda.get_device_name(0),
                'packages': {name:metadata.version(name) for name in ['torch','transformers','peft','accelerate','datasets','bitsandbytes']}}

    def save_manifest():
        (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    save_manifest()
    try:
        set_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        data = Path(config.train_json).read_bytes()
        rows = json.loads(data)
        if not isinstance(rows, list) or not rows:
            raise ValueError('Training JSON must be a nonempty array; the original JSONL preparation is unchanged')
        manifest['train_data_sha256'] = hashlib.sha256(data).hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision,
                                                   token=os.getenv('HF_TOKEN'), trust_remote_code=False)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer needs a pad or EOS token')
        tokenizer.padding_side = 'right'
        manifest['chat_template_sha256'] = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        selected = rows[:min(config.max_samples, 8 if config.smoke else config.max_samples)]
        encoded = [encode_example(row, tokenizer, config.cutoff) for row in selected]
        manifest.update(train_examples=len(encoded), answer_tokens=sum(sum(x!=-100 for x in row['labels'][1:]) for row in encoded))
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(config.model, revision=config.revision,
            token=os.getenv('HF_TOKEN'), trust_remote_code=False, use_safetensors=True,
            device_map={'': torch.cuda.current_device()}, dtype=dtype, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=config.rank,
                              lora_alpha=config.alpha, target_modules='all-linear', lora_dropout=0.0,
                              revision=config.revision))
        manifest.update(status='training', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        save_manifest()
        args = TrainingArguments(output_dir=str(output),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=1 if config.smoke else config.accumulation_steps,
            learning_rate=config.learning_rate, num_train_epochs=config.epochs,
            max_steps=2 if config.smoke else -1, lr_scheduler_type='cosine', logging_steps=1 if config.smoke else 10,
            save_strategy='no' if config.smoke else 'steps', save_steps=50, save_total_limit=2,
            eval_strategy='no', bf16=dtype==torch.bfloat16, fp16=dtype==torch.float16,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            optim='adamw_torch', report_to='none', seed=config.seed, dataloader_num_workers=0)
        trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(encoded),
                          data_collator=AssistantCollator(tokenizer.pad_token_id), processing_class=tokenizer)
        metrics = trainer.train().metrics
        adapter = output/'adapter'
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        trainer.save_state()
        (output/'train_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        manifest.update(status='complete', steps=trainer.state.global_step,
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        return output
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        manifest['seconds'] = time.perf_counter()-started
        save_manifest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = TrainingConfig(**json.loads(args.config.read_text()))
    print(train(config))
