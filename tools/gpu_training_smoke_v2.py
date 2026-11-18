"""Credential-free NF4/LoRA integration check, not a model-quality benchmark."""
import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from project1_training_v2 import AssistantCollator
from training_environment import validate


def run(output_root):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, set_seed

    validate(Path(__file__).with_name('requirements-training-v2.hashes.txt'))
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; no CPU result may substitute for this test')
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output_root)/stamp
    output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with tempfile.TemporaryDirectory(prefix='qlora-v2-smoke-') as directory:
        tiny = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=2))
        tiny.save_pretrained(directory, safe_serialization=True)
        del tiny
        model = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=False,
            use_safetensors=True, dtype=dtype, device_map={'': 0}, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        assert model.is_loaded_in_4bit
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16, target_modules='all-linear'))
        before = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        row = {'input_ids': [1,3,4,5,6,7,8,2], 'attention_mask': [1]*8,
               'labels': [-100]*4+[6,7,8,2]}
        args = TrainingArguments(output_dir=str(output/'trainer'), max_steps=2,
            per_device_train_batch_size=2, learning_rate=2e-4, save_strategy='no', eval_strategy='no',
            bf16=dtype==torch.bfloat16, fp16=dtype==torch.float16, report_to='none', optim='adamw_torch',
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, seed=42)
        trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list([row]*4),
                          data_collator=AssistantCollator(0))
        metrics = trainer.train().metrics
        assert trainer.state.global_step == 2 and torch.isfinite(torch.tensor(metrics['train_loss']))
        assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if p.requires_grad)
        model.save_pretrained(output/'adapter', safe_serialization=True)
        summary = {'status': 'passed', 'purpose': 'authored random tiny model; NF4 integration, not quality',
            'steps': 2, 'loss_finite': True, 'lora_updated': True, 'gpu': torch.cuda.get_device_name(0),
            'cuda': torch.version.cuda, 'seconds': time.perf_counter()-started,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
            'packages': {n:metadata.version(n) for n in ['torch','transformers','peft','accelerate','datasets','bitsandbytes']},
            'lock_sha256': hashlib.sha256(Path(__file__).with_name('requirements-training-v2.hashes.txt').read_bytes()).hexdigest()}
        (output/'gpu_smoke.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
        return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args().output_root)
