import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('training_v2', Path(__file__).parents[1]/'tools/project1_training_v2.py')
training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = training
spec.loader.exec_module(training)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        prefix = [1, 2, 3, 4]
        return {'input_ids': prefix + ([5, 6, 7] if len(messages)>1 else [])}


def test_assistant_mask_and_response_aware_cutoff():
    row = {'instruction': 'Question', 'output': 'Answer'}
    encoded = training.encode_example(row, Tokenizer(), 5)
    assert len(encoded['input_ids']) == 5
    assert encoded['labels'] == [-100, -100, -100, 5, 6]
    assert training.encode_example(row, Tokenizer(), 20)['labels'] == [-100]*4+[5, 6, 7]


def test_template_and_configuration_fail_closed():
    class BadTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if len(messages)==1 else [9, 2, 3]
    with pytest.raises(ValueError, match='boundary'):
        training.encode_example({'instruction':'q', 'output':'a'}, BadTokenizer(), 10)
    with pytest.raises(ValueError, match='commit'):
        training.TrainingConfig('m', 'main', 'data', 'out').validate()


def test_tiny_real_trainer_updates_only_adapter(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, set_seed

    # Check production kwargs too: the tiny test alone would miss removed API arguments.
    import ast
    import inspect
    supported = inspect.signature(TrainingArguments).parameters
    tree = ast.parse(inspect.getsource(training))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'TrainingArguments']
    assert calls and all(k.arg in supported for call in calls for k in call.keywords)

    set_seed(42)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        pad_token_id=0, eos_token_id=7))
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
                                           target_modules=['q_proj','v_proj']))
    before = {name:p.detach().clone() for name,p in model.named_parameters()}
    rows = [training.encode_example({'instruction':'q', 'output':'a'}, Tokenizer(), 16)]*4
    batch = training.AssistantCollator(0)([rows[0], {k:v[:-1] for k,v in rows[0].items()}])
    assert batch['labels'][1,-1].item() == -100 and batch['attention_mask'][1,-1].item() == 0
    args = TrainingArguments(output_dir=str(tmp_path/'trainer'), use_cpu=True, max_steps=2,
        per_device_train_batch_size=2, learning_rate=.01, save_strategy='no', report_to='none',
        optim='adamw_torch', disable_tqdm=True, dataloader_pin_memory=False)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                      data_collator=training.AssistantCollator(0))
    metrics = trainer.train().metrics
    assert torch.isfinite(torch.tensor(metrics['train_loss']))
    assert trainer.state.global_step == 2
    assert all(torch.equal(before[n], p) for n,p in model.named_parameters() if not p.requires_grad)
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if p.requires_grad)
    model.save_pretrained(tmp_path/'adapter', safe_serialization=True)
    assert (tmp_path/'adapter/adapter_model.safetensors').exists()
