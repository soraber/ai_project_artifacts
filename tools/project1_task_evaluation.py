"""Matched base/adapter task smoke evaluation, without executing generated code."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

from agent_task_evaluation import compare_tasks


def evaluate_adapter_tasks(model_name, model_revision, adapter_dir, output_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    started = time.perf_counter()
    adapter_dir = Path(adapter_dir)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision, token=os.getenv("HF_TOKEN"))
    base = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision,
        token=os.getenv("HF_TOKEN"), device_map="auto", dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()
    def generate(prompt, adapter_enabled):
        from contextlib import nullcontext
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)
        batch = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=2048).to(next(model.parameters()).device)
        with torch.inference_mode(), (nullcontext() if adapter_enabled else model.disable_adapter()):
            output = model.generate(**batch, do_sample=False, max_new_tokens=256, use_cache=True,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        result = compare_tasks(generate)
        result.update(model=model_name, model_revision=model_revision,
                      seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(0),
                      decoding={"do_sample": False, "max_new_tokens": 256, "input_max_tokens": 2048},
                      adapter_weights_sha256=hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest())
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        return result
    finally:
        del model, base
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_adapter_tasks(args.model, args.revision, args.adapter, args.output)
    print(json.dumps(result["metrics"], indent=2))
