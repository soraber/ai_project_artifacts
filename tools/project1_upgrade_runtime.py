"""Run unchanged dataset cells and versioned training in a separate Colab interpreter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(notebook_path, persistent_root):
    from security_preflight import require_gpu_clearance
    require_gpu_clearance()
    from huggingface_hub import HfApi, get_token
    import datasets
    import torch
    import yaml

    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Enter HF_TOKEN in the runner notebook's hidden input first")
    os.environ["HF_TOKEN"] = token
    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    api = HfApi(token=token)
    model_revision = api.model_info(model).sha
    dataset_name = "HuggingFaceH4/stack-exchange-preferences"
    dataset_revision = api.dataset_info(dataset_name).sha
    root = Path(persistent_root)
    if not (Path("/content/drive/MyDrive")).is_dir():
        raise RuntimeError("Mount the approved Google Drive before running")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / "runs" / "upgrade-v1" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    alias = Path("/content/fine_tuning_pipeline")
    if alias.exists() or alias.is_symlink():
        raise FileExistsError("Existing /content/fine_tuning_pipeline needs review before creating a new run")
    alias.symlink_to(run_dir, target_is_directory=True)
    os.chdir(root)
    registry = data_dir / "dataset_info.json"
    if not registry.exists():
        registry.write_text("{}\n")
    nb = json.loads(Path(notebook_path).read_text())
    cells = {cell.get("id"): "".join(cell["source"]) for cell in nb["cells"]}
    manifest = {"run_id": stamp, "status": "preparing", "model": model,
                "model_revision": model_revision, "dataset": dataset_name,
                "dataset_revision": dataset_revision, "gpu": torch.cuda.get_device_name(0),
                "dataset_cell_sha256": hashlib.sha256(cells["253ca33c"].encode()).hexdigest()}
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
           "HF_HUB_DISABLE_XET": "1", "WANDB_DISABLED": "true"}
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_dir / "environment.lock.txt").write_text(freeze)
    namespace = {"__name__": "__main__"}
    real_load_dataset = datasets.load_dataset

    def pinned_load_dataset(name, *args, **kwargs):
        if name == dataset_name:
            kwargs["revision"] = dataset_revision
        return real_load_dataset(name, *args, **kwargs)

    try:
        datasets.load_dataset = pinned_load_dataset
        for cell_id in ["470a4fd2", "253ca33c", "08fa3947", "b3948dc6", "a7f02586", "c3a77256", "e1480002"]:
            source = cells[cell_id]
            # beautifulsoup4 is installed in the isolated environment before execution.
            if cell_id == "253ca33c":
                source = source.replace("%pip install -q beautifulsoup4\n", "")
            print(f"Executing notebook cell {cell_id}", flush=True)
            exec(compile(source, f"notebook:{cell_id}", "exec"), namespace)
            if cell_id == "470a4fd2":
                namespace.update(DATA_DIR=data_dir, DATASET_JSONL=data_dir / "domain_sft_50k.jsonl",
                                 EVAL_JSONL=data_dir / "eval_500.jsonl", MODEL_REVISION=model_revision)
        config_path = namespace["SFT_YAML"]
        config = yaml.safe_load(config_path.read_text())
        config.update(model_revision=model_revision, dataset_dir=str(data_dir),
                      overwrite_output_dir=False, report_to="none")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest.update(status="training", train_rows=len(namespace["train_rows"]),
                        eval_rows=len(namespace["eval_rows"]), training_config=config,
                        dataset_jsonl_sha256=hashlib.sha256(namespace["DATASET_JSONL"].read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        cli = Path(sys.executable).parent / "llamafactory-cli"
        subprocess.run([str(cli), "train", str(config_path)], check=True, env=env, timeout=6 * 3600)
        manifest["status"] = "evaluating"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        exec(compile(cells["18ed85c3"], "notebook:18ed85c3", "exec"), namespace)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        datasets.load_dataset = real_load_dataset
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("Persistent run:", run_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--persistent-root", default="/content/drive/MyDrive/projects/MyProject1")
    args = parser.parse_args()
    run(args.notebook, args.persistent_root)
