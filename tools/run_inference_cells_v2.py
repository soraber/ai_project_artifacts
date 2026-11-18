"""Execute the reviewed inference cells in a fresh process on Colab."""
import argparse
import ast
import asyncio
import gc
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path


EXPECTED_FILES = {
    2: ("project2_summary.json", "project2_eval_results.json"),
    3: ("project3_summary.json", "router_per_case.json", "mcp_integration_v2.json"),
    5: ("baseline/project5_final_summary.json", "candidate/project5_final_summary.json"),
    8: ("longmemeval/results/summary.json", "longmemeval/results/per_case.json"),
}


async def run(project, cells_path, run_root, benchmark=None):
    from training_environment import validate

    tools_dir = Path(__file__).resolve().parent
    lock_path = tools_dir / "requirements-inference-v2.hashes.txt"
    validate(lock_path)
    import torch

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select the approved A100 runtime")
    if not Path("/content/drive").is_mount():
        raise RuntimeError("Google Drive must be mounted before evaluation")
    run_id = run_root.name
    if not re.fullmatch(r"\d{8}T\d{12}Z", run_id):
        raise ValueError("Expected a timestamp run ID")
    project_dir = Path(f"/content/drive/MyDrive/projects/MyProject{project}")
    if run_root != project_dir / "outputs/runs/upgrade-v2" / run_id:
        raise ValueError("Run root must be in the Drive folder for this project")
    os.environ.update(
        PROJECT_REPO=str(tools_dir.parent),
        PORTFOLIO_PROJECT_DIR=str(project_dir),
        PORTFOLIO_RUN_ID=run_id,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_path = run_root / "run_manifest.json"
    manifest = {
        "project": project,
        "run_id": run_id,
        "status": "running",
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version.split()[0],
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "cell_bundle_sha256": hashlib.sha256(cells_path.read_bytes()).hexdigest(),
        "completed_cells": [],
    }
    if benchmark:
        manifest["benchmark"] = benchmark

    def save():
        pending = manifest_path.with_suffix(".pending")
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        pending.replace(manifest_path)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    save()
    try:
        cells = json.loads(cells_path.read_text())
        if project in (2, 3):
            await execute(project, cells, manifest, save, retrieval_refresh=bool(benchmark))
        else:
            work = tools_dir.parent / run_id
            shutil.copytree(tools_dir / "source", work)
            project_root = work / f"project{project}"
            os.chdir(project_root)
            sys.path.insert(0, str(project_root / "src"))
            os.environ["PYTHONPATH"] = str(project_root / "src")
            if project == 5:
                manifest["model_runs"] = []
                for label, model in (("baseline", None),
                                     ("candidate", "Qwen/Qwen3-4B-Instruct-2507")):
                    manifest.update(active_model=label, completed_cells=[])
                    result_dir = await execute(project, cells, manifest, save, project_root, model)
                    shutil.copytree(result_dir, run_root / label)
                    manifest["model_runs"].append({"arm": label, "completed_cells": manifest["completed_cells"]})
                    gc.collect()
                    torch.cuda.empty_cache()
                summaries = [
                    json.loads((run_root / arm / "project5_final_summary.json").read_text())
                    for arm in ("baseline", "candidate")
                ]
                baseline, candidate = summaries
                configs = [{key: value for key, value in row["config"].items() if key != "local_model"}
                           for row in summaries]
                if configs[0] != configs[1] or baseline["test_cases_sha256"] != candidate["test_cases_sha256"]:
                    raise ValueError("Project 5 model arms did not use matched settings and holdouts")
                manifest["matched_holdout_sha256"] = baseline["test_cases_sha256"]
            else:
                prepare_longmemeval(project_dir, project_root)
                os.environ["PORTFOLIO_LONGMEMEVAL"] = "1"
                result_dir = await execute(project, cells, manifest, save, project_root)
                shutil.copytree(result_dir, run_root / "longmemeval")
        expected = EXPECTED_FILES[project] if not benchmark else (
            "retriever_refresh_summary.json", "retriever_refresh_per_case.json")
        missing = [name for name in expected if not (run_root / name).is_file()]
        if missing:
            raise RuntimeError(f"Expected evaluation files are missing: {missing}")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest.update(seconds=time.perf_counter() - started,
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        save()


async def execute(project, cells, manifest, save, project_root=None, model=None, retrieval_refresh=False):
    module = types.ModuleType(f"project{project}_inference")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(display=print, PROJECT_ROOT=project_root)
    try:
        for cell in cells:
            number = cell["number"]
            manifest["active_cell"] = number
            save()
            print(f"Project {project} | Cell {number} ({cell['id']})", flush=True)
            code = compile(cell["source"], f"project{project}:cell{number}",
                           "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)
            result = eval(code, namespace)
            if inspect.isawaitable(result):
                await result
            if project in (5, 8) and number == 3 and not namespace["RUN_LOCAL_GPU_EVAL"]:
                raise RuntimeError("This launcher requires the notebook's local GPU mode")
            if project == 5 and number == 3 and model:
                namespace["config"] = namespace["config"].model_copy(update={"local_model": model})
                print(f"Candidate planner: {model}", flush=True)
            manifest["completed_cells"].append(number)
        if retrieval_refresh:
            from project2_retriever_refresh import evaluate
            evaluate(namespace)
        return namespace.get("result_dir") if project == 5 else namespace.get("longmem_run")
    finally:
        sys.modules.pop(module.__name__, None)


def prepare_longmemeval(project_dir, project_root):
    from huggingface_hub import hf_hub_download
    from project8_agent.longmemeval import DATASET, REVISION

    cache = project_dir / "data/cache"
    frozen = cache / "longmemeval-v2"
    if not (frozen / "selected.json").is_file():
        raw = hf_hub_download(DATASET, "longmemeval_s_cleaned.json", repo_type="dataset",
                              revision=REVISION, local_dir=cache / "benchmark-downloads")
        subprocess.run([sys.executable, str(project_root / "tools/prepare_longmemeval.py"),
                        "--input", raw, "--output", str(frozen),
                        "--evidence", str(cache / "longmemeval_preparation.json")],
                       check=True, timeout=180)
    prep = json.loads((cache / "longmemeval_preparation.json").read_text())
    selected = (frozen / "selected.json").read_bytes()
    if prep["selected_sha256"] != hashlib.sha256(selected).hexdigest() or len(json.loads(selected)) != 6:
        raise ValueError("LongMemEval selection differs from its frozen six-case manifest")
    dest = project_root / "data/cache/longmemeval-v2"
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "selected.json").write_bytes(selected)


if __name__ == "__main__":
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=int, choices=EXPECTED_FILES, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("retriever-refresh",))
    args = parser.parse_args()
    if args.benchmark and args.project != 2:
        parser.error("The retriever refresh applies to Project 2")
    with Path("/content/portfolio-inference-v2.lock").open("w") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args.project, args.cells, args.output, args.benchmark))
