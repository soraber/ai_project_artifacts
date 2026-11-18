"""Print an append-only patch for launching an original notebook on Colab."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    2: ("02_rag_document_qa_system.ipynb", (3, 4, 6, 7, 9, 11, 13, 15, 17),
        ("conditional_hyde.py",)),
    3: ("03_llm_agent_system.ipynb", (3, 4, 6, 8, 10, 12, 14, 16),
        ("project3_mcp.py", "project3_mcp_check.py")),
    5: ("project5/project5_governed_text_to_sql.ipynb", (3, 4, 5, 6, 7, 9), ()),
    8: ("project8/project8_long_term_memory.ipynb", (3, 13), ()),
}
AGENTIC = ROOT.parent / "ai_agentic_attemptings"
COMMON = ("requirements-inference-v2.hashes.txt", "training_environment.py", "run_inference_cells_v2.py")
LAUNCH = '''from datetime import datetime, timezone
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from google.colab import drive

mount = Path("/content/drive")
if not mount.is_mount():
    drive.mount(str(mount), timeout_ms=180000)
if not mount.is_mount():
    raise RuntimeError("Connect this notebook to the mounted MyPro runtime")
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
if "A100" not in gpu:
    raise RuntimeError(f"Expected the A100, found {gpu}")
if globals().get("inference_worker") is not None and inference_worker.poll() is None:
    raise RuntimeError("This notebook already has an active evaluation worker")
if subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
    raise RuntimeError("Another process is using the GPU")

runtime = Path("/content/portfolio_inference_v2")
bundle_dir = runtime / f"project{PROJECT}" / BUNDLE_SHA256
tools_dir = bundle_dir / "tools"
tools_dir.mkdir(parents=True, exist_ok=True)
bundle = gzip.decompress(base64.b64decode(BUNDLE))
if hashlib.sha256(bundle).hexdigest() != BUNDLE_SHA256:
    raise RuntimeError("Notebook package hash mismatch")
for name, contents in json.loads(bundle).items():
    destination = (tools_dir / name).resolve()
    if not destination.is_relative_to(tools_dir):
        raise ValueError("Unexpected path in notebook package")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents)

bootstrap = runtime / "bootstrap"
subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(bootstrap),
                "--no-deps", "uv==0.12.17"], check=True, timeout=180)
uv_cache = Path("/content/project1_training_v2/uv-cache")
if not uv_cache.is_dir():
    uv_cache = runtime / "uv-cache"
uv_env = dict(os.environ, PYTHONPATH=str(bootstrap), UV_CACHE_DIR=str(uv_cache),
              UV_HTTP_RETRIES="1")

def uv(*arguments):
    subprocess.run([sys.executable, "-m", "uv", *map(str, arguments)], env=uv_env,
                   check=True, timeout=600)

venv = runtime / "venv"
python = venv / "bin/python"
lock = tools_dir / "requirements-inference-v2.hashes.txt"
fingerprint = hashlib.sha256(lock.read_bytes()).hexdigest()
marker = venv / ".portfolio-lock"
if not python.is_file():
    uv("venv", "--python", "3.11.17", venv)
if not marker.is_file() or marker.read_text() != fingerprint:
    uv("pip", "install", "--python", python, "--require-hashes", "-r", lock)
    uv("pip", "check", "--python", python)
    marker.write_text(fingerprint)

project_root = mount / f"MyDrive/projects/MyProject{PROJECT}"
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
run_parent = project_root / "outputs/runs/upgrade-v2"
run_parent.mkdir(parents=True, exist_ok=True)
run_dir = run_parent / run_id
inference_log = run_parent / f"{run_id}.log"
inference_env = {
    k: v for k, v in os.environ.items()
    if not any(word in k.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
}
inference_env.update(
    HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_XET="1",
    HF_HUB_DISABLE_TELEMETRY="1", PYTHONUNBUFFERED="1", OTEL_SDK_DISABLED="true",
)
with inference_log.open("x") as log:
    benchmark_args = ["--benchmark", BENCHMARK] if BENCHMARK else []
    inference_worker = subprocess.Popen(
        [str(python), str(tools_dir / "run_inference_cells_v2.py"), "--project",
         str(PROJECT), "--cells", str(tools_dir / "cells.json"), "--output", str(run_dir), *benchmark_args],
        env=inference_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
print({"project": PROJECT, "run_id": run_id, "pid": inference_worker.pid})
'''

STATUS = '''from pathlib import Path
import json

manifest_path = run_dir / "run_manifest.json"
if manifest_path.is_file():
    print(json.dumps(json.loads(manifest_path.read_text()), indent=2))
print("Worker exit:", inference_worker.poll())
if inference_worker.poll() not in (None, 0):
    with inference_log.open("rb") as log:
        log.seek(max(0, inference_log.stat().st_size - 4096))
        print(log.read().decode(errors="replace"))
'''


def notebook_patch(project, refresh=False, benchmark=None):
    name, numbers, extra = PROJECTS[project]
    prefix = "mypro-inference"
    if benchmark == "retriever-refresh":
        numbers = (3, 4, 6, 7, 9, 11, 15)
        extra = ("project2_retriever_refresh.py", "evaluation_upgrade.py")
        prefix = "mypro-retriever-refresh"
    repo = AGENTIC if project in (5, 8) else ROOT
    notebook = json.loads((repo / name).read_text())
    existing = next((c for c in notebook["cells"] if c.get("id") == f"{prefix}-launch"), None)
    if existing and not refresh:
        raise ValueError("The original notebook already has a generated launcher")
    if refresh and not existing:
        raise ValueError("There is no launcher to refresh")
    selected = []
    for number in numbers:
        cell = notebook["cells"][number - 1]
        if cell["cell_type"] != "code":
            raise ValueError(f"Expected code in cell {number}")
        source = "".join(cell["source"])
        if project in (5, 8) and number == 3:
            backend = [node.value for node in ast.parse(source).body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "EVAL_BACKEND"
                               for target in node.targets)]
            if len(backend) != 1 or ast.literal_eval(backend[0]) != "local_gpu":
                raise ValueError("Preserve the existing backend selection. This launcher needs local_gpu.")
        compile(source, name, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)
        selected.append({"number": number, "id": cell["id"], "source": source})
    payload = {name: (ROOT / "tools" / name).read_text() for name in COMMON + extra}
    if project in (5, 8):
        root = repo / f"project{project}"
        files = list((root / "src").rglob("*.py")) + [root / "config/default.json"]
        files.append(repo / "tools/run_manifest.py")
        if project == 5:
            files.append(root / "tools/generate_dataset.py")
        else:
            files.append(root / "tools/prepare_longmemeval.py")
        for path in files:
            payload[f"source/{path.relative_to(repo).as_posix()}"] = path.read_text()
    payload["cells.json"] = json.dumps(selected, indent=2)
    bundle = json.dumps(payload, sort_keys=True).encode()
    data = base64.b64encode(gzip.compress(bundle, mtime=0)).decode()
    launch = (f"PROJECT = {project}\nBUNDLE_SHA256 = {hashlib.sha256(bundle).hexdigest()!r}\n"
              f"BUNDLE = {data!r}\nBENCHMARK = {benchmark!r}\n" + LAUNCH)
    status = STATUS
    if benchmark:
        status += '''elif inference_worker.poll() == 0:
    summary = json.loads((run_dir / "retriever_refresh_summary.json").read_text())
    print("Retriever refresh:", json.dumps(summary))
'''
    new_cells = []
    for identifier, source in (("launch", launch), ("status", status)):
        compile(source, identifier, "exec")
        new_cells.append(dict(cell_type="code", execution_count=None, id=f"{prefix}-{identifier}",
                              metadata={}, outputs=[], source=source.splitlines(keepends=True)))
    if refresh:
        old = json.dumps(existing["source"], indent=1).splitlines()[1:-1]
        new = json.dumps(new_cells[0]["source"], indent=1).splitlines()[1:-1]
        print(f"*** Begin Patch\n*** Update File: {repo.name}/{name}\n@@")
        print("\n".join("-   " + line for line in old))
        print("\n".join("+   " + line for line in new))
        print("*** End Patch")
        return
    lines = json.dumps(new_cells, indent=1).splitlines()[1:-1]
    print(f"*** Begin Patch\n*** Update File: {repo.name}/{name}\n@@")
    print("-  }\n+  },")
    print("\n".join("+ " + line for line in lines))
    print('  ],\n  "metadata": {\n*** End Patch')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=int, choices=PROJECTS)
    parser.add_argument("--refresh", action="store_true", help="Replace only the generated launch cell source")
    parser.add_argument("--retriever-refresh", action="store_true", help="Add the Project 2 retrieval-only comparison")
    args = parser.parse_args()
    if args.retriever_refresh and args.project != 2:
        parser.error("The retriever refresh applies to Project 2")
    notebook_patch(args.project, args.refresh, "retriever-refresh" if args.retriever_refresh else None)
