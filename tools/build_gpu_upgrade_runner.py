"""Package reviewed sources into a transient jobs-only VS Code/Colab controller."""
from __future__ import annotations

import base64
import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT.parent
AGENTIC = JOBS / 'ai_agentic_attemptings'


def build(reviewed_existing_sha256=None):
    files = {}
    for name in ['02_rag_document_qa_system.ipynb', '03_llm_agent_system.ipynb']:
        files['artifacts/' + name] = ROOT / name
    for name in ['gpu_upgrade_runtime.py', 'project1_task_evaluation.py',
                 'agent_task_evaluation.py', 'requirements-upgrade-inference.in',
                 'requirements-upgrade-inference.lock.txt']:
        files['artifacts/tools/' + name] = ROOT / 'tools' / name
    for number in [5, 8]:
        project = AGENTIC / f'project{number}'
        paths = [*project.glob('*.ipynb'), project / 'pyproject.toml',
                 *project.glob('src/**/*.py'), *project.glob('tests/**/*.py'),
                 *project.glob('config/*.json')]
        if number == 5:
            paths.append(project / 'tools/generate_dataset.py')
        else:
            paths.extend(project / name for name in ['data/lifecycle_cases.json',
                         'data/locomo_selection.json', 'data/cache/locomo_subset.json',
                         'data/cache/locomo/LICENSE.txt', 'tools/generate_report.py',
                         'tools/analyze_saved_ablation.py'])
        for path in paths:
            files['agentic/' + str(path.relative_to(AGENTIC))] = path
    contents = {}
    for name, path in files.items():
        data = path.read_bytes()
        if path.suffix == '.ipynb':
            notebook = json.loads(data)
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    cell.update(outputs=[], execution_count=None)
            # Runtime connection metadata is not needed by the subprocess executor.
            notebook['metadata'] = {'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python 3'}}
            data = (json.dumps(notebook, indent=1) + '\n').encode()
        if re.search(rb'\b(?:hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})', data):
            raise ValueError(f'Credential-like text found in {name}; nothing staged')
        contents[name] = data
    contents['source_manifest.json'] = json.dumps({name: hashlib.sha256(data).hexdigest()
                                                  for name, data in contents.items()}, indent=2).encode()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    encoded = base64.b64encode(payload.getvalue()).decode()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    setup = '''import os, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parent
venv = root / 'venv'
bootstrap = '/content/project1_upgrade_runtime/bootstrap'
env = dict(os.environ, PYTHONPATH=bootstrap, UV_CACHE_DIR='/content/project1_upgrade_runtime/uv-cache', UV_HTTP_RETRIES='0')
def uv(*args):
    subprocess.run([sys.executable, '-m', 'uv', *args], env=env, check=True, timeout=540)
if not (venv / 'bin/python').is_file():
    uv('venv', '--python', '3.11', '--seed', str(venv))
uv('pip', 'install', '--python', str(venv / 'bin/python'), '-r', str(root / 'artifacts/tools/requirements-upgrade-inference.lock.txt'))
uv('pip', 'check', '--python', str(venv / 'bin/python'))
subprocess.run([str(venv / 'bin/python'), '-c', "import torch,crewai,transformers; from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn; print({'gpu':torch.cuda.get_device_name(0),'transformers':transformers.__version__})"], check=True, timeout=120)
'''
    source = f'''import base64, io, json, os, pathlib, subprocess, sys, zipfile
GPU_WORKSPACE = pathlib.Path('/content/portfolio_upgrade_{stamp}')
if 'gpu_suite_process' in globals() and gpu_suite_process.poll() is None:
    raise RuntimeError('A suite is already waiting/running; inspect its progress instead')
GPU_WORKSPACE.mkdir(parents=True, exist_ok=False)
bundle = base64.b64decode({encoded!r})
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    for member in archive.infolist():
        target = GPU_WORKSPACE / member.filename
        if not target.resolve().is_relative_to(GPU_WORKSPACE.resolve()):
            raise PermissionError('Unsafe package path')
    archive.extractall(GPU_WORKSPACE)
(GPU_WORKSPACE / 'setup_inference.py').write_text({setup!r})
if not os.environ.get('HF_TOKEN'):
    raise RuntimeError('Use the same authenticated Project 1 kernel; do not paste a key into code')
gpu_suite_log_path = pathlib.Path('/content/drive/MyDrive/projects/MyProject1/upgrade_gpu_suite.log')
gpu_suite_log = gpu_suite_log_path.open('a', buffering=1)
gpu_suite_process = subprocess.Popen([sys.executable, '-u', str(GPU_WORKSPACE / 'artifacts/tools/gpu_upgrade_runtime.py'), '--workspace', str(GPU_WORKSPACE)], stdout=gpu_suite_log, stderr=subprocess.STDOUT, env=dict(os.environ), start_new_session=True)
print({{'pid': gpu_suite_process.pid, 'status': 'waiting for Project 1 to finish; no concurrent GPU jobs'}})
'''
    cells = [
        {'cell_type': 'markdown', 'id': 'gpu-suite-description', 'metadata': {}, 'source': [
            '# Upgraded GPU Evaluation Queue\n',
            'Use the same authenticated AI-project kernel. This waits for Project 1 training/NLL,\n',
            'then runs task smoke, Project 2/3, and Project 5/8 evaluations sequentially.\n',
            'No paid API calls or generated-code execution. Exact dependency resolution is saved.\n']},
        {'cell_type': 'code', 'id': 'gpu-suite-launch', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': source.splitlines(keepends=True)},
        {'cell_type': 'code', 'id': 'gpu-suite-progress', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': ["print('Suite exit:', gpu_suite_process.poll())\n",
                                  "print(gpu_suite_log_path.read_text()[-7000:])\n"]},
    ]
    target = JOBS / '.upgrade_runtime/gpu_upgrade_suite.ipynb'
    target.parent.mkdir(exist_ok=True)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python 3 (ipykernel)'}}, 'cells': cells}
    if target.exists():
        existing = json.loads(target.read_text())
        if (hashlib.sha256(target.read_bytes()).hexdigest() != reviewed_existing_sha256
                or any(cell.get('outputs') or cell.get('execution_count') is not None
                       for cell in existing['cells'] if cell['cell_type'] == 'code')):
            raise FileExistsError('Preserving existing controller. Review it before regenerating.')
    target.write_text(json.dumps(notebook, indent=1) + '\n')
    print({'controller': str(target), 'files': len(files), 'compressed_bytes': len(payload.getvalue())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviewed-existing-sha256')
    build(parser.parse_args().reviewed_existing_sha256)
