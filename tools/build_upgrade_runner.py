"""Package reviewed local sources into an ephemeral, standalone Colab control notebook."""
import base64
import json
from pathlib import Path
from security_preflight import require_gpu_clearance

require_gpu_clearance()

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "01_llm_fine_tuning_pipeline.ipynb").read_text())
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell.update(outputs=[], execution_count=None)
payload = base64.b64encode(json.dumps(notebook).encode()).decode()
runner = base64.b64encode((ROOT / "tools/project1_upgrade_runtime.py").read_bytes()).decode()


def cell(identifier, source):
    return {"cell_type": "code", "id": identifier, "source": source.splitlines(keepends=True),
            "execution_count": None, "metadata": {}, "outputs": []}


cells = [cell("isolated-setup", '''import subprocess, sys, os, pathlib, base64, time
RUNTIME = pathlib.Path('/content/project1_upgrade_runtime')
RUNTIME.mkdir(exist_ok=True)
VENV = RUNTIME / 'venv'
UV_TARGET = RUNTIME / 'bootstrap'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(UV_TARGET), '--no-deps', 'uv'], check=True, timeout=180)
uv_env = dict(os.environ, PYTHONPATH=str(UV_TARGET), UV_CACHE_DIR=str(RUNTIME/'uv-cache'), UV_HTTP_RETRIES='0')
def uv(*args, timeout=600):
    subprocess.run([sys.executable, '-m', 'uv', *args], env=uv_env, check=True, timeout=timeout)
uv('venv', '--python', '3.11', '--seed', str(VENV))
PYTHON = VENV / 'bin/python'
uv('pip', 'install', '--python', str(PYTHON), 'llamafactory==0.9.5', 'bitsandbytes', 'beautifulsoup4', 'ipykernel', timeout=600)
uv('pip', 'check', '--python', str(PYTHON))
subprocess.run([str(PYTHON), '-c', "import torch,peft,transformers; print({'cuda':torch.cuda.is_available(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0)})"], check=True, timeout=120)
print('Isolated environment ready; Colab base packages were not changed')
'''), cell("secure-hf-auth", '''import getpass, os, subprocess
if not os.environ.get('HF_TOKEN'):
    token = getpass.getpass('Hugging Face read token (hidden; never paste into chat): ')
    if not token: raise RuntimeError('HF token is required for the gated Llama baseline')
    os.environ['HF_TOKEN'] = token
    del token
subprocess.run([str(PYTHON), '-c', "import os; from huggingface_hub import HfApi; info=HfApi(token=os.environ['HF_TOKEN']).model_info('meta-llama/Meta-Llama-3-8B-Instruct'); print('Gated model access verified:',info.sha)"], check=True, timeout=60)
'''), cell("stage-reviewed-sources", f'''(RUNTIME/'project1.ipynb').write_bytes(base64.b64decode({payload!r}))
(RUNTIME/'run.py').write_bytes(base64.b64decode({runner!r}))
print('Reviewed notebook and runner staged; dataset preparation source preserved')
'''), cell("train-and-evaluate", '''from google.colab import drive
drive.mount('/content/drive', timeout_ms=180000)
log_path = pathlib.Path('/content/drive/MyDrive/projects/MyProject1/upgrade_training.log')
log_path.parent.mkdir(parents=True, exist_ok=True)
if 'training_process' in globals() and training_process.poll() is None:
    raise RuntimeError('Training is already running; inspect the progress cell')
training_log = log_path.open('a', buffering=1)
training_process = subprocess.Popen([str(PYTHON), '-u', str(RUNTIME/'run.py'), '--notebook', str(RUNTIME/'project1.ipynb')],
    stdout=training_log, stderr=subprocess.STDOUT, env=dict(os.environ, HF_HUB_DISABLE_XET='1'), start_new_session=True)
print({'pid': training_process.pid, 'log': str(log_path), 'status': 'started, not yet complete'})
'''), cell("training-progress", '''print('Process exit:', training_process.poll())
print(log_path.read_text()[-9000:])
''')]
output = ROOT.parent / ".upgrade_runtime/project1_upgrade_run.ipynb"
output.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n")
print(output)
