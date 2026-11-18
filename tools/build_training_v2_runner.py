"""Build a separate controller; never overwrite the historical GPU notebooks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    payload = {}
    for name in ['project1_training_v2.py', 'training_environment.py', 'gpu_training_smoke_v2.py',
                 'requirements-training-v2.hashes.txt']:
        content = (ROOT/'tools'/name).read_bytes()
        payload[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'data': base64.b64encode(content).decode()}
    source1 = '''import subprocess, sys, os, pathlib, base64, hashlib
RUNTIME = pathlib.Path('/content/project1_training_v2')
RUNTIME.mkdir(exist_ok=True)
''' + f'payload = {payload!r}\n' + '''for name, item in payload.items():
    content = base64.b64decode(item['data'])
    assert hashlib.sha256(content).hexdigest() == item['sha256']
    (RUNTIME/name).write_bytes(content)
BOOTSTRAP = RUNTIME/'bootstrap'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(BOOTSTRAP), '--no-deps', 'uv==0.12.17'],
               check=True, timeout=180)
uv_env = dict(os.environ, PYTHONPATH=str(BOOTSTRAP), UV_CACHE_DIR=str(RUNTIME/'uv-cache'), UV_HTTP_RETRIES='0')
def uv(*args):
    subprocess.run([sys.executable, '-m', 'uv', *args], env=uv_env, check=True, timeout=600)
VENV = RUNTIME/'venv'
PYTHON = VENV/'bin/python'
if not PYTHON.exists():
    uv('venv', '--python', '3.11.17', str(VENV))
uv('pip', 'install', '--python', str(PYTHON), '--require-hashes', '-r', str(RUNTIME/'requirements-training-v2.hashes.txt'))
uv('pip', 'check', '--python', str(PYTHON))
subprocess.run([str(PYTHON), '-c', "from training_environment import validate; print(validate('requirements-training-v2.hashes.txt'))"],
               cwd=RUNTIME, check=True, timeout=60)
print('Isolated v2 environment ready; no shared packages or old runtime were replaced')
'''
    source2 = '''from google.colab import drive
drive.mount('/content/drive', timeout_ms=180000)
PERSIST = pathlib.Path('/content/drive/MyDrive/projects/MyProject1')
if not pathlib.Path('/content/drive/MyDrive').is_dir():
    raise RuntimeError('Drive must be mounted before writing persistent evidence')
SMOKE_ROOT = PERSIST/'runs/upgrade-v2/gpu_integration'
# Remove model/API credentials: this authored random model needs no remote weights.
safe_env = {k:v for k,v in os.environ.items() if not any(word in k.upper() for word in ('TOKEN', 'SECRET', 'API_KEY', 'PASSWORD'))}
safe_env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
subprocess.run([str(PYTHON), str(RUNTIME/'gpu_training_smoke_v2.py'), '--output-root', str(SMOKE_ROOT)],
               cwd=RUNTIME, env=safe_env, check=True, timeout=600)
'''
    cells = [{'cell_type':'markdown', 'id':'v2-scope', 'metadata':{}, 'source':[
        '# Project 1: Patched Trainer GPU Smoke\n',
        'This performs two steps on an authored tiny random model. It needs no HF token, does not download a dataset, and does not retrain the historical adapter.\n'] }]
    for identifier, source in [('v2-install', source1), ('v2-nf4-smoke', source2)]:
        cells.append({'cell_type':'code','id':identifier,'metadata':{},'source':source.splitlines(keepends=True),
                      'outputs':[],'execution_count':None})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump({'cells': cells, 'nbformat':4, 'nbformat_minor':5,
                   'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}}, handle, indent=1)
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT.parent/'.upgrade_runtime/project1_training_v2_smoke.ipynb')
    build(parser.parse_args().output)
