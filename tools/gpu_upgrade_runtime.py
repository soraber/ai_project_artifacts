"""Sequential isolated GPU evaluations after the approved Project 1 retraining.

No API key is persisted. Only the user-approved MyProject1 Drive subtree is used.
Historical repo outputs are not read as new results or overwritten.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path


PERSISTENT_ROOT = Path('/content/drive/MyDrive/projects/MyProject1')
P1_PYTHON = Path('/content/project1_upgrade_runtime/venv/bin/python')


def utc_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def read_run_manifest(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.2)


def bounded_process(command, env, timeout):
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


class Tee(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)


async def execute_cells(path, selected, namespace, evidence_dir):
    notebook = json.loads(path.read_text())
    available = {cell.get('id') for cell in notebook['cells']}
    if set(selected) - available:
        raise ValueError('Selected cell is missing from the staged notebook')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell.update(outputs=[], execution_count=None)
    count = 0
    for cell in notebook['cells']:
        cell_id = cell.get('id')
        if cell_id not in selected:
            continue
        count += 1
        source = ''.join(cell['source'])
        if cell_id in ('P05-C02', 'P08-C02'):
            modes = [node.value.value for node in ast.parse(source).body
                     if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                     and any(isinstance(target, ast.Name) and target.id == 'EVAL_BACKEND' for target in node.targets)]
            if modes != ['local_gpu']:
                raise RuntimeError('Preserving the edited mode: this queue only permits local_gpu, not paid APIs')
        print(f'Executing {path.name} cell {cell_id}', flush=True)
        captured = Tee()
        try:
            with contextlib.redirect_stdout(captured):
                code = compile(source, f'{path.name}:{cell_id}', 'exec',
                               flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)
                result = eval(code, namespace)
                if inspect.isawaitable(result):
                    await result
        finally:
            cell.update(execution_count=count, outputs=[{'output_type': 'stream', 'name': 'stdout',
                                                        'text': captured.getvalue().splitlines(keepends=True)}])
            (evidence_dir / 'executed_notebook.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')


async def evaluate_stage(stage, workspace, evidence_dir):
    import torch
    evidence_dir.mkdir(parents=True, exist_ok=False)
    module = types.ModuleType('_portfolio_' + stage)
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace['display'] = lambda value: print(str(value)[:6000])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    manifest = {'stage': stage, 'started_utc': utc_stamp(), 'status': 'running',
                'gpu': torch.cuda.get_device_name(0), 'python': sys.version.split()[0],
                'seed_note': 'Notebook seeds retained; authored checks and model scores are distinct.'}
    try:
        if stage == 'project2':
            path = workspace / 'artifacts/02_rag_document_qa_system.ipynb'
            selected = ['3d8e1486', '33724c09', '724765b3', '031a5acb', '814ef015',
                        'bdf0279c', '01d9a1ca', '45f0634e']
        elif stage == 'project3':
            path = workspace / 'artifacts/03_llm_agent_system.ipynb'
            selected = ['87bbb7e4', None, 'c576a49d', '3396dd38', 'a24313e8', 'b3a5ffa9']
            # Discover the tool-definition cell by its already-reviewed source signature.
            nb = json.loads(path.read_text())
            selected[1] = next(cell['id'] for cell in nb['cells']
                               if 'def save_workspace_report(' in ''.join(cell['source']))
        else:
            number = stage.removeprefix('project')
            root = workspace / 'agentic' / stage
            path = next(root.glob('*.ipynb'))
            namespace.update(PROJECT_ROOT=root)
            os.chdir(root)
            sys.path.insert(0, str(root / 'src'))
            # Notebook dataset/test subprocesses need the same package lookup path.
            os.environ['PYTHONPATH'] = str(root / 'src')
            selected = [f'P0{number}-C{index:02d}' for index in range(2, 9)]
            if stage == 'project8':
                # The original pinned public subset is packaged, not re-downloaded.
                selected.remove('P08-C03')
                namespace['subset_path'] = root / 'data/cache/locomo_subset.json'
        manifest['notebook_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        await execute_cells(path, selected, namespace, evidence_dir)
        source_output = Path(namespace.get('OUT_DIR') or namespace['result_dir'])
        if stage == 'project8':
            source_output = source_output.parent
        for path in source_output.rglob('*'):
            if path.is_file() and path.suffix in ['.json', '.jsonl', '.csv', '.md']:
                relative = path.relative_to(source_output)
                destination = evidence_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        manifest.update(seconds=time.perf_counter()-start,
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), finished_utc=utc_stamp())
        (evidence_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))


def queue(workspace, inference_python, resume_stages=None):
    alias = Path('/content/fine_tuning_pipeline')
    p1_run = alias.resolve(strict=True)
    if not p1_run.is_relative_to(PERSISTENT_ROOT):
        raise PermissionError('Project 1 run is outside the authorized Drive subtree')
    manifest_path = p1_run / 'run_manifest.json'
    deadline = time.monotonic() + 6 * 3600
    while True:
        manifest = read_run_manifest(manifest_path)
        if manifest['status'] == 'complete':
            break
        if manifest['status'] == 'failed':
            raise RuntimeError('Project 1 failed; review its log before starting further GPU work')
        if time.monotonic() >= deadline:
            raise TimeoutError('Project 1 wait exceeded six hours; no automatic retraining')
        time.sleep(60)
    suite = PERSISTENT_ROOT / 'upgrade_evidence' / utc_stamp()
    suite.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_DISABLE_XET='1', CREWAI_DISABLE_TELEMETRY='true',
               OTEL_SDK_DISABLED='true', AI_PROJECT_SKIP_INSTALL='1', PYTHONDONTWRITEBYTECODE='1')
    stages = []
    suite_manifest = {'status': 'running', 'stages': stages,
                      'source_manifest_sha256': hashlib.sha256((workspace / 'source_manifest.json').read_bytes()).hexdigest()}
    shutil.copyfile(workspace / 'source_manifest.json', suite / 'source_manifest.json')
    def save():
        (suite / 'suite_manifest.json').write_text(json.dumps(suite_manifest, indent=2))
    save()
    try:
        if resume_stages is None:
            bounded_process([str(P1_PYTHON), '-u', str(workspace / 'artifacts/tools/project1_task_evaluation.py'),
                '--revision', manifest['model_revision'], '--adapter', str(p1_run / 'outputs/llama3_lora_adapter'),
                '--output', str(suite / 'project1_agent_task_smoke.json')], env, 1200)
            stages.append({'stage': 'project1_task_smoke', 'status': 'complete'})
            save()
            bounded_process([sys.executable, str(workspace / 'setup_inference.py')], env, 600)
        else:
            if not inference_python.is_file():
                raise FileNotFoundError('Resume requires the already validated inference environment')
            bounded_process([str(inference_python), '-m', 'pip', 'check'], env, 120)
        freeze = subprocess.check_output([str(inference_python), '-m', 'pip', 'freeze'], text=True, env=env)
        (suite / 'inference_environment.lock.txt').write_text(freeze)
        for stage in (resume_stages if resume_stages is not None else ['project2', 'project3', 'project5', 'project8']):
            output = suite / stage
            try:
                bounded_process([str(inference_python), '-u', str(Path(__file__)), '--stage', stage,
                                 '--workspace', str(workspace), '--output', str(output)], env, 3600)
                stages.append({'stage': stage, 'status': 'complete'})
            except Exception as exc:
                stages.append({'stage': stage, 'status': 'failed', 'error_type': type(exc).__name__})
                # One failed execution is recorded; no blind retries or dependency repair loop.
            save()
        suite_manifest['status'] = 'complete_with_failures' if any(s['status'] == 'failed' for s in stages) else 'complete'
    except BaseException as exc:
        suite_manifest.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        save()
        print('Saved evaluation suite:', suite, flush=True)


if __name__ == '__main__':
    from security_preflight import require_gpu_clearance
    require_gpu_clearance()
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--stage', choices=['queue', 'project2', 'project3', 'project5', 'project8'], default='queue')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume-stages', nargs='+', choices=['project2', 'project3', 'project5', 'project8'])
    args = parser.parse_args()
    if args.stage == 'queue':
        import fcntl
        def terminate(signum, frame):
            raise KeyboardInterrupt('Queue termination requested')
        signal.signal(signal.SIGTERM, terminate)
        # A kernel reconnect must not create a second competing GPU queue.
        with Path('/content/portfolio_upgrade_queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue(args.workspace, args.workspace / 'venv/bin/python', args.resume_stages)
    else:
        asyncio.run(evaluate_stage(args.stage, args.workspace, args.output))
