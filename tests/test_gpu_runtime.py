import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('gpu_runtime', ROOT / 'tools/gpu_upgrade_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_runner_records_only_selected_cells_and_supports_await(tmp_path):
    path = tmp_path / 'notebook.ipynb'
    path.write_text(json.dumps({'cells': [
        {'id': 'a', 'cell_type': 'code', 'source': ['import asyncio\nawait asyncio.sleep(0)\nx=7\nprint(x)'],
         'outputs': [], 'execution_count': None},
        {'id': 'skip', 'cell_type': 'code', 'source': ['raise RuntimeError("not selected")'],
         'outputs': [{'old': 'result'}], 'execution_count': 5}]}))
    namespace = {}
    asyncio.run(runtime.execute_cells(path, ['a'], namespace, tmp_path))
    saved = json.loads((tmp_path / 'executed_notebook.ipynb').read_text())
    assert namespace['x'] == 7
    assert saved['cells'][1]['outputs'] == []
    assert saved['cells'][0]['outputs'][0]['text'] == ['7\n']


def test_queue_does_not_override_user_changed_api_mode(tmp_path):
    path = tmp_path / 'notebook.ipynb'
    path.write_text(json.dumps({'cells': [{'id': 'P05-C02', 'cell_type': 'code',
        'source': ['EVAL_BACKEND="openai"\nraise AssertionError("must not execute")']}]}))
    with pytest.raises(RuntimeError, match='only permits local_gpu'):
        asyncio.run(runtime.execute_cells(path, ['P05-C02'], {}, tmp_path))


def test_all_runtime_tools_compile_without_importing_gpu_libraries():
    for filename in ['gpu_upgrade_runtime.py', 'build_gpu_upgrade_runner.py',
                     'project1_upgrade_runtime.py', 'project1_task_evaluation.py']:
        compile((ROOT / 'tools' / filename).read_text(), filename, 'exec')


def test_notebook_annotations_do_not_inherit_runner_future_flags(tmp_path):
    path = tmp_path / 'typed.ipynb'
    source = ('from typing import Literal\nfrom pydantic import BaseModel\n'
              'class Decision(BaseModel):\n    tool: Literal["search", "abstain"]\n'
              'result = Decision.model_validate({"tool": "search"}).tool\n')
    path.write_text(json.dumps({'cells': [{'id': 'typed', 'cell_type': 'code', 'source': [source]}]}))
    namespace = {'__name__': '__main__'}
    asyncio.run(runtime.execute_cells(path, ['typed'], namespace, tmp_path))
    assert namespace['result'] == 'search'
