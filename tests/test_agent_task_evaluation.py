import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("task_eval", Path(__file__).parents[1] / "tools/agent_task_evaluation.py")
task_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_eval)


@pytest.mark.parametrize("raw", [
    '{"tool":"calculator","arguments":{"operation":"add","a":NaN,"b":2}}',
    '{"tool":"calculator","arguments":{"operation":"add","a":true,"b":2}}',
    '{"tool":"shell","arguments":{}}',
    '{"tool":"abstain","arguments":{"reason":"extra"}}',
])
def test_rejects_invalid_contracts(raw):
    with pytest.raises(ValueError): task_eval.validate_decision(raw)


def test_no_generated_code_runs_without_isolated_executor():
    calls = []
    def generate(prompt, enabled):
        calls.append(enabled)
        return json.dumps({"tool": "python", "arguments": {"code": "raise RuntimeError('never run')"}})
    result = task_eval.compare_tasks(generate)
    assert len(calls) == 12
    assert result["metrics"]["base"]["code_tests_passed"] == {"mean": None, "count": 0}
    assert result["metrics"]["tuned"]["tool_correct"]["mean"] == 1/6
