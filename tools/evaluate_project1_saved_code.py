"""Score saved code responses only inside Project 6's validated Docker boundary."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from agent_task_evaluation import TASKS, validate_decision


def evaluate(source, sandbox_source, work_dir, output):
    if output.exists():
        raise FileExistsError('Preserving existing code-test evidence')
    sys.path.insert(0, str(sandbox_source.resolve()))
    from project6_agent.runner import run_tests, sandbox_preflight
    sandbox = sandbox_preflight('project6-test-runner:upgrade-v1')
    smoke = json.loads(source.read_text())
    task = next(task for task in TASKS if task['id'] == 'code')
    records = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for row in smoke['records']:
        if row['task_id'] != 'code':
            continue
        result = {'system': row['system'], 'task_id': 'code', 'execution_attempted': False,
                  'task_passed': False, 'reason': None}
        try:
            decision = validate_decision(row['response'])
            if decision['tool'] != 'python':
                raise ValueError('Expected the Python tool')
        except (ValueError, KeyError) as exc:
            result['reason'] = 'invalid_model_contract:' + type(exc).__name__
            records.append(result)
            continue
        with tempfile.TemporaryDirectory(prefix='project1-code-', dir=work_dir) as temporary:
            root = Path(temporary)
            (root / 'candidate.py').write_text(decision['arguments']['code'])
            (root / 'test_candidate.py').write_text(task['tests'])
            outcome = run_tests(root, ['test_candidate.py'], timeout_seconds=20,
                                backend='docker', image=sandbox['image_id'])
            result.update(execution_attempted=True, task_passed=outcome.passed,
                          test_outcome=outcome.model_dump())
        records.append(result)
    if {row['system'] for row in records} != {'base', 'tuned'}:
        raise ValueError('Missing base/tuned saved code response')
    result = {'source_smoke_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'sandbox': sandbox, 'records': records,
              'runner_sha256': hashlib.sha256((sandbox_source / 'project6_agent/runner.py').read_bytes()).hexdigest(),
              'authored_tests_sha256': hashlib.sha256(task['tests'].encode()).hexdigest(),
              'limitations': 'One authored add-function task with two assertions, scored from frozen outputs. '
                             'Invalid tool contracts count as failed tasks without executing text. '
                             'No additional model calls or hidden-test feedback to the model.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sandbox-source', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source, args.sandbox_source, args.work_dir, args.output), indent=2))
