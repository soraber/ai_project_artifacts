"""Matched router arms; authored smoke checks are not semantic quality ratings."""
import random
import statistics
import time


def evaluate_router_arms(cases, execute, usage, seed=2026):
    records = []
    rng = random.Random(seed)
    for case in cases:
        arms = [False, True]
        rng.shuffle(arms)
        for constrained in arms:
            started = time.perf_counter()
            usage_start = len(usage)
            error = None
            try:
                result = execute(case['task'], allow_write=case.get('allow_write', False),
                                 constrained=constrained)
            except Exception as exc:
                error = type(exc).__name__
                result = {'planned_tool': None, 'planner': {}, 'answer': '',
                          'observation': {}, 'status': 'error', 'trace': []}
            observation = result['observation']
            answer = result['answer'].lower()
            planner = result['planner']
            fallback = bool(planner.get('fallback_used', False))
            routed = result['planned_tool'] == case['tool']
            if 'total' in case:
                success = abs(observation.get('total_budget', -1) - case['total']) < 1e-6
            elif case.get('saved'):
                success = observation.get('status') == 'saved'
            elif case.get('blocked'):
                success = observation.get('status') == 'blocked'
            elif case.get('unsupported'):
                success = 'insufficient' in answer
            elif case['tool'] in ['abstain', 'request_clarification']:
                success = routed and not result['trace']
            else:
                success = all(term in answer for term in case.get('terms', []))
            calls = usage[usage_start:]
            records.append({
                'id': case['id'], 'system': 'constrained' if constrained else 'prompt_only',
                'task': case['task'], 'expected_tool': case['tool'],
                'planned_tool': result['planned_tool'], 'routing_correct': routed,
                'model_routing_correct': routed and not fallback and error is None,
                'fallback_used': fallback, 'planner_valid_json': planner.get('valid_json', False),
                'arguments_valid': planner.get('arguments_valid', False),
                'task_success': bool(success and error is None),
                'safety_case': bool(case.get('blocked')),
                'safety_blocked': observation.get('status') == 'blocked',
                'latency_seconds': time.perf_counter() - started,
                'input_tokens': sum(call['input_tokens'] for call in calls),
                'output_tokens': sum(call['output_tokens'] for call in calls),
                'calls': len(calls), 'error': error,
                'answer': result['answer'], 'planner': planner,
                'observation': observation, 'trace': result['trace'],
            })
    summaries = {}
    for system in ['prompt_only', 'constrained']:
        rows = [row for row in records if row['system'] == system]
        latency = sorted(row['latency_seconds'] for row in rows)
        safety = [row['safety_blocked'] for row in rows if row['safety_case']]
        summaries[system] = {
            'test_cases': len(rows), 'errors': sum(row['error'] is not None for row in rows),
            **{name + '_rate': statistics.fmean(row[name] for row in rows)
               for name in ['routing_correct', 'model_routing_correct', 'fallback_used',
                            'planner_valid_json', 'arguments_valid', 'task_success']},
            'safety_block_rate': statistics.fmean(safety) if safety else None,
            'latency_p50_seconds': statistics.median(latency),
            'latency_p95_seconds': latency[min(len(latency)-1, int(.95*len(latency)))],
            'input_tokens': sum(row['input_tokens'] for row in rows),
            'output_tokens': sum(row['output_tokens'] for row in rows),
        }
    return records, summaries
