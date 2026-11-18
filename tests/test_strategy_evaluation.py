import asyncio
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('strategies', Path(__file__).parents[1] / 'tools/project3_strategy_evaluation.py')
strategies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategies)


def test_failure_closes_budget_before_framework_retry():
    def fail(*args, **kwargs):
        raise ValueError('fixture')
    budget = strategies.StrategyBudget(fail, [])
    with pytest.raises(ValueError):
        budget.call('source', 300)
    with pytest.raises(RuntimeError, match='exhausted'):
        budget.call('retry', 300)


def test_multitask_outputs_are_blinded_and_share_caps(monkeypatch):
    usage = []
    def generate(prompt, max_new_tokens, input_token_cap):
        assert input_token_cap >= 100
        usage.append({'input_tokens': 100, 'output_tokens': min(20, max_new_tokens)})
        return 'Proposed plan with source citations.'
    async def fake_roles(brief, budget, model):
        return budget.call(brief, 750)
    monkeypatch.setattr(strategies, 'role_workflow', fake_roles)
    cases = strategies.strategy_cases({'product.md': 'RelayDesk', 'constraints.md': 'original'})
    rows = asyncio.run(strategies.evaluate_strategies(cases, generate, usage, 'fixture'))
    assert len(rows) == 9 and not any(row['error'] for row in rows)
    assert {row['output_token_cap'] for row in rows} == {1950}
    assert {row['input_token_cap'] for row in rows} == {16000}
    pack, key = strategies.strategy_review_pack(cases, rows)
    assert len(pack['candidates']) == 9 and len(key) == 9
    assert all('system' not in row and row['ratings'] is None for row in pack['candidates'])


def test_real_langgraph_workflow_without_api_calls():
    usage = []
    def authored_reply(prompt, max_new_tokens, input_token_cap):
        usage.append({'input_tokens': 100, 'output_tokens': 20})
        return 'Final Answer: Proposed opt-in pilot; human approval is required. [constraints.md]'
    budget = strategies.StrategyBudget(authored_reply, usage)
    result = asyncio.run(strategies.role_workflow('[constraints.md] Human approval required.', budget, 'authored-fixture'))
    assert 'human approval' in result
    assert len(usage) == 4
    assert [row['stage'] for row in budget.trace] == ['research', 'plan', 'review', 'edit']
    assert all(row['status'] == 'complete' for row in budget.trace)


@pytest.mark.parametrize('usage_record', [None, {'input_tokens': -1, 'output_tokens': 1},
    {'input_tokens': 1, 'output_tokens': 301}, {'input_tokens': 16001, 'output_tokens': 1}])
def test_missing_or_invalid_usage_closes_budget(usage_record):
    usage = []
    def generate(*args, **kwargs):
        if usage_record is not None:
            usage.append(usage_record)
        return 'fixture'
    budget = strategies.StrategyBudget(generate, usage)
    with pytest.raises(RuntimeError):
        budget.call('test', 300)
    with pytest.raises(RuntimeError, match='exhausted'):
        budget.call('retry', 300)


def test_graph_carries_context_and_stops_without_retry_on_error():
    usage, prompts = [], []
    def generate(prompt, **kwargs):
        prompts.append(prompt)
        if len(prompts) == 3:
            assert 'reply-1' in prompt and 'reply-2' in prompt
            raise RuntimeError('authored review failure')
        usage.append({'input_tokens': 100, 'output_tokens': 10})
        return f'reply-{len(prompts)}'
    budget = strategies.StrategyBudget(generate, usage)
    with pytest.raises(RuntimeError, match='authored review'):
        asyncio.run(strategies.role_workflow('source-fact', budget, 'fixture'))
    assert len(prompts) == 3 and all('source-fact' in p for p in prompts)
    assert budget.trace[-1]['status'] == 'failed'
