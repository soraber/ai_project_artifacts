import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lmformatenforcer import JsonSchemaParser
from pydantic import BaseModel, Field

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('router', ROOT / 'tools/project3_router.py')
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)


class BudgetInput(BaseModel):
    channel: str
    weeks: int = Field(ge=1, le=12)
    weekly_budget: float = Field(gt=0, le=12000)


class SaveInput(BaseModel):
    filename: str
    content: str
    confirm_write: bool = False


TOOLS = {'calculate_campaign_budget': SimpleNamespace(args_schema=BudgetInput),
         'save_workspace_report': SimpleNamespace(args_schema=SaveInput)}


def test_constrained_and_control_share_prompt_budget_and_semantic_checks():
    calls = []
    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return '{"tool":"request_clarification","arguments":{"reason":"What weekly spend?"}}'
    for constrained in [False, True]:
        decision, trace = router.plan_with_schema('Calculate an email budget', TOOLS, generate, constrained)
        assert decision.tool == 'request_clarification' and not trace['fallback_used']
    assert calls[0][0] == calls[1][0]
    assert calls[0][1]['max_new_tokens'] == calls[1][1]['max_new_tokens'] == 192
    assert calls[0][1]['json_schema'] is None
    assert calls[1][1]['json_schema'] == router.decision_schema(TOOLS)


@pytest.mark.parametrize('arguments', [
    {'channel': 'email', 'weeks': True, 'weekly_budget': 500},
    {'channel': 'email', 'weeks': 2, 'weekly_budget': float('nan')},
    {'channel': 'email', 'weeks': 2, 'weekly_budget': 500, 'extra': 'ignore'},
    {'channel': 'email', 'weeks': 2},
])
def test_invalid_semantics_do_not_become_default_spend(arguments):
    raw = json.dumps({'tool': 'calculate_campaign_budget', 'arguments': arguments})
    decision, trace = router.plan_with_schema('Calculate budget', TOOLS, lambda *a, **k: raw)
    assert decision.tool == 'request_clarification'
    assert trace['fallback_used'] and trace['attempts'] == 2 and not trace['arguments_valid']


def test_model_cannot_grant_write_permission():
    with pytest.raises(ValueError, match='approval'):
        router.validate_decision(json.dumps({'tool': 'save_workspace_report', 'arguments': {
            'filename': 'x.md', 'content': 'x', 'confirm_write': True}}), TOOLS)


def test_real_grammar_parser_accepts_typed_branch_and_rejects_unknown_tool():
    schema = router.decision_schema(TOOLS)
    raw = json.dumps({'tool': 'calculate_campaign_budget', 'arguments': {
        'channel': 'email', 'weeks': 2, 'weekly_budget': 500}})
    parser = JsonSchemaParser(schema)
    for character in raw:
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()
    parser = JsonSchemaParser(schema)
    invalid = '{"tool": "shell"'
    accepted = True
    for character in invalid:
        if character not in parser.get_allowed_characters():
            accepted = False
            break
        parser = parser.add_character(character)
    assert not accepted
