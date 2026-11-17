import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from automated_review import (GroundingRating, ReviewBudget, grounding_request,
    select_grounding_groups, summarize_grounding, validate_grounding, run_review,
    StrategyRating)


def test_budget_reserves_failure_and_prevents_group_overspend():
    budget = ReviewBudget(0.1, 2)
    request = {"text": "fixture"}
    amount = budget.reserve(request, GroundingRating)
    assert budget.charged_upper_bound == amount
    budget.settle(amount, SimpleNamespace(input_tokens=10, output_tokens=10))
    assert budget.actual_cost == pytest.approx(0.00024)
    budget.reserve(request, GroundingRating)
    assert not budget.can_afford([request], GroundingRating)
    with pytest.raises(RuntimeError, match="budget"):
        budget.reserve(request, GroundingRating)


def test_supported_claim_requires_exact_source_quote():
    entry = {"retrieved_evidence": {"one": "Boston is the venue."}}
    rating = GroundingRating(is_abstention=False, evidence_sufficient="yes", relevance=2,
        claims=[{"text": "Boston", "verdict": "supported", "evidence_id": "one",
                 "supporting_quote": "Paris", "reason": "fixture"}], reason="fixture")
    with pytest.raises(ValueError, match="quote"):
        validate_grounding(rating, entry)
    rating.claims[0].supporting_quote = "Boston"
    validate_grounding(rating, entry)
    rating.is_abstention = True
    with pytest.raises(ValueError, match="abstention"):
        validate_grounding(rating, entry)


def test_prompt_does_not_expose_system_identity():
    entry = {"system": "advanced", "review_id": "secret-key", "question": "Where?",
             "answer": "Boston", "retrieved_evidence": {"one": "Boston"}}
    request = grounding_request(entry)
    assert "system" not in request and "review_id" not in request


def test_sample_uses_all_arms_and_separates_absence_probes():
    entries, key, records = [], {}, []
    for index in range(5):
        for arm in ["dense", "hybrid"]:
            rid = f"{index}-{arm}"
            entries.append({"review_id": rid, "answer": "A", "question": "Q"})
            key[rid] = {"system": arm, "example_id": index}
            records.append({**key[rid], "answer": "A", "question": "Q", "stratum":
                            "authored_absence_probe" if index == 0 else "natural_qa"})
    groups = select_grounding_groups({"entries": entries}, key, records, natural_count=2)
    assert len(groups) == 3 and groups[0][0] == 0
    assert all(len(group) == 2 for _, group in groups)
    assert groups == select_grounding_groups({"entries": entries}, key, records, natural_count=2)
    entries[0]["answer"] = "changed"
    with pytest.raises(ValueError, match="match"):
        select_grounding_groups({"entries": entries}, key, records)


def test_abstention_and_uncertainty_do_not_inflate_claim_support():
    row = {"system": "dense", "stratum": "natural_qa", "status": "rated",
           "rating": {"is_abstention": False, "relevance": 1,
                      "claims": [{"verdict": "supported"}, {"verdict": "uncertain"}]}}
    score = summarize_grounding([row])[0]
    assert score["supported_claim_fraction"] == 0.5 and score["uncertain_claims"] == 1
    row["rating"].update(is_abstention=True, claims=[])
    assert summarize_grounding([row])[0]["supported_claim_fraction"] is None


def make_strategy_pack(path):
    path.mkdir()
    candidates = [{"candidate": f"c{i}", "text": "A proposed plan.",
                   "sources": {"brief.md": "Budget is undecided."}} for i in range(4)]
    (path / "blind_review.json").write_text(json.dumps({"requirements": "Do not invent inputs",
        "rubric": {"uncertainty_handling": "Request undecided inputs"}, "candidates": candidates}))
    (path / "review_key.json").write_text(json.dumps({f"c{i}": {"system": "BLINDED_ARM",
        "case_id": f"case{i}"} for i in range(4)}))


def test_full_review_keeps_blinding_and_usage(tmp_path):
    source = tmp_path / "source"
    make_strategy_pack(source)
    rating = StrategyRating(**{name: {"score": 1, "reason": "Partially conditional"}
                               for name in StrategyRating.model_fields})
    class Usage:
        input_tokens, output_tokens = 100, 200
        def model_dump(self):
            return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
    def parse(**kwargs):
        assert "BLINDED_ARM" not in json.dumps(kwargs["input"])
        assert kwargs["store"] is False
        return SimpleNamespace(usage=Usage(), output_parsed=rating, output_text=rating.model_dump_json())
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    manifest = run_review(client, "project3", source, tmp_path / "review")
    assert manifest["status"] == "complete" and manifest["api_calls"] == 4
    assert manifest["estimated_cost_usd"] == pytest.approx(4 * 0.0044)


def test_three_failures_stop_and_keep_reservations(tmp_path):
    source = tmp_path / "source"
    make_strategy_pack(source)
    def fail(**kwargs):
        raise TimeoutError("fixture")
    client = SimpleNamespace(responses=SimpleNamespace(parse=fail))
    manifest = run_review(client, "project3", source, tmp_path / "review")
    assert manifest["status"] == "stopped_after_three_review_failures"
    assert manifest["api_calls"] == 3 and manifest["failures"] == 3
    assert 0 < manifest["cost_with_failed_call_reservations_usd"] < 1
    assert manifest["estimated_cost_usd"] == 0


def test_resume_keeps_prior_spending_and_skips_successes(tmp_path):
    source = tmp_path / 'source'
    make_strategy_pack(source)
    rating = StrategyRating(**{name: {'score': 1, 'reason': 'fixture'} for name in StrategyRating.model_fields})
    class Usage:
        input_tokens, output_tokens = 10, 10
        def model_dump(self):
            return {'input_tokens': 10, 'output_tokens': 10}
    calls = []
    def parse(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError('first attempt failed')
        return SimpleNamespace(usage=Usage(), output_parsed=rating, output_text=rating.model_dump_json())
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    previous = run_review(client, 'project3', source, tmp_path / 'first')
    resumed = run_review(client, 'project3', source, tmp_path / 'second', resume_from=tmp_path / 'first')
    assert len(calls) == 5 and resumed['api_calls'] == 1
    assert resumed['carried_valid_ratings'] == 3
    assert resumed['cumulative_api_calls'] == 5 and resumed['cumulative_review_failures'] == 1
    assert resumed['budget_cap_usd'] == pytest.approx(1 - previous['cumulative_reserved_cost_usd'])
    assert len(json.loads((tmp_path / 'second/ratings.json').read_text())) == 4
