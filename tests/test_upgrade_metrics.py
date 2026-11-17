import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("metrics", Path(__file__).parents[1]/"tools/evaluation_upgrade.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_fingerprint_changes_with_content_and_settings():
    assert metrics.fingerprint({"text":"a","model":"m"}) != metrics.fingerprint({"text":"b","model":"m"})
    assert metrics.fingerprint({"text":"a","model":"m"}) != metrics.fingerprint({"text":"a","model":"new"})


def test_id_membership_is_not_claim_support():
    result=metrics.rag_measurements({"reference":"Boston"},"Paris [wiki::1::0]",["wiki::1::0"],{"wiki::1::0":"Boston"})
    assert result["citation_id_validity"] == 1
    assert result["claim_support"] is None
    assert result["token_f1"] == 0


def test_ablation_pairs_examples_and_counts_errors():
    cases=[{"question":"Where?","reference":"Boston"},{"question":"Absent?","reference":"Insufficient evidence.","answerable":False}]
    corpus={"wiki::1::0":"Boston"}
    def failing(query): raise RuntimeError("fixture")
    rows,summary=metrics.run_rag_ablation(cases,{"dense":lambda q:["wiki::1::0"],"failure":failing},lambda q,ids:"Boston [wiki::1::0]",corpus)
    assert len(rows)==4 and summary["failure"]["errors"]==2
    assert summary["failure"]["paired_f1_vs_dense"]["pairs"]==2


def test_fallback_is_not_counted_as_model_routing_success():
    result={"planner":{"valid_json":False,"attempts":2},"planned_tool":"search","status":"ok"}
    score=metrics.agent_accounting(result,{"tool":"search"})
    assert score["fallback_used"] and not score["model_routing_correct"]


def test_paired_interval_is_deterministic():
    assert metrics.paired_mean_interval([0,0],[1,1])["ci95"]==[1,1]


def test_yes_no_accuracy_is_not_answer_substring_retrieval():
    score = metrics.rag_measurements({'reference': 'yes'}, 'Yes, it is. [wiki::1::0]',
                                    ['wiki::1::0'], {'wiki::1::0': 'Lincoln was the sixteenth president.'})
    assert score['binary_answer_correct'] == 1 and score['answer_hit_proxy'] is None


def test_substring_retrieval_requires_word_boundaries_and_has_denominators():
    score = metrics.rag_measurements({'reference': 'York'}, 'York', ['wiki::1'], {'wiki::1': 'Yorkshire'})
    assert score['answer_hit_proxy'] == 0
    _, summaries = metrics.run_rag_ablation(
        [{'question': 'q', 'reference': 'yes'}, {'question': 'q2', 'reference': 'York'}],
        {'dense': lambda q: ['wiki::1']}, lambda q, ids: 'Yes', {'wiki::1': 'York'})
    assert summaries['dense']['metric_counts']['answer_hit_proxy'] == 1
    assert summaries['dense']['metric_counts']['binary_answer_correct'] == 1


def test_relevance_labels_cannot_silently_refer_to_different_chunks():
    with pytest.raises(ValueError, match='Unknown'):
        metrics.validate_relevance_labels([{'question': 'q', 'reference': 'a', 'answerable': True,
            'relevant_chunk_ids': ['wiki::missing'], 'label_origin': 'reviewer'}], {'wiki::1': 'a'})


def test_unknown_retrieval_ids_are_recorded_as_errors():
    _, summary = metrics.run_rag_ablation([{'question': 'q', 'reference': 'a'}],
        {'dense': lambda q: ['wiki::missing']}, lambda q, ids: 'a', {'wiki::1': 'a'})
    assert summary['dense']['errors'] == 1 and summary['dense']['token_f1'] == 0
