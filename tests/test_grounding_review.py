import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('review', Path(__file__).parents[1] / 'tools/grounding_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_review_pack_is_blind_and_requires_real_evidence():
    rows = [{'system': 'dense', 'example_id': 0, 'question': 'Where?',
             'answer': 'Boston [wiki::1::0]', 'retrieved_chunk_ids': ['wiki::1::0']}]
    pack, key = review.grounding_review_pack(rows, {'wiki::1::0': 'The venue is in Boston.'})
    entry = pack['entries'][0]
    assert 'system' not in entry and key[entry['review_id']]['system'] == 'dense'
    assert review.score_review(entry, entry) == {'supported': None, 'claims': None}
    rated = {**entry, 'reviewer': 'test reviewer', 'claims': [{'text': 'Boston', 'supported': True,
             'cited_chunk_id': 'wiki::1::0', 'supporting_quote': 'Paris', 'rationale': 'fixture'}]}
    with pytest.raises(ValueError, match='quote'):
        review.score_review(rated, entry)
    rated['claims'][0]['supporting_quote'] = 'in Boston'
    assert review.score_review(rated, entry) == {'supported': 1, 'claims': 1}


def test_review_identity_changes_with_answer_or_evidence():
    row = {'question': 'Where?', 'answer': 'Boston', 'retrieved_chunk_ids': ['wiki::1::0']}
    assert review.evidence_identity(row, {'wiki::1::0': 'Boston'}) != review.evidence_identity(row, {'wiki::1::0': 'Paris'})
