"""Blind evidence-support review. A valid citation ID is never a support label."""
import hashlib
import json
import random


def evidence_identity(row, corpus):
    evidence = {chunk: corpus[chunk] for chunk in row['retrieved_chunk_ids']}
    payload = [row['question'], row['answer'], evidence]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def grounding_review_pack(rows, corpus, seed=2026):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    entries, key = [], {}
    for position, row in enumerate(shuffled):
        review_id = f'review-{position:04d}'
        entries.append({'review_id': review_id, 'identity': evidence_identity(row, corpus),
                        'question': row['question'], 'answer': row['answer'],
                        'retrieved_evidence': {chunk: corpus[chunk] for chunk in row['retrieved_chunk_ids']},
                        'claims': None, 'reviewer': None})
        key[review_id] = {'system': row['system'], 'example_id': row['example_id']}
    return {'rubric_version': 'claim-support-v1', 'instructions': (
        'Split factual assertions into atomic claims. For each claim record text, supported '
        '(boolean), cited_chunk_id, supporting_quote, and rationale. A quote must appear '
        'verbatim in the cited retrieved chunk; judge entailment, not word overlap. '
        'Use claims=[] for a pure abstention and name the reviewer. Leave unreviewed entries null.'),
        'entries': entries}, key


def score_review(review, original):
    if review['identity'] != original['identity']:
        raise ValueError('Stale answer/evidence review')
    claims = review.get('claims')
    if claims is None:
        return {'supported': None, 'claims': None}
    if not review.get('reviewer') or not isinstance(claims, list):
        raise ValueError('Reviewed claims require a reviewer and a list')
    for claim in claims:
        if type(claim.get('supported')) is not bool or not claim.get('text') or not claim.get('rationale'):
            raise ValueError('Each claim requires a decision and reason')
        if claim['supported']:
            quote = claim.get('supporting_quote')
            evidence = original['retrieved_evidence'].get(claim.get('cited_chunk_id'), '')
            if not quote or quote not in evidence:
                raise ValueError('Support must cite an exact quote from retrieved evidence')
    return {'supported': sum(claim['supported'] for claim in claims), 'claims': len(claims)}
