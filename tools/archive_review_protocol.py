"""Recover exact public review inputs and verify their recorded request hashes."""
import argparse
import json
from pathlib import Path

from automated_review import SYSTEM, digest, grounding_request, strategy_request


QUOTE_CLARIFICATION = (
    "The quote MUST be one CONTINUOUS, character-for-character substring copied from that "
    "chunk. Do not shorten it with ellipses (...), paraphrase, join separated passages, "
    "normalize punctuation, or alter whitespace. Copy the full sentence if needed. "
)


def archive_protocol(source, review):
    manifest = json.loads((review / 'run_manifest.json').read_text())
    assert digest(SYSTEM) == manifest['system_prompt_sha256']
    is_rag = manifest['project'] == 'project2'
    pack = json.loads((source / ('grounding_review.json' if is_rag else 'blind_review.json')).read_text())
    entries = {e['review_id' if is_rag else 'candidate']: e for e in pack['entries' if is_rag else 'candidates']}
    requests = []
    for row in json.loads((review / 'ratings.json').read_text()):
        entry = entries[row['review_id']]
        request = grounding_request(entry) if is_rag else strategy_request(entry, pack)
        if digest(request) != row['request_sha256'] and is_rag:
            request['rubric'] = request['rubric'].replace(QUOTE_CLARIFICATION, '')
        if digest(request) != row['request_sha256']:
            raise ValueError('Cannot reconstruct the exact recorded request')
        requests.append({'review_id': row['review_id'], 'request_sha256': row['request_sha256'],
                         'user_message': request})
    protocol = {'system_message': SYSTEM, 'requests': requests,
                'provenance': 'Reconstructed from frozen evidence; every request matches its recorded SHA-256. '
                              'System identities are not included in the model input.'}
    path = review / 'review_protocol.json'
    content = json.dumps(protocol, indent=2)
    if path.exists() and path.read_text() != content:
        raise FileExistsError('Existing review protocol differs')
    path.write_text(content)
    return {'verified_request_hashes': len(requests), 'file': str(path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    print(archive_protocol(args.source, args.review))
