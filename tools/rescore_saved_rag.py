"""Correct retrieval-proxy eligibility from frozen outputs without new model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation_upgrade import rag_measurements, summarize_rag_rows


def rescore(source):
    records = json.loads((source / 'project2_eval_results.json').read_text())
    pack = json.loads((source / 'grounding_review.json').read_text())
    corpus = {}
    for entry in pack['entries']:
        for chunk, text in entry['retrieved_evidence'].items():
            if chunk in corpus and corpus[chunk] != text:
                raise ValueError('Conflicting frozen evidence')
            corpus[chunk] = text
    for row in records:
        example = {'reference': row['reference']}
        if row['stratum'] == 'authored_absence_probe':
            example['answerable'] = False
        corrected = rag_measurements(example, row['answer'], row['retrieved_chunk_ids'], corpus)
        for metric in ['answer_hit_proxy', 'reciprocal_rank_proxy']:
            row[metric] = corrected[metric]
    variants = list(dict.fromkeys(r['system'] for r in records))
    summary = json.loads((source / 'project2_summary.json').read_text())
    summary.update(evaluation_version='upgrade-v1-metric-correction-v2',
                   metrics=summarize_rag_rows(records, variants, summary['evaluation_seed']),
                   metrics_by_stratum={name: summarize_rag_rows([r for r in records if r['stratum'] == name],
                       variants, summary['evaluation_seed']) for name in sorted({r['stratum'] for r in records})},
                   metric_correction='Whole-word reference matching; yes/no and authored absence cases excluded '
                                     'only from substring retrieval proxies. All 63 cases remain in answer scores. '
                                     'Saved answers, timing and judge inputs are unchanged; no new model calls.',
                   parent_records_sha256=hashlib.sha256((source / 'project2_eval_results.json').read_bytes()).hexdigest(),
                   metric_source_sha256=hashlib.sha256((Path(__file__).parent / 'evaluation_upgrade.py').read_bytes()).hexdigest())
    output = source / 'metrics_v2'
    output.mkdir(exist_ok=False)
    (output / 'project2_eval_results.json').write_text(json.dumps(records, indent=2))
    (output / 'project2_summary.json').write_text(json.dumps(summary, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    print(rescore(parser.parse_args().source))
