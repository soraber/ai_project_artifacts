"""CPU-testable measurement helpers, embedded into the standalone notebooks."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def paired_mean_interval(before, after, seed=2026, resamples=2000):
    if not before or len(before) != len(after):
        raise ValueError("Paired evaluation requires nonempty, equally sized samples")
    if resamples < 40 or any(not math.isfinite(value) for value in [*before, *after]):
        raise ValueError("At least 40 resamples and finite observations are required")
    delta = [right-left for left, right in zip(before, after)]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(delta, k=len(delta))) for _ in range(resamples))
    return {"mean_change": statistics.fmean(delta),
            "ci95": [means[int(.025*resamples)], means[min(resamples-1, int(.975*resamples))]],
            "pairs": len(delta), "resamples": resamples}


def normalized_words(text):
    return re.findall(r"[a-z0-9]+", re.sub(r"\[wiki::[^\]]+\]", " ", text.lower()))


def overlap_f1(answer, reference):
    predicted, gold = normalized_words(answer), normalized_words(reference)
    if not predicted or not gold:
        return float(predicted == gold)
    common = sum((Counter(predicted) & Counter(gold)).values())
    return 2*common/(len(predicted)+len(gold))


def rag_measurements(example, answer, retrieved_ids, corpus):
    reference = example["reference"]
    target = " ".join(normalized_words(reference))
    hits = [rank for rank, item in enumerate(retrieved_ids, 1)
            if target and f" {target} " in " " + " ".join(normalized_words(corpus[item])) + " "]
    retrieval_proxy_defined = target not in ("yes", "no") and example.get("answerable") is not False
    cited = set(re.findall(r"\[(wiki::[^\]]+)\]", answer))
    relevant = set(example.get("relevant_chunk_ids", []))
    binary = re.match(r"^\s*(yes|no)\b", answer.lower())
    return {
        "answer_hit_proxy": float(bool(hits)) if retrieval_proxy_defined else None,
        "reciprocal_rank_proxy": (1/hits[0] if hits else 0.0) if retrieval_proxy_defined else None,
        "token_f1": overlap_f1(answer, reference),
        "binary_answer_correct": float(bool(binary) and binary.group(1) == target) if target in ("yes", "no") else None,
        "citation_id_validity": len(cited & set(retrieved_ids))/len(cited) if cited else 0.0,
        "relevance_recall": len(relevant & set(retrieved_ids))/len(relevant) if relevant else None,
        "abstention_correct": float("insufficient evidence" in answer.lower()) if example.get("answerable") is False else None,
        "claim_support": None,
    }


def run_rag_ablation(examples, variants, answerer, corpus, seed=2026, usage=None):
    """Retrievers return IDs; the same answerer and final context budget serve every arm."""
    rows = []
    rng = random.Random(seed)
    for index, example in enumerate(examples):
        names = list(variants)
        rng.shuffle(names)
        for name in names:
            started = time.perf_counter()
            usage_start = len(usage) if usage is not None else 0
            error, ids, answer = None, [], ""
            try:
                ids = variants[name](example["question"])
                if not isinstance(ids, list) or any(item not in corpus for item in ids):
                    raise ValueError("Retriever returned unknown chunk IDs")
                answer = answerer(example["question"], ids)
            except Exception as exc:
                error = type(exc).__name__
                ids = [item for item in (ids or []) if item in corpus]
            calls = usage[usage_start:] if usage is not None else []
            rows.append({"example_id": index, "system": name, "question": example["question"],
                         "reference": example["reference"], "answer": answer,
                         "retrieved_chunk_ids": ids, "error": error,
                         "stratum": example.get("stratum", "natural_qa"),
                         "input_tokens": sum(call['input_tokens'] for call in calls),
                         "output_tokens": sum(call['output_tokens'] for call in calls),
                         "generation_calls": len(calls),
                         "latency_seconds": time.perf_counter()-started,
                         **rag_measurements(example, answer, ids, corpus)})
    return rows, summarize_rag_rows(rows, variants, seed)


def summarize_rag_rows(rows, variants, seed=2026):
    summaries = {}
    for name in variants:
        selected = [row for row in rows if row["system"] == name]
        metrics, metric_counts = {}, {}
        for key in ["answer_hit_proxy", "reciprocal_rank_proxy", "token_f1", "binary_answer_correct", "citation_id_validity", "relevance_recall", "abstention_correct"]:
            values = [row[key] for row in selected if row[key] is not None]
            metrics[key] = statistics.fmean(values) if values else None
            metric_counts[key] = len(values)
        times = sorted(row["latency_seconds"] for row in selected)
        summaries[name] = {**metrics, "metric_counts": metric_counts, "examples": len(selected),
                           "generation_calls": sum(row['generation_calls'] for row in selected),
                           "input_tokens": sum(row['input_tokens'] for row in selected),
                           "output_tokens": sum(row['output_tokens'] for row in selected),
                           "errors": sum(row["error"] is not None for row in selected),
                           "latency_p50_seconds": statistics.median(times),
                           "latency_p95_seconds": times[min(len(times)-1, int(.95*len(times)))]}
    baseline = [row["token_f1"] for row in rows if row["system"] == "dense"]
    for name in variants:
        selected = [row["token_f1"] for row in rows if row["system"] == name]
        summaries[name]["paired_f1_vs_dense"] = paired_mean_interval(baseline, selected, seed)
    return summaries


def validate_relevance_labels(labels, corpus):
    if not isinstance(labels, list):
        raise ValueError("Held-out labels must be a list")
    for item in labels:
        if not all(isinstance(item.get(key), str) and item[key].strip()
                   for key in ['question', 'reference', 'label_origin']):
            raise ValueError("Labels require question, reference, and label_origin")
        if type(item.get('answerable')) is not bool:
            raise ValueError("answerable must be an explicit boolean label")
        ids = item.get('relevant_chunk_ids')
        if not isinstance(ids, list) or any(chunk not in corpus for chunk in ids):
            raise ValueError("Unknown or missing relevant_chunk_ids")
        if bool(ids) != item['answerable']:
            raise ValueError("Answerability and relevant evidence labels conflict")
    return [{**item, 'stratum': 'independently_labeled'} for item in labels]


def agent_accounting(result, case):
    planner = result["planner"]
    fallback = bool(planner.get("fallback_used", not planner.get("valid_json", False)))
    routed = result["planned_tool"] == case["tool"]
    return {"fallback_used": fallback, "model_routing_correct": routed and not fallback,
            "model_parse_valid": bool(planner.get("valid_json", False)),
            "planner_attempts": planner.get("attempts", 0),
            "tool_execution_status": result.get("status", "unknown")}
