"""Matched retrieval-only screening using the saved Project 2 holdout."""
import hashlib
import json
import random
import statistics
import time

from evaluation_upgrade import normalized_words, paired_mean_interval


def normalize_text(text):
    return " ".join(normalized_words(text))


def evaluate(state):
    import faiss
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from torch.nn.functional import normalize
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    project_dir = state["PROJECT_DIR"]
    holdout_dir = project_dir / "outputs/runs/upgrade-v2/20261118T092816338337Z"
    saved = json.loads((holdout_dir / "project2_eval_results.json").read_text())
    holdout = [row for row in saved if row["system"] == "dense" and row["stratum"] == "natural_qa"]
    if len(holdout) != 60 or len({row["example_id"] for row in holdout}) != 60:
        raise ValueError("Expected the 60 unique natural QA cases from the completed run")
    chunks = state["chunk_texts"]
    device = state["DEVICE"]
    candidates, top_k = state["CANDIDATE_K"], state["ADVANCED_K"]
    qwen = "Qwen/Qwen3-Embedding-0.6B"
    bge = "BAAI/bge-reranker-v2-m3"
    api = HfApi()
    revisions = {name: api.model_info(name).sha for name in (qwen, bge)}
    tokenizer = AutoTokenizer.from_pretrained(qwen, revision=revisions[qwen], padding_side="left")
    encoder = AutoModel.from_pretrained(qwen, revision=revisions[qwen], use_safetensors=True,
                                        trust_remote_code=False, dtype=torch.bfloat16).to(device).eval()
    rerank_tokenizer = AutoTokenizer.from_pretrained(bge, revision=revisions[bge])
    reranker = AutoModelForSequenceClassification.from_pretrained(
        bge, revision=revisions[bge], use_safetensors=True, trust_remote_code=False,
        dtype=torch.bfloat16,
    ).to(device).eval()

    @torch.inference_mode()
    def qwen_encode(texts):
        vectors = []
        for start in range(0, len(texts), 32):
            batch = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                              max_length=512, return_tensors="pt").to(device)
            hidden = encoder(**batch).last_hidden_state[:, -1, :].float()
            vectors.append(normalize(hidden, p=2, dim=1).cpu().numpy())
        return np.concatenate(vectors)

    cache_spec = json.dumps({
        "revision": revisions[qwen], "texts": chunks, "pooling": "last_token_left_padded",
        "dtype": "bfloat16", "normalized": True, "max_length": 512,
    }, sort_keys=True).encode()
    cache = state["CACHE_DIR"] / f"qwen3_{hashlib.sha256(cache_spec).hexdigest()}.npy"
    if cache.is_file():
        vectors = np.load(cache, allow_pickle=False)
    else:
        vectors = qwen_encode(chunks)
        np.save(cache, vectors, allow_pickle=False)
    if len(vectors) != len(chunks) or not np.isfinite(vectors).all():
        raise ValueError("Cached Qwen embeddings are invalid")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    def qwen_rank(question):
        query = f"Instruct: Retrieve encyclopedia passages that answer the question.\nQuery: {question}"
        scores, positions = index.search(qwen_encode([query]), candidates)
        return [(int(i), float(s)) for i, s in zip(positions[0], scores[0])]

    def hybrid(question, dense):
        return state["reciprocal_rank_fusion"](
            [dense(question), state["sparse_rank"](question, candidates)], candidates,
        )

    def e5_hybrid(question):
        return hybrid(question, lambda query: state["dense_rank"](query, candidates))

    @torch.inference_mode()
    def bge_rank(question, ids):
        pairs = [[question, chunks[index]] for index in ids]
        batch = rerank_tokenizer(pairs, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt").to(device)
        scores = reranker(**batch).logits.flatten().float().cpu().numpy()
        return [ids[int(i)] for i in np.argsort(-scores, kind="stable")[:top_k]]

    arms = {
        "e5_hybrid": lambda q: e5_hybrid(q)[:top_k],
        "qwen3_hybrid": lambda q: hybrid(q, qwen_rank)[:top_k],
        "e5_minilm_rerank": lambda q: state["rerank"](q, e5_hybrid(q), top_k),
        "e5_bge_rerank": lambda q: bge_rank(q, e5_hybrid(q)),
    }
    normalized = [f" {normalize_text(chunk)} " for chunk in chunks]
    for retrieve in arms.values():
        retrieve("Who wrote the novel Pride and Prejudice?")
    rng = random.Random(state["SEED"])
    records = []
    for example in holdout:
        reference = normalize_text(example["reference"])
        if reference in ("", "yes", "no"):
            continue
        order = list(arms)
        rng.shuffle(order)
        for arm in order:
            start = time.perf_counter()
            ids, error = [], None
            try:
                ids = arms[arm](example["question"])
            except Exception as exc:
                error = type(exc).__name__
            seconds = time.perf_counter() - start
            rank = next((r for r, idx in enumerate(ids, start=1)
                         if f" {reference} " in normalized[idx]), None)
            records.append({"example_id": example["example_id"], "arm": arm,
                            "answer_hit": int(rank is not None), "reciprocal_rank": 1 / rank if rank else 0,
                            "latency_seconds": seconds, "error": error,
                            "retrieved_chunk_ids": [state["chunks"][i].metadata["chunk_id"] for i in ids]})

    metrics = {}
    for arm in arms:
        group = [r for r in records if r["arm"] == arm]
        reference_arm = "e5_minilm_rerank" if arm.endswith("_rerank") else "e5_hybrid"
        ref = [r["answer_hit"] for r in records if r["arm"] == reference_arm]
        durations = sorted(r["latency_seconds"] for r in group)
        metrics[arm] = {"answer_hit_at_5_proxy": statistics.mean(r["answer_hit"] for r in group),
                        "mrr_at_5_proxy": statistics.mean(r["reciprocal_rank"] for r in group),
                        "paired_hit_change": paired_mean_interval(ref, [r["answer_hit"] for r in group], state["SEED"]),
                        "p50_seconds": statistics.median(durations), "p95_seconds": durations[int(.95 * len(durations))],
                        "errors": sum(r["error"] is not None for r in group)}
    summary = {"cases": len(records) // len(arms), "natural_holdout_cases": len(holdout),
               "holdout_run": holdout_dir.name, "metrics": metrics,
               "revisions": {**revisions, state["EMBED_MODEL"]: state["EMBED_REVISION"],
                             state["RERANK_MODEL"]: state["RERANK_REVISION"]},
               "context_candidates": candidates, "retrieved_chunks": top_k, "generator_calls": 0,
               "scope": "Reference-substring retrieval proxy; does not measure answer quality or claim support."}
    output = state["OUT_DIR"]
    (output / "retriever_refresh_per_case.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "retriever_refresh_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)
