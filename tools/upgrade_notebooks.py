"""Targeted upgrade: preserve unrelated cells, kernel metadata, and historical files."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS = (ROOT / "tools/evaluation_upgrade.py").read_text().replace("from __future__ import annotations\n", "")


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one source match: {before[:80]}")
    return source.replace(before, after, 1)


INFERENCE_PREFLIGHT = '''# Use tools/build_gpu_upgrade_runner.py with the connected AI-project kernel.
# The runner creates a separate Python environment and captures its resolved lock.
import subprocess, sys
from importlib.metadata import version
if sys.prefix == sys.base_prefix:
    raise RuntimeError("Shared Colab kernel selected. Use the isolated GPU runner; do not repair global packages.")
for name, expected in [('transformers', '4.57.6'), ('crewai', '1.15.41'), ('lm-format-enforcer', '0.11.3')]:
    actual = version(name)
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected}, got {actual}; recreate the isolated inference environment")
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
print('Isolated inference dependencies verified')
'''


def project1(cells):
    s = cells["18ed85c3"]
    s = replace_once(s, 'EVAL_RESULTS_JSON = OUT_DIR / "base_vs_tuned_eval.json"\nMAX_EVAL_SAMPLES = 100  # Use 500 for the final portfolio result.', '''from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
RUN_DIR = OUT_DIR / "runs" / "upgrade-v1" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RUN_DIR.mkdir(parents=True, exist_ok=False)
EVAL_RESULTS_JSON = RUN_DIR / "base_vs_tuned_eval.json"
MAX_EVAL_SAMPLES = None  # Score the full existing holdout; do not regenerate it.
evaluation_started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()''')
    s = replace_once(s, 'if len(eval_examples) > MAX_EVAL_SAMPLES:', 'available_examples = len(eval_examples)\nif MAX_EVAL_SAMPLES is not None and len(eval_examples) > MAX_EVAL_SAMPLES:')
    s = replace_once(s, 'per_example = []\nfor index', 'per_example = []\nskipped_examples = []\nfor index')
    s = replace_once(s, 'if batch is None:\n        continue', 'if batch is None:\n        skipped_examples.append({"sample_index": index, "reason": "no_answer_tokens"})\n        continue')
    s = replace_once(s, '    per_example.append({', '''    if not math.isfinite(base_nll) or not math.isfinite(tuned_nll):
        raise RuntimeError(f"Nonfinite loss at holdout sample {index}; do not publish a partial score")
    per_example.append({''')
    s = replace_once(s, 'summary = {', '''# Resample examples as paired clusters, then recompute the token-weighted estimand.
weights = np.array([row["answer_tokens"] for row in per_example], dtype=float)
weighted_deltas = np.empty(BOOTSTRAP_SAMPLES)
for i in range(BOOTSTRAP_SAMPLES):
    positions = bootstrap_rng.integers(0, len(per_example), len(per_example))
    weighted_deltas[i] = np.average(paired_improvements[positions], weights=weights[positions])
weighted_ci = np.percentile(weighted_deltas, [2.5, 97.5]).tolist()

summary = {''')
    s = replace_once(s, '"adapter_dir": str(ADAPTER_DIR),', '''"adapter_dir": ADAPTER_DIR.name,
    "evaluation_version": "upgrade-v1",
    "evaluation_file_sha256": hashlib.sha256(EVAL_JSON.read_bytes()).hexdigest(),
    "adapter_config_sha256": hashlib.sha256((ADAPTER_DIR / "adapter_config.json").read_bytes()).hexdigest(),
    "examples_available": available_examples,
    "examples_selected": len(eval_examples),
    "examples_skipped": len(skipped_examples),
    "cutoff_length": EVAL_MAX_LENGTH,
    "bootstrap_resamples": BOOTSTRAP_SAMPLES,
    "seed": SEED,
    "token_weighted_nll_improvement_95pct_ci": weighted_ci,
    "uncertainty_note": "Paired example-cluster bootstrap; macro CI and token-weighted CI are distinct estimands",
    "scope": "Reference-answer fit, not agent task completion; no generation decoding is used for NLL",
    "seconds": time.perf_counter() - evaluation_started,
    "gpu": torch.cuda.get_device_name(0),
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "packages": {name: version(name) for name in ["torch", "transformers", "peft", "bitsandbytes"]},''')
    s = replace_once(s, '{"summary": summary, "per_example": per_example}', '{"summary": summary, "per_example": per_example, "skipped_examples": skipped_examples}')
    return {"c2e49511": '''# Fail closed instead of uninstalling unrelated Colab packages.
import sys, os, subprocess
from pathlib import Path
from importlib.metadata import version
if sys.prefix == sys.base_prefix:
    raise RuntimeError(
        "Use the isolated project1_upgrade_runtime venv kernel, or execute the "
        "standalone runner created by tools/build_upgrade_runner.py. "
        "This cell will not repair or uninstall packages in Colab's shared environment."
    )
assert version("llamafactory") == "0.9.5", "Use the tested isolated training environment"
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
work_dir = Path("/content/project1_notebook_work")
(work_dir / "data").mkdir(parents=True, exist_ok=True)
registry = work_dir / "data/dataset_info.json"
if not registry.exists(): registry.write_text("{}\\n")
os.chdir(work_dir)
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
print("Isolated interpreter:", sys.executable)
''', "18ed85c3": s, "257323c9": '''## Paired Holdout Evaluation (Upgrade v1)

Cell 20 evaluates every example in the existing holdout without changing dataset preparation.
Both systems score identical answer tokens on the same base checkpoint with the adapter
disabled/enabled. Macro and token-weighted NLL uncertainty are reported separately.
This evaluates reference-answer fit, not tool-use or general agent quality. New artifacts
go into a timestamped `outputs/runs/upgrade-v1` directory; historical results stay intact.
Restore the saved adapter and held-out JSON before running; do not rerun training to evaluate.
'''}


RAG_EVAL = '''# Shared helpers are embedded so this notebook remains standalone in Colab.
__HELPERS__

def answer_text(value):
    return str(value[0]) if isinstance(value, (list, tuple)) and value else str(value)

eligible = [
    {"question": str(row["question"]).strip(), "reference": answer_text(row["answer"]).strip()}
    for row in qa_ds if str(row.get("question", "")).strip() and answer_text(row.get("answer", "")).strip()
]
eval_examples = random.Random(SEED).sample(eligible, min(EVAL_SIZE, len(eligible)))
corpus_by_id = {chunk.metadata["chunk_id"]: chunk.page_content for chunk in chunks}
# Optional independently labeled JSON: question, reference, relevant_chunk_ids, answerable.
# Do not label absence merely because a reference substring was not found.
label_path = PROJECT_DIR / "heldout_relevance.json"
if label_path.exists():
    labeled = json.loads(label_path.read_text())
    eval_examples.extend(validate_relevance_labels(labeled, corpus_by_id))

# Authored absence probes are reported separately from public natural QA.
for question in ["What private passphrase did I choose this morning?",
                 "What is the unpublished result of my personal medical test?",
                 "What is the current balance of my private bank account?"]:
    eval_examples.append({'question': question, 'reference': 'Insufficient evidence.',
                          'answerable': False, 'stratum': 'authored_absence_probe'})

corpus_by_id = {chunk.metadata["chunk_id"]: chunk.page_content for chunk in chunks}
position_by_id = {chunk.metadata["chunk_id"]: i for i, chunk in enumerate(chunks)}
def ids_at(positions):
    return [chunks[i].metadata["chunk_id"] for i in positions]
def hybrid_positions(question):
    return reciprocal_rank_fusion([dense_rank(question, CANDIDATE_K), sparse_rank(question, CANDIDATE_K)], CANDIDATE_K)
def hyde_positions(question):
    hypothetical = generate_batch([hyde_prompt(question)], max_new_tokens=96)[0]
    return advanced_retrieve(question, hypothetical)
variants = {
    "dense": lambda q: ids_at(baseline_retrieve(q)),
    "hybrid": lambda q: ids_at(hybrid_positions(q)[:ADVANCED_K]),
    "hybrid_rerank": lambda q: ids_at(rerank(q, hybrid_positions(q), ADVANCED_K)),
    "hyde_hybrid_rerank": lambda q: ids_at(hyde_positions(q)),
}
def shared_answerer(question, ids):
    indices = [position_by_id[item] for item in ids]
    return generate_batch([baseline_answer_prompt(question, indices)], max_new_tokens=96)[0]

# Warm each inference path, excluding these examples and timings from the evaluation.
for retrieve in variants.values():
    shared_answerer("What is the purpose of this encyclopedia?", retrieve("What is the purpose of this encyclopedia?"))
records, arm_summaries = run_rag_ablation(eval_examples, variants, shared_answerer, corpus_by_id, SEED, rag_generation_usage)
results = pd.DataFrame(records)
summary = pd.DataFrame(arm_summaries).T
run_summary = {
    "evaluation_version": "upgrade-v1", "dataset": DATASET_NAME, "dataset_revision": DATASET_REVISION,
    "examples": len(eval_examples), "evaluation_seed": SEED, "generator_model": GENERATOR_MODEL,
    "generator_revision": getattr(generator_model.config, "_commit_hash", None),
    "embedding_fingerprint": embedding_fingerprint,
    "reranker_revision": getattr(rerank_model.config, "_commit_hash", None),
    "answer_prompt_sha256": fingerprint(baseline_answer_prompt("QUESTION", [])),
    "evaluation_sha256": fingerprint(eval_examples),
    "decoding": {"do_sample": False, "max_new_tokens": 96, "input_max_tokens": 3072},
    "context_chunks": ADVANCED_K, "gpu": torch.cuda.get_device_name(0),
    "metrics": arm_summaries,
    "metrics_by_stratum": {name: summarize_rag_rows([row for row in records if row['stratum'] == name], variants, SEED)
                           for name in sorted({row['stratum'] for row in records})},
    "limitations": "Substring hit/MRR and token F1 are proxies. Citation ID validity is not claim support. Human claim-support labels remain pending.",
}
results.to_json(OUT_DIR / "project2_eval_results.json", orient="records", indent=2)
(OUT_DIR / "project2_summary.json").write_text(json.dumps(run_summary, indent=2))
display(summary)
print("Saved versioned ablations:", OUT_DIR)
__GROUNDING_REVIEW__
review_pack, review_key = grounding_review_pack(records, corpus_by_id, SEED)
(OUT_DIR / 'grounding_review.json').write_text(json.dumps(review_pack, indent=2))
(OUT_DIR / 'grounding_review_key.json').write_text(json.dumps(review_key, indent=2))
'''


def project2(cells):
    config = replace_once(cells["3d8e1486"], "OUT_DIR = PROJECT_DIR / 'outputs'", "from datetime import datetime, timezone\nOUT_DIR = PROJECT_DIR / 'outputs/runs/upgrade-v1' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')")
    config = replace_once(config, "BASELINE_K = 3", "BASELINE_K = 5")
    data = cells['33724c09'].replace("CACHE_DIR / 'mini_wikipedia_corpus'", "CACHE_DIR / f'mini_wikipedia_corpus_{DATASET_REVISION}'").replace("CACHE_DIR / 'mini_wikipedia_qa'", "CACHE_DIR / f'mini_wikipedia_qa_{DATASET_REVISION}'")
    generator = replace_once(cells['bdf0279c'], 'compute_dtype =', '''GENERATOR_REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
rag_generation_usage = []
compute_dtype =''')
    generator = generator.replace('local_files_only=True', 'local_files_only=True, revision=GENERATOR_REVISION')
    generator = replace_once(generator, "rendered, padding=True, truncation=True, max_length=3072, return_tensors='pt'",
                              "rendered, padding=True, truncation=False, return_tensors='pt'")
    generator = replace_once(generator, '        generated = generator_model.generate(', '''        if encoded['input_ids'].shape[1] > 3072:
            raise ValueError("RAG input exceeds the matched 3072-token budget; no silent truncation")
        generated = generator_model.generate(''')
    generator = replace_once(generator, '        answers.extend(', '''        for row_index in range(len(batch_prompts)):
            padding = (new_tokens[row_index] == generator_tokenizer.pad_token_id).nonzero()
            output_length = int(padding[0]) + 1 if len(padding) else new_tokens.shape[1]
            rag_generation_usage.append({
                'input_tokens': int(encoded['attention_mask'][row_index].sum()),
                'output_tokens': output_length})
        answers.extend(''')
    embedding = replace_once(cells["814ef015"], "encoder = E5Encoder(EMBED_MODEL, DEVICE)", '''from huggingface_hub import HfApi
import hashlib
EMBED_REVISION = HfApi().model_info(EMBED_MODEL).sha
encoder = E5Encoder(EMBED_MODEL, DEVICE)''')
    embedding = embedding.replace("from_pretrained(model_name)", "from_pretrained(model_name, revision=EMBED_REVISION)")
    embedding = replace_once(embedding, "embedding_cache = CACHE_DIR / 'e5_small_v2_chunk_embeddings.npy'", '''embedding_spec = {"model": EMBED_MODEL, "revision": EMBED_REVISION,
                  "texts": chunk_texts, "prefix": "passage: ", "pooling": "mean",
                  "normalized": True, "max_length": 512,
                  "chunk_size": 900, "chunk_overlap": 120}
embedding_fingerprint = hashlib.sha256(json.dumps(embedding_spec, sort_keys=True).encode()).hexdigest()
embedding_cache = CACHE_DIR / f"embeddings_{embedding_fingerprint}.npy"''')
    retrieval = cells["01d9a1ca"]
    retrieval = replace_once(retrieval, 'rerank_tokenizer =', 'RERANK_REVISION = HfApi().model_info(RERANK_MODEL).sha\nrerank_tokenizer =')
    retrieval = retrieval.replace('from_pretrained(RERANK_MODEL)', 'from_pretrained(RERANK_MODEL, revision=RERANK_REVISION)')
    start = retrieval.index('def advanced_answer_prompt(')
    end = retrieval.index("\nprint('Retriever components", start)
    retrieval = retrieval[:start] + "# Both arms use the exact same answer instruction.\nadvanced_answer_prompt = baseline_answer_prompt\n" + retrieval[end:]
    providers = replace_once(cells['e9f2a964'], '    %pip install -q --upgrade-strategy only-if-needed "openai>=1,<3"\n', '')
    start = providers.index('    benchmark_prompts = [')
    end = providers.index('    for provider, model_name', start)
    providers = providers[:start] + '''    benchmark_rows = [row for row in records if row['system'] == 'hybrid_rerank'
                      and row['stratum'] == 'natural_qa'][:PROVIDER_EVAL_SIZE]
    benchmark_prompts = [baseline_answer_prompt(row['question'], [position_by_id[item]
                        for item in row['retrieved_chunk_ids']]) for row in benchmark_rows]
''' + providers[end:]
    providers = replace_once(providers, "'token_f1': token_f1(answer, references[i]),", "'token_f1': overlap_f1(answer, benchmark_rows[i]['reference']),")
    providers = replace_once(providers, "'citation_validity': citation_validity(answer, advanced_indices[i]),", "'citation_validity': rag_measurements(benchmark_rows[i], answer, benchmark_rows[i]['retrieved_chunk_ids'], corpus_by_id)['citation_id_validity'],")
    return {'67c7857a': INFERENCE_PREFLIGHT, "3d8e1486": config, '33724c09': data, "814ef015": embedding,
            'bdf0279c': generator, "01d9a1ca": retrieval,
            'e9f2a964': providers,
            "45f0634e": RAG_EVAL.replace("__HELPERS__", HELPERS).replace(
                '__GROUNDING_REVIEW__', (ROOT / 'tools/grounding_review.py').read_text())}


def project3(cells):
    setup = INFERENCE_PREFLIGHT
    config = replace_once(cells["87bbb7e4"], "OUT_DIR = PROJECT_DIR / 'outputs'", "from datetime import datetime, timezone\nOUT_DIR = PROJECT_DIR / 'outputs/runs/upgrade-v1' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')")
    config = replace_once(config, "local_generation_calls = 0", "local_generation_calls = 0\ngeneration_usage = []")
    config = replace_once(config, "# Reuse Project 2's model", '''MODEL_REVISION = os.environ.get('PROJECT_MODEL_REVISION', '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
# Reuse Project 2's model''')
    config = config.replace('local_files_only=True', 'local_files_only=True, revision=MODEL_REVISION')
    config = replace_once(config, "    return agent_tokenizer.decode(new_tokens[0]", "    generation_usage.append({'input_tokens': int(encoded['input_ids'].numel()), 'output_tokens': int(new_tokens.numel()), 'output_token_cap': max_new_tokens})\n    return agent_tokenizer.decode(new_tokens[0]")
    config = replace_once(config, 'temperature=0.0):', 'temperature=0.0, json_schema=None, input_token_cap=None):')
    config = replace_once(config, "        rendered, truncation=True, max_length=7168, return_tensors='pt'",
                          "        rendered, truncation=False, return_tensors='pt'")
    config = replace_once(config, '    generated = agent_model.generate(', '''    context_limit = min(8192, agent_model.config.max_position_embeddings)
    if input_token_cap is not None and encoded['input_ids'].numel() > input_token_cap:
        raise RuntimeError("Aggregate input-token budget exhausted")
    if encoded['input_ids'].shape[1] + max_new_tokens > context_limit:
        raise ValueError("Prompt plus output budget exceeds the context window; no silent truncation")
    generation_kwargs = {}
    if json_schema is not None:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
        if not hasattr(generate_local, 'enforcer_tokenizer_data'):
            generate_local.enforcer_tokenizer_data = build_token_enforcer_tokenizer_data(agent_tokenizer)
        generation_kwargs['prefix_allowed_tokens_fn'] = build_transformers_prefix_allowed_tokens_fn(
            generate_local.enforcer_tokenizer_data, JsonSchemaParser(json_schema))
    generated = agent_model.generate(
        **generation_kwargs,''')
    router = (ROOT / "tools/project3_router.py").read_text() + '''
from langchain_core.runnables import RunnableLambda

def plan_action(task, constrained=True):
    return plan_with_schema(task, LANGCHAIN_TOOLS, generate_local, constrained=constrained)

''' + cells["c576a49d"][cells["c576a49d"].index('def grounded_answer('):]
    start = router.index("    if 'insufficient local evidence' in answer.lower():")
    end = router.index("    return answer", start)
    router = router[:start] + router[end:]
    router = replace_once(router, 'def run_local_agent(task, allow_write=False):',
                          'def run_local_agent(task, allow_write=False, constrained=True):')
    router = replace_once(router, 'decision, planner = plan_action(task)',
                          'decision, planner = plan_action(task, constrained=constrained)')
    router = replace_once(router, "        if decision.tool == 'search_knowledge_base':", '''        if decision.tool in ['request_clarification', 'abstain']:
            status = decision.tool
            observation = {'status': status, **arguments}
            answer = arguments['reason']
        elif decision.tool == 'search_knowledge_base':''')
    router = replace_once(router, "request.get('allow_write', False))",
                          "request.get('allow_write', False), request.get('constrained', True))")

    crew = (ROOT / 'tools/project3_strategy_evaluation.py').read_text() + '''
strategy_tasks = strategy_cases(KNOWLEDGE_BASE)
strategy_study = await evaluate_strategies(strategy_tasks, generate_local, generation_usage, MODEL_NAME, SEED)
(OUT_DIR / 'strategy_per_case.json').write_text(json.dumps(strategy_study, indent=2))
# Keep the legacy first-case coverage table separately from the multi-task review.
standard = {row['system']: row for row in strategy_study if row['case_id'] == 'standard_launch'}
baseline_strategy, self_review_strategy, crew_strategy = [standard[name]['answer'] for name in ['one_shot', 'self_review', 'crewai']]
baseline_strategy_seconds, self_review_strategy_seconds, crew_strategy_seconds = [standard[name]['latency_seconds'] for name in ['one_shot', 'self_review', 'crewai']]
baseline_usage, self_review_usage, crew_usage = [standard[name]['usage'] for name in ['one_shot', 'self_review', 'crewai']]
full_brief = '\\n\\n'.join(f'[{name}]\\n{text}' for name, text in sorted(KNOWLEDGE_BASE.items()))
for name, row in standard.items():
    (OUT_DIR / f'{name}_strategy.md').write_text(row['answer'] + '\\n')
print([{'case': row['case_id'], 'system': row['system'], 'error': row['error'],
        'seconds': round(row['latency_seconds'], 2)} for row in strategy_study])
'''

    evaluation = HELPERS + "\n" + cells["b3a5ffa9"]
    evaluation = replace_once(evaluation, 'task_success = any(term in answer_lower', 'task_success = all(term in answer_lower')
    evaluation = replace_once(evaluation, "        'id': case['id'],", "        **agent_accounting(result, case),\n        'id': case['id'],")
    evaluation = replace_once(evaluation, "    'routing_accuracy': float(agent_results['routing_correct'].mean()),", "    'harness_routing_accuracy': float(agent_results['routing_correct'].mean()),\n    'model_routing_accuracy': float(agent_results['model_routing_correct'].mean()),\n    'fallback_rate': float(agent_results['fallback_used'].mean()),")
    evaluation = replace_once(evaluation, "crew_dimensions, crew_strategy_score = strategy_rubric(crew_strategy)", "crew_dimensions, crew_strategy_score = strategy_rubric(crew_strategy)\nself_review_dimensions, self_review_score = strategy_rubric(self_review_strategy)")
    evaluation = replace_once(evaluation, "    {'system': 'crewai', 'score': crew_strategy_score, **crew_dimensions},", "    {'system': 'crewai', 'score': crew_strategy_score, **crew_dimensions},\n    {'system': 'single_agent_self_review', 'score': self_review_score, **self_review_dimensions},")
    evaluation = replace_once(evaluation, "    'model': MODEL_NAME,", "    'model': MODEL_NAME,\n    'evaluation_version': 'upgrade-v1',\n    'strategy_score_definition': 'Keyword coverage smoke proxy, NOT independent content quality',\n    'aggregate_output_token_cap_per_arm': TOTAL_OUTPUT_BUDGET,\n    'usage_by_arm': {'one_shot': baseline_usage, 'self_review': self_review_usage, 'crewai': crew_usage},")
    evaluation = replace_once(evaluation, 'strategy_lift = 100 * (crew_strategy_score - baseline_strategy_score) / max(baseline_strategy_score, 1e-9)',
                              'strategy_lift = 100 * (crew_strategy_score - baseline_strategy_score) / baseline_strategy_score if baseline_strategy_score else None')
    evaluation = replace_once(evaluation, 'print(f"CrewAI strategy quality improvement: {strategy_lift:.2f}%")',
                              'print("First-case keyword proxy change (%), undefined if baseline is zero:", strategy_lift)')
    evaluation += '''
review_pack, review_key = strategy_review_pack(strategy_tasks, strategy_study, SEED)
(OUT_DIR / "blind_review.json").write_text(json.dumps(review_pack, indent=2))
(OUT_DIR / "review_key.json").write_text(json.dumps(review_key, indent=2))
'''
    start = evaluation.index('evaluation_records = []')
    end = evaluation.index('def strategy_rubric(', start)
    router_evaluation = (ROOT / 'tools/project3_router_evaluation.py').read_text()
    evaluation = evaluation[:start] + router_evaluation + '''
AGENT_TESTS.extend([
    {'id': 'c1', 'task': 'Calculate an email campaign budget.', 'tool': 'request_clarification'},
    {'id': 'c2', 'task': 'Save a report.', 'tool': 'request_clarification'},
    {'id': 'a1', 'task': 'Book a flight to Boston tomorrow.', 'tool': 'abstain'},
])
evaluation_records, local_agent_metrics = evaluate_router_arms(
    AGENT_TESTS, run_local_agent, generation_usage, seed=SEED)
agent_results = pd.DataFrame(evaluation_records)
(OUT_DIR / 'router_per_case.json').write_text(json.dumps(evaluation_records, indent=2))
local_agent_metrics['paired_task_success_constrained_vs_prompt_only'] = paired_mean_interval(
    [row['task_success'] for row in evaluation_records if row['system'] == 'prompt_only'],
    [row['task_success'] for row in evaluation_records if row['system'] == 'constrained'], SEED)

''' + evaluation[end:]
    evaluation = replace_once(evaluation, "    'model': MODEL_NAME,", '''    'model': MODEL_NAME,
    'model_revision': MODEL_REVISION,
    'router_schema_sha256': fingerprint(decision_schema(LANGCHAIN_TOOLS)),
    'router_prompt_sha256': fingerprint(ROUTER_INSTRUCTION),
    'knowledge_base_sha256': fingerprint(KNOWLEDGE_BASE),
    'router_cases_sha256': fingerprint(AGENT_TESTS),
    'gpu': torch.cuda.get_device_name(0),
    'decoding': {'do_sample': False, 'router_max_new_tokens': 192, 'router_attempts': 2},''')
    evaluation = replace_once(evaluation, "    'strategy': {", '''    'strategy_tasks': len(strategy_tasks),
    'strategy_cases_sha256': fingerprint(strategy_tasks),
    'aggregate_input_token_cap_per_arm': TOTAL_INPUT_BUDGET,
    'strategy_independent_content_ratings': None,
    'strategy_budget_note': 'Matched maximum token budgets, not equal actual compute. Record actual input/output use per case; no independent content scores yet.',
    'strategy': {''')
    return {"4b855404": setup, "87bbb7e4": config, "c576a49d": router,
            "a24313e8": crew, "b3a5ffa9": evaluation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviewed', action='append', default=[], metavar='CELL_ID=SHA256',
                        help='Allow replacement of an explicitly reviewed current cell source')
    args = parser.parse_args()
    reviewed = dict(item.split('=', 1) for item in args.reviewed)
    changed = []
    for filename, transform in [
        ("01_llm_fine_tuning_pipeline.ipynb", project1),
        ("02_rag_document_qa_system.ipynb", project2),
        ("03_llm_agent_system.ipynb", project3),
    ]:
        path = ROOT / filename
        original_text = subprocess.check_output(["git", "show", f"HEAD:{filename}"], cwd=ROOT, text=True)
        original = json.loads(original_text)
        second_line = original_text.splitlines()[1]
        json_indent = len(second_line) - len(second_line.lstrip()) or 2
        before = {cell.get("id"): "".join(cell["source"]) for cell in original["cells"]}
        replacements = transform(before)
        notebook = json.loads(path.read_text())
        for position, cell in enumerate(notebook["cells"], 1):
            cell_id = cell.get("id")
            if cell_id not in replacements:
                continue
            current, target = "".join(cell["source"]), replacements[cell_id]
            if current == target:
                continue
            if current != before[cell_id] and reviewed.get(cell_id) != hashlib.sha256(current.encode()).hexdigest():
                raise RuntimeError(f"User-edited cell requires review: {filename} cell {position} {cell_id}")
            cell["source"] = target.splitlines(keepends=True)
            if cell["cell_type"] == "code":
                cell.update(outputs=[], execution_count=None)
            changed.append({"notebook": filename, "cell": position, "id": cell_id})
        # Never change the user's dataset-preparation source or kernel metadata.
        if filename.startswith("01_"):
            assert "".join(notebook["cells"][7]["source"]) == before["253ca33c"]
            description = '''## Isolated Training Environment (Upgrade v1)

Shared Colab packages are never removed to resolve trainer conflicts. Cell 4 checks
an isolated LLaMA-Factory 0.9.5 environment and stops if the shared kernel is selected.
For VS Code + Colab, run `python tools/build_upgrade_runner.py` locally and open the
generated runner notebook in the same Colab server. It installs a separate Python
environment, prompts for credentials without saving them, and runs the reviewed
dataset/training/evaluation cells in a child interpreter. The original dataset
preparation source is preserved. The run captures exact resolved dependencies.
'''
            current = "".join(notebook["cells"][2]["source"])
            if current not in ["".join(original["cells"][2]["source"]), description]:
                raise RuntimeError("Preserving user-edited environment description")
            if current != description:
                notebook["cells"][2]["source"] = description.splitlines(keepends=True)
                changed.append({"notebook": filename, "cell": 3, "id": None})
            task_helpers = (ROOT / "tools/agent_task_evaluation.py").read_text().replace("from __future__ import annotations\n", "")
            task_runner = (ROOT / "tools/project1_task_evaluation.py").read_text().split('\nif __name__ == "__main__":')[0]
            task_runner = task_runner.replace("from __future__ import annotations\n", "").replace("from agent_task_evaluation import compare_tasks\n", "")
            additions = [
                {"cell_type": "markdown", "id": "p1-task-smoke-description", "metadata": {}, "source": [
                    "## 7. Agent Task Smoke Comparison\n",
                    "Score base versus adapter with identical deterministic decoding on six authored tasks.\n",
                    "Tool selection, argument validity, and abstention are separate from NLL. This small set\n",
                    "is not an independent benchmark. Generated-code correctness remains unmeasured until\n",
                    "a separately validated isolated executor is available; no generated code runs in Colab.\n"]},
                {"cell_type": "code", "id": "p1-task-smoke-evaluation", "metadata": {}, "execution_count": None,
                 "outputs": [], "source": (task_helpers + "\n" + task_runner + '''
task_comparison = evaluate_adapter_tasks(MODEL_NAME, MODEL_REVISION, ADAPTER_DIR,
                                        RUN_DIR / "agent_task_smoke.json")
print(json.dumps(task_comparison["metrics"], indent=2))
''').splitlines(keepends=True)},
            ]
            for addition in additions:
                existing = next((cell for cell in notebook["cells"] if cell.get("id") == addition["id"]), None)
                if existing is None:
                    notebook["cells"].append(addition)
                    changed.append({"notebook": filename, "cell": len(notebook["cells"]), "id": addition["id"]})
                elif existing["source"] != addition["source"]:
                    raise RuntimeError("Review changes to the existing task-evaluation cell before replacement")
        path.write_text(json.dumps(notebook, indent=json_indent) + "\n")
    print(json.dumps(changed, indent=2))


if __name__ == "__main__":
    main()
