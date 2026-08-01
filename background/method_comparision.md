# Method Comparison and Alternatives

This guide compares design choices **within each project**. Each section starts
from the implementation in the associated notebook, then considers replacements
for that project's methods, functions, and metrics. It does not rank methods or
scores across different projects because their objectives and evaluation units
are different.

## Project 1: Efficient LLM Fine-Tuning

### Current Procedure

The implemented path validates and caches Stack Exchange records, formats them
for LLaMA-Factory, trains a QLoRA adapter on a 4-bit Llama 3 8B base model, and
compares adapter-on and adapter-off answer likelihood on the same held-out
examples. Important functions are `validate_cached_jsonl()`, `clean_rows()`,
`to_llamafactory()`, `encode_eval_example()`, and `forward_nll()`.

### Method and Function Alternatives

| Current choice | Alternative | Advantages of changing | Disadvantages of changing | Prefer the alternative when |
| --- | --- | --- | --- | --- |
| Custom streaming, validation, and JSONL caching with `validate_cached_jsonl()` and `clean_rows()` | Hugging Face `Dataset.map()` / `filter()` with Arrow caching | Parallel transforms, schema features, fingerprints, and easier dataset inspection | More library coupling; cache behavior can be less obvious in ephemeral Colab storage | The preprocessing pipeline has many transforms, multiple datasets, or distributed workers |
| Supervised fine-tuning on instruction-answer pairs | Continued pretraining on domain text | Learns domain vocabulary and writing patterns without requiring instruction pairs | Does not directly teach instruction following; objective and evaluation need redesign | Large quantities of clean domain text exist but paired answers do not |
| Supervised fine-tuning | DPO, ORPO, or another preference objective | Can optimize preferred behavior and reduce undesirable answer styles | Requires trustworthy chosen/rejected pairs; more sensitive to preference-data quality | Preference labels are available and SFT has already established basic task behavior |
| QLoRA with NF4 base weights | BF16 LoRA without base quantization | Simpler numerical behavior and a cleaner base-model comparison | Substantially higher GPU memory use | The model fits comfortably in memory and maximum numerical fidelity matters |
| QLoRA with NF4 base weights | Full-parameter fine-tuning | Maximum adaptation capacity; every model weight can change | Large memory, optimizer-state, storage, and catastrophic-forgetting costs | Sufficient multi-GPU compute and a large, high-quality domain dataset justify the expense |
| LoRA applied to all target modules | Smaller target set or lower LoRA rank | Fewer trainable parameters, smaller adapters, and faster updates | May underfit or miss task-relevant transformations | Deployment size or training cost matters more than maximum adaptation |
| LLaMA-Factory CLI and YAML | Hugging Face Trainer or TRL | Direct control over callbacks, custom losses, and research experiments | More implementation and maintenance work | The experiment needs a custom objective or behavior not exposed by LLaMA-Factory |
| LLaMA-Factory CLI and YAML | Unsloth or a DeepSpeed/FSDP stack | Unsloth can improve single-GPU efficiency; DeepSpeed/FSDP supports larger distributed runs | Additional compatibility constraints and a more complex failure surface | Training speed is the bottleneck or the model must span multiple GPUs |
| Sequence packing | Length bucketing without packing | Simpler example boundaries and easier debugging of labels | More padding and lower token throughput | Examples are already long or packing complicates loss masking |
| Gradient checkpointing | Store all activations | Faster backward pass because activations are not recomputed | Higher peak GPU memory | The model and batch fit in memory with room to spare |
| Custom `encode_eval_example()` with answer-only masking | Evaluate loss over prompt and answer together | Simpler encoding and compatibility with generic evaluators | Prompt length can dominate the score and obscure answer quality | The intended metric is whole-conversation likelihood rather than answer quality |
| `forward_nll(adapter_enabled)` for a paired adapter-on/off comparison | Load two separately materialized models | Supports comparisons across unrelated model checkpoints | More memory and more opportunities for tokenizer, quantization, or revision mismatch | Comparing models that cannot share one base-model instance |
| Minimal FastAPI generation endpoint | vLLM, Text Generation Inference, or Triton | Continuous batching, production metrics, higher throughput, and stronger serving controls | More deployment complexity and adapter-management decisions | Serving performance, concurrency, or operational reliability is part of the goal |

### Metric Alternatives

| Current metric | Alternative | Advantages of changing | Disadvantages of changing |
| --- | --- | --- | --- |
| Answer-token NLL | Exact match or token F1 | Directly evaluates generated answer overlap | Penalizes valid paraphrases and depends strongly on decoding |
| Perplexity | Code execution, unit tests, or pass rate | Measures whether software answers actually work | Applies only to executable tasks and needs secure sandboxes plus test cases |
| Perplexity | Human preference review | Captures usefulness, clarity, and factual concerns missed by likelihood | Expensive, slower, and subject to reviewer variation |
| Paired example win rate | Mean or median paired NLL change | Preserves the magnitude of improvements rather than only their direction | A few large changes can influence the mean; the median can hide broad small gains |
| Percentile paired-bootstrap interval | Paired t interval | Fast and analytically simple | Relies more heavily on distributional assumptions |
| Percentile paired-bootstrap interval | Wilcoxon signed-rank or permutation test | Provides a hypothesis test with fewer parametric assumptions | Produces a significance result rather than a directly interpretable effect-size interval |
| One trained adapter and one evaluation sample | Multiple training seeds with hierarchical confidence intervals | Captures training randomness as well as evaluation-sample uncertainty | Multiplies GPU time and experiment-management cost |

### Recommended Changes by Goal

- For a stronger job-application claim, keep QLoRA but add executable-code tests,
  factuality review, the full 500-example holdout, and multiple seeds.
- For lower cost, reduce LoRA rank or target modules and measure whether the NLL
  and generation metrics remain acceptable.
- For production serving, preserve the adapter artifact but replace the minimal
  FastAPI path with vLLM or Text Generation Inference and run load tests.

## Project 2: RAG Document QA

### Current Procedure

The implemented baseline uses E5 embeddings and exact FAISS inner-product search.
The advanced path adds HyDE, BM25, reciprocal-rank fusion, cross-encoder
reranking, and evidence-first generation. Key functions include `dense_rank()`,
`sparse_rank()`, `reciprocal_rank_fusion()`, `rerank()`,
`baseline_retrieve()`, `advanced_retrieve()`, `token_f1()`,
`retrieval_metrics()`, and `citation_validity()`.

### Method and Function Alternatives

| Current choice | Alternative | Advantages of changing | Disadvantages of changing | Prefer the alternative when |
| --- | --- | --- | --- | --- |
| Recursive 900-character chunks with 120-character overlap | Fixed token chunks | Aligns chunk size with model token budgets and is simple to batch | Can split headings, sentences, and semantic units | Documents have weak structure and predictable tokenization is most important |
| Recursive character chunks | Semantic or structure-aware chunks by section, paragraph, or embedding change | Better topical coherence and potentially cleaner citations | More preprocessing, model calls, and tuning; chunk sizes may vary widely | Documents have reliable headings or long sections containing multiple topics |
| `intfloat/e5-small-v2` dense embeddings | Larger E5 or BGE embedding model | Usually improves semantic retrieval quality | More GPU memory, encoding time, storage, and query latency | Retrieval quality matters more than low-cost indexing and serving |
| Local E5 embeddings | Hosted embedding API | Easier scaling and access to managed high-quality models | Recurring cost, network latency, privacy review, and vendor dependency | Operations simplicity is worth external data processing and usage cost |
| Exact `faiss.IndexFlatIP` | FAISS HNSW or IVF/PQ approximate search | Much faster and smaller at large corpus sizes | Requires index tuning and can miss nearest neighbors | The corpus grows beyond the point where exact scanning meets latency targets |
| In-memory FAISS plus NumPy cache | Qdrant, Milvus, Weaviate, Pinecone, or another vector database | Persistence, metadata filters, updates, replication, access control, and service APIs | Operational overhead or managed-service cost | The application needs multi-user production retrieval and incremental updates |
| BM25 via `sparse_rank()` | SPLADE or another learned sparse retriever | Retains lexical-style indexing while adding learned term expansion | Larger indexes and additional model inference | Exact terms matter but standard BM25 recall is insufficient |
| Raw query plus one HyDE passage | Multi-query expansion | Generates several query perspectives and can improve recall without assuming one answer | More generation and retrieval calls; duplicate or drifting candidates | Questions are ambiguous and one hypothetical passage is unstable |
| Raw query plus one HyDE passage | Deterministic query rewriting or no expansion | Lower latency, easier caching, and less hallucination-driven query drift | May lose vocabulary bridging between short questions and long passages | Latency is strict or HyDE ablations show little retrieval benefit |
| Unweighted reciprocal-rank fusion | Tuned weighted RRF | Can emphasize the stronger retriever for this domain | Needs a separate development set and may overfit | Dense and sparse retrieval have consistently different reliability |
| Reciprocal-rank fusion | Normalize and combine raw scores | Preserves score magnitude that rank-only fusion discards | Dense and BM25 scores are not naturally comparable and calibration can drift | Scores can be calibrated reliably across queries and model revisions |
| Cross-encoder reranking with `rerank()` | No reranker | Large latency and compute reduction | Lower top-k precision and potentially weaker generation context | The first-stage retriever is already strong or latency dominates quality |
| Cross-encoder reranking | ColBERT late interaction | Better scaling than a full cross-encoder while retaining token-level matching | More complex indexing and larger storage than a bi-encoder | The corpus is large and reranking every candidate is too expensive |
| Cross-encoder reranking | LLM reranker | Can apply nuanced relevance instructions and domain criteria | High latency, cost, nondeterminism, and harder calibration | Candidate sets are small and relevance requires complex reasoning |
| Local Llama 3 8B evidence-first generator | Smaller local instruct model | Lower latency and memory use | May reduce answer quality, citation compliance, and robustness | The retrieved evidence is simple and serving cost is critical |
| Citation IDs checked by `citation_validity()` | Claim-level entailment verifier | Tests whether citations actually support generated claims | Adds model calls, thresholds, and false-positive/negative decisions | Citation reliability is a production requirement rather than a formatting check |

### Metric Alternatives

| Current metric | Alternative | Advantages of changing | Disadvantages of changing |
| --- | --- | --- | --- |
| Answer-hit based on answer-string containment | Recall@k using human-validated gold passages | Measures retrieval directly and accepts paraphrased evidence | Requires passage-level relevance labels |
| Reciprocal rank | nDCG@k | Handles multiple relevance levels and rewards good ordering across the list | Requires graded relevance judgments |
| Token F1 | BERTScore or embedding similarity | Gives more credit to semantically correct paraphrases | Model-dependent and less transparent than token overlap |
| Token F1 | Exact match | Very simple and strict | Under-credits correct wording variations |
| Citation validity | Citation precision, recall, and entailment | Separates citation syntax, coverage, and factual support | More annotation and verification cost |
| Declared composite score | Report a metric vector or Pareto frontier | Avoids hiding tradeoffs behind subjective weights | Harder to summarize with one portfolio number |
| Mean seconds per question | Median, p95, throughput, and stage-level latency | Reveals tail behavior and identifies HyDE, retrieval, reranking, or generation bottlenecks | Requires more runs and careful warm-up controls |

### Recommended Changes by Goal

- For higher retrieval validity, add gold passage judgments and report Recall@k,
  MRR, and nDCG before changing the retriever.
- For lower latency, cache HyDE, batch reranking, test no-HyDE and no-reranker
  ablations, then move to ANN only when corpus scale requires it.
- For production, replace the local FAISS reconstruction path with a persistent
  vector database and add metadata access-control filters plus citation entailment.

## Project 3: Local LLM Agent System

### Current Procedure

The implemented controller asks a local Llama planner for JSON, validates an
`AgentDecision`, retries once, and uses `fallback_decision()` if needed.
`run_local_agent()` invokes one of three typed tools through policy and tracing
boundaries. A separate sequential CrewAI path decomposes strategy work into
researcher, strategist, critic, and editor roles.

### Method and Function Alternatives

| Current choice | Alternative | Advantages of changing | Disadvantages of changing | Prefer the alternative when |
| --- | --- | --- | --- | --- |
| Prompted JSON from `plan_action()` | Native function calling or structured-output API | Higher schema compliance and less JSON-repair code | Provider or model feature dependency; still requires authorization checks | The selected model reliably supports constrained tool calls |
| Prompted JSON | Grammar-constrained local decoding | Keeps inference local while guaranteeing syntactic structure | Integration complexity and grammar/model compatibility concerns | Native provider tools are unavailable but valid structure is mandatory |
| Pydantic `AgentDecision` validation | Plain dictionary checks | Fewer dependencies and less code for tiny prototypes | Weaker error messages, coercion rules, and generated schemas | The contract is very small and never crosses a service boundary |
| One retry plus `fallback_decision()` | More LLM retries | May recover unusual tasks without expanding deterministic rules | Higher latency and no guarantee of correctness; repeated failures waste compute | Malformed output is rare and each task is high value |
| One retry plus deterministic fallback | Human approval or clarification | Safer for ambiguous, consequential actions | Breaks full automation and adds user wait time | The action is destructive, expensive, or legally sensitive |
| BM25 `search_knowledge_base()` | Dense or hybrid retrieval | Better semantic matching when query and source vocabulary differ | Adds embedding models, indexes, and retrieval tuning | The knowledge base grows or contains many paraphrased concepts |
| Deterministic `calculate_campaign_budget()` | Let the LLM calculate in text | Fewer tool calls and simpler demonstrations | Arithmetic errors, poor auditability, and inconsistent formatting | Almost never appropriate when deterministic code is available |
| Approval-gated `save_workspace_report()` with path confinement | Unrestricted filesystem tool | Greater flexibility for an autonomous development environment | Severe traversal, overwrite, exfiltration, and destructive-action risk | Only inside a strongly isolated disposable sandbox with additional controls |
| In-process MCP-style manifest and dispatcher | Real MCP server/client transport | Standard discovery, process isolation, reusable tools, and transport-level testing | More deployment, versioning, timeout, and authentication work | Multiple clients or processes must use the same tool service |
| Per-call `traced()` records | OpenTelemetry traces and centralized audit storage | Cross-service correlation, dashboards, retention controls, and production observability | Infrastructure cost and privacy/redaction responsibilities | The agent runs as a maintained service rather than a notebook prototype |
| Extractive `grounded_answer()` fallback | Always use generative answers | More natural language and synthesis | Greater hallucination risk when evidence is weak | Evidence is rich and an entailment verifier protects the output |
| Sequential CrewAI roles | Single well-structured agent prompt | Much lower latency and simpler state management | Less role specialization and fewer inspectable review stages | The task is routine or interactive latency matters |
| Sequential CrewAI roles | Graph workflow with conditional branches | Explicit state, retries, early exits, and selective role execution | More orchestration code and testing | Different tasks need different role paths or the critic should trigger revisions |
| Sequential CrewAI roles | Parallel independent agents followed by synthesis | Reduces wall-clock time and increases viewpoint diversity | Duplicate work, higher total token use, and conflict-resolution needs | Research subtasks are independent and parallel compute is available |
| Local Llama 3 8B for every role | Smaller role-specific models | Lower latency and memory cost; models can match task complexity | More deployment artifacts and inconsistent behavior across roles | Some roles are classification, extraction, or formatting tasks that do not need 8B capacity |

### Metric Alternatives

| Current metric | Alternative | Advantages of changing | Disadvantages of changing |
| --- | --- | --- | --- |
| Tool-routing accuracy | Per-tool precision, recall, and confusion matrix | Reveals which tools are confused and handles imbalanced test suites | Needs more cases per tool |
| Planner valid-JSON rate | Schema-valid tool-call rate before any repair | More directly measures structured planner reliability | Still does not measure whether the selected action is correct |
| End-to-end task success | Partial-credit task rubric | Distinguishes complete failure from useful but incomplete execution | Requires careful scoring rules and possibly human review |
| Citation rate | Claim-level groundedness and citation entailment | Detects unsupported claims even when a citation is present | More expensive and less deterministic |
| Safety-block rate | Attack success rate over a larger adversarial suite | Covers prompt injection, indirect injection, and permission escalation more realistically | Building and maintaining representative attacks is substantial work |
| Median and p95 latency | Stage latency, throughput, tokens, and cost per successful task | Identifies planner, tool, or generation bottlenecks and supports capacity planning | Requires instrumentation and controlled load testing |
| Lexical strategy rubric | Blinded human scoring | Better captures usefulness, coherence, and business quality | Reviewer cost and inter-rater disagreement |
| Lexical strategy rubric | Calibrated independent LLM judge | Scales beyond manual review and can use detailed criteria | Judge bias, model dependence, prompt sensitivity, and possible self-preference |
| One 12-case suite | Larger stratified and adversarial evaluation over multiple seeds | Produces stronger reliability and safety evidence | Higher runtime and test-maintenance cost |

### Recommended Changes by Goal

- For planner reliability, use constrained structured output first, while keeping
  Pydantic validation and deterministic authorization outside the model.
- For production tools, expose the contracts through a real MCP server, add
  timeouts and cancellation, and export redacted OpenTelemetry traces.
- For lower CrewAI latency, route simple requests to one agent, parallelize
  independent research, and invoke critic/editor stages only when quality gates
  fail.

## Related Project Material

- [Project 1 background](project1_background.md)
- [Project 2 background](project2_background.md)
- [Project 3 background](project3_background.md)
- [Repository overview](../README.md)
