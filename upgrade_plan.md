# Upgrade Plan: Projects 1-3

> **Updated:** 2026-11-17 02:35 EST<br>
> **Version:** 1.5: controlled GPU evidence and bounded automated reviews<br>
> **Status:** Implementation started; unchecked items are not completed or validated.

## Scope And Rules

The read-only audit found mostly current core methods with older model baselines,
fragile dependency management, and evaluation limitations. This plan preserves
the existing experiments rather than treating new code as new measured results.

- **P0:** Safety, reproducibility, or evidence defects to address first.
- **P1:** Correctness, integration, and controlled evaluation.
- **P2:** Optimizations justified by the new measurements.
- Implement steps in priority order. Keep historical outputs intact and separate
  new evaluation artifacts by run/version.
- Preserve user-modified cells, especially Project 1 dataset preparation. Report
  every changed notebook cell by position and stable cell ID.
- Never claim a model improvement from an unexecuted notebook. Mark code checks,
  deterministic tests, and model-backed evaluations separately.
- Stop and ask for help after three failed attempts on the same issue. Record
  the issue, attempts, and resolution in the debug log.
- Keep existing local-only artifact policies. No automatic commit or push.

## Shared Checklist

- [x] **S1 / P0:** Record affected notebook cells and preserve baseline outputs.
- [ ] **S2 / P0:** Replace destructive environment repair with an isolated,
  reproducible setup; pin source revisions and generate a tested dependency lock.
- [x] **S3 / P1:** Add deterministic regression checks and notebook validation.
- [ ] **S4 / P1:** Record model/dataset/prompt revisions, decoding settings,
  hardware, sample counts, latency, and cost with every new evaluation.
- [ ] **S5 / P2:** Compare newer small models under matched task and resource
  budgets. Keep the old checkpoints as historical baselines.

**Acceptance:** A fresh supported environment reproduces the smoke checks without
uninstalling unrelated packages. Model-backed claims have saved per-case evidence.

## Project 1: Fine-Tuning

**Evidence:** [Notebook](01_llm_fine_tuning_pipeline.ipynb), setup cell 4
(`c2e49511`), training cell 17 (`e1480002`), evaluation cell 20 (`18ed85c3`).

**Finding:** QLoRA, NF4, packing, and paired bootstrap remain useful. The current
Llama 3 checkpoint is an older baseline; the 100-example NLL/perplexity result
measures reference-answer fit, not agent task completion. Installing a moving
LLaMA-Factory checkout into a manually repaired Colab environment is fragile.

### Ordered Checklist

- [x] **P1-01 / P0:** Make setup reproducible without deleting unrelated Colab
  packages. Resolve a tested environment and immutable trainer revision.
- [x] **P1-02 / P1:** Preserve the dataset-preparation cell and cached JSONL.
  Expand the paired NLL evaluation to the full available holdout, expose the
  actual evaluated count, and keep answer-token masking and uncertainty estimates.
- [x] **P1-03 / P1:** Add a separately versioned task-evaluation interface for
  tool selection, argument correctness, abstention, and executable code tests.
  Generated code must run only in an isolated executor, never directly in Colab.
- [x] **P1-04 / P1:** Execute the new held-out comparison and save per-example
  records, confidence intervals, failures, and resource measurements.
- [ ] **P1-05 / P2:** Compare a newer small model and, only if error analysis
  supports it, tool-trajectory SFT against Q&A SFT under a fixed GPU-hour budget.

**Acceptance:** Distribution-fit results and task-success results are reported
separately. No broad "agent improvement" claim is inferred from perplexity alone.

## Project 2: RAG Document QA

**Evidence:** [Notebook](02_rag_document_qa_system.ipynb), retrieval cell 15
(`01d9a1ca`) and evaluation cell 17 (`45f0634e`).

**Finding:** BM25, dense retrieval, RRF, reranking, and FAISS remain appropriate.
HyDE is run unconditionally; baseline and advanced prompts/context counts differ.
Citation validity checks ID membership rather than whether evidence supports the
answer. Reference-substring retrieval metrics are proxies, not relevance labels.

### Ordered Checklist

- [x] **P2-01 / P0:** Use identical answer prompts, decoding settings, and final
  context budgets when comparing retrieval variants. Label proxy metrics honestly.
- [x] **P2-02 / P1:** Add dense, hybrid, hybrid-plus-reranker, and HyDE ablations;
  retain a no-HyDE path and measure generation plus retrieval latency together.
- [x] **P2-03 / P1:** Add a held-out relevance/claim-support evaluation input,
  unanswerable cases, and paired uncertainty. Do not equate valid IDs with grounding.
- [x] **P2-04 / P1:** Fingerprint embedding caches with model revision, corpus,
  chunking, and normalization settings rather than checking row count alone.
- [x] **P2-05 / P1:** Run the new ablations and save their individual metrics,
  failures, p50/p95 latency, and evaluation sample counts.
- [ ] **P2-06 / P2:** Test conditional HyDE and newer embedding/reranking models.
  Add a vector database only when filtering, updates, tenancy, or scale needs it.

**Acceptance:** Gains can be attributed to a changed component; citations are
evaluated for support separately from formatting; difficult/absent answers remain
in the benchmark instead of being silently filtered out.

## Project 3: Tool-Using And Multi-Agent Workflow

**Evidence:** [Notebook](03_llm_agent_system.ipynb), router cell 8, MCP-style
manifest cell 10 (`3396dd38`), CrewAI cell 12 (`a24313e8`), and evaluation
cell 14 (`b3a5ffa9`).

**Finding:** Prompt-only JSON and rule fallbacks can hide routing failures. The
four-role workflow has a larger generation budget and a keyword-based rubric;
the current comparison does not isolate the value of multiple agents. The local
tool manifest is MCP-style, not a demonstrated interoperable MCP service.

### Ordered Checklist

- [x] **P3-01 / P0:** Separate model decisions, parse failures, fallbacks, tool
  execution, and end-task success in evaluation records.
- [x] **P3-02 / P1:** Add schema-constrained generation/native tool calling with
  semantic validation and explicit unsupported/ambiguous-task behavior.
- [x] **P3-03 / P1:** Implement one-shot, single-agent self-review, and role-based
  workflows under matched aggregate output/cost budgets and shared requirements.
- [x] **P3-04 / P1:** Add a blinded, versioned content rubric and multiple tasks.
  Treat keyword checks as smoke tests, not independent quality judgments.
- [x] **P3-05 / P1:** Run comparisons and report task success, fallbacks, safety,
  p50/p95 latency, and cost with per-task evidence.
- [ ] **P3-06 / P2:** Integrate a real MCP client/server using the current SDK
  and supported transport, borrowing the tested boundary from Project 7.

**Acceptance:** Fallback behavior is visible; any claimed multi-agent benefits must survive an
equal-budget comparison; interoperability is claimed only after client/server tests.

## Suggested Execution Order

1. Shared reproducibility and validation work.
2. Project 2 comparison/metric corrections and Project 3 fallback accounting.
3. Project 1 expanded evaluation without changing dataset preparation.
4. Controlled model-backed evaluations; leave blocked runs explicitly pending.
5. Newer models, adaptive retrieval, and protocol integration after correctness.

## Primary References

- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B): candidate,
  not a drop-in replacement or a guaranteed improvement.
- [Qwen3 embedding and reranking](https://github.com/QwenLM/Qwen3-Embedding).
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).
- [MCP transport specification](https://modelcontextprotocol.io/specification/latest/basic/transports).

## Implementation Record

Projects 1-3 completed new GPU comparisons, and Projects 2/3 received bounded
automated reviews. Completion of an
experiment does not mean that its method improved the result.

| Project | Notebook Cells Changed (1-Based) | Implemented / Pending |
| --- | --- | --- |
| 1 | 3 (environment markdown), 4 `c2e49511`, 19 `257323c9`, 20 `18ed85c3`; added 23 `p1-task-smoke-description`, 24 `p1-task-smoke-evaluation` | Fail-closed environment preflight; full available holdout, skip counts, paired token-weighted CI; separate six-task tool/argument/abstention smoke interface. Generated code is not executed without an isolated evaluator. Cell 8 `253ca33c` is unchanged. |
| 2 | 3 `67c7857a`, 4 `3d8e1486`, 6 `33724c09`, 11 `814ef015`, 13 `bdf0279c`, 15 `01d9a1ca`, 17 `45f0634e`, 19 `e9f2a964` | Four-arm GPU comparison and 60 blinded automated ratings complete; human/gold relevance labels remain pending. Cell 17 also excludes binary/absence cases from substring retrieval proxies and reports metric denominators; frozen outputs are rescored in `metrics_v2/`. |
| 3 | 3 `4b855404`, 4 `87bbb7e4`, 8 `c576a49d`, 12 `a24313e8`, 14 `b3a5ffa9` | Real grammar-constrained router, 15 paired cases, three workflow arms on three authored briefs, and nine blinded automated ratings completed. No demonstrated general multi-agent quality advantage; real MCP integration still pending in this project. |

- **Local validation:** 42 CPU metric/router/task/runtime/notebook tests pass on Python 3.11;
  the optional real-CrewAI test is skipped in that environment. All three strategy
  tests, including the real CrewAI 1.15.41 local-LLM bridge, pass in a separate
  credential-free environment with authored responses. Those unit tests make no API calls.
  LM Format Enforcer 0.11.3's actual JSON parser is tested; Transformers/GPU
  integration is now exercised by the completed GPU comparison. Constraints do not
  establish factual correctness or authorization.
- **CI:** Read-only, secret-free workflow added with immutable action commits;
  not pushed or executed on GitHub. CPU dependency snapshot is not a GPU lock.
- **A100:** Connection verified (40 GB); isolated LLaMA-Factory 0.9.5 environment
  installed and CUDA/import preflight passed. The base Colab environment was not altered.
- **Storage:** User-approved `MyDrive/projects/MyProject1/`; original artifacts absent.
  `tools/build_upgrade_runner.py` packages reviewed sources into a local transient
  runner. It preserves dataset-cell source, pins dataset/model revisions at run time,
  and writes per-run environment/configuration/status records to Drive.
- **Project 1 completed run:** `20261117T032756029425Z`, fresh retraining, not the
  original adapter. Training took 9,756.33 seconds for 701 steps on one A100;
  environment lock and source/data revisions are exported. The final holdout
  contains 500 scored examples, zero skips, and 95,502 answer tokens. Token-weighted
  NLL fell from 2.083986 to 1.719413; perplexity from 8.0364 to 5.5812 (30.55%).
  Paired token-weighted NLL reduction CI: [0.344408, 0.386997]; the macro-example
  CI [0.515079, 0.567639] answers a different weighting question.
  **Task regression:** valid schema/tool selection fell from 5/6 to 0/6 on the
  six authored prompts; valid argument matches fell from 4/5 to 0/5. A separate
  frozen-output Docker check passed the base model's one authored code test and
  rejected the tuned response before execution for invalid JSON. This adapter is not ready for agent
  deployment. Likely task-distribution mismatch or instruction-retention loss
  needs a separate development study, not rewriting these failures as successes.
  The user's dataset-preparation cell and raw cached JSONL remain unchanged.
  See [new evidence](output/project1/runs/upgrade-v1/20261117T032756029425Z/).
- **Next GPU runs:** `tools/build_gpu_upgrade_runner.py` prepares a sequential queue
  for the Project 1 task smoke and Projects 2/3/5/8. A Linux/Python 3.11 solver lock
  is generated; installation, dependency checks and CUDA imports succeeded. The
  queue completed Projects 2/3; runner-only integration fixes were needed for
  dynamic annotations, subprocess import paths and a missing test helper. Projects
  5/8 also completed in the sibling repository; their results are documented there.
  CrewAI's MCP 1.x requirements are kept separate from Project 7's tested MCP 2.x environment.
- **Review gates:** Three authored strategy briefs are not an independent benchmark.
  Shared maximum token budgets do not equalize actual GPU compute. RAG ID validity,
  lexical answer scores, and quote membership do not establish semantic claim support.
  User-approved automated review is bounded at $2 for Project 2 and $1 for Project 3.
  `tools/automated_review.py` blinds system identities, validates exact support
  quotes, retains uncertain labels/failures, and samples complete matched arm groups.
  Reviews completed for $0.455684 (Project 2, including rejected attempts) and
  $0.130832 (Project 3), totaling $0.586516. Three abbreviated supporting quotes
  triggered the requested stop; user-approved retry clarified continuous verbatim
  copying and preserved 40 valid ratings. All 60 final RAG and nine strategy ratings
  remain explicitly automated and not human-verified. Exact request hashes verify
  the archived review inputs. Human calibration and gold relevance remain pending.

See [upgrade debugging record](debug_log/upgrade_debug_log.md).
