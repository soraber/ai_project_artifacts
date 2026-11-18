# Upgrade Debug Log

Updated: 2026-11-17 EST. Local code changes and authorized remote execution; no push.

## Local Follow-Up: 2026-11-17 12:40 EST

- User deferred training/GPU work and approved LangGraph migration. Replaced only
  the new role workflow; historical CrewAI results and prior GPU artifacts remain.
  The graph propagates prior outputs, accounts for measured tokens, and stops on
  errors/missing usage. Eight local strategy tests pass without API calls.
- Added conditional HyDE as a fifth arm. Inspection found that invalid retrieval
  IDs after the top three could evade validation; validate the whole rank list
  and reject unknown/duplicate reranker outputs. Four control tests pass.
- Added real MCP 2.3 stdio discovery/search/calculation tests; write tools are not
  exposed. Two tests pass, including symlink escape rejection. This is a local
  protocol boundary, not remote OAuth or model-task accuracy.
- Final CPU/inference locks were regenerated after adding MCP and audited again:
  zero reported advisories. Final CPU suite: 57 passed, one real-torch test skipped
  in both macOS and network-disabled Linux ARM64. Three separate trainer tests
  pass, including a real tiny CPU LoRA update. A Transformers deprecation warning
  is retained; it does not indicate a failed test or a verified CUDA environment.
- Changed notebook cells: P1 3-5 and 15-20; P2 3/4/13/16/17;
  P3 2-4 and 11-14, with new 15-16. Full IDs are in SECURITY_UPGRADE.md.
  P1 dataset cell 8 source SHA256 is unchanged. No historical PDF was regenerated.
- No unresolved issue reached three failed attempts. No paid API evaluation,
  full training, GPU model run, commit, or push occurred in this follow-up.

## Earlier Execution Log

| Issue | Attempts / Evidence | Resolution / Status |
| --- | --- | --- |
| New Colab runtime had no Project 1 adapter or holdout | One scoped artifact probe | User authorized fresh training and persistence in MyDrive/projects/MyProject1/. Historical scores remain unchanged. |
| Drive mount timed out during authorization | One failed 60-second mount | User approved the Drive grant; authorization completed and the second mount succeeded. |
| Colab Secrets unavailable from VS Code | One warning from Hugging Face get_token | Use the masked getpass input and runtime-only environment variable. Never print or save the token in a notebook. |
| Router unit-test fixture omitted its prompt constant | One test failure | Completed the fixture. No model run was involved. |
| Retrieval comparison mixed prompts and context sizes | Audit finding | Shared answer prompt, five final chunks per arm, explicit dense/hybrid/reranker/HyDE ablations. |
| Citation-ID validity was treated as grounding | Audit finding | Separate ID validity, optional labeled relevance, and unmeasured claim support. Remove the arbitrary aggregate composite. |
| Routing fallback could hide model failure | Audit finding | Validate tool arguments before execution; record fallback/model routing separately and preserve abstention. |
| External notebook edit reset the selected cell to setup | One accidental execution of setup; uv refused the existing environment | No environment was replaced. Training continued normally. Verify selected cell before every UI execution. |
| Guessed arithmetic fallback used a default spend | Code review | Replace it with clarification; grammar constraints and strict semantic checks are independently tested. |
| Variable-length strategy output could exceed the context window | Code review | Reserve the full requested output budget and reject oversize input rather than silently truncate source evidence. |
| Executed-notebook export could retain later unexecuted cell outputs | One new runner unit-test failure | Clear all code outputs before execution, then checkpoint only newly produced outputs. |
| Mac locked during runtime monitoring | One UI observation | User unlocked the Mac; monitoring resumed without restarting training. |
| Same Colab server opened a different notebook kernel | One queue launch stopped before starting a process because HF_TOKEN was absent | Staged the already-reviewed files, then launched from the authenticated Project 1 controller. Added temporary controller cells 6-7; no token copied to source or notebook output. |

## Changed Notebook Cells

- Project 1: cells 3, 4 (`c2e49511`), 19 and 20 (`257323c9`, `18ed85c3`),
  plus new cells 23-24 (`p1-task-smoke-description`, `p1-task-smoke-evaluation`). Dataset preparation
  cell 8 (`253ca33c`) remains unchanged.
- Project 2: cells 3, 4, 6, 11, 13, 15, 17, 19 (`67c7857a`, `3d8e1486`, `33724c09`,
  `814ef015`, `bdf0279c`, `01d9a1ca`, `45f0634e`, `e9f2a964`).
- Project 3: cells 3, 4, 8, 12, 14 (`4b855404`, `87bbb7e4`, `c576a49d`, `a24313e8`, `b3a5ffa9`).
- Separate temporary runner under the jobs workspace has setup, secure auth,
  source staging, training/evaluation launch, and progress cells. It is not a
  replacement for the user's edited dataset cell.

No issue has remained unresolved after three attempts. The isolated A100 training
started at 2026-11-17 03:33 UTC, with 701 optimizer steps and checkpoint interval 50.
Artifacts and the full 50,500-row raw cache persist in the approved Drive folder.
The local Projects 1-3 suite now has 32 passing tests, including the real JSON grammar
parser, and one optional CrewAI integration test skipped in the core environment.
The three strategy tests also pass in the separate CrewAI 1.15.41 environment,
including its actual four-role execution with authored responses and no credentials.
Internal library deprecation warnings were observed, but no integration failure.
This is not a model-performance result. Fresh 500-example NLL evaluation
will run after training; the separately added agent-task smoke suite runs afterward.

The Linux inference dependency solver succeeded with Transformers 4.57.6, PyTorch
2.14.1, CrewAI 1.15.41, and LM Format Enforcer 0.11.3. Its GPU import/model checks
remain pending. Training reached step 562/701 at about 05:44 UTC. The sequential
queue was launched and verified alive, waiting for Project 1 training/NLL before
agent-task smoke, Projects 2/3, and Projects 5/8. Completion is not yet verified.
# 2026-11-17: Completed Retraining And Bounded Review Preparation

## Final Controlled Evaluations And Reviews

- Project 3's corrected run `20261117T064159242835Z` finished in 773.90 seconds.
  JSON/schema validity improved from 8/15 to 14/15; authored task checks from 5/15
  to 9/15. All nine strategy outputs were produced, but later automated content
  ratings did not establish a CrewAI advantage. Raw memos, calls, timing and
  failures remain separate from the historical keyword proxy.
- Project 2's judge stopped after three non-verbatim quotes (each abbreviated
  with `...`). It had accepted 40 ratings and spent $0.265492. After the user's
  explicit approval, a clarified continuous-quote instruction rated only the
  remaining 20 outputs for another $0.190192. The old failures remain intact;
  the completed review carries forward the first 40 ratings and prior spending.
- Project 3's nine automated reviews cost $0.130832. Combined review spend is
  $0.586516, below the approved $3 cap. No human-verified claim is made. Archived
  review inputs match all recorded request hashes, including both quote protocols.
- RAG measurement correction: yes/no reference strings are not meaningful
  substring-retrieval targets, and substring matching lacked word boundaries.
  Updated **Project 2 cell 17 `45f0634e`** and `evaluation_upgrade.py`; the
  corrected `metrics_v2/` evidence is rescored from frozen outputs without model
  calls. All questions remain in answer metrics; 28 non-binary cases contribute
  retrieval proxies and 32 binary cases contribute yes/no accuracy. Judge inputs
  and original per-case records are unchanged. Added denominator regression checks.
- Project 1's saved code response was checked through the already validated
  Project 6 Docker boundary, pinned by image ID. The base response passed one
  authored test with two assertions; the tuned response failed JSON validation
  and was never executed. No model regeneration or test-feedback loop occurred.
- Latest CPU verification: 42 passed, one optional CrewAI-environment skip.
  No changes were made to Project 1's protected dataset-preparation cell or the
  local-only interview PDF. No commit/push occurred.

- Project 1 finished a fresh 701-step run on the A100 in 9,756.33 seconds; the
  full 500-example holdout had no skipped rows. Saved environment versions,
  adapter configuration, trainer state, source/data revisions and per-example NLL
  under `output/project1/runs/upgrade-v1/20261117T032756029425Z/`.
- Lower perplexity (8.0364 -> 5.5812) did not transfer to the authored agent smoke:
  valid tool decisions regressed from 5/6 to 0/6. All raw failures are retained.
  No retuning on those test cases or new deployment-readiness claim is made.
  Code generation has not been executed by this smoke harness.
- Separate inference environment installed and passed dependency/CUDA preflight;
  the queue proceeded to Project 2. The shared Colab interpreter was not changed.
- Added a post-run, blinded automated-review CLI with $2/$1 project caps within
  the approved $3 total, no SDK retries, conservative failed-call reservations,
  exact evidence-quote checks and explicit uncertain labels. Five unit tests pass.
  No source notebook cell changed in this review-preparation step. Transient
  controller cells 8/9 only inventory/export small evidence within the approved
  Drive subtree and block credential-like strings before transfer.

## P0 Dependency And Trainer Follow-Up: 2026-11-17

- The advisory scan reopened environment hardening: pytest 8.4.2 and both historical
  GPU dependency closures had findings. CPU locks now use pytest 9.0.3 and
  distribution hashes. Legacy GPU launchers stop before constructing a new run.
- LLaMA-Factory 0.9.5 constrains Transformers to at most 5.6.0. A solver attempt
  with the patched floor failed; the user approved direct Transformers/PEFT training.
  The new lock audits clean. SoupSieve's patched release requires Python >=3.11.5;
  resolving against 3.11.0 failed once, then succeeded with an explicit microversion.
- The new Trainer API removed `overwrite_output_dir`. Signature inspection caught
  it before any GPU launch; directory creation now enforces non-overwrite. A real
  two-step CPU test verifies adapter-only updates and safetensors output. A test
  checks production TrainingArguments keywords against the installed API.
- The notebook migration initially encountered a legacy cell without an ID; changed
  the structured traversal to preserve such cells. Updated cells 3-5 and 15-20 only.
  Cell 8 source/hash and unrelated cell outputs remain intact. Evaluation refuses
  smoke adapters and requires the training model revision.
- The new CPU suite passes 45 tests, with optional CrewAI/tiny-torch integration
  tests skipped there; all three trainer tests pass in their separate CPU venv.
- VS Code currently reports zero active Colab sessions. Requested A100 reconnection;
  generated a separate credential-free NF4 tiny-model controller. Its GPU result
  is pending, not inferred from CPU tests. No full retraining was launched.
- Inspection confirmed `crewai-core` contains shared utilities, not Agent/Crew/Task.
  Chroma advisories in the old closure lack fixed-version entries; no blanket
  suppression was added. Requested approval before changing the role framework.

## First GPU Suite: Runner Integration Failures

The first suite `20261117T061830369288Z` completed Project 2 in 652.25 seconds,
including model/data setup and 63 questions x four arms. It had no per-case RAG
errors. All three absence probes received the expected abstention in every arm.
The following failures each occurred once; they are distinct integration issues,
not failed model answers or evidence for model quality:

- **Project 3:** Dynamic notebook compilation inherited `from __future__ import
  annotations` from the runner, while Pydantic resolved types against the wrong
  `__main__` module. `AgentDecision` could not resolve `Literal`. The runner now
  disables inherited compile flags and gives each stage a registered module
  namespace. A regression test reproduces the typed-cell execution successfully.
- **Project 5:** The dataset-generation subprocess could not import
  `project5_agent`: changing parent `sys.path` does not change child imports.
  The stage now sets `PYTHONPATH` to that project's `src` before launching its
  existing notebook subprocesses.
- **Project 8:** Test collection expected `tools/generate_report.py`, which was
  omitted from the packaged runtime. Added that reviewed helper to the bundle;
  report rendering dependencies are not needed just to import its pure scorer.
- A runner-only repair cell (transient controller **cell 10**, ID
  `gpu-resume-runner-fixes-v1`) updates source hashes and restarts only Projects
  3/5/8 in a new evidence directory. The training and completed Project 2 run
  are not rerun. No project source notebook cell changed for these runner fixes.
- Blinded Project 2 automated review started against the completed evidence;
  its first calls returned valid ratings within the approved budget.

## Project 1 MyPro Training: 2026-11-17 To 2026-11-18

- The notebook's original dataset preparation stayed in cell 8. Cells 25-27
  verified the mounted Drive split and A100 smoke result, checked gated model
  access, and launched the full run as a separate process on MyPro.
- The notebook initially showed a stale status after a VS Code window switch.
  A status refresh also selected the existing evaluation cell by mistake. That
  cell stopped at its missing-run guard. The background worker continued.
- Reloaded the VS Code window and selected the existing Project 01 Jupyter
  session on MyPro. Cell 28 now reads the run log and refreshes every 30 seconds.
  Its step counter advanced across the reload. Drive's separate file listing
  lagged the mounted folder during training and later showed the final files.
- Run `20261118T033050536430Z` completed 3,106 steps and scored all 500 holdout
  rows. Verified the 80.1 MiB adapter and paired evaluation JSON under
  `MyDrive/projects/MyProject1/runs/upgrade-v2/20261118T033050536430Z/`.

## MyPro Inference Follow-Up: 2026-11-18

- The Project 2 notebook launcher installed the separate, hash-locked
  Python 3.11.17 inference environment. Run `20261118T092816338337Z`
  completed 63 cases across five retrieval arms with zero errors.
- Project 3's first attempt `20261118T094559405926Z` stopped at cell 8.
  LM Format Enforcer 0.11.3 imported `PreTrainedTokenizerBase` from the
  Transformers 4 path. Cell 4 temporarily exposes the moved class during the
  integration import and restores the module afterward. The refreshed launcher
  in cell 17 ran `20261118T094818859687Z` through routing, LangGraph and MCP
  stdio. The worker exited successfully in 708.80 seconds.
- The notebooks save each run under the corresponding MyProject2 or MyProject3
  Drive folder. The repo keeps compact metrics snapshots and the original
  notebook outputs for review.
