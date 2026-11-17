# Upgrade Debug Log

Updated: 2026-11-17 EST. Local code changes and authorized remote execution; no push.

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
