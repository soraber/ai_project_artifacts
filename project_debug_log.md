# AI Projects Debug Log

This log records reproducible engineering issues, root causes, fixes, and
verification steps. Never add API keys, access tokens, cookies, or other secrets.

## Project 1 - LLM Fine-Tuning Pipeline

### Colab dependency conflicts around Gradio, Pydantic, and Starlette

- **Symptom:** `pip` reported incompatible requirements among Gradio,
  `hf-gradio`, Google ADK, `python-fasthtml`, Google GenAI, and Pydantic.
- **Root cause:** Colab preinstalls applications with dependency constraints that
  conflict with the versions supported by LLaMA-Factory.
- **Fix:** Use one constraints file for every install; pin Pydantic 2.12.3,
  Gradio 5.50.0, and Gradio Client 1.14.0; remove unrelated conflicting Colab
  applications and remove `hf-gradio` after replacing Gradio 6.x.
- **Affected cells:** Project 1 Cells 3-5.
- **Verification:** Run `python -m pip check`, print package versions, and import
  the pinned core stack.

### PyTorch kernel did not have CUDA support

- **Symptom:** `torch.cuda.get_device_name(0)` raised `Torch not compiled with
  CUDA enabled`.
- **Root cause:** VS Code was attached to a local CPU kernel instead of the Colab
  GPU runtime.
- **Fix:** Create/select the Colab A100 runtime and attach the notebook to its
  live remote Python kernel.
- **Verification:** `torch.cuda.is_available()` is true and the device name is
  `NVIDIA A100-SXM4-40GB`.

### Gated Meta Llama model returned HTTP 401

- **Symptom:** Loading `meta-llama/Meta-Llama-3-8B-Instruct` failed with
  `401 Unauthorized`.
- **Root cause:** The runtime lacked a valid Hugging Face token or the account had
  not accepted the model terms.
- **Fix:** Accept the model terms, authenticate with a read-only token through a
  hidden login prompt, and verify access with `HfApi.model_info` before training.
- **Affected cell:** Project 1 Cell 6.
- **Verification:** Print the model ID and a short revision hash, never the token.

### VS Code Colab extension could not fetch Colab Secrets

- **Symptom:** Secret retrieval timed out with `Secrets can only be fetched when
  running from the Colab UI`.
- **Root cause:** Colab Secrets are only exposed to code executed in the browser
  Colab UI, not through the VS Code extension transport.
- **Fix:** Use `huggingface_hub.login()` or a hidden `getpass` prompt in the live
  runtime. Do not store the token in the notebook.
- **Verification:** Confirm authenticated API access without printing credentials.

### Evaluation examples lost answer tokens at the cutoff

- **Symptom:** Evaluation raised `No evaluation examples contained answer tokens
  within the cutoff length`.
- **Root cause:** Long prompts consumed the sequence budget before answer tokens.
- **Fix:** Encode prompt and answer separately, reserve explicit answer capacity,
  truncate the prompt side, and mask only prompt/padding tokens.
- **Verification:** Count evaluated examples and answer tokens before scoring.

### Tokenizer Encoding object caused dtype inference failure

- **Symptom:** `torch.tensor(...)` raised `Could not infer dtype of
  tokenizers.Encoding`.
- **Root cause:** The low-level tokenizer object was passed instead of a plain list
  of token IDs.
- **Fix:** Request ordinary tokenizer outputs and convert `input_ids` to Python
  lists before constructing tensors.
- **Verification:** Run one encoded example and assert integer tensor dtype/shape.

## Project 2 - RAG Document QA System

### Pinned dataset revision exposed only a default builder configuration

- **Date:** 2026-07-22
- **Symptom:** Cell 6 raised `BuilderConfig 'text-corpus' not found. Available:
  ['default']` under `datasets 4.0.0`.
- **Root cause:** The pinned repository revision uses separate parquet directories,
  while the current datasets builder did not reconstruct the two README configs.
- **Fix:** List the pinned repository files, download corpus and QA parquet files
  explicitly with `hf_hub_download`, and load them through the parquet builder.
  Save both datasets with `save_to_disk` so later runs require no download.
- **Affected cell:** Project 2 Cell 6 only.
- **Verification:** Loaded 3,200 corpus rows and 918 QA rows from the pinned revision.

### Qwen 3B download did not complete through the VS Code Colab proxy

- **Date:** 2026-07-22
- **Symptom:** Cell 13 remained in Hugging Face shard download/reconstruction for
  approximately 10 minutes without completing or raising an exception.
- **Root cause:** Multi-gigabyte Xet-backed model transfer was unreliable through
  the VS Code-to-Colab connection.
- **Fix:** Interrupt at the requested 10-minute ceiling, disable Xet transfer with
  `HF_HUB_DISABLE_XET=1`, and use the public Qwen2.5-1.5B-Instruct model. Both
  baseline and advanced systems still share the same generator, preserving the
  controlled comparison.
- **Affected cells:** Project 2 Cells 4 and 12.
- **Initial verification:** A kernel restart confirmed that Cells 3-11 replay from cache,
  but the 1.5B transfer also remained in shard reconstruction. The operation was
  interrupted at the requested debugging time limit.
- **Resolution:** Reused Project 1's cached Meta Llama 3 8B weights in NF4 mode;
  Cell 13 and the full evaluation then completed on the A100.
- **Next controlled options:** Download the public Qwen model once from the Colab
  browser UI, where transfers are more reliable, then resume at Cell 13; or change
  the generator deliberately to the gated Meta Llama weights already cached by
  Project 1 and document that added access requirement.

### Advanced answer prompt exposed reasoning and depressed answer F1

- **Date:** 2026-07-22
- **Symptom:** The seed-42 development run improved answer-hit from 0.783 to
  0.900 and reciprocal rank from 0.711 to 0.845, but verbose advanced answers
  reduced token F1 from 0.259 to 0.112. Composite lift was only 2.05%.
- **Root cause:** The evidence-first prompt asked the model to identify evidence
  and resolve conflicts without prohibiting visible reasoning. Llama returned
  long analyses that were factually grounded but mismatched concise references.
- **Fix:** Require internal evidence selection and output only the shortest answer
  plus a valid chunk citation. Use seed 2026 as a fresh final holdout so the
  development sample is not reused for the portfolio result.
- **Affected cells:** Project 2 Cells 4, 15, and 17.
- **Verification:** On the untouched seed-2026 holdout, token F1 improved
  from 0.317 to 0.409 and the composite score improved from 0.567 to 0.667,
  producing a measured 17.55% lift.

### Colab A100 endpoint expired before the final-holdout rerun

- **Date:** 2026-07-22
- **Symptom:** VS Code failed at Cell 3 with `Invalid response: 404 Not Found`
  while starting the previously connected remote kernel. One retry produced the
  same result.
- **Root cause:** The Colab runtime/server URL expired or was terminated; notebook
  code cannot recreate that external endpoint.
- **Fix:** Reconnect the Project 2 notebook to a new live Colab A100 kernel, then
  run all cells. Dataset and embedding caches persist only if the same Colab VM is
  recovered; otherwise Cells 6 and 11 rebuild them once.
- **Affected cells:** None. This is a remote-kernel lifecycle issue.
- **Preserved evidence:** The completed seed-42 development metrics are stored at
  `output/project2/project2_seed42_development_summary.json`.
- **Resolution:** Reconnecting to the same A100 VM restored the endpoint and its
  caches. The final seed-2026 run completed successfully.

## Project 3 - LLM Agent System

### CrewAI synchronous kickoff conflicted with the notebook event loop

- **Date:** 2026-07-22
- **Symptom:** Project 3 Cell 12 raised `Agent execution was invoked
  synchronously from within a running event loop` under CrewAI 1.15.5.
- **Root cause:** The VS Code Jupyter/Colab notebook already owns an asyncio event
  loop, while current CrewAI rejects nested synchronous `crew.kickoff()` calls.
- **Fix:** Use top-level notebook await with `await crew.kickoff_async()`.
- **Affected cell:** Project 3 Cell 12 only.
- **Verification:** Async kickoff completed all four role calls. A subsequent
  quality pass increased role-specific output budgets because the shared 320-token
  limit truncated the final memo before its timeline, KPI, and safeguard sections.

### Colab endpoint expired before the Project 3 quality rerun

- **Date:** 2026-07-22
- **Symptom:** The corrected rerun failed before Cell 3 with `Invalid response:
  404 Not Found`; one connection retry produced the same result.
- **Root cause:** The remote Colab server URL expired after the first complete
  Project 3 run. This is external to notebook code.
- **Fix:** Reconnect the notebook to a live A100 kernel and rerun all cells.
- **Affected cells:** None.
- **Preserved evidence:** The completed first-run metrics are stored at
  `output/project3/project3_first_run_summary.json` and explicitly labeled as
  development results rather than the final quality claim.
- **Resolution:** Reconnecting to the live A100 runtime restored the endpoint.
  The corrected rerun completed Cells 3-14 with no failed evaluation cases;
  final metrics and report artifacts are stored under `output/project3/`.
