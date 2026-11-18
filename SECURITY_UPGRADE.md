# Dependency Security And Trainer Migration

Updated: 2026-11-18. This follow-up supersedes the execution commands in the
historical reports. Existing measurements and saved adapters remain unchanged.

## Status

| Environment | Verification | Status |
| --- | --- | --- |
| CPU regression | Hash-locked distributions; advisory audit; macOS and Linux ARM64 tests | Ready for CPU checks |
| Transformers/PEFT training v2 | Linux x86_64 hash lock; advisory audit; real tiny CPU LoRA training | A100 smoke, full training, and holdout evaluation passed |
| Historical LLaMA-Factory training | Audit found vulnerable dependencies; wrapper caps Transformers at 5.6.0 | Legacy launcher blocked |
| LangGraph inference v2 | Hash lock; zero reported advisories; real local graph tests | Project 2/3 A100 evaluations passed |
| Historical shared inference / CrewAI | Audit found vulnerable dependencies, including transitive ChromaDB | Legacy launcher remains blocked |

An audit with zero reported findings is point-in-time evidence, not a guarantee.
Hash verification pins artifacts; it does not prove their code is trustworthy.
CPU regression validation does not establish CUDA compatibility. MyPro runs
provide the separate CUDA evidence.
Audit inputs, results, and lock fingerprints are in [security evidence](output/security/p0-20261117/audit_summary.json).

## CPU Checks

Run from this repository. These commands do not change a system or shared kernel:

```sh
uv venv --python 3.11.17 .venv-upgrade
uv pip install --python .venv-upgrade/bin/python --require-hashes -r requirements-upgrade-cpu.hashes.txt
uv pip check --python .venv-upgrade/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-upgrade/bin/python -m pytest -q -p no:cacheprovider tests
```

The real LangGraph tests run in this CPU environment; only the real-torch trainer
test skips. The separate trainer CPU test ran with the patched model libraries.
The final suite passes 57 tests; the separate model environment passes all three
trainer tests. Those CPU updates validate loss masks and adapter-only changes,
not NF4/CUDA kernels. No model was downloaded for the tiny random-weight test.
CI also audits the hashed CPU lock, but the changed workflow has not run on GitHub.

## Project 1: New Trainer

The user approved removal of the LLaMA-Factory wrapper. The new runner uses
Transformers Trainer, PEFT, NF4 double quantization, all-linear LoRA, masked prompt
and padding labels, BF16 on supported GPUs, and non-reentrant checkpointing.
PEFT's preparation helper handles the frozen low-bit model and layer norms.
Sequence packing is disabled so assistant boundaries and loss masks are explicit.
This changes the training recipe: compare new runs separately, not as bitwise
reproductions of the historical packed run.

1. Generate the **separate** controller locally:

   ```sh
   python3 tools/build_training_v2_runner.py
   ```

   It creates `../.upgrade_runtime/project1_training_v2_smoke.ipynb` and refuses
   to overwrite it. Open it in VS Code and select the reconnected A100 server.
   The historical `project1_upgrade_run.ipynb` is not rewritten.

2. Run controller cell 2 `v2-install`. It installs a pinned uv bootstrap into a
   dedicated directory, then creates a separate Python 3.11.17 venv and installs
   the Linux training lock with `--require-hashes`. The bootstrap is version-pinned;
   the full training dependencies are additionally distribution-hash verified.

3. Run controller cell 3 `v2-nf4-smoke`. It mounts Drive and performs two steps
   with a tiny randomly initialized model using real GPU NF4 kernels. It strips
   model/API credential variables, uses offline mode, and does not need an HF
   token, dataset download, or full training. Evidence is written only under
   `MyDrive/projects/MyProject1/runs/upgrade-v2/gpu_integration/`.

4. For actual notebook training, install/select the same isolated environment:

   ```sh
   uv venv --python 3.11.17 .venv-training-v2
   uv pip install --python .venv-training-v2/bin/python --require-hashes -r tools/requirements-training-v2.hashes.txt
   uv pip check --python .venv-training-v2/bin/python
   ```

   These are **Linux CUDA** commands. Select that interpreter as the notebook
   kernel, set `PROJECT1_REPO` to the runtime repository location, and mount Drive.
   No automatic git pull is performed. Use a hidden HF prompt for the gated model;
   never put credentials in notebook source, shell history, or saved config files.

5. In the main notebook, run setup and existing data cells. Cell 8 is byte-for-byte
   unchanged and retains its cache. Cell 17 defaults to a two-step smoke. A new
   full training run requires deliberately setting `SMOKE_ONLY = False`; it saves
   to a unique v2 directory. Cell 18 checks the subprocess exit status. Cell 20
   rejects smoke adapters and mismatched model revisions before holdout evaluation.

Changed main-notebook cells: **3 `c7f465ce`, 4 `c2e49511`, 5 `920adf65`,
15 `b60448d2`, 16 `516f9375`, 17 `e1480002`, 18 `c4d89854`,
19 `257323c9`, 20 `18ed85c3`**. The exact protected cell-8 SHA256 is
`e0eb053df6f13160eafc7e1ee91d3bb5864482eda2bdfbd69faae06c88d29231`.

The runner saves a model commit, data/template hashes, environment, seed, settings,
timing, peak allocated GPU memory, and adapter weights separately. It never uses
the holdout to select checkpoints. A completed smoke is integration evidence only.

## Project 2: Conditional HyDE

Cells **3 `67c7857a`, 4 `3d8e1486`, 13 `bdf0279c`, 16 `67e3a2f8`,
17 `45f0634e`** now select the patched isolated inference lock and add a fifth
retrieval arm. `tools/conditional_hyde.py` skips hypothesis generation when two
of the top three dense and positive-scoring BM25 IDs agree. This is an uncalibrated
rank heuristic, not an answer-confidence estimate. Dense probing, any hypothesis
call, fusion, and reranking all remain inside measured latency. Final answer
prompt, context count, and decoding are shared; warm-up trace entries are excluded.
The four historical arms remain available, and new outputs use `upgrade-v2`.
Four local tests validate gating, malformed candidates, and retained failures.
Project 2 cells 20 and 21 launch and check the original notebook in a separate
hash-verified Python 3.11.17 process. The verified five-arm run on MyPro used the
cached Llama revision and saved its manifest and per-case results under
`MyDrive/projects/MyProject2/outputs/runs/upgrade-v2/20261118T092816338337Z/`.

## Project 3: Approved LangGraph Migration

The audit reports Chroma server authorization/code-execution advisories in the
CrewAI dependency closure without fixed versions in those advisory records. This
project does not expose a Chroma HTTP server, so that exploit surface differs
from its local role workflow. Nevertheless, no blanket advisory ignore is added.
`crewai-core` is only shared utilities, not a replacement for Agent/Crew/Task.
The user approved explicit LangGraph orchestration for the new implementation:
research -> plan -> review -> edit. Four bounded calls share source evidence and
measured token budgets; no delegation, hidden retries, or memory service is used.
Per-node traces save timing and prompt/answer hashes. The real graph runs in CPU
tests using authored replies, including error stopping and context propagation.
The new inference lock no longer includes CrewAI or Chroma and has zero reported
advisories. The new launcher is in Project 3 cell 17. Legacy launchers remain blocked.

Main Project 3 notebook edits: cells **2 `2cb02c52`, 3 `4b855404`, 4 `87bbb7e4`,
11 `13fffca2`, 12 `a24313e8`, 13 `f3348159`, 14 `b3a5ffa9`**. New results use
`upgrade-v2` and arm name `langgraph_roles`. Historical CrewAI figures stay labeled
CrewAI; no framework-quality superiority is inferred from the migration.

New cells **15 `p3-mcp-v2-description`, 16 `p3-mcp-v2-check`** exercise a real
MCP 2.3.0 stdio client/server. `tools/project3_mcp.py` exposes only bounded
knowledge search and budget calculation, without write authority. Two local
tests validate discovery, valid/invalid calls, unknown write-tool rejection,
and symlink escape rejection. This is a separate protocol result, not a proxy
for the model's tool-use quality. It does not implement remote authentication.

Projects 2/3 use `PROJECT_REPO` to locate the runtime checkout. The MyPro launch
cells stage the selected notebook sources and install the separate CUDA environment
with `--require-hashes -r tools/requirements-inference-v2.hashes.txt`.

## References

- [PEFT low-bit training](https://huggingface.co/docs/peft/en/developer_guides/quantization).
- [Transformers Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).
- [pytest advisory and patched floor](https://github.com/advisories/GHSA-6w46-j5rx-g56g).
- [pip-audit](https://pypi.org/project/pip-audit/).
- [Chroma advisory](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c).
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api).
