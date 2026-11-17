# Isolated Upgrade Runs

Run these commands from this repository, with its sibling `ai_agentic_attemptings`
available when running the combined GPU suite:

```sh
python tools/build_upgrade_runner.py
# Open ../.upgrade_runtime/project1_upgrade_run.ipynb in VS Code.
# Select AI-project; use its masked prompt for the gated Hugging Face model.
```

Do not rerun this builder or the training launch cell while the approved training
process is active. The persistent training log and adapter are under the
user-approved `MyDrive/projects/MyProject1/` subtree. The main notebook's dataset
preparation cell 8 remains unchanged.

After staging the current reviewed upgrades:

```sh
python tools/build_gpu_upgrade_runner.py
# Open ../.upgrade_runtime/gpu_upgrade_suite.ipynb on the same authenticated kernel.
# Run cell 2 once, then use cell 3 to inspect progress.
```

This queue waits for the Project 1 manifest to report completed training and NLL
evaluation. It then runs the separate base/adapter task smoke evaluation, creates
an isolated inference environment, and evaluates Projects 2, 3, 5, and 8 in
separate sequential child processes. No GPU jobs run concurrently, no paid API
calls are made, and no generated Python is executed. It packages only reviewed
source/config/test files and the pinned public LoCoMo subset, never cached keys or
historical notebook outputs. Source hashes, environment versions, per-case scores,
executed notebook copies and runtime manifests are saved as new evidence.

The queue records failed stages without repeatedly repairing or rerunning them.
Investigate one failure at a time; ask for help after three failed attempts or
ten minutes on one debugging issue. Do not interpret a completed evaluation as
all tasks passing: inspect per-case errors and sample counts.

## Dependencies

Project 1 uses isolated LLaMA-Factory 0.9.5. Projects 2/3/5/8 install the Linux/Python
3.11 solver lock `requirements-upgrade-inference.lock.txt` in a different environment
and save the actual `inference_environment.lock.txt`. The solver lock is not
runtime-validated until installation and model execution succeed. PyTorch 2.14.1
uses the CUDA 13.0 family already verified in the Colab base runtime.
LM Format Enforcer is pinned with
Transformers 4.57.6 because the published grammar integration predates
Transformers 5. Project 7's tested MCP 2.3.0 environment remains separate from
CrewAI 1.15.41, which declares MCP <2 and Pydantic <2.13.

- [LLaMA-Factory package](https://pypi.org/project/llamafactory/0.9.5/)
- [CrewAI package](https://pypi.org/project/crewai/1.15.41/)
- [LM Format Enforcer integration](https://github.com/noamgat/lm-format-enforcer)
- [Transformers 4.57.6](https://pypi.org/project/transformers/4.57.6/)

The shared Colab interpreter is not repaired or downgraded. Projects 2 and 3 setup
cells fail closed on that interpreter; use the queue or a separately registered
kernel backed by the prepared inference environment.

## Bounded Automated Review

After importing a completed GPU stage's small evidence files, run separately:

```sh
python tools/automated_review.py --project project2 \
  --source output/project2/runs/upgrade-v1/RUN_ID \
  --output output/project2/runs/upgrade-v1/RUN_ID/automated_review
python tools/automated_review.py --project project3 \
  --source output/project3/runs/upgrade-v1/RUN_ID \
  --output output/project3/runs/upgrade-v1/RUN_ID/automated_review
```

The key is requested through a masked prompt or read from `OPENAI_API_KEY`; it is
not stored. Do not place a real key directly in a shell command or notebook.
The approved run uses GPT-5.6 Sol with at most $2/$1 respectively, no SDK retries,
and worst-case reservations for calls that fail before usage is known. It stops
after three review failures. Confirm current prices before a future execution.

Project 2 selects all three authored absence questions and up to twelve seeded
natural questions, preserving all four retrieval arms for each. Conservative
budget preflight may stop before the full planned sample. Project 3 reviews all
nine strategy outputs. Arm labels are joined only after each model response; the
judge receives no key mapping. Exact quote checks establish text provenance, not
semantic correctness. Keep the source hashes, raw ratings, failure counts, actual
reviewed sample size, and usage together. These are small automated reviews, not
human-labeled benchmarks or production guarantees. Repeated runs consume another
budget and need separate authorization; changing the output path does not reset
the user's total spending permission.
