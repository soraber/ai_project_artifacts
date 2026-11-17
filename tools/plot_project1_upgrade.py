"""Plot new measurements separately from the historical Project 1 report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run):
    evaluation_paths = list(run.glob("outputs/runs/upgrade-v1/*/base_vs_tuned_eval.json"))
    if len(evaluation_paths) != 1:
        raise ValueError("Select a run with exactly one final NLL evaluation")
    evaluation = json.loads(evaluation_paths[0].read_text())["summary"]
    smoke = json.loads((run / "project1_agent_task_smoke.json").read_text())
    state = json.loads((run / "outputs/llama3_lora_adapter/trainer_state.json").read_text())
    history = [row for row in state["log_history"] if "loss" in row]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    axes[0].plot([r["step"] for r in history], [r["loss"] for r in history], color="#087F8C", linewidth=1.8)
    axes[0].set(title="Training: reference-answer adaptation", xlabel="Optimizer step", ylabel="Logged training loss")
    values = [evaluation["base_perplexity"], evaluation["tuned_perplexity"]]
    bars = axes[1].bar(["Base", "Tuned"], values, color=["#536273", "#087F8C"], width=.55)
    axes[1].bar_label(bars, fmt="%.2f", padding=4)
    axes[1].set(title="Holdout: lower perplexity", ylabel="Answer-only perplexity", ylim=(0, max(values)*1.25))
    axes[1].text(.5, .87, f"{evaluation['perplexity_reduction_pct']:.2f}% reduction",
                 transform=axes[1].transAxes, ha="center")
    values = [smoke["metrics"][s]["tool_correct"]["mean"]*100 for s in ["base", "tuned"]]
    bars = axes[2].bar(["Base", "Tuned"], values, color=["#536273", "#C34846"], width=.55)
    labels = [f"{round(smoke['metrics'][s]['tool_correct']['mean'] * smoke['metrics'][s]['tool_correct']['count'])}/{smoke['metrics'][s]['tool_correct']['count']}"
              for s in ["base", "tuned"]]
    axes[2].bar_label(bars, labels=labels, padding=4)
    axes[2].set(title="Authored smoke: tool-action regression", ylabel="Correct tool action (%)", ylim=(0, 110))
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Project 1: distribution fit improved; agent readiness did not", fontsize=15)
    fig.supxlabel("New run: 500 holdout examples; six separate authored agent prompts. These are different tasks and metrics.", fontsize=10)
    destination = run / "assets/fit_and_task_tradeoff.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    print(plot(parser.parse_args().run))
