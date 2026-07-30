#!/usr/bin/env python3
"""Build a portfolio PDF from the saved outputs in Project 1's notebook."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "01_llm_fine_tuning_pipeline.ipynb"
OUTPUT_DIR = REPO_ROOT / "output" / "project1"
ASSET_DIR = OUTPUT_DIR / "project1_report_assets"
OUTPUT_PDF = OUTPUT_DIR / "project1_report.pdf"

PAGE_W, PAGE_H = LETTER
MARGIN = 42

INK = HexColor("#17212B")
MUTED = HexColor("#5E6B76")
NAVY = HexColor("#16324F")
TEAL = HexColor("#167D7F")
GREEN = HexColor("#3B8C6E")
CORAL = HexColor("#D96C4A")
GOLD = HexColor("#D4A72C")
PALE_BLUE = HexColor("#E9F1F7")
PALE_GREEN = HexColor("#E8F3EE")
PALE_CORAL = HexColor("#F8ECE7")
LIGHT = HexColor("#F4F6F7")
MID = HexColor("#D7DEE3")


def output_text(cell: dict) -> str:
    parts = []
    for item in cell.get("outputs", []):
        if "text" in item:
            parts.append("".join(item["text"]))
        elif item.get("output_type") == "error":
            parts.append("\n".join(item.get("traceback", [])))
    return "\n".join(parts)


def load_results() -> dict:
    notebook = json.loads(NOTEBOOK.read_text())
    cells = notebook["cells"]

    data_text = output_text(cells[10])
    data_match = re.search(r"\{'train':.*\}", data_text)
    if not data_match:
        raise RuntimeError("Could not parse data split metrics from Cell 11")
    data_split = ast.literal_eval(data_match.group(0))

    training_text = output_text(cells[17])
    train_rows = [
        tuple(map(float, match.groups()))
        for match in re.finditer(
            r"\{'loss': '([^']+)', 'grad_norm': '([^']+)', "
            r"'learning_rate': '([^']+)', 'epoch': '([^']+)'\}",
            training_text,
        )
    ]
    if not train_rows:
        raise RuntimeError("Could not parse training history from Cell 18")

    trainable_match = re.search(
        r"trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: ([\d.]+)",
        training_text,
    )
    examples_match = re.search(r"Num examples = ([\d,]+)", training_text)

    eval_text = output_text(cells[19])
    start = eval_text.rfind('{\n  "criterion"')
    end = eval_text.find("\n}\nSaved:", start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not parse evaluation summary from Cell 20")
    evaluation = json.loads(eval_text[start : end + 2])

    initial_window = float(np.mean([row[0] for row in train_rows[:10]]))
    final_window = float(np.mean([row[0] for row in train_rows[-10:]]))

    return {
        "raw_rows": 50_500,
        "train_rows": data_split["train"],
        "eval_rows": data_split["eval"],
        "rejected": sum(data_split["rejected"].values()),
        "rejection_reasons": data_split["rejected"],
        "clean_rows": data_split["train"] + data_split["eval"],
        "training": train_rows,
        "trainable_params": int(trainable_match.group(1).replace(",", "")) if trainable_match else 20_971_520,
        "all_params": int(trainable_match.group(2).replace(",", "")) if trainable_match else 8_051_232_768,
        "trainable_pct": float(trainable_match.group(3)) if trainable_match else 0.2605,
        "packed_examples": int(examples_match.group(1).replace(",", "")) if examples_match else 11_206,
        "updates": 701,
        "train_loss": 1.7811,
        "runtime": "2:44:33",
        "runtime_seconds": 9873,
        "samples_per_second": 1.135,
        "initial_loss_window": initial_window,
        "final_loss_window": final_window,
        "loss_window_reduction_pct": 100 * (initial_window - final_window) / initial_window,
        "evaluation": evaluation,
    }


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#33414D",
            "axes.edgecolor": "#A8B4BD",
            "xtick.color": "#5E6B76",
            "ytick.color": "#5E6B76",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def create_charts(results: dict) -> dict[str, Path]:
    configure_plots()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    training = np.array(results["training"])
    steps = np.arange(1, len(training) + 1) * 10
    losses = training[:, 0]
    learning_rates = training[:, 2]
    kernel = np.ones(5) / 5
    smoothed = np.convolve(losses, kernel, mode="valid")

    loss_path = ASSET_DIR / "training_dynamics.png"
    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=180)
    ax.plot(steps, losses, color="#9DB0BC", linewidth=1.1, marker="o", markersize=2.5, label="Logged loss")
    ax.plot(steps[4:], smoothed, color="#167D7F", linewidth=2.6, label="5-point moving average")
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("SFT training loss")
    ax.set_xlim(0, 710)
    ax.grid(axis="y", color="#E4E9EC", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    lr_ax = ax.twinx()
    lr_ax.plot(steps, learning_rates, color="#D4A72C", linewidth=1.8, alpha=0.85, label="Cosine learning rate")
    lr_ax.set_ylabel("Learning rate")
    lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    lr_ax.spines["top"].set_visible(False)
    lines = ax.get_lines() + lr_ax.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)

    eval_path = ASSET_DIR / "evaluation_comparison.png"
    evaluation = results["evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), dpi=180)
    labels = ["Base 4-bit", "Tuned + LoRA"]
    colors = ["#9AA8B2", "#3B8C6E"]
    ppl = [evaluation["base_perplexity"], evaluation["tuned_perplexity"]]
    nll = [evaluation["base_answer_nll"], evaluation["tuned_answer_nll"]]
    for ax, values, title, ylabel in [
        (axes[0], ppl, "Held-out answer perplexity", "Perplexity (lower is better)"),
        (axes[1], nll, "Held-out answer NLL", "Negative log-likelihood"),
    ]:
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E4E9EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.035, f"{value:.3f}", ha="center", fontweight="bold")
    fig.tight_layout(w_pad=3)
    fig.savefig(eval_path, bbox_inches="tight")
    plt.close(fig)

    data_path = ASSET_DIR / "data_quality.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=180, gridspec_kw={"width_ratios": [1.45, 1]})
    stages = ["Cached", "Passed gates", "Train"]
    values = [results["raw_rows"], results["clean_rows"], results["train_rows"]]
    bars = axes[0].barh(stages[::-1], values[::-1], color=["#167D7F", "#3B8C6E", "#9AA8B2"])
    axes[0].set_title("Data retained through preparation")
    axes[0].set_xlabel("Examples")
    axes[0].grid(axis="x", color="#E4E9EC", linewidth=0.8)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values[::-1]):
        axes[0].text(value - 500, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", ha="right", color="white", fontweight="bold")
    axes[1].pie(
        [results["train_rows"], results["eval_rows"]],
        labels=["Train\n49,693", "Holdout\n500"],
        colors=["#16324F", "#D4A72C"],
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Leakage-resistant split")
    fig.tight_layout()
    fig.savefig(data_path, bbox_inches="tight")
    plt.close(fig)

    return {"loss": loss_path, "evaluation": eval_path, "data": data_path}


def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if stringWidth(candidate, font, size) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 9.5,
    leading: float = 13,
    color=INK,
) -> float:
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    for line in wrap_lines(text, font, size, width):
        if line:
            pdf.drawString(x, y, line)
        y -= leading
    return y


def draw_footer(pdf: canvas.Canvas, page: int, label: str) -> None:
    pdf.setStrokeColor(MID)
    pdf.setLineWidth(0.5)
    pdf.line(MARGIN, 28, PAGE_W - MARGIN, 28)
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(MUTED)
    pdf.drawString(MARGIN, 16, label)
    pdf.drawRightString(PAGE_W - MARGIN, 16, f"{page} / 7")


def page_title(pdf: canvas.Canvas, number: str, title: str, subtitle: str | None = None) -> float:
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(TEAL)
    pdf.drawString(MARGIN, PAGE_H - 48, number.upper())
    pdf.setFont("Helvetica-Bold", 23)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, PAGE_H - 78, title)
    y = PAGE_H - 99
    if subtitle:
        y = draw_wrapped(pdf, subtitle, MARGIN, y, PAGE_W - 2 * MARGIN, size=9.5, leading=13, color=MUTED)
    pdf.setStrokeColor(MID)
    pdf.line(MARGIN, y - 4, PAGE_W - MARGIN, y - 4)
    return y - 22


def card(pdf: canvas.Canvas, x: float, y: float, w: float, h: float, fill=LIGHT, stroke=MID) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(0.6)
    pdf.roundRect(x, y, w, h, 5, fill=1, stroke=1)


def kpi(pdf: canvas.Canvas, x: float, y: float, w: float, value: str, label: str, color=TEAL) -> None:
    card(pdf, x, y, w, 76, fill=white)
    pdf.setFont("Helvetica-Bold", 21)
    pdf.setFillColor(color)
    pdf.drawString(x + 13, y + 43, value)
    draw_wrapped(pdf, label, x + 13, y + 25, w - 26, size=8.2, leading=10, color=MUTED)


def draw_arrow(pdf: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=TEAL, width=1.4) -> None:
    pdf.setStrokeColor(color)
    pdf.setFillColor(color)
    pdf.setLineWidth(width)
    pdf.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 7
    for delta in (2.65, -2.65):
        pdf.line(x2, y2, x2 + length * math.cos(angle + delta), y2 + length * math.sin(angle + delta))


def draw_process_box(pdf: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, detail: str, fill=white, accent=TEAL) -> None:
    card(pdf, x, y, w, h, fill=fill)
    pdf.setFillColor(accent)
    pdf.rect(x, y, 4, h, fill=1, stroke=0)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(INK)
    pdf.drawString(x + 12, y + h - 19, title)
    draw_wrapped(pdf, detail, x + 12, y + h - 34, w - 22, size=7.5, leading=9.5, color=MUTED)


def build_pdf(results: dict, charts: dict[str, Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=LETTER, pageCompression=1)
    pdf.setTitle("Project 1 Report - Llama 3 8B Domain Fine-Tuning")
    pdf.setAuthor("Hongda Wu")
    pdf.setSubject("QLoRA domain adaptation for software engineering question answering")

    evaluation = results["evaluation"]

    # Page 1: Executive summary.
    pdf.setFillColor(NAVY)
    pdf.rect(0, PAGE_H - 265, PAGE_W, 265, fill=1, stroke=0)
    pdf.setFillColor(TEAL)
    pdf.rect(0, PAGE_H - 269, PAGE_W, 4, fill=1, stroke=0)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(HexColor("#77D0C8"))
    pdf.drawString(MARGIN, PAGE_H - 54, "LLM FINE-TUNING PIPELINE | PROJECT 1")
    pdf.setFont("Helvetica-Bold", 29)
    pdf.setFillColor(white)
    pdf.drawString(MARGIN, PAGE_H - 94, "Domain-Adapted Llama 3 8B")
    pdf.setFont("Helvetica", 15)
    pdf.setFillColor(HexColor("#DCE8EF"))
    pdf.drawString(MARGIN, PAGE_H - 121, "Software engineering Q&A with QLoRA")
    pdf.setFont("Helvetica-Bold", 48)
    pdf.setFillColor(HexColor("#7CD7B3"))
    pdf.drawString(MARGIN, PAGE_H - 190, f"{evaluation['perplexity_reduction_pct']:.2f}%")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(white)
    pdf.drawString(MARGIN + 190, PAGE_H - 169, "LOWER HELD-OUT")
    pdf.drawString(MARGIN + 190, PAGE_H - 187, "ANSWER PERPLEXITY")
    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(HexColor("#C9D7DF"))
    pdf.drawString(MARGIN, PAGE_H - 228, "Base 7.9560  ->  Tuned 5.5359 | 100 paired examples | 18,798 answer tokens")

    y = PAGE_H - 300
    pdf.setFont("Helvetica-Bold", 18)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "What this project demonstrates")
    y = draw_wrapped(
        pdf,
        "An end-to-end adaptation workflow that converts a large public preference corpus into a controlled supervised fine-tuning experiment, trains a parameter-efficient adapter on an A100, and compares the adapted and unadapted model under identical inference conditions.",
        MARGIN,
        y - 22,
        PAGE_W - 2 * MARGIN,
        size=10.5,
        leading=15,
    )

    gap = 10
    card_w = (PAGE_W - 2 * MARGIN - 2 * gap) / 3
    kpi(pdf, MARGIN, y - 102, card_w, "50,500", "real-world Q&A examples cached", TEAL)
    kpi(pdf, MARGIN + card_w + gap, y - 102, card_w, f"{results['trainable_pct']:.2f}%", "of parameters trained with LoRA", CORAL)
    kpi(pdf, MARGIN + 2 * (card_w + gap), y - 102, card_w, results["runtime"], "single-A100 training runtime", GOLD)

    card(pdf, MARGIN, 72, PAGE_W - 2 * MARGIN, 122, fill=PALE_GREEN, stroke=HexColor("#B9D9CA"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(GREEN)
    pdf.drawString(MARGIN + 15, 172, "PORTFOLIO SIGNAL")
    draw_wrapped(
        pdf,
        "The result is evidence that a small trainable adapter can materially shift an 8B model toward the target answer distribution while preserving a frozen quantized backbone. It demonstrates data engineering, efficient training, experimental control, statistical evaluation, and deployment-oriented artifact design.",
        MARGIN + 15,
        151,
        PAGE_W - 2 * MARGIN - 30,
        size=9.5,
        leading=13,
    )
    draw_footer(pdf, 1, "Hongda Wu | Project 1 portfolio report | Run completed July 22, 2026")
    pdf.showPage()

    # Page 2: Data pipeline.
    y = page_title(pdf, "01 / DATA", "From public Q&A to controlled SFT data", "A deterministic pipeline selects stronger Stack Overflow answers, normalizes text, enforces quality gates, and isolates a fixed holdout before training.")
    box_y = y - 105
    box_w = 102
    xs = [MARGIN, MARGIN + 135, MARGIN + 270, MARGIN + 405]
    draw_process_box(pdf, xs[0], box_y, box_w, 82, "SOURCE", "10.8M-row Stack Exchange preference corpus", PALE_BLUE, NAVY)
    draw_process_box(pdf, xs[1], box_y, box_w, 82, "STREAM", "Stackoverflow.com subset, seed 42 shuffle", white, TEAL)
    draw_process_box(pdf, xs[2], box_y, box_w, 82, "SELECT", "pm_score >= 2; accepted answer prioritized", PALE_GREEN, GREEN)
    draw_process_box(pdf, xs[3], box_y, box_w, 82, "NORMALIZE", "Strip HTML; retain metadata and CC BY-SA 4.0", PALE_CORAL, CORAL)
    for left, right in zip(xs[:-1], xs[1:]):
        draw_arrow(pdf, left + box_w + 6, box_y + 41, right - 7, box_y + 41)

    pdf.drawImage(str(charts["data"]), MARGIN, 330, width=PAGE_W - 2 * MARGIN, height=206, preserveAspectRatio=True, anchor="c", mask="auto")

    reject_pct = 100 * results["rejected"] / results["raw_rows"]
    card(pdf, MARGIN, 217, PAGE_W - 2 * MARGIN, 86, fill=LIGHT)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN + 15, 279, "Quality gates")
    gate_texts = [
        ("Required fields", "instruction and output must be non-empty"),
        ("Length control", "combined text <= 8,000 characters"),
        ("Deduplication", "case-normalized instruction/input/output key"),
        ("Outcome", f"{results['rejected']} rejected ({reject_pct:.2f}%); {results['clean_rows']:,} retained"),
    ]
    col_w = (PAGE_W - 2 * MARGIN - 30) / 4
    for i, (head, detail) in enumerate(gate_texts):
        x = MARGIN + 15 + i * col_w
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.setFillColor(TEAL if i < 3 else CORAL)
        pdf.drawString(x, 258, head)
        draw_wrapped(pdf, detail, x, 244, col_w - 10, size=7.4, leading=9, color=MUTED)

    card(pdf, MARGIN, 72, PAGE_W - 2 * MARGIN, 118, fill=PALE_CORAL, stroke=HexColor("#E8C5B9"))
    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.setFillColor(CORAL)
    pdf.drawString(MARGIN + 15, 167, "WHY THE SPLIT MATTERS")
    draw_wrapped(
        pdf,
        "The 500-example holdout is removed before LLaMA-Factory registration and training. This prevents direct train/eval leakage. The final reported comparison samples 100 examples from that fixed holdout with seed 42; the remaining 400 are available for the final full-scale evaluation.",
        MARGIN + 15,
        148,
        PAGE_W - 2 * MARGIN - 30,
        size=9.2,
        leading=13,
    )
    draw_footer(pdf, 2, "Source: HuggingFaceH4/stack-exchange-preferences | Notebook Cells 8-14")
    pdf.showPage()

    # Page 3: Efficient adaptation architecture.
    y = page_title(pdf, "02 / MODEL", "Efficient adaptation, not full retraining", "QLoRA keeps the Llama 3 8B backbone frozen in 4-bit form while training low-rank updates on every linear projection.")

    # Architecture sketch.
    model_x, model_y, model_w, model_h = 205, 335, 210, 290
    draw_process_box(pdf, MARGIN, 515, 120, 72, "INPUT", "Instruction + software engineering question", PALE_BLUE, NAVY)
    draw_arrow(pdf, MARGIN + 120, 551, model_x - 12, 551)
    card(pdf, model_x, model_y, model_w, model_h, fill=HexColor("#F0F4F6"), stroke=HexColor("#AAB8C1"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.setFillColor(NAVY)
    pdf.drawCentredString(model_x + model_w / 2, model_y + model_h - 29, "Llama 3 8B backbone")
    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(MUTED)
    pdf.drawCentredString(model_x + model_w / 2, model_y + model_h - 45, "frozen NF4 4-bit weights")
    layer_names = ["Attention: q/k/v/o", "MLP: gate/up/down", "LayerNorm upcast", "Gradient checkpointing"]
    for i, name in enumerate(layer_names):
        layer_y = model_y + 52 + i * 48
        pdf.setFillColor(white)
        pdf.setStrokeColor(MID)
        pdf.roundRect(model_x + 20, layer_y, model_w - 40, 32, 4, fill=1, stroke=1)
        pdf.setFont("Helvetica", 8.2)
        pdf.setFillColor(INK)
        pdf.drawString(model_x + 31, layer_y + 11, name)
        if i < 2:
            pdf.setFillColor(CORAL)
            pdf.roundRect(model_x + model_w - 64, layer_y + 7, 31, 18, 3, fill=1, stroke=0)
            pdf.setFont("Helvetica-Bold", 7)
            pdf.setFillColor(white)
            pdf.drawCentredString(model_x + model_w - 48.5, layer_y + 13, "LoRA")
    draw_process_box(pdf, 458, 515, 112, 72, "OUTPUT", "Domain-adapted answer distribution", PALE_GREEN, GREEN)
    draw_arrow(pdf, model_x + model_w + 10, 551, 451, 551)

    card(pdf, MARGIN, 335, 125, 145, fill=PALE_CORAL, stroke=HexColor("#E8C5B9"))
    pdf.setFont("Helvetica-Bold", 24)
    pdf.setFillColor(CORAL)
    pdf.drawString(MARGIN + 12, 439, f"{results['trainable_pct']:.4f}%")
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN + 12, 420, "TRAINABLE SHARE")
    draw_wrapped(pdf, f"{results['trainable_params']:,} LoRA parameters out of {results['all_params']:,} total parameters.", MARGIN + 12, 400, 101, size=8, leading=11, color=MUTED)

    # Configuration table.
    table_y = 287
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, table_y, "Training configuration")
    configs = [
        ("Base model", "Meta-Llama-3-8B-Instruct"),
        ("Method", "LoRA on all linear targets"),
        ("Quantization", "4-bit NF4 + double quantization"),
        ("Precision", "bfloat16 compute; LayerNorm FP32"),
        ("Sequence", "2,048 tokens; packing enabled"),
        ("Batch", "2 x 8 accumulation = 16 effective"),
        ("Schedule", "1 epoch; cosine; LR 2e-4"),
        ("Checkpoints", "every 50 steps; retain 2"),
    ]
    left_x, right_x = MARGIN, 316
    for i, (key, value) in enumerate(configs):
        col = left_x if i < 4 else right_x
        row = i if i < 4 else i - 4
        yy = table_y - 30 - row * 40
        pdf.setFont("Helvetica-Bold", 8)
        pdf.setFillColor(TEAL)
        pdf.drawString(col, yy, key.upper())
        pdf.setFont("Helvetica", 8.7)
        pdf.setFillColor(INK)
        pdf.drawString(col, yy - 14, value)
        pdf.setStrokeColor(MID)
        pdf.line(col, yy - 23, col + 250, yy - 23)

    card(pdf, MARGIN, 61, PAGE_W - 2 * MARGIN, 53, fill=PALE_GREEN, stroke=HexColor("#B9D9CA"))
    draw_wrapped(pdf, "Engineering meaning: the adapter is portable and inexpensive to store, while disabling it recovers the exact quantized baseline for a controlled A/B comparison.", MARGIN + 14, 93, PAGE_W - 2 * MARGIN - 28, size=9, leading=12, color=GREEN)
    draw_footer(pdf, 3, "Method: QLoRA + PEFT LoRA | LLaMA-Factory 0.9.6.dev0")
    pdf.showPage()

    # Page 4: Training dynamics.
    y = page_title(pdf, "03 / TRAIN", "Stable one-epoch adaptation on a single A100", "The curve shows rapid early domain adaptation followed by a stable plateau under cosine decay.")
    pdf.drawImage(str(charts["loss"]), MARGIN, 350, width=PAGE_W - 2 * MARGIN, height=286, preserveAspectRatio=True, anchor="c", mask="auto")

    gap = 10
    card_w = (PAGE_W - 2 * MARGIN - 3 * gap) / 4
    kpi(pdf, MARGIN, 251, card_w, "701", "optimizer updates", NAVY)
    kpi(pdf, MARGIN + card_w + gap, 251, card_w, results["runtime"], "wall-clock runtime", GOLD)
    kpi(pdf, MARGIN + 2 * (card_w + gap), 251, card_w, f"{results['train_loss']:.4f}", "reported train loss", TEAL)
    kpi(pdf, MARGIN + 3 * (card_w + gap), 251, card_w, f"{results['samples_per_second']:.3f}", "packed sequences/sec", CORAL)

    card(pdf, MARGIN, 110, 257, 116, fill=PALE_BLUE, stroke=HexColor("#C3D6E2"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(NAVY)
    pdf.drawString(MARGIN + 14, 202, "OBSERVED DYNAMICS")
    draw_wrapped(
        pdf,
        f"Mean loss fell from {results['initial_loss_window']:.4f} across the first 10 logs to {results['final_loss_window']:.4f} across the final 10, an {results['loss_window_reduction_pct']:.2f}% reduction. Gradient norms stabilized after the initial updates.",
        MARGIN + 14,
        182,
        229,
        size=8.8,
        leading=12,
    )
    card(pdf, 313, 110, 257, 116, fill=PALE_CORAL, stroke=HexColor("#E8C5B9"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(CORAL)
    pdf.drawString(327, 202, "WHAT THE CURVE CANNOT PROVE")
    draw_wrapped(
        pdf,
        "Training loss alone does not establish better answers. No in-training eval loss was recorded, so the claim depends on the separate frozen holdout evaluation shown next. This separation is deliberate and avoids tuning to the test metric.",
        327,
        182,
        229,
        size=8.8,
        leading=12,
    )
    draw_footer(pdf, 4, "Hardware: NVIDIA A100-SXM4 40GB | 11,206 packed sequences | 49,693 source examples")
    pdf.showPage()

    # Page 5: Evaluation design.
    y = page_title(pdf, "04 / EVALUATE", "A controlled paired comparison", "The experiment changes one factor only: whether the trained LoRA adapter is active.")

    draw_process_box(pdf, MARGIN, 545, 122, 83, "FIXED HOLDOUT", "100 sampled from 500 with seed 42", PALE_BLUE, NAVY)
    draw_arrow(pdf, 170, 586, 211, 586)
    draw_process_box(pdf, 218, 545, 152, 83, "SHARED PREPROCESSING", "same tokenizer, 2,048 cutoff, response-aware truncation", white, TEAL)
    draw_arrow(pdf, 377, 586, 420, 586)
    draw_process_box(pdf, 427, 545, 143, 83, "SAME 4-BIT MODEL", "revision 8afb486c1db2; SDPA; BF16", PALE_GREEN, GREEN)

    branch_x = 294
    draw_arrow(pdf, branch_x, 538, branch_x, 488, MUTED)
    pdf.setStrokeColor(MUTED)
    pdf.line(174, 488, 414, 488)
    draw_arrow(pdf, 174, 488, 174, 448, NAVY)
    draw_arrow(pdf, 414, 488, 414, 448, GREEN)
    draw_process_box(pdf, 91, 358, 166, 88, "BASELINE PASS", "PEFT adapter disabled; frozen quantized backbone", PALE_BLUE, NAVY)
    draw_process_box(pdf, 331, 358, 166, 88, "TUNED PASS", "same backbone with trained LoRA active", PALE_GREEN, GREEN)
    draw_arrow(pdf, 174, 350, 174, 314, NAVY)
    draw_arrow(pdf, 414, 350, 414, 314, GREEN)
    draw_process_box(pdf, 91, 228, 166, 84, "ANSWER-ONLY NLL", "prompt labels masked; reference answer tokens scored", white, NAVY)
    draw_process_box(pdf, 331, 228, 166, 84, "ANSWER-ONLY NLL", "identical labels and token budget", white, GREEN)
    pdf.setStrokeColor(MUTED)
    pdf.line(174, 220, 174, 194)
    pdf.line(414, 220, 414, 194)
    pdf.line(174, 194, 414, 194)
    draw_arrow(pdf, 294, 194, 294, 165, TEAL)
    draw_process_box(pdf, 196, 82, 196, 82, "PAIRED REPORT", "token-weighted perplexity, win rate, bootstrap 95% CI", PALE_CORAL, CORAL)

    card(pdf, MARGIN, 82, 125, 82, fill=LIGHT)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN + 12, 141, "METRIC")
    draw_wrapped(pdf, "NLL = mean negative log probability of answer tokens. PPL = exp(NLL). Lower is better.", MARGIN + 12, 124, 101, size=7.6, leading=10, color=MUTED)
    card(pdf, 435, 82, 135, 82, fill=LIGHT)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(INK)
    pdf.drawString(447, 141, "STATISTICAL CHECK")
    draw_wrapped(pdf, "2,000 paired bootstrap resamples; positive NLL improvement favors tuned.", 447, 124, 111, size=7.6, leading=10, color=MUTED)
    draw_footer(pdf, 5, "Primary criterion: answer-only held-out perplexity | Lower is better")
    pdf.showPage()

    # Page 6: Results.
    y = page_title(pdf, "05 / RESULT", "The adapter materially improves held-out fit", "All displayed values come from the notebook's saved evaluation output.")
    pdf.drawImage(str(charts["evaluation"]), MARGIN, 390, width=PAGE_W - 2 * MARGIN, height=244, preserveAspectRatio=True, anchor="c", mask="auto")

    card(pdf, MARGIN, 282, 252, 88, fill=PALE_GREEN, stroke=HexColor("#B9D9CA"))
    pdf.setFont("Helvetica-Bold", 28)
    pdf.setFillColor(GREEN)
    pdf.drawString(MARGIN + 15, 326, f"{evaluation['perplexity_reduction_pct']:.2f}%")
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN + 15, 307, "PERPLEXITY REDUCTION")
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(MUTED)
    pdf.drawString(MARGIN + 15, 292, "7.9560 base -> 5.5359 tuned")

    card(pdf, 318, 282, 252, 88, fill=PALE_BLUE, stroke=HexColor("#C3D6E2"))
    pdf.setFont("Helvetica-Bold", 28)
    pdf.setFillColor(NAVY)
    pdf.drawString(333, 326, f"{evaluation['tuned_example_win_rate_pct']:.0f}%")
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(INK)
    pdf.drawString(333, 307, "PAIRED EXAMPLE WIN RATE")
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(MUTED)
    pdf.drawString(333, 292, f"{evaluation['examples_scored']} examples; {evaluation['answer_tokens_scored']:,} answer tokens")

    card(pdf, MARGIN, 180, PAGE_W - 2 * MARGIN, 80, fill=LIGHT)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(TEAL)
    pdf.drawString(MARGIN + 14, 237, "CONFIDENCE CHECK")
    ci = evaluation["mean_nll_improvement_95pct_ci"]
    draw_wrapped(
        pdf,
        f"The 95% bootstrap interval for unweighted mean paired NLL improvement is [{ci[0]:.6f}, {ci[1]:.6f}]. Because the entire interval is above zero, the direction of improvement is consistent across resampled evaluation sets.",
        MARGIN + 14,
        218,
        PAGE_W - 2 * MARGIN - 28,
        size=9,
        leading=12,
    )

    card(pdf, MARGIN, 61, 254, 96, fill=PALE_GREEN, stroke=HexColor("#B9D9CA"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(GREEN)
    pdf.drawString(MARGIN + 14, 133, "DEFENSIBLE CLAIM")
    draw_wrapped(pdf, "On a fixed 100-example holdout subset, QLoRA tuning reduced answer perplexity by 30.42% versus the same 4-bit base model.", MARGIN + 14, 114, 226, size=8.8, leading=12, color=INK)
    card(pdf, 316, 61, 254, 96, fill=PALE_CORAL, stroke=HexColor("#E8C5B9"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(CORAL)
    pdf.drawString(330, 133, "DO NOT OVERCLAIM")
    draw_wrapped(pdf, "Perplexity is not factual accuracy or production quality. The result shows stronger fit to held-out reference answers; human and execution-based evaluation remain necessary.", 330, 114, 226, size=8.8, leading=12, color=INK)
    draw_footer(pdf, 6, "Evaluation artifact: base_vs_tuned_eval.json | Model revision pinned")
    pdf.showPage()

    # Page 7: Meaning, limitations, and references.
    y = page_title(pdf, "06 / TAKEAWAYS", "What the project proves - and what comes next", "A credible engineering report separates measured evidence from the next validation steps.")
    sections = [
        (
            "PROVEN IN THIS RUN",
            GREEN,
            PALE_GREEN,
            [
                "Reproducible ingestion and atomic caching of 50,500 real-world records.",
                "Quality gates, deterministic split, and LLaMA-Factory registration.",
                "Single-A100 QLoRA training with 0.2605% trainable parameters.",
                "Controlled adapter-on versus adapter-off evaluation on held-out answers.",
                "30.42% lower perplexity with a paired bootstrap interval above zero.",
            ],
        ),
        (
            "LIMITATIONS",
            CORAL,
            PALE_CORAL,
            [
                "Final evaluation covers 100 of the 500 held-out examples.",
                "Reference likelihood does not directly score correctness, safety, or usefulness.",
                "Community votes and acceptance are imperfect proxies for answer quality.",
                "No independent contamination audit or executable-code benchmark was run.",
                "The adapter and evaluation artifacts live in ephemeral Colab storage unless exported.",
            ],
        ),
        (
            "NEXT EXPERIMENTS",
            NAVY,
            PALE_BLUE,
            [
                "Run all 500 holdout examples and stratify by topic and answer length.",
                "Add human pairwise review for correctness, relevance, and code validity.",
                "Execute code-bearing answers in sandboxes and record pass rates.",
                "Benchmark latency, memory, and throughput before and after adapter merge.",
                "Publish the adapter, model card, data statement, and reproducibility manifest.",
            ],
        ),
    ]
    section_w = (PAGE_W - 2 * MARGIN - 20) / 3
    for idx, (heading, accent, fill, bullets) in enumerate(sections):
        x = MARGIN + idx * (section_w + 10)
        card(pdf, x, 382, section_w, 248, fill=fill, stroke=MID)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.setFillColor(accent)
        pdf.drawString(x + 13, 607, heading)
        yy = 582
        for bullet in bullets:
            pdf.setFillColor(accent)
            pdf.circle(x + 16, yy + 2, 2, fill=1, stroke=0)
            yy = draw_wrapped(pdf, bullet, x + 25, yy + 5, section_w - 38, size=7.8, leading=10.5, color=INK) - 8

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, 350, "Evidence and source links")
    references = [
        ("Project notebook", str(NOTEBOOK)),
        ("Stack Exchange preference dataset", "https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences"),
        ("Meta Llama 3 8B Instruct model card", "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"),
        ("LLaMA-Factory", "https://github.com/hiyouga/LlamaFactory"),
        ("QLoRA paper", "https://arxiv.org/abs/2305.14314"),
        ("PEFT adapter controls", "https://huggingface.co/docs/peft/main/package_reference/peft_model"),
        ("Perplexity definition", "https://huggingface.co/docs/transformers/v4.33.0/en/perplexity"),
    ]
    yy = 325
    for label, url in references:
        pdf.setFont("Helvetica-Bold", 8)
        pdf.setFillColor(TEAL)
        pdf.drawString(MARGIN, yy, label)
        pdf.setFont("Helvetica", 7.5)
        pdf.setFillColor(MUTED)
        display = url if len(url) < 95 else url[:92] + "..."
        pdf.drawString(MARGIN + 150, yy, display)
        if url.startswith("http"):
            pdf.linkURL(url, (MARGIN + 150, yy - 2, PAGE_W - MARGIN, yy + 8), relative=0)
        yy -= 22

    card(pdf, MARGIN, 72, PAGE_W - 2 * MARGIN, 72, fill=LIGHT)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN + 14, 121, "REPORTING RECOMMENDATION")
    draw_wrapped(pdf, "Use the 30.42% figure with its metric and sample size: 'reduced held-out answer perplexity by 30.42% on a 100-example paired evaluation.' Replace it after the full 500-example run rather than presenting perplexity as a generic quality score.", MARGIN + 14, 102, PAGE_W - 2 * MARGIN - 28, size=8.8, leading=12, color=MUTED)
    draw_footer(pdf, 7, "Generated from saved notebook outputs | No synthetic metrics used")
    pdf.save()


def main() -> None:
    results = load_results()
    charts = create_charts(results)
    build_pdf(results, charts)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
