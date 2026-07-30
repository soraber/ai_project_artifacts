#!/usr/bin/env python3
"""Build the Project 2 portfolio report from executed notebook outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "02_rag_document_qa_system.ipynb"
OUTPUT_DIR = REPO_ROOT / "output" / "project2"
ASSET_DIR = OUTPUT_DIR / "project2_report_assets"
OUTPUT_PDF = OUTPUT_DIR / "project2_report.pdf"

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
            value = item["text"]
            parts.append("".join(value) if isinstance(value, list) else value)
        elif item.get("output_type") == "error":
            parts.append("\n".join(item.get("traceback", [])))
    return "\n".join(parts)


def marker_json(text: str, marker: str):
    match = re.search(rf"^{re.escape(marker)}=(.+)$", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(
            f"Could not find {marker} in Cell 17. Run Cells 3-17 and save the notebook first."
        )
    return json.loads(match.group(1))


def load_results() -> tuple[dict, list[dict]]:
    notebook = json.loads(NOTEBOOK.read_text())
    text = output_text(notebook["cells"][16])
    return marker_json(text, "PROJECT2_SUMMARY_JSON"), marker_json(
        text, "PROJECT2_SAMPLES_JSON"
    )


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


def create_charts(summary: dict) -> dict[str, Path]:
    configure_plots()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {row["system"]: row for row in summary["metrics"]}
    baseline = metrics["baseline"]
    advanced = metrics["advanced"]

    quality_path = ASSET_DIR / "quality_metrics.png"
    labels = ["Answer hit", "Reciprocal rank", "Token F1", "Valid citations"]
    keys = ["answer_hit", "reciprocal_rank", "token_f1", "citation_validity"]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=180)
    first = ax.bar(x - width / 2, [baseline[key] for key in keys], width, label="Dense baseline", color="#98A7B2")
    second = ax.bar(x + width / 2, [advanced[key] for key in keys], width, label="Advanced RAG", color="#167D7F")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (higher is better)")
    ax.grid(axis="y", color="#E4E9EC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    for bars in [first, second]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{bar.get_height():.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(quality_path, bbox_inches="tight")
    plt.close(fig)

    tradeoff_path = ASSET_DIR / "score_latency_tradeoff.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), dpi=180)
    systems = ["Dense\nbaseline", "Advanced\nRAG"]
    colors = ["#98A7B2", "#3B8C6E"]
    composite = [baseline["composite_score"], advanced["composite_score"]]
    latency = [
        baseline["retrieval_seconds_per_question"] + baseline["generation_seconds_per_question"],
        advanced["retrieval_seconds_per_question"] + advanced["generation_seconds_per_question"],
    ]
    for ax, values, title, ylabel in [
        (axes[0], composite, "Transparent composite quality", "Composite score"),
        (axes[1], latency, "End-to-end latency", "Seconds per question"),
    ]:
        bars = ax.bar(systems, values, color=colors, width=0.56)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E4E9EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.28 if max(values) else 1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.04, f"{value:.3f}", ha="center", fontweight="bold")
    fig.tight_layout(w_pad=3)
    fig.savefig(tradeoff_path, bbox_inches="tight")
    plt.close(fig)
    return {"quality": quality_path, "tradeoff": tradeoff_path}


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


def draw_wrapped(pdf, text, x, y, width, font="Helvetica", size=9.3, leading=12.5, color=INK):
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    for line in wrap_lines(str(text), font, size, width):
        if line:
            pdf.drawString(x, y, line)
        y -= leading
    return y


def draw_footer(pdf, page):
    pdf.setStrokeColor(MID)
    pdf.line(MARGIN, 28, PAGE_W - MARGIN, 28)
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(MUTED)
    pdf.drawString(MARGIN, 16, "Project 2 | RAG Document QA System")
    pdf.drawRightString(PAGE_W - MARGIN, 16, f"{page} / 6")


def page_title(pdf, number, title, subtitle=None):
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


def metric_card(pdf, x, y, width, height, value, label, fill):
    pdf.setFillColor(fill)
    pdf.roundRect(x, y, width, height, 6, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(x + 13, y + height - 27, value)
    pdf.setFont("Helvetica", 8.2)
    pdf.setFillColor(MUTED)
    pdf.drawString(x + 13, y + 12, label)


def draw_pipeline(pdf, y):
    stages = [
        ("Question", "raw query", PALE_BLUE),
        ("HyDE", "synthetic passage", PALE_CORAL),
        ("Hybrid", "E5 + BM25 + RRF", PALE_GREEN),
        ("Rerank", "cross-encoder", PALE_BLUE),
        ("Answer", "Llama + citations", PALE_GREEN),
    ]
    gap = 10
    width = (PAGE_W - 2 * MARGIN - gap * 4) / 5
    height = 72
    for index, (title, detail, fill) in enumerate(stages):
        x = MARGIN + index * (width + gap)
        pdf.setFillColor(fill)
        pdf.setStrokeColor(MID)
        pdf.roundRect(x, y - height, width, height, 5, fill=1, stroke=1)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 9.5)
        pdf.drawCentredString(x + width / 2, y - 27, title)
        draw_wrapped(pdf, detail, x + 6, y - 45, width - 12, size=7.4, leading=9, color=MUTED)
        if index < len(stages) - 1:
            arrow_x = x + width + 2
            pdf.setStrokeColor(TEAL)
            pdf.setFillColor(TEAL)
            pdf.line(arrow_x, y - height / 2, arrow_x + gap - 4, y - height / 2)
            pdf.line(arrow_x + gap - 7, y - height / 2 + 3, arrow_x + gap - 4, y - height / 2)
            pdf.line(arrow_x + gap - 7, y - height / 2 - 3, arrow_x + gap - 4, y - height / 2)
    return y - height


def build_report(summary: dict, samples: list[dict], charts: dict[str, Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {row["system"]: row for row in summary["metrics"]}
    baseline = metrics["baseline"]
    advanced = metrics["advanced"]
    lift = summary["relative_improvement_percent"]
    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=LETTER)
    pdf.setTitle("Project 2 Report - RAG Document QA System")

    # Page 1: executive result
    pdf.setFillColor(NAVY)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(TEAL)
    pdf.rect(0, PAGE_H - 16, PAGE_W, 16, fill=1, stroke=0)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN, PAGE_H - 68, "LLM & AGENT PROJECTS / PROJECT 2")
    pdf.setFont("Helvetica-Bold", 31)
    pdf.drawString(MARGIN, PAGE_H - 117, "Evidence-Grounded RAG")
    pdf.drawString(MARGIN, PAGE_H - 154, "Document QA System")
    draw_wrapped(
        pdf,
        "A measured retrieval upgrade on real Wikipedia QA: HyDE, hybrid search, cross-encoder reranking, evidence-first generation, and traceable citations.",
        MARGIN,
        PAGE_H - 190,
        PAGE_W - 2 * MARGIN,
        size=12,
        leading=17,
        color=HexColor("#DDE8EE"),
    )
    card_y = 385
    metric_card(pdf, MARGIN, card_y, 158, 82, f"{lift:+.1f}%", "MEASURED COMPOSITE LIFT", white)
    metric_card(pdf, MARGIN + 172, card_y, 158, 82, f"{summary['evaluation_examples']}", "HELD-OUT QA EXAMPLES", white)
    metric_card(pdf, MARGIN + 344, card_y, 142, 82, f"{summary['chunks']:,}", "SEARCHABLE CHUNKS", white)
    pdf.setFillColor(HexColor("#233E59"))
    pdf.roundRect(MARGIN, 150, PAGE_W - 2 * MARGIN, 185, 7, fill=1, stroke=0)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN + 18, 307, "Portfolio claim, backed by artifacts")
    y = 282
    bullets = [
        "Real public benchmark pinned to an immutable dataset revision",
        "Same local generator in both systems to isolate retrieval improvements",
        "Per-example predictions, retrieved chunk IDs, metrics, and timing saved",
        "Optional OpenAI and DeepInfra adapters use identical retrieved contexts",
    ]
    for bullet in bullets:
        pdf.setFillColor(GOLD)
        pdf.circle(MARGIN + 23, y + 3, 2.6, fill=1, stroke=0)
        y = draw_wrapped(pdf, bullet, MARGIN + 34, y + 6, PAGE_W - 2 * MARGIN - 52, size=9.6, leading=15, color=white) - 5
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(HexColor("#AFC1CE"))
    pdf.drawString(MARGIN, 48, "Hongda Wu | AI Agent Engineer portfolio | Reproducible notebook experiment")
    pdf.showPage()

    # Page 2: problem and data
    y = page_title(pdf, "01 / Problem", "What this project demonstrates", "Production-oriented RAG is a retrieval and evidence problem, not merely an LLM call.")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Engineering objective")
    y = draw_wrapped(pdf, "Improve factual document QA while keeping the experiment auditable. The baseline and advanced variants share the same corpus, evaluation set, generator, decoding settings, and metric code.", MARGIN, y - 19, PAGE_W - 2 * MARGIN, size=10, leading=14)
    y -= 13
    metric_card(pdf, MARGIN, y - 95, 158, 76, f"{summary['corpus_rows']:,}", "WIKIPEDIA PASSAGES", PALE_BLUE)
    metric_card(pdf, MARGIN + 172, y - 95, 158, 76, f"{summary['qa_rows']:,}", "AVAILABLE QUESTIONS", PALE_GREEN)
    metric_card(pdf, MARGIN + 344, y - 95, 142, 76, "CC BY 3.0", "DATASET LICENSE", PALE_CORAL)
    y -= 126
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Why RAG Mini-Wikipedia")
    y = draw_wrapped(pdf, "It is public, compact enough for Colab, and contains a real corpus plus held-out questions. The dataset does not publish gold passage IDs, so this implementation explicitly evaluates whether the reference answer occurs in a retrieved passage rather than inventing relevance labels.", MARGIN, y - 19, PAGE_W - 2 * MARGIN, size=9.7, leading=14)
    y -= 13
    pdf.setFillColor(LIGHT)
    pdf.roundRect(MARGIN, y - 143, PAGE_W - 2 * MARGIN, 132, 6, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN + 14, y - 31, "Reproducibility contract")
    details = [
        f"Dataset revision: {summary['dataset_revision'][:12]}...",
        f"Deterministic final-holdout seed: {summary.get('evaluation_seed', 2026)}",
        "Downloaded datasets and corpus embeddings are cached",
        "Raw per-example outputs are saved as CSV; summary is saved as JSON",
    ]
    line_y = y - 52
    for detail in details:
        pdf.setFillColor(TEAL)
        pdf.circle(MARGIN + 18, line_y + 2, 2.2, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(MARGIN + 28, line_y, detail)
        line_y -= 20
    draw_footer(pdf, 2)
    pdf.showPage()

    # Page 3: architecture
    y = page_title(pdf, "02 / System", "From question to cited answer", "The advanced route improves recall first, then precision, then grounded generation.")
    y = draw_pipeline(pdf, y) - 30
    columns = [
        ("Baseline", ["E5 query embedding", "FAISS top-3", "Direct context prompt"], PALE_BLUE),
        ("Advanced", ["HyDE passage expansion", "E5 + BM25 with RRF", "Cross-encoder top-5", "Evidence-first cited answer"], PALE_GREEN),
    ]
    col_w = (PAGE_W - 2 * MARGIN - 18) / 2
    for index, (title, items, fill) in enumerate(columns):
        x = MARGIN + index * (col_w + 18)
        height = 165
        pdf.setFillColor(fill)
        pdf.roundRect(x, y - height, col_w, height, 6, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(x + 15, y - 27, title)
        line_y = y - 53
        for item in items:
            pdf.setFillColor(TEAL if index else NAVY)
            pdf.circle(x + 19, line_y + 2, 2.4, fill=1, stroke=0)
            pdf.setFillColor(INK)
            pdf.setFont("Helvetica", 9.2)
            pdf.drawString(x + 30, line_y, item)
            line_y -= 25
    y -= 195
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Design rationale")
    rationale = "Dense retrieval supplies semantic matching; BM25 recovers exact entities and numbers; reciprocal-rank fusion avoids incomparable raw scores; the cross-encoder spends extra compute only on a small candidate set; HyDE bridges terse questions and passage-style documents."
    draw_wrapped(pdf, rationale, MARGIN, y - 19, PAGE_W - 2 * MARGIN, size=9.7, leading=14)
    draw_footer(pdf, 3)
    pdf.showPage()

    # Page 4: evaluation
    y = page_title(pdf, "03 / Evaluation", "A score that can be defended", "Four complementary metrics expose retrieval, answer overlap, and citation behavior.")
    metric_defs = [
        ("Answer-hit@k", "Reference answer text appears in at least one retrieved chunk.", "25%"),
        ("Reciprocal rank", "Rewards systems that place answer-bearing evidence earlier.", "15%"),
        ("Token F1", "Standard lexical overlap between generated and reference answers.", "45%"),
        ("Citation validity", "Cited chunk IDs must belong to the supplied context.", "15%"),
    ]
    for title, detail, weight in metric_defs:
        pdf.setFillColor(LIGHT)
        pdf.roundRect(MARGIN, y - 62, PAGE_W - 2 * MARGIN, 52, 5, fill=1, stroke=0)
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(MARGIN + 13, y - 31, title)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica", 8.7)
        pdf.drawString(MARGIN + 128, y - 31, detail)
        pdf.setFillColor(CORAL)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawRightString(PAGE_W - MARGIN - 13, y - 31, weight)
        y -= 68
    y -= 7
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Validity controls")
    controls = [
        "Only answer-grounded questions are sampled; yes/no references are excluded.",
        "The same deterministic 60 examples are used for both systems.",
        "Generator model, decoding, corpus, and preprocessing remain fixed.",
        "The improvement percentage is derived from saved outputs, never inserted as a target.",
    ]
    y -= 23
    for control in controls:
        pdf.setFillColor(GREEN)
        pdf.circle(MARGIN + 4, y + 3, 2.4, fill=1, stroke=0)
        y = draw_wrapped(pdf, control, MARGIN + 14, y + 6, PAGE_W - 2 * MARGIN - 14, size=9.2, leading=13) - 5
    draw_footer(pdf, 4)
    pdf.showPage()

    # Page 5: measured results
    y = page_title(pdf, "04 / Results", "Measured quality and tradeoffs", f"Advanced RAG changed the composite score from {baseline['composite_score']:.3f} to {advanced['composite_score']:.3f} ({lift:+.1f}%).")
    pdf.drawImage(str(charts["quality"]), MARGIN, y - 255, width=PAGE_W - 2 * MARGIN, height=225, preserveAspectRatio=True, anchor="c")
    pdf.drawImage(str(charts["tradeoff"]), MARGIN, y - 480, width=PAGE_W - 2 * MARGIN, height=205, preserveAspectRatio=True, anchor="c")
    draw_footer(pdf, 5)
    pdf.showPage()

    # Page 6: evidence, caveats, sources
    y = page_title(pdf, "05 / Evidence", "What the result means", "A portfolio result is strongest when the claim, limitations, and next experiment are all visible.")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Representative held-out example")
    if samples:
        sample = samples[0]
        y = draw_wrapped(pdf, "Question: " + sample["question"], MARGIN, y - 19, PAGE_W - 2 * MARGIN, font="Helvetica-Bold", size=9, leading=12)
        y = draw_wrapped(pdf, "Reference: " + sample["reference"], MARGIN, y - 5, PAGE_W - 2 * MARGIN, size=8.8, leading=12, color=MUTED)
        y = draw_wrapped(pdf, "Baseline: " + sample["baseline_answer"], MARGIN, y - 7, PAGE_W - 2 * MARGIN, size=8.6, leading=11.5)
        y = draw_wrapped(pdf, "Advanced: " + sample["advanced_answer"], MARGIN, y - 7, PAGE_W - 2 * MARGIN, size=8.6, leading=11.5)
    y -= 10
    pdf.setFillColor(PALE_CORAL)
    pdf.roundRect(MARGIN, y - 106, PAGE_W - 2 * MARGIN, 96, 6, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN + 14, y - 31, "Caveats and next step")
    caveat = "Answer-hit is a proxy because the dataset has no gold passage labels, and token F1 can under-credit valid paraphrases. A production follow-up should add human relevance judgments or an independently validated LLM judge, repeat across seeds, and measure cost under expected traffic."
    draw_wrapped(pdf, caveat, MARGIN + 14, y - 50, PAGE_W - 2 * MARGIN - 28, size=8.7, leading=12)
    y -= 134
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Sources and implementation references")
    references = [
        ("RAG Mini-Wikipedia dataset", "https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia"),
        ("E5 text embeddings", "https://huggingface.co/intfloat/e5-small-v2"),
        ("HyDE retrieval paper", "https://arxiv.org/abs/2212.10496"),
        ("MS MARCO MiniLM cross-encoder", "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ("Meta Llama 3 8B Instruct", "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"),
    ]
    y -= 20
    for label, url in references:
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 8.2)
        pdf.drawString(MARGIN, y, label)
        pdf.linkURL(url, (MARGIN, y - 2, MARGIN + 180, y + 9), relative=0)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 7.3)
        pdf.drawRightString(PAGE_W - MARGIN, y, url)
        y -= 18
    draw_footer(pdf, 6)
    pdf.save()


def main() -> None:
    summary, samples = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "project2_final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (OUTPUT_DIR / "project2_representative_samples.json").write_text(
        json.dumps(samples, indent=2) + "\n"
    )
    charts = create_charts(summary)
    build_report(summary, samples, charts)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
