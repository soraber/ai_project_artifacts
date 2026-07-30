#!/usr/bin/env python3
"""Build the Project 3 portfolio report from executed notebook outputs."""

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


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "ai_project_artifacts" / "03_llm_agent_system.ipynb"
OUTPUT_DIR = ROOT / "ai_project_artifacts" / "output" / "project3"
ASSET_DIR = OUTPUT_DIR / "project3_report_assets"
OUTPUT_PDF = OUTPUT_DIR / "project3_report.pdf"

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
            f"Could not find {marker} in Cell 14. Run Cells 3-14 and save the notebook first."
        )
    return json.loads(match.group(1))


def load_results() -> tuple[dict, list[dict], str]:
    notebook = json.loads(NOTEBOOK.read_text())
    text = output_text(notebook["cells"][13])
    return (
        marker_json(text, "PROJECT3_SUMMARY_JSON"),
        marker_json(text, "PROJECT3_SAMPLES_JSON"),
        marker_json(text, "PROJECT3_CREW_MEMO"),
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

    agent = summary["local_agent"]
    agent_path = ASSET_DIR / "agent_quality_metrics.png"
    labels = ["Routing", "Valid JSON", "Task\nsuccess", "Search\ncitations", "Safety\nblocks"]
    keys = [
        "routing_accuracy",
        "planner_valid_json_rate",
        "task_success_rate",
        "search_citation_rate",
        "safety_block_rate",
    ]
    values = [agent[key] for key in keys]
    colors = ["#167D7F", "#D4A72C", "#3B8C6E", "#16324F", "#D96C4A"]
    fig, ax = plt.subplots(figsize=(9.2, 3.75), dpi=180)
    bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.6)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Success rate")
    ax.set_title("Local agent evaluation (12 deterministic cases)")
    ax.grid(axis="y", color="#E4E9EC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value * 100:.1f}%",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(agent_path, bbox_inches="tight")
    plt.close(fig)

    strategy = summary["strategy"]
    strategy_path = ASSET_DIR / "strategy_quality.png"
    labels = ["Citations", "Audience", "Channels", "Timeline", "KPIs", "Safeguards", "Risks"]
    keys = [
        "source_citations",
        "audience_coverage",
        "channel_plan",
        "four_week_timeline",
        "measurable_kpis",
        "safeguards",
        "risks_and_mitigations",
    ]
    baseline = [strategy["single_agent_dimensions"][key] for key in keys]
    crew = [strategy["crewai_dimensions"][key] for key in keys]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=180, gridspec_kw={"width_ratios": [2.15, 1]})
    axes[0].bar(x - width / 2, baseline, width, label="Single agent", color="#98A7B2")
    axes[0].bar(x + width / 2, crew, width, label="CrewAI", color="#167D7F")
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Rubric credit")
    axes[0].set_title("Strategy rubric dimensions")
    axes[0].grid(axis="y", color="#E4E9EC", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper center", ncol=2)

    totals = [strategy["single_agent_score"], strategy["crewai_score"]]
    total_bars = axes[1].bar(["Single\nagent", "CrewAI"], totals, color=["#98A7B2", "#3B8C6E"], width=0.58)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Weighted score / 100")
    axes[1].set_title("Overall quality")
    axes[1].grid(axis="y", color="#E4E9EC", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar, value in zip(total_bars, totals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value:.1f}",
            ha="center",
            fontweight="bold",
        )
    fig.tight_layout(w_pad=2.2)
    fig.savefig(strategy_path, bbox_inches="tight")
    plt.close(fig)
    return {"agent": agent_path, "strategy": strategy_path}


def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    lines = []
    for paragraph in str(text).split("\n"):
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
    pdf.drawString(MARGIN, 16, "Project 3 | Local LLM Agent and CrewAI System")
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
        ("Question", "natural language", PALE_BLUE),
        ("Planner", "local Llama JSON", PALE_CORAL),
        ("Validate", "schema + retry", PALE_GREEN),
        ("Tool", "typed execution", PALE_BLUE),
        ("Answer", "evidence + trace", PALE_GREEN),
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


def draw_crew_pipeline(pdf, y):
    roles = [
        ("Researcher", "retrieve facts", PALE_BLUE),
        ("Strategist", "draft plan", PALE_GREEN),
        ("Critic", "find gaps", PALE_CORAL),
        ("Editor", "final memo", PALE_GREEN),
    ]
    gap = 18
    width = (PAGE_W - 2 * MARGIN - gap * 3) / 4
    height = 86
    for index, (title, detail, fill) in enumerate(roles):
        x = MARGIN + index * (width + gap)
        pdf.setFillColor(fill)
        pdf.setStrokeColor(MID)
        pdf.roundRect(x, y - height, width, height, 6, fill=1, stroke=1)
        pdf.setFillColor(TEAL if index != 2 else CORAL)
        pdf.circle(x + 20, y - 22, 10, fill=1, stroke=0)
        pdf.setFillColor(white)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawCentredString(x + 20, y - 25, str(index + 1))
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(x + 36, y - 26, title)
        draw_wrapped(pdf, detail, x + 12, y - 53, width - 24, size=8.2, leading=10, color=MUTED)
        if index < len(roles) - 1:
            arrow_x = x + width + 3
            pdf.setStrokeColor(TEAL)
            pdf.line(arrow_x, y - height / 2, arrow_x + gap - 6, y - height / 2)
            pdf.line(arrow_x + gap - 9, y - height / 2 + 3, arrow_x + gap - 6, y - height / 2)
            pdf.line(arrow_x + gap - 9, y - height / 2 - 3, arrow_x + gap - 6, y - height / 2)
    return y - height


def build_report(summary: dict, samples: list[dict], memo: str, charts: dict[str, Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agent = summary["local_agent"]
    strategy = summary["strategy"]
    lift = strategy["relative_improvement_percent"]
    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=LETTER)
    pdf.setTitle("Project 3 Report - Local LLM Agent and CrewAI System")

    # Page 1: executive result
    pdf.setFillColor(NAVY)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(TEAL)
    pdf.rect(0, PAGE_H - 16, PAGE_W, 16, fill=1, stroke=0)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN, PAGE_H - 68, "LLM & AGENT PROJECTS / PROJECT 3")
    pdf.setFont("Helvetica-Bold", 31)
    pdf.drawString(MARGIN, PAGE_H - 117, "Local LLM Agent System")
    pdf.drawString(MARGIN, PAGE_H - 154, "with MCP-Style Tools")
    draw_wrapped(
        pdf,
        "A portfolio-grade LangChain controller and CrewAI marketing workflow, powered by a locally cached Llama 3 8B model with typed tools, policy gates, traces, and measured evaluation.",
        MARGIN,
        PAGE_H - 190,
        PAGE_W - 2 * MARGIN,
        size=11.6,
        leading=16,
        color=HexColor("#DDE8EE"),
    )
    card_y = 385
    metric_card(pdf, MARGIN, card_y, 158, 82, f"{agent['task_success_rate'] * 100:.0f}%", "TASK SUCCESS", white)
    metric_card(pdf, MARGIN + 172, card_y, 158, 82, f"{agent['safety_block_rate'] * 100:.0f}%", "SAFETY BLOCK RATE", white)
    metric_card(pdf, MARGIN + 344, card_y, 142, 82, f"+{lift:.1f}%", "STRATEGY LIFT", white)
    pdf.setFillColor(HexColor("#233E59"))
    pdf.roundRect(MARGIN, 150, PAGE_W - 2 * MARGIN, 185, 7, fill=1, stroke=0)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN + 18, 307, "What the implementation proves")
    bullets = [
        "Local Llama planner emits auditable structured tool decisions",
        "LangChain typed tools separate model judgment from deterministic actions",
        "MCP-style schemas expose capabilities, side effects, and safety annotations",
        "CrewAI role decomposition improved the saved strategy rubric score",
    ]
    y = 282
    for bullet in bullets:
        pdf.setFillColor(GOLD)
        pdf.circle(MARGIN + 23, y + 3, 2.6, fill=1, stroke=0)
        y = draw_wrapped(pdf, bullet, MARGIN + 34, y + 6, PAGE_W - 2 * MARGIN - 52, size=9.6, leading=15, color=white) - 5
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(HexColor("#AFC1CE"))
    pdf.drawString(MARGIN, 48, "Hongda Wu | AI Agent Engineer portfolio | Reproducible Colab notebook")
    pdf.showPage()

    # Page 2: problem and scenario
    y = page_title(pdf, "01 / Problem", "A realistic business agent", "RelayDesk AI supplies a compact, inspectable knowledge base for testing agent orchestration without external API dependence.")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Engineering objective")
    y = draw_wrapped(
        pdf,
        "Answer product questions, calculate campaign budgets, and draft approved reports while preventing unauthorized writes. A second workflow turns the same evidence into a 30-day marketing strategy through specialized CrewAI roles.",
        MARGIN,
        y - 19,
        PAGE_W - 2 * MARGIN,
        size=10,
        leading=14,
    )
    y -= 13
    metric_card(pdf, MARGIN, y - 95, 158, 76, str(summary["knowledge_base_files"]), "KNOWLEDGE FILES", PALE_BLUE)
    metric_card(pdf, MARGIN + 172, y - 95, 158, 76, str(len(summary["langchain_tools"])), "TYPED TOOLS", PALE_GREEN)
    metric_card(pdf, MARGIN + 344, y - 95, 142, 76, "LOCAL", "MODEL EXECUTION", PALE_CORAL)
    y -= 126
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Scenario and constraints")
    scenario = [
        ("Audience", "Ecommerce founders and support operations leads at teams of 5 to 50."),
        ("Product", "Workflow assistant for support replies, policy retrieval, and campaign briefs."),
        ("Safety", "Human approval is required before customer messages, campaign publication, or file writes."),
        ("Privacy", "Customer PII must not be uploaded to public tools."),
    ]
    y -= 19
    for title, detail in scenario:
        pdf.setFillColor(LIGHT)
        pdf.roundRect(MARGIN, y - 48, PAGE_W - 2 * MARGIN, 40, 5, fill=1, stroke=0)
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 9.2)
        pdf.drawString(MARGIN + 12, y - 31, title)
        draw_wrapped(pdf, detail, MARGIN + 85, y - 27, PAGE_W - 2 * MARGIN - 100, size=8.5, leading=10.5)
        y -= 52
    draw_footer(pdf, 2)
    pdf.showPage()

    # Page 3: LangChain and MCP architecture
    y = page_title(pdf, "02 / Agent", "Auditable tool orchestration", "The LLM chooses an action, but schemas, deterministic code, and policy checks control execution.")
    y = draw_pipeline(pdf, y) - 27
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Three typed capabilities")
    tools = [
        ("Search", "BM25 retrieval over four local Markdown files; answers cite source filenames.", PALE_BLUE),
        ("Budget", "Deterministic arithmetic returns weekly spend, total, and cap compliance.", PALE_GREEN),
        ("Write", "Path confinement and explicit confirm_write approval block unsafe side effects.", PALE_CORAL),
    ]
    col_w = (PAGE_W - 2 * MARGIN - 20) / 3
    for index, (title, detail, fill) in enumerate(tools):
        x = MARGIN + index * (col_w + 10)
        pdf.setFillColor(fill)
        pdf.roundRect(x, y - 126, col_w, 107, 6, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(x + 12, y - 43, title)
        draw_wrapped(pdf, detail, x + 12, y - 63, col_w - 24, size=8.1, leading=11, color=MUTED)
    y -= 156
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Why the MCP-style contract matters")
    y = draw_wrapped(
        pdf,
        "Each tool publishes an input schema, output schema, description, and annotations for read-only, destructive, idempotent, and open-world behavior. This makes discovery explicit and gives a future MCP server a stable interface without coupling business logic to one agent framework.",
        MARGIN,
        y - 19,
        PAGE_W - 2 * MARGIN,
        size=9.4,
        leading=13.5,
    )
    y -= 8
    pdf.setFillColor(PALE_GREEN)
    pdf.roundRect(MARGIN, y - 75, PAGE_W - 2 * MARGIN, 63, 6, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(MARGIN + 13, y - 35, "Failure recovery")
    draw_wrapped(pdf, "Malformed planner JSON triggers validation, one constrained retry, then a deterministic fallback route. Every tool call records arguments, status, and latency.", MARGIN + 114, y - 31, PAGE_W - 2 * MARGIN - 128, size=8.4, leading=11)
    draw_footer(pdf, 3)
    pdf.showPage()

    # Page 4: CrewAI architecture
    y = page_title(pdf, "03 / CrewAI", "Role-specialized strategy generation", "Sequential roles convert retrieved evidence into a business plan, then challenge and revise it.")
    y = draw_crew_pipeline(pdf, y) - 28
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Why use multiple agents")
    rationale = [
        ("Separation of concerns", "Research emphasizes evidence; strategy emphasizes synthesis; critique emphasizes gaps and unsupported claims."),
        ("Explicit handoffs", "Each role receives the prior artifact, making intermediate reasoning inspectable and replaceable."),
        ("Quality control", "The critic asks for missing safeguards, budget alignment, citations, and measurable outcomes before editing."),
    ]
    y -= 22
    for title, detail in rationale:
        pdf.setFillColor(LIGHT)
        pdf.roundRect(MARGIN, y - 62, PAGE_W - 2 * MARGIN, 52, 5, fill=1, stroke=0)
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 9.2)
        pdf.drawString(MARGIN + 13, y - 32, title)
        draw_wrapped(pdf, detail, MARGIN + 135, y - 27, PAGE_W - 2 * MARGIN - 150, size=8.3, leading=10.5)
        y -= 68
    y -= 4
    pdf.setFillColor(PALE_CORAL)
    pdf.roundRect(MARGIN, y - 91, PAGE_W - 2 * MARGIN, 80, 6, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.drawString(MARGIN + 13, y - 34, "Measured tradeoff")
    tradeoff = f"Quality rose {lift:.1f}%, while generation time increased from {strategy['single_agent_seconds']:.1f}s to {strategy['crewai_seconds']:.1f}s. Role decomposition is appropriate for high-value asynchronous planning, not every low-latency request."
    draw_wrapped(pdf, tradeoff, MARGIN + 13, y - 53, PAGE_W - 2 * MARGIN - 26, size=8.7, leading=12)
    draw_footer(pdf, 4)
    pdf.showPage()

    # Page 5: measured results
    y = page_title(pdf, "04 / Results", "Measured quality and tradeoffs", f"All 12 agent cases passed; CrewAI improved the weighted strategy score from {strategy['single_agent_score']:.1f} to {strategy['crewai_score']:.1f}.")
    pdf.drawImage(str(charts["agent"]), MARGIN, y - 245, width=PAGE_W - 2 * MARGIN, height=215, preserveAspectRatio=True, anchor="c")
    pdf.drawImage(str(charts["strategy"]), MARGIN, y - 485, width=PAGE_W - 2 * MARGIN, height=220, preserveAspectRatio=True, anchor="c")
    draw_footer(pdf, 5)
    pdf.showPage()

    # Page 6: evidence, caveats, and sources
    y = page_title(pdf, "05 / Evidence", "What the result means", "The report keeps the measured claim, a representative trace, and limitations together.")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Representative grounded answer")
    if samples:
        sample = samples[0]
        y = draw_wrapped(pdf, "Question: " + sample["task"], MARGIN, y - 19, PAGE_W - 2 * MARGIN, font="Helvetica-Bold", size=9.1, leading=12)
        y = draw_wrapped(pdf, "Planned tool: " + sample["planned_tool"], MARGIN, y - 5, PAGE_W - 2 * MARGIN, size=8.7, leading=11.5, color=MUTED)
        y = draw_wrapped(pdf, "Answer: " + sample["answer"], MARGIN, y - 6, PAGE_W - 2 * MARGIN, size=8.9, leading=12)
    y -= 9
    pdf.setFillColor(PALE_CORAL)
    pdf.roundRect(MARGIN, y - 125, PAGE_W - 2 * MARGIN, 114, 6, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN + 14, y - 34, "Caveats")
    caveat = (
        "The agent suite has only 12 deterministic cases and the strategy score is a lexical rubric, not a human business review. "
        "The final memo still contains unsupported improvement claims and earned no KPI credit under the strict rubric. "
        "A production follow-up should add human grading, prompt-injection tests, repeated seeds, token/cost telemetry, and a real MCP transport layer."
    )
    draw_wrapped(pdf, caveat, MARGIN + 14, y - 53, PAGE_W - 2 * MARGIN - 28, size=8.6, leading=12)
    y -= 151
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Saved evidence")
    evidence = [
        "Final summary JSON with evaluation seed, metrics, timings, and rubric dimensions",
        "Representative agent samples with planned tools and grounded responses",
        "Full CrewAI memo plus the executed notebook outputs and debugging log",
    ]
    y -= 20
    for item in evidence:
        pdf.setFillColor(GREEN)
        pdf.circle(MARGIN + 4, y + 3, 2.3, fill=1, stroke=0)
        y = draw_wrapped(pdf, item, MARGIN + 14, y + 6, PAGE_W - 2 * MARGIN - 14, size=8.5, leading=11.5) - 3
    y -= 6
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(INK)
    pdf.drawString(MARGIN, y, "Sources and implementation references")
    references = [
        ("LangChain agents", "https://docs.langchain.com/oss/python/langchain/agents"),
        ("MCP tools specification", "https://modelcontextprotocol.io/specification/2025-11-25/server/tools"),
        ("CrewAI documentation", "https://docs.crewai.com/"),
        ("Meta Llama 3 8B Instruct", "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"),
    ]
    y -= 19
    for label, url in references:
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica-Bold", 8.1)
        pdf.drawString(MARGIN, y, label)
        pdf.linkURL(url, (MARGIN, y - 2, MARGIN + 170, y + 9), relative=0)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 7.1)
        pdf.drawRightString(PAGE_W - MARGIN, y, url)
        y -= 17
    draw_footer(pdf, 6)
    pdf.showPage()
    pdf.save()


def main() -> None:
    summary, samples, memo = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "project3_final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "project3_representative_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    (OUTPUT_DIR / "project3_crewai_memo.md").write_text(memo.rstrip() + "\n")
    charts = create_charts(summary)
    build_report(summary, samples, memo, charts)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
