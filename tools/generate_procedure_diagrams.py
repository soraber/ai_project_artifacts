"""Generate full-width SVG procedure diagrams for the repository README."""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1600
HEIGHT = 540


DIAGRAMS = [
    {
        "project": "Project 1",
        "title": "Efficient LLM Fine-Tuning Procedure",
        "subtitle": "Controlled QLoRA training and paired adapter-on/off evaluation",
        "output": ROOT / "output/project1/project1_report_assets/procedure_diagram.svg",
        "stages": [
            ("Data", ["Stack Exchange: 50,500", "validate_cached_jsonl()", "clean_rows()"]),
            ("Split and format", ["49,693 train / 500 holdout", "to_llamafactory()", "cached JSONL"]),
            ("Efficient model", ["Llama 3 8B + QLoRA", "NF4 4-bit / BF16", "0.26% trainable"]),
            ("Training", ["llamafactory-cli train", "packing + grad. accumulation", "checkpointing + cosine LR"]),
            ("Paired evaluation", ["encode_eval_example()", "forward_nll(on/off)", "same 100 examples"]),
            ("Inference decision", ["base vs. tuned comparison", "2,000 paired bootstraps", "qualified result claim"]),
        ],
        "metrics": [
            ("Answer NLL", "2.0739 -> 1.7112"),
            ("Perplexity", "7.9560 -> 5.5359"),
            ("Relative change", "30.42% reduction"),
            ("Paired win rate", "100%"),
            ("NLL improvement 95% CI", "[0.4751, 0.5900]"),
            ("Evaluation scale", "100 / 18,798 tokens"),
        ],
    },
    {
        "project": "Project 2",
        "title": "RAG Document QA Procedure",
        "subtitle": "Dense baseline versus HyDE, hybrid retrieval, fusion, and reranking",
        "output": ROOT / "output/project2/project2_report_assets/procedure_diagram.svg",
        "stages": [
            ("Corpus and chunks", ["3,200 Wikipedia rows", "900 chars / 120 overlap", "3,492 stable chunk IDs"]),
            ("Build indexes", ["E5Encoder.encode()", "FAISS IndexFlatIP", "lexical_tokens() + BM25"]),
            ("Expand query", ["hyde_prompt()", "generate_batch()", "raw + hypothetical text"]),
            ("Hybrid retrieval", ["dense_rank() + sparse_rank()", "reciprocal_rank_fusion()", "20 candidates per path"]),
            ("Rerank and answer", ["rerank(): cross-encoder", "advanced_answer_prompt()", "Llama top-5 + citations"]),
            ("Controlled evaluation", ["retrieval_metrics()", "token_f1()", "citation_validity()"]),
        ],
        "metrics": [
            ("Answer-hit", "0.833 -> 0.933"),
            ("Reciprocal rank", "0.792 -> 0.881"),
            ("Token F1", "0.317 -> 0.409"),
            ("Citation validity", "0.650 -> 0.783"),
            ("Composite", "+17.55% relative"),
            ("Latency / question", "0.805 s -> 1.749 s"),
        ],
    },
    {
        "project": "Project 3",
        "title": "Local LLM Agent Procedure",
        "subtitle": "Validated tools, deterministic policy controls, traces, and role decomposition",
        "output": ROOT / "output/project3/project3_report_assets/procedure_diagram.svg",
        "stages": [
            ("Plan action", ["plan_action()", "local Llama JSON decision", "100-token planner budget"]),
            ("Validate and recover", ["AgentDecision / Pydantic", "extract_json_object()", "retry + fallback_decision()"]),
            ("Execute typed tool", ["search_knowledge_base()", "calculate_campaign_budget()", "save_workspace_report()"]),
            ("Enforce and trace", ["traced() wrapper", "approval + path confinement", "latency / status record"]),
            ("Return response", ["run_local_agent()", "grounded_answer()", "MCP-style dispatcher"]),
            ("Role workflow", ["CrewAI sequential process", "research > strategy > critique", "edit + strategy_rubric()"]),
        ],
        "metrics": [
            ("Routing accuracy", "100%"),
            ("Native valid JSON", "66.7%"),
            ("Task success", "100%"),
            ("Citation / safety", "100% / 100%"),
            ("Median / p95", "4.86 s / 7.24 s"),
            ("Crew strategy", "+35.71%; 4.58x slower"),
        ],
    },
]


BOX_COLORS = ["#eef6ff", "#f2fbf5", "#fff8e8", "#f7f2ff", "#eef9f8", "#fff3f3"]
ACCENTS = ["#2563eb", "#15803d", "#b45309", "#7e22ce", "#0f766e", "#be123c"]


def text_element(x, y, value, css_class, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" class="{css_class}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def render(diagram):
    margin = 35
    gap = 26
    box_width = 233
    box_y = 128
    box_height = 220
    arrow_y = box_y + box_height // 2
    metric_y = 404
    metric_height = 104

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        f'<title>{escape(diagram["project"] + ": " + diagram["title"])}</title>',
        f'<desc>{escape(diagram["subtitle"])}. Methods, functions, and evaluation metrics are shown.</desc>',
        """<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#475569"/>
  </marker>
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; fill: #172033; }
    .title { font-size: 32px; font-weight: 700; }
    .subtitle { font-size: 18px; fill: #475569; }
    .section { font-size: 14px; font-weight: 700; fill: #64748b; letter-spacing: 1px; }
    .stage-title { font-size: 20px; font-weight: 700; }
    .stage-detail { font-size: 15px; fill: #334155; }
    .metric-title { font-size: 15px; font-weight: 600; fill: #cbd5e1; }
    .metric-value { font-size: 18px; font-weight: 700; fill: #ffffff; }
  </style>
</defs>""",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        text_element(margin, 48, f'{diagram["project"]}: {diagram["title"]}', "title"),
        text_element(margin, 78, diagram["subtitle"], "subtitle"),
        text_element(margin, 109, "METHODS AND FUNCTIONS", "section"),
    ]

    for index, (title, details) in enumerate(diagram["stages"]):
        x = margin + index * (box_width + gap)
        if index:
            previous_right = x - gap
            parts.append(
                f'<line x1="{previous_right + 4}" y1="{arrow_y}" x2="{x - 7}" y2="{arrow_y}" '
                'stroke="#475569" stroke-width="3" marker-end="url(#arrow)"/>'
            )
        parts.extend(
            [
                f'<rect x="{x}" y="{box_y}" width="{box_width}" height="{box_height}" rx="8" '
                f'fill="{BOX_COLORS[index]}" stroke="#cbd5e1" stroke-width="2"/>',
                f'<rect x="{x}" y="{box_y}" width="8" height="{box_height}" rx="4" fill="{ACCENTS[index]}"/>',
                text_element(x + 22, box_y + 38, f"{index + 1}. {title}", "stage-title"),
            ]
        )
        for line_index, detail in enumerate(details):
            parts.append(text_element(x + 22, box_y + 84 + line_index * 38, detail, "stage-detail"))

    parts.extend(
        [
            text_element(margin, 385, "RECORDED EVALUATION METRICS", "section"),
            f'<rect x="{margin}" y="{metric_y}" width="{WIDTH - 2 * margin}" height="{metric_height}" rx="8" fill="#1e293b"/>',
        ]
    )

    metric_width = (WIDTH - 2 * margin) / len(diagram["metrics"])
    for index, (title, value) in enumerate(diagram["metrics"]):
        x = margin + metric_width * index
        if index:
            parts.append(
                f'<line x1="{x}" y1="{metric_y + 18}" x2="{x}" y2="{metric_y + metric_height - 18}" '
                'stroke="#475569" stroke-width="1"/>'
            )
        center = x + metric_width / 2
        parts.append(text_element(center, metric_y + 39, title, "metric-title", "middle"))
        parts.append(text_element(center, metric_y + 75, value, "metric-value", "middle"))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    for diagram in DIAGRAMS:
        diagram["output"].parent.mkdir(parents=True, exist_ok=True)
        diagram["output"].write_text(render(diagram), encoding="utf-8")
        print(diagram["output"].relative_to(ROOT))


if __name__ == "__main__":
    main()
