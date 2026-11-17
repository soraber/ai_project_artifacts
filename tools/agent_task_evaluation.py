"""Small authored task smoke set, separate from held-out language-model NLL.

Generated Python is never executed here. A caller may supply a separately
validated isolated executor; otherwise code correctness remains unmeasured.
"""
from __future__ import annotations

import json
import math
import statistics
import time


TASKS = [
    {"id": "add", "task": "Use calculator to add 17 and 29.", "tool": "calculator", "arguments": {"operation": "add", "a": 17, "b": 29}},
    {"id": "multiply", "task": "Use calculator to multiply 12 by 8.", "tool": "calculator", "arguments": {"operation": "multiply", "a": 12, "b": 8}},
    {"id": "lookup", "task": "Look up the return policy using query 'return policy'.", "tool": "lookup", "arguments": {"query": "return policy"}},
    {"id": "missing_fact", "task": "What is our company's unreleased 2027 revenue? No source or data has been supplied.", "tool": "abstain", "arguments": {}},
    {"id": "missing_amount", "task": "Add my expenses. I have not supplied their amounts.", "tool": "abstain", "arguments": {}},
    {"id": "code", "task": "Return Python defining add(a, b), which adds its two inputs.", "tool": "python", "tests": "from candidate import add\ndef test_add():\n    assert add(2,3)==5\n    assert add(-4,1)==-3\n"},
]

PROMPT = '''Return exactly one JSON object with keys tool and arguments, with no other text.
Allowed contracts:
calculator: arguments {operation: add or multiply, a: finite number, b: finite number}
lookup: arguments {query: nonempty string}
abstain: arguments {} when required evidence/inputs are missing
python: arguments {code: Python source text}; do not call a host shell.
Task: {task}
'''


def validate_decision(raw):
    def reject_constant(value):
        raise ValueError(f"Non-JSON number: {value}")
    value = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
        raise ValueError("Expected exactly tool and arguments")
    tool, args = value["tool"], value["arguments"]
    if not isinstance(args, dict):
        raise ValueError("Arguments must be an object")
    if tool == "calculator":
        if set(args) != {"operation", "a", "b"} or args["operation"] not in {"add", "multiply"}:
            raise ValueError("Invalid calculator contract")
        if any(type(args[key]) not in (int, float) or not math.isfinite(args[key]) for key in ["a", "b"]):
            raise ValueError("Calculator operands must be finite numbers")
    elif tool in {"lookup", "python"}:
        key = "query" if tool == "lookup" else "code"
        if set(args) != {key} or not isinstance(args[key], str) or not args[key].strip():
            raise ValueError("Missing query or source code")
    elif tool != "abstain" or args:
        raise ValueError("Unsupported action or nonempty abstention arguments")
    return value


def compare_tasks(generate, code_executor=None):
    """generate(prompt, adapter_enabled) must use matched deterministic decoding."""
    records = []
    for task in TASKS:
        for system, enabled in [("base", False), ("tuned", True)]:
            started = time.perf_counter()
            row = {"task_id": task["id"], "system": system, "schema_valid": False,
                   "tool_correct": False, "arguments_correct": None if task["tool"] == "python" else False,
                   "abstention_correct": None, "code_tests_passed": None, "error": None}
            try:
                raw = generate(PROMPT.replace("{task}", task["task"]), enabled)
                row["response"] = raw
                decision = validate_decision(raw)
                row["schema_valid"] = True
                row["tool_correct"] = decision["tool"] == task["tool"]
                if "arguments" in task:
                    row["arguments_correct"] = row["tool_correct"] and decision["arguments"] == task["arguments"]
                if task["tool"] == "abstain":
                    row["abstention_correct"] = row["tool_correct"]
                if task["tool"] == "python" and code_executor is not None:
                    row["code_tests_passed"] = (bool(code_executor(decision["arguments"]["code"], task["tests"]))
                                                if row["tool_correct"] else False)
            except Exception as exc:
                row["error"] = type(exc).__name__
                if task["tool"] == "abstain": row["abstention_correct"] = False
            row["latency_seconds"] = time.perf_counter() - started
            records.append(row)
    metrics = {}
    for system in ["base", "tuned"]:
        selected = [row for row in records if row["system"] == system]
        metrics[system] = {}
        for key in ["schema_valid", "tool_correct", "arguments_correct", "abstention_correct", "code_tests_passed"]:
            values = [row[key] for row in selected if row[key] is not None]
            metrics[system][key] = {"mean": statistics.fmean(values) if values else None, "count": len(values)}
    return {"evaluation_version": "authored-task-smoke-v1", "records": records, "metrics": metrics,
            "limitations": "Six authored smoke tasks, not an independent agent benchmark. Code scoring requires an isolated executor; never execute generated code in Colab."}
