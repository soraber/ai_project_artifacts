import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).parents[1]


def source(filename, cell_id):
    notebook = json.loads((ROOT / filename).read_text())
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell.get("id") == cell_id))


def router_namespace():
    code = source("03_llm_agent_system.ipynb", "c576a49d")
    tree = ast.parse(code)
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(BaseModel=BaseModel, ConfigDict=ConfigDict, Field=Field,
                     Any=Any, Literal=Literal, re=re, json=json,
                     ValidationError=ValidationError, ROUTER_INSTRUCTION="Choose one tool")
    exec(compile(ast.Module(definitions, type_ignores=[]), "router", "exec"), namespace)
    return namespace


def test_router_rejects_invalid_tool_arguments_before_execution():
    class Search(BaseModel):
        query: str
        top_k: int = Field(ge=1, le=5)
    namespace = router_namespace()
    namespace["LANGCHAIN_TOOLS"] = {"search_knowledge_base": SimpleNamespace(args_schema=Search)}
    namespace["generate_local"] = lambda *args, **kwargs: '{"tool":"search_knowledge_base","arguments":{"query":"x","top_k":-1}}'
    decision, trace = namespace["plan_action"]("Who are the buyers?")
    assert trace["fallback_used"] and not trace["arguments_valid"]
    assert decision.tool == "request_clarification"


def test_abstention_is_not_overridden_by_word_overlap():
    namespace = router_namespace()
    namespace["generate_local"] = lambda *args, **kwargs: "Insufficient local evidence."
    assert namespace["grounded_answer"]("product revenue", {"results": [
        {"source": "product.md", "snippet": "This product has no verified revenue information."}
    ]}) == "Insufficient local evidence."


@pytest.mark.parametrize("filename", [
    "01_llm_fine_tuning_pipeline.ipynb", "02_rag_document_qa_system.ipynb", "03_llm_agent_system.ipynb"
])
def test_notebook_python_compiles(filename):
    notebook = json.loads((ROOT / filename).read_text())
    for index, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        # Shell/magic commands are environment checks, not Python syntax.
        lines = "".join(cell["source"]).splitlines()
        python_lines = []
        continuation = False
        for line in lines:
            if continuation or line.lstrip().startswith(("!", "%")):
                continuation = line.rstrip().endswith("\\")
                python_lines.append("# IPython command")
            else:
                python_lines.append(line)
        compile("\n".join(python_lines), f"{filename}:cell{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
