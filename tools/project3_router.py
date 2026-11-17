"""Typed router shared by the notebook and CPU contract tests."""
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["search_knowledge_base", "calculate_campaign_budget",
                  "save_workspace_report", "request_clarification", "abstain"]
    arguments: dict[str, Any]


class NonActionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=300)


def decision_schema(tools):
    branches = []
    schemas = {name: tool.args_schema.model_json_schema() for name, tool in tools.items()}
    schemas.update({name: NonActionInput.model_json_schema()
                    for name in ["request_clarification", "abstain"]})
    for name, arguments in schemas.items():
        arguments["additionalProperties"] = False
        if name == "save_workspace_report":
            arguments["properties"]["confirm_write"] = {"type": "boolean", "enum": [False]}
        branches.append({"type": "object", "additionalProperties": False,
                         "required": ["tool", "arguments"],
                         "properties": {"tool": {"type": "string", "enum": [name]},
                                        "arguments": arguments}})
    return {"anyOf": branches}


def validate_decision(raw, tools):
    # Require the entire response to be JSON, not a JSON-looking substring.
    decision = AgentDecision.model_validate(json.loads(raw), strict=True)
    schema = (NonActionInput if decision.tool in ["request_clarification", "abstain"]
              else tools[decision.tool].args_schema)
    if set(decision.arguments) - set(schema.model_fields):
        raise ValueError("Unknown tool argument")
    decision.arguments = schema.model_validate(decision.arguments, strict=True).model_dump()
    if decision.tool == "save_workspace_report" and decision.arguments.get("confirm_write"):
        raise ValueError("The model cannot grant write approval")
    return decision


ROUTER_INSTRUCTION = """Select one action and return only its JSON object.
Search local knowledge for business facts, including facts that might be absent.
Calculate only if channel, duration in weeks, and weekly spend are supplied.
Use request_clarification with a short reason when required details are missing;
never invent a spend, duration, filename, or report content.
Use abstain for tasks outside these tools, such as sending messages or booking travel.
For report saving set confirm_write=false. A separate controller owns permission.
"""


def plan_with_schema(task, tools, generate, constrained=True, max_attempts=2):
    if max_attempts not in (1, 2):
        raise ValueError("Router permits at most two attempts")
    schema = decision_schema(tools)
    prompt = ROUTER_INSTRUCTION + "\nSchema: " + json.dumps(schema) + "\nTask: " + task
    failures = []
    for attempt in range(max_attempts):
        suffix = "" if not failures else "\nInvalid output: return a complete schema-valid JSON object."
        raw = generate(prompt + suffix, max_new_tokens=192,
                       json_schema=schema if constrained else None)
        parsed = False
        try:
            json.loads(raw)
            parsed = True
            decision = validate_decision(raw, tools)
            return decision, {"valid_json": True, "arguments_valid": True,
                              "fallback_used": False, "constrained": constrained,
                              "attempts": attempt + 1, "raw": raw,
                              "attempt_failures": failures}
        except (ValueError, ValidationError, KeyError) as exc:
            failures.append({"attempt": attempt + 1, "json_parsed": parsed,
                             "error_type": type(exc).__name__})
    # A safe non-action replaces the historical guessed-$500 arithmetic fallback.
    decision = AgentDecision(tool="request_clarification", arguments={
        "reason": "I could not validate an action. Please clarify the request and required details."})
    return decision, {"valid_json": parsed, "arguments_valid": False,
                      "fallback_used": True, "constrained": constrained,
                      "attempts": max_attempts, "raw": raw, "attempt_failures": failures}
