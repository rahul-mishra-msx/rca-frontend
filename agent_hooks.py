"""Hooks invoked from agent SSE events (see ``communication.fold_event`` ``hook`` channel)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from agents_schema import UserPromptStructuredOutput

# Emitted when a node finishes streaming ``structured_output`` (tool structured output complete).
HOOK_TOOL_STRUCTURED_OUTPUT_COMPLETE = "tool_structured_output_complete"

# Tool name from ``agents_schema`` / agent runtime (matches ``contentBlockStart.toolUse.name``).
USER_PROMPT_STRUCTURED_TOOL_NAME = "UserPromptStructuredOutput"


@dataclass
class AgentHookContext:
    """Runtime callbacks injected by the UI (e.g. NiceGUI page)."""

    set_agents_header_title: Callable[[str], None] | None = None


def _title_from_user_prompt_structured(structured: dict[str, Any]) -> str | None:
    """Parse ``UserPromptStructuredOutput`` and return ``title`` (see ``agents_schema.py``)."""
    try:
        model = UserPromptStructuredOutput.model_validate(structured)
        return (model.title or "").strip() or None
    except ValidationError:
        raw_title = str(structured.get("title") or "").strip()
        if raw_title:
            return raw_title
        summary = str(structured.get("user_goal_summary") or "").strip()
        if not summary:
            return None
        return summary[:60] + ("…" if len(summary) > 60 else "")


def _is_user_prompt_structured_payload(structured: dict[str, Any]) -> bool:
    """Heuristic: matches ``UserPromptStructuredOutput`` (``intent`` + ``user_goal_summary``)."""
    return "intent" in structured and "user_goal_summary" in structured


def on_tool_structured_output_complete(
    ctx: AgentHookContext,
    *,
    node_id: str,
    structured: dict[str, Any],
    tool_name: str | None = None,
) -> None:
    """Default handler: when ``user_prompt`` emits structured tool output, set the agents page title."""
    if node_id != "user_prompt":
        return
    if tool_name is not None and tool_name != USER_PROMPT_STRUCTURED_TOOL_NAME:
        return
    if tool_name is None and not _is_user_prompt_structured_payload(structured):
        return
    title = _title_from_user_prompt_structured(structured)
    if not title:
        return
    if ctx.set_agents_header_title:
        ctx.set_agents_header_title(title)


def dispatch_agent_hook(payload: Any, *, ctx: AgentHookContext) -> None:
    """Run handlers for a ``hook`` channel payload from ``fold_event``."""
    if not isinstance(payload, dict):
        return
    kind = payload.get("kind")
    if kind != HOOK_TOOL_STRUCTURED_OUTPUT_COMPLETE:
        return
    node_id = str(payload.get("node_id") or "")
    structured = payload.get("structured")
    if not isinstance(structured, dict):
        structured = {}
    tool_name = payload.get("tool_name")
    tool_name = str(tool_name) if tool_name else None
    on_tool_structured_output_complete(
        ctx, node_id=node_id, structured=structured, tool_name=tool_name
    )


__all__ = [
    "HOOK_TOOL_STRUCTURED_OUTPUT_COMPLETE",
    "USER_PROMPT_STRUCTURED_TOOL_NAME",
    "AgentHookContext",
    "dispatch_agent_hook",
    "on_tool_structured_output_complete",
]
