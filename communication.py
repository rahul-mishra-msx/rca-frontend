"""Translate AgentCore / multi-agent SSE payloads into frontend channel events.

The agent emits Server-Sent Events: each event body is JSON (after the ``data:`` field).
This module parses those payloads and maps them to UI channels:

- ``text`` — assistant token stream (narrative only in the Assistant panel)
- ``assistant_message`` — full assistant text blocks from a completed ``message`` event
- ``structured`` — JSON string of ``structured_output`` for Activity + final report
- ``hook`` — dict for UI hooks (e.g. ``tool_structured_output_complete`` when structured tool output is done)
- ``activity`` — human-readable log lines (tools, nodes, token usage)
- ``status`` — dict for live spinner / agent name / tool line
- ``raw`` — non-JSON payload fallback

Values that are not JSON-serializable (e.g. custom objects in a decoded payload) are coerced via
``default`` to ``"<TypeName>"``, or omitted when serialization still fails (e.g. circular refs).

``compile_final_rca_report`` turns the final structured RCA JSON (when it matches the required schema)
into Markdown for the report panel; otherwise it falls back to a JSON code block.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Captured ./logs sometimes contain Python repr() debug lines instead of JSON.
_RE_DELTA_TEXT_SINGLE = re.compile(r"'text':\s*'((?:[^'\\]|\\.)*)'")
_RE_DELTA_TEXT_DOUBLE = re.compile(r'"text":\s*"((?:[^"\\]|\\.)*)"')

# Final RCA report JSON shape (agent structured output).
RCA_REPORT_REQUIRED_KEYS = frozenset({
    "title",
    "problem_statement",
    "solution_summary",
    "analysis_summary",
    "hypothesis_summary",
    "references",
    "relevant_code_files",
})
_RCA_STRING_FIELDS = (
    "title",
    "problem_statement",
    "solution_summary",
    "analysis_summary",
    "hypothesis_summary",
)


def _json_default_non_serializable(o: Any) -> str:
    """Coerce values ``json.dumps`` cannot encode into a string (logged, not raised)."""
    return f"<{type(o).__qualname__}>"


def _safe_json_dumps(
    value: Any,
    *,
    indent: int | None = 2,
    max_len: int | None = 2000,
) -> str | None:
    """Serialize for the UI. Returns ``None`` if encoding fails (caller should ignore that fragment)."""
    try:
        text = json.dumps(value, indent=indent, default=_json_default_non_serializable, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    if max_len is not None and len(text) > max_len:
        return text[:max_len] + "\n…"
    return text


class SseDataAssembler:
    """Buffer ``data:`` field values until a blank line ends the SSE event (HTML SSE)."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def feed_line(self, line: str) -> list[str]:
        """Return completed event payload strings (after ``data:`` is stripped and joined)."""
        line = line.rstrip("\r")
        if line == "":
            if not self._parts:
                return []
            payload = "\n".join(self._parts)
            self._parts.clear()
            return [payload]
        if line.startswith(":"):
            return []
        colon = line.find(":")
        if colon < 0:
            return []
        field = line[:colon].strip()
        if field.lower() != "data":
            return []
        value = line[colon + 1 :].lstrip()
        self._parts.append(value)
        return []

    def flush(self) -> str | None:
        if not self._parts:
            return None
        payload = "\n".join(self._parts)
        self._parts.clear()
        return payload


def _unescape_sse_text_fragment(s: str) -> str:
    return s.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def _extract_text_from_stream_debug_repr(payload: str) -> str | None:
    """Recover token text from captured log lines that are Python repr, not JSON."""
    if "multiagent_node_stream" not in payload:
        return None
    for pat in (_RE_DELTA_TEXT_SINGLE, _RE_DELTA_TEXT_DOUBLE):
        m = pat.search(payload)
        if m:
            return _unescape_sse_text_fragment(m.group(1))
    return None


def parse_sse_data_value(payload: str) -> dict[str, Any] | None:
    """Parse one SSE event body: the string after ``data:`` (already stripped from the line)."""
    payload = payload.strip()
    if not payload:
        return None
    if payload == "[DONE]":
        return {"type": "done"}
    try:
        parsed: Any = json.loads(payload)
    except json.JSONDecodeError:
        text = _extract_text_from_stream_debug_repr(payload)
        if text is not None:
            return {
                "type": "multiagent_node_stream",
                "node_id": "user_prompt",
                "event": {"event": {"contentBlockDelta": {"delta": {"text": text}}}},
            }
        return {"type": "raw", "text": payload}

    if isinstance(parsed, dict):
        return parsed
    raw_text = _safe_json_dumps(parsed, max_len=None)
    return {"type": "raw", "text": raw_text if raw_text is not None else ""}


def _structured_output_to_dict(value: Any) -> dict[str, Any]:
    """Coerce ``structured_output`` payload to a plain ``dict`` for hooks and JSON."""
    if isinstance(value, dict):
        return value
    dumped = _safe_json_dumps(value, max_len=None)
    if not dumped:
        return {}
    try:
        out = json.loads(dumped)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def pretty_node(name: str) -> str:
    """Display label for a graph ``node_id`` (e.g. ``user_prompt`` → ``User Prompt``)."""
    if not name:
        return "—"
    return name.replace("_", " ").title()


def fold_event(obj: dict[str, Any]) -> list[tuple[str, Any]]:
    """Map one parsed agent event dict to ordered ``(channel, payload)`` tuples for the UI.

    Channels
    --------
    text, assistant_message
        String fragments for the Assistant panel (narrative only).
    structured
        JSON string for Activity log and final report assembly.
    activity
        Log lines (tools, lifecycle, usage).
    status
        ``dict`` with keys: ``node``, ``tool``, ``tool_detail``, ``phase``, optional ``role``,
        ``node_type``, ``tool_use_id``, ``result``.
    raw
        Unparsed line text for Activity.
    hook
        ``dict`` with ``kind`` (e.g. ``tool_structured_output_complete``), ``node_id``, ``structured``,
        optional ``tool_name`` — for UI hooks (see ``agent_hooks``).
    """
    rows: list[tuple[str, Any]] = []
    t = obj.get("type")
    node_id = str(obj.get("node_id") or "")

    if t == "done":
        rows.append(("activity", "— stream complete —"))
        rows.append(("status", {"node": "", "tool": None, "tool_detail": "", "phase": "idle"}))
        return rows
    if t == "raw":
        rows.append(("raw", obj.get("text", "")))
        return rows

    if t == "multiagent_node_start":
        nty = obj.get("node_type", "")
        rows.append(
            ("status", {"node": node_id, "tool": None, "tool_detail": "", "phase": "node_start", "node_type": nty})
        )
        rows.append(("activity", f"▶ Agent **{pretty_node(node_id)}** started ({nty})"))
        return rows

    if t == "multiagent_node_stop":
        rows.append(("status", {"node": node_id, "tool": None, "tool_detail": "", "phase": "node_stop"}))
        rows.append(("activity", f"■ Agent **{pretty_node(node_id)}** finished"))
        return rows

    if t != "multiagent_node_stream":
        dumped = _safe_json_dumps(obj, max_len=2000)
        if dumped is not None:
            rows.append(("activity", dumped))
        else:
            rows.append(("activity", f"(omitted: non-JSON-serializable event, type={t!r})"))
        return rows

    ev = obj.get("event") or {}
    inner = ev.get("event")
    if isinstance(inner, dict):
        if "messageStart" in inner:
            role = (inner.get("messageStart") or {}).get("role", "")
            rows.append(
                (
                    "status",
                    {"node": node_id, "tool": None, "tool_detail": "", "phase": "generating", "role": role},
                )
            )
        if "contentBlockStart" in inner:
            cbs = inner["contentBlockStart"] or {}
            start = cbs.get("start") or {}
            tu = start.get("toolUse") or {}
            tname = tu.get("name") or ""
            tid = tu.get("toolUseId") or ""
            rows.append(
                (
                    "status",
                    {
                        "node": node_id,
                        "tool": tname,
                        "tool_detail": "",
                        "phase": "tool_call",
                        "tool_use_id": tid,
                    },
                )
            )
            rows.append(("activity", f"**{pretty_node(node_id)}** → calling tool `{tname}`"))
        if "contentBlockDelta" in inner:
            d = inner["contentBlockDelta"].get("delta") or {}
            if "text" in d:
                rows.append(("text", str(d["text"])))
            elif "toolUse" in d:
                tu = d["toolUse"] or {}
                chunk = tu.get("input")
                if chunk:
                    rows.append(
                        (
                            "status",
                            {
                                "node": node_id,
                                "tool": tu.get("name"),
                                "tool_detail": str(chunk),
                                "phase": "tool_stream",
                            },
                        )
                    )
        if "messageStop" in inner:
            ms = inner["messageStop"] or {}
            sr = ms.get("stopReason", "")
            rows.append(("status", {"node": node_id, "tool": None, "tool_detail": "", "phase": f"stop:{sr}"}))
        if "metadata" in inner:
            meta = inner["metadata"]
            usage = meta.get("usage", {})
            lat = meta.get("metrics", {}).get("latencyMs")
            bit = f"tokens {usage}"
            if lat is not None:
                bit += f" · {lat}ms"
            rows.append(("activity", bit))

    if "structured_output" in ev:
        so = ev["structured_output"]
        dumped = _safe_json_dumps(so, max_len=None)
        if dumped is not None:
            rows.append(("structured", dumped))
        rows.append(("status", {"node": node_id, "tool": None, "tool_detail": "", "phase": "structured"}))
        so_dict = _structured_output_to_dict(so)
        tool_name: str | None = None
        res = ev.get("result")
        if isinstance(res, dict):
            msg = res.get("message")
            if isinstance(msg, dict):
                for block in msg.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    tu = block.get("toolUse")
                    if isinstance(tu, dict):
                        tool_name = str(tu.get("name") or "") or None
                        break
        rows.append(
            (
                "hook",
                {
                    "kind": "tool_structured_output_complete",
                    "node_id": node_id,
                    "structured": so_dict,
                    "tool_name": tool_name,
                },
            )
        )

    if "message" in ev:
        msg = ev["message"]
        content = msg.get("content") or []
        text_only: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                text_only.append(str(block["text"]))
            elif "toolUse" in block:
                tu = block["toolUse"]
                name = tu.get("name", "")
                inp = tu.get("input")
                rows.append(("activity", f"[tool] {name}: {inp}"))
            elif "toolResult" in block:
                tr = block["toolResult"]
                st = tr.get("status", "")
                rows.append(
                    ("status", {"node": node_id, "tool": None, "tool_detail": "", "phase": "tool_result", "result": st})
                )
                rows.append(("activity", f"[tool result] {st}: {tr.get('content')}"))
        if text_only:
            rows.append(("assistant_message", "\n".join(text_only)))

    return rows


def is_rca_report_schema(obj: Any) -> bool:
    """Return True if ``obj`` matches the expected RCA final structured report (required keys + types)."""
    if not isinstance(obj, dict):
        return False
    if not RCA_REPORT_REQUIRED_KEYS.issubset(obj.keys()):
        return False
    for k in _RCA_STRING_FIELDS:
        if not isinstance(obj.get(k), str):
            return False
    refs = obj.get("references")
    paths = obj.get("relevant_code_files")
    if not isinstance(refs, list) or not isinstance(paths, list):
        return False
    return all(isinstance(x, str) for x in refs) and all(isinstance(x, str) for x in paths)


def structured_rca_to_markdown(data: dict[str, Any]) -> str:
    """Render a validated RCA report dict as Markdown (field order follows the schema)."""
    title = str(data["title"]).strip() or "RCA report"
    lines: list[str] = [f"# {title}", ""]

    def section(heading: str, body: str) -> None:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.strip() or "_—_")
        lines.append("")

    # Section order matches the JSON schema property order.
    section("Problem statement", str(data["problem_statement"]))
    section("Solution summary", str(data["solution_summary"]))
    section("Analysis summary", str(data["analysis_summary"]))
    section("Hypothesis summary", str(data["hypothesis_summary"]))

    lines.append("## References")
    lines.append("")
    refs = data["references"]
    if not refs:
        lines.append("_None listed._")
    else:
        for r in refs:
            r = str(r).strip()
            if r.startswith(("http://", "https://")):
                lines.append(f"- [{r}]({r})")
            else:
                lines.append(f"- {r}")
    lines.append("")

    lines.append("## Relevant code files")
    lines.append("")
    files = data["relevant_code_files"]
    if not files:
        lines.append("_None listed._")
    else:
        for path in files:
            lines.append(f"- `{str(path).strip()}`")

    return "\n".join(lines)


def compile_final_rca_report(narrative: str, structured_json: str | None) -> str:
    """Build the full final report: narrative plus structured output as Markdown or raw JSON fallback."""
    narrative_block = (narrative or "").strip() or "_No narrative text was returned._"
    intro = "## Assistant narrative\n\n" + narrative_block

    if not structured_json or not structured_json.strip():
        return intro

    try:
        parsed: Any = json.loads(structured_json)
    except json.JSONDecodeError:
        return (
            intro
            + "\n\n---\n\n#### Structured output (unparsed)\n\n```\n"
            + structured_json.strip()
            + "\n```"
        )

    if isinstance(parsed, dict) and is_rca_report_schema(parsed):
        return intro + "\n\n---\n\n" + structured_rca_to_markdown(parsed)

    dumped = _safe_json_dumps(parsed, max_len=None)
    if dumped is not None:
        return intro + "\n\n---\n\n#### Structured output\n\n```json\n" + dumped + "\n```"
    return intro + "\n\n---\n\n_Structured payload could not be serialized._"


__all__ = [
    "SseDataAssembler",
    "compile_final_rca_report",
    "fold_event",
    "is_rca_report_schema",
    "parse_sse_data_value",
    "pretty_node",
    "structured_rca_to_markdown",
]
