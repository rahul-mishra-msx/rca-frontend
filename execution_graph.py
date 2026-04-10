"""Parse ``graph.md`` (agent DAG) for the execution graph UI."""

from __future__ import annotations

import re
from pathlib import Path

# Lines like: (Agent, Parent, node_id) or (Agent, Parent, node_id, ONE|BOTH)
_ROW_RE = re.compile(
    r"^\(\s*(.*?)\s*,\s*(.*?)\s*,\s*([\w_]+)\s*(?:,\s*([A-Za-z_]+)\s*)?\)\s*$"
)


def load_graph_md(path: Path | str | None = None) -> list[dict[str, str]]:
    """Load node rows from ``graph.md``: agent name, parent id, node id, edge direction.

    ``edge`` is ``ONE`` (parent → child) or ``BOTH`` (bidirectional). Missing column defaults to ``ONE``.
    """
    p = Path(path) if path is not None else Path(__file__).resolve().parent / "graph.md"
    text = p.read_text(encoding="utf-8")
    nodes: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        agent, parent, node_id = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        raw_dir = (m.group(4) or "ONE").strip().upper()
        edge = raw_dir if raw_dir in ("ONE", "BOTH") else "ONE"
        nodes.append({"agent": agent, "parent": parent, "id": node_id, "edge": edge})
    return nodes


def horizontal_layout_order(nodes: list[dict[str, str]]) -> list[str]:
    """Left-to-right order for a single-row timeline (children of ``analyser`` stay in file order)."""
    ids = [n["id"] for n in nodes]
    # Ensure main spine + branch order as in graph.md
    return ids


def node_labels_by_id(nodes: list[dict[str, str]]) -> dict[str, str]:
    return {n["id"]: n["agent"] for n in nodes}


def _mermaid_escape_label(text: str) -> str:
    """Make text safe inside Mermaid node labels."""
    return text.replace("\\", "\\\\").replace('"', "'").replace("\n", " ").replace("\r", "")


def _ellipsize_label(text: str, max_chars: int = 20) -> str:
    """Shorten labels for compact horizontal layout (Unicode ellipsis)."""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    if max_chars < 2:
        return "…"
    return t[: max_chars - 1] + "…"


def _stadium_node(nid: str, display_label: str) -> str:
    """Mermaid stadium / pill node ``id([\"label\"])`` (ellipse-like); quotes keep spaces safe."""
    inner = display_label.replace("\\", "\\\\").replace('"', "'")
    return f'{nid}([\"{inner}\"])'


def mermaid_edge_arrow(edge: str) -> str:
    """Map ``graph.md`` edge kind to Mermaid link syntax (``ONE`` → ``-->``, ``BOTH`` → ``<-->``)."""
    if (edge or "ONE").upper() == "BOTH":
        return "<-->"
    return "-->"


def build_mermaid_source(
    nodes: list[dict[str, str]],
    *,
    active_node_id: str | None = None,
    highlight_active: bool = True,
    label_max_chars: int = 20,
) -> str:
    """Build a Mermaid ``flowchart LR`` mirroring ``graph.md``.

    Each row's ``edge`` field selects the link: ``ONE`` uses ``-->``, ``BOTH`` uses ``<-->``.
    Nodes use stadium shapes with ellipsized labels. ``graph_start`` stands in for ``START``.
    Active node gets class ``rcaActive``; others ``rcaIdle``.
    """
    _init = (
        '%%{init: {"themeVariables": {"fontSize": "12px", "fontFamily": "system-ui, sans-serif"}, '
        '"flowchart": {"nodeSpacing": 22, "rankSpacing": 28, "padding": 4, "curve": "basis", '
        '"useMaxWidth": false}}}%%'
    )

    if not nodes:
        return f'{_init}\nflowchart LR\n  empty([\"No graph data\"])'

    lines: list[str] = [_init, "flowchart LR"]
    known_ids = {n["id"] for n in nodes}
    start_declared = False

    for n in nodes:
        nid = n["id"]
        pid = n["parent"]
        arrow = mermaid_edge_arrow(n.get("edge", "ONE"))
        label = _ellipsize_label(_mermaid_escape_label(n["agent"]), label_max_chars)
        node_lhs = _stadium_node(nid, label)
        if pid == "START":
            if not start_declared:
                lines.append(f"  {_stadium_node('graph_start', 'START')} {arrow} {node_lhs}")
                start_declared = True
            else:
                lines.append(f"  graph_start {arrow} {node_lhs}")
        else:
            lines.append(f"  {pid} {arrow} {node_lhs}")

    all_ids = ["graph_start", *[x["id"] for x in nodes]]
    lines.append(
        "  classDef rcaIdle fill:#1e293b,stroke:#475569,color:#cbd5e1,stroke-width:1px"
    )
    lines.append(
        "  classDef rcaActive fill:#14532d,stroke:#4ade80,color:#ecfdf7,"
        "stroke-width:2.5px,stroke-dasharray:6 4"
    )

    if highlight_active and active_node_id and active_node_id in known_ids:
        idle = [i for i in all_ids if i != active_node_id]
        lines.append(f"  class {','.join(idle)} rcaIdle")
        lines.append(f"  class {active_node_id} rcaActive")
    else:
        lines.append(f"  class {','.join(all_ids)} rcaIdle")

    return "\n".join(lines)


__all__ = [
    "build_mermaid_source",
    "horizontal_layout_order",
    "load_graph_md",
    "mermaid_edge_arrow",
    "node_labels_by_id",
]
