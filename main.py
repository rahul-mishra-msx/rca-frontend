"""NiceGUI frontend for the AgentCore RCA agent (POST /invocations, SSE response)."""

from __future__ import annotations

import copy
import os
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from nicegui import app, context, ui

from agent_hooks import AgentHookContext, dispatch_agent_hook
from communication import (
    SseDataAssembler,
    compile_final_rca_report,
    fold_event,
    parse_sse_data_value,
    pretty_node,
)
from execution_graph import build_mermaid_source, load_graph_md

# Defaults aligned with local curl and sample logs under ./logs
DEFAULT_API_BASE = "http://localhost:8080"
DEFAULT_PROMPT = "can you analyse what is wrong with the recent deployment"
DEFAULT_ARCH: list[dict[str, str]] = [
    {
        "type": "lambda",
        "value": "/aws/lambda/rca-testing-function",
        "desc": "main lambda function",
    },
]
DEFAULT_REPO = {"ownerId": "rahul-mishra-msx", "name": "rca-erros"}

# Set on Sessions when starting a run; Agents page consumes and starts analysis after load.
STORAGE_AUTO_START_RUN = "auto_start_run"
STORAGE_UI_SETTINGS = "ui_settings"

DEFAULT_AGENT_HOST = "localhost"
DEFAULT_AGENT_PORT = 8080

MERMAID_CSS = """
        .rca-mermaid svg g.node.rcaActive path,
        .rca-mermaid svg g.node.rcaActive polygon,
        .rca-mermaid svg g.node.rcaActive rect {
            animation: rca-mermaid-processing 1.15s ease-in-out infinite !important;
        }
        .rca-mermaid svg g.node.rcaActive .nodeLabel,
        .rca-mermaid svg g.node.rcaActive .label {
            animation: rca-mermaid-label 1.15s ease-in-out infinite !important;
        }
        @keyframes rca-mermaid-processing {
            0%, 100% { stroke-opacity: 1; filter: drop-shadow(0 0 2px rgba(74, 222, 128, 0.9)); }
            50% { stroke-opacity: 0.55; filter: drop-shadow(0 0 10px rgba(34, 197, 94, 0.95)); }
        }
        @keyframes rca-mermaid-label {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.72; }
        }
        .rca-mermaid svg .nodeLabel text {
            font-size: 12px !important;
            font-weight: 700 !important;
        }
        .rca-mermaid svg foreignObject .nodeLabel,
        .rca-mermaid svg foreignObject .nodeLabel * {
            font-size: 12px !important;
            font-weight: 700 !important;
        }
        .rca-mermaid {
            max-height: 240px;
            display: flex;
            justify-content: center;
            align-items: center;
            width: auto;
            max-width: 100%;
        }
        .rca-mermaid svg {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .rca-agents-title-loading {
            letter-spacing: 0.35em;
            animation: rca-title-dots 1s ease-in-out infinite;
            opacity: 0.9;
        }
        @keyframes rca-title-dots {
            0%, 100% { opacity: 0.35; }
            50% { opacity: 1; }
        }
"""


def apply_colors_only() -> None:
    ui.colors(primary="#6366f1", secondary="#22d3ee")


def apply_dark_from_storage() -> None:
    s = load_ui_settings()
    dm = ui.dark_mode()
    if s["dark_mode"]:
        dm.enable()
    else:
        dm.disable()


def apply_global_theme() -> None:
    apply_colors_only()
    apply_dark_from_storage()


def _coerce_port(value: Any) -> int:
    if value is None or value == "":
        return DEFAULT_AGENT_PORT
    try:
        p = int(value)
        return p if 1 <= p <= 65535 else DEFAULT_AGENT_PORT
    except (TypeError, ValueError):
        return DEFAULT_AGENT_PORT


def load_ui_settings() -> dict[str, Any]:
    raw = app.storage.user.get(STORAGE_UI_SETTINGS)
    if not isinstance(raw, dict):
        raw = {}
    host = str(raw.get("agent_host", DEFAULT_AGENT_HOST) or DEFAULT_AGENT_HOST).strip()
    port = _coerce_port(raw.get("agent_port", DEFAULT_AGENT_PORT))
    dark = raw.get("dark_mode")
    if not isinstance(dark, bool):
        dark = True
    return {"agent_host": host, "agent_port": port, "dark_mode": dark}


def save_ui_settings(*, agent_host: str, agent_port: int, dark_mode: bool) -> None:
    app.storage.user[STORAGE_UI_SETTINGS] = {
        "agent_host": (agent_host or DEFAULT_AGENT_HOST).strip() or DEFAULT_AGENT_HOST,
        "agent_port": int(agent_port),
        "dark_mode": bool(dark_mode),
    }


def agent_base_from_host_port(host: str, port: int) -> str:
    h = (host or "").strip() or DEFAULT_AGENT_HOST
    if h.startswith("http://") or h.startswith("https://"):
        return h.rstrip("/")
    return f"http://{h}:{int(port)}"


def get_agent_api_base() -> str:
    s = load_ui_settings()
    return agent_base_from_host_port(s["agent_host"], s["agent_port"])


async def ping_agent(base_url: str) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/ping"
    try:
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            text = (resp.text or "").strip()
            if resp.status_code >= 400:
                return False, f"HTTP {resp.status_code}" + (f": {text[:200]}" if text else "")
            detail = text if text else f"HTTP {resp.status_code}"
            if len(detail) > 400:
                detail = detail[:400] + "…"
            return True, detail
    except httpx.ConnectError:
        return False, "connection refused — agent is not up or address is wrong"
    except httpx.TimeoutException:
        return False, "request timed out"
    except OSError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def default_rca_config() -> dict[str, Any]:
    return {
        "api_base": get_agent_api_base(),
        "prompt": DEFAULT_PROMPT,
        "session_id": str(uuid.uuid4())[:8],
        "owner": DEFAULT_REPO["ownerId"],
        "repo_name": DEFAULT_REPO["name"],
        "arch": [copy.deepcopy(x) for x in DEFAULT_ARCH],
    }


def load_rca_from_storage() -> dict[str, Any]:
    cfg = default_rca_config()
    stored = app.storage.user.get("rca")
    if isinstance(stored, dict):
        for k in cfg:
            if k == "api_base":
                continue
            if k in stored:
                if k == "arch" and isinstance(stored["arch"], list):
                    cfg["arch"] = [copy.deepcopy(x) for x in stored["arch"] if isinstance(x, dict)]
                else:
                    cfg[k] = stored[k]
    cfg["api_base"] = get_agent_api_base()
    return cfg


def save_rca_to_storage(
    *,
    prompt: str,
    session_id: str,
    owner: str,
    repo_name: str,
    arch: list[dict[str, str]],
) -> None:
    app.storage.user["rca"] = {
        "prompt": prompt,
        "session_id": session_id,
        "owner": owner,
        "repo_name": repo_name,
        "arch": [copy.deepcopy(x) for x in arch],
    }


def ensure_past_sessions() -> list[dict[str, str]]:
    ps = app.storage.user.get("past_sessions")
    if not isinstance(ps, list):
        ps = []
        app.storage.user["past_sessions"] = ps
    return ps  # type: ignore[return-value]


def append_past_session(session_id: str) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    ps = ensure_past_sessions()
    if not any(isinstance(x, dict) and x.get("sessionId") == sid for x in ps):
        ps.insert(0, {"sessionId": sid})


def is_logged_in() -> bool:
    return bool((app.storage.user.get("username") or "").strip())


def logout() -> None:
    preserved = app.storage.user.get(STORAGE_UI_SETTINGS)
    app.storage.user.clear()
    if preserved is not None:
        app.storage.user[STORAGE_UI_SETTINGS] = preserved
    ui.navigate.to("/")


AGENTS_HEADER_LOADING = "···"


def render_auth_header(page_title: str, *, agents_page: bool) -> None:
    uname = (app.storage.user.get("username") or "User").strip()
    with ui.header().classes(
        "items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700 text-gray-100"
    ):
        ui.label(page_title).classes("text-lg font-medium text-indigo-200 rca-header-title")
        with ui.row().classes("items-center gap-1 rca-header-user"):
            with ui.button().props("flat no-caps dense"):
                ui.icon("account_circle").classes("text-2xl")
                ui.label(uname).classes("text-sm ml-1")
                with ui.menu():
                    ui.menu_item("Logout", on_click=logout)
                    if agents_page:
                        ui.menu_item("New session", on_click=lambda: ui.navigate.to("/sessions"))


def render_agents_auth_header() -> Any:
    """Agents page header: title label starts as loading dots; updated via hooks (user_prompt structured output)."""
    uname = (app.storage.user.get("username") or "User").strip()
    with ui.header().classes(
        "items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700 text-gray-100"
    ):
        title_lbl = ui.label(AGENTS_HEADER_LOADING).classes(
            "text-lg font-medium text-indigo-200 rca-header-title rca-agents-title-loading"
        )
        with ui.row().classes("items-center gap-1 rca-header-user"):
            with ui.button().props("flat no-caps dense"):
                ui.icon("account_circle").classes("text-2xl")
                ui.label(uname).classes("text-sm ml-1")
                with ui.menu():
                    ui.menu_item("Logout", on_click=logout)
                    ui.menu_item("New session", on_click=lambda: ui.navigate.to("/sessions"))
    return title_lbl


def build_payload(
    prompt: str,
    session_id: str,
    arch: list[dict[str, str]],
    repo_owner: str = "",
    repo_name: str = "",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt": prompt,
        "sessionId": session_id,
        "arch": arch,
    }
    owner = (repo_owner or "").strip()
    name = (repo_name or "").strip()
    if owner or name:
        body["repo"] = {"ownerId": owner, "name": name}
    return body


def _arch_editor(arch_state: list[dict[str, str]], refreshable: Any) -> None:
    for i, row in enumerate(arch_state):

        def remove_at(idx: int) -> None:
            if len(arch_state) > 1:
                arch_state.pop(idx)
                refreshable.refresh()

        with ui.row().classes("w-full items-end gap-1 mb-2"):
            t = ui.input(value=row["type"]).props("dense").classes("w-20")
            v = ui.input(value=row["value"]).props("dense").classes("flex-1")
            d = ui.input(value=row["desc"]).props("dense").classes("flex-1")
            ui.button(icon="remove", on_click=lambda idx=i: remove_at(idx)).props("flat dense round size=sm")

            def sync() -> None:
                row["type"] = t.value or ""
                row["value"] = v.value or ""
                row["desc"] = d.value or ""

            t.on("update:model-value", sync)
            v.on("update:model-value", sync)
            d.on("update:model-value", sync)

    ui.button(
        "Add row",
        icon="add",
        on_click=lambda: (
            arch_state.append({"type": "lambda", "value": "", "desc": ""}),
            refreshable.refresh(),
        ),
    ).props("flat dense sm")


@ui.page("/")
def landing_page() -> None:
    apply_global_theme()

    def open_cognito() -> None:
        ui.notify("Cognito login is not wired yet.", type="info")

    def open_developer_login() -> None:
        app.storage.user["username"] = "Developer"
        ensure_past_sessions()
        ui.navigate.to("/sessions")

    def open_settings() -> None:
        s = load_ui_settings()
        with ui.dialog() as dlg, ui.card().classes("w-[min(100vw-2rem,24rem)] p-6"):
            ui.label("Settings").classes("text-lg font-semibold mb-3")
            host_in = ui.input("Agent host", value=s["agent_host"]).props("dense").classes("w-full")
            port_in = ui.input("Agent port", value=str(s["agent_port"])).props("dense").classes("w-full")
            ui.label(
                "Hostname (e.g. localhost) or full URL (http/https). Port is used only for bare hostnames."
            ).classes("text-xs text-gray-500 mb-2")
            dark_sw = ui.switch("Dark mode", value=s["dark_mode"])
            ui.label("Save calls GET …/ping; settings are stored only when the agent responds.").classes(
                "text-xs text-gray-500 mt-2"
            )

            async def save_settings() -> None:
                host = (host_in.value or "").strip()
                port = _coerce_port(port_in.value)
                base = agent_base_from_host_port(host, port)
                ok, msg = await ping_agent(base)
                if not ok:
                    ui.notify(f"Agent is not up. {msg}", type="negative")
                    return
                snippet = msg.replace("\n", " ").strip()
                if len(snippet) > 300:
                    snippet = snippet[:300] + "…"
                ui.notify(f"Agent OK: {snippet}", type="positive")
                save_ui_settings(
                    agent_host=host or DEFAULT_AGENT_HOST,
                    agent_port=port,
                    dark_mode=bool(dark_sw.value),
                )
                apply_dark_from_storage()
                dlg.close()

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Save", on_click=save_settings, icon="save")
        dlg.open()

    with ui.column().classes("w-full min-h-screen relative bg-gray-950"):
        with ui.row().classes("absolute top-4 left-4 z-20"):
            ui.button(icon="settings", on_click=open_settings).props("flat round").classes(
                "rca-main-settings text-gray-300"
            )
        with ui.column().classes(
            "w-full flex-1 justify-center items-center gap-8 p-8 pt-20 min-h-screen"
        ):
            ui.label("RCA").classes("text-3xl font-bold text-indigo-300")
            with ui.row().classes("gap-8 flex-wrap justify-center items-stretch"):
                with ui.element("div").classes(
                    "rca-login-cognito cursor-pointer rounded-xl border border-gray-600 bg-gray-900 p-10 "
                    "w-[min(100vw-2rem,20rem)] hover:border-indigo-500 transition"
                ).on("click", open_cognito):
                    with ui.column().classes("items-center gap-2"):
                        ui.icon("vpn_key", size="lg").classes("text-indigo-400")
                        ui.label("Login via Cognito").classes("text-lg font-medium text-center")
                with ui.element("div").classes(
                    "rca-login-developer cursor-pointer rounded-xl border border-gray-600 bg-gray-900 p-10 "
                    "w-[min(100vw-2rem,20rem)] hover:border-cyan-500 transition"
                ).on("click", open_developer_login):
                    with ui.column().classes("items-center gap-2"):
                        ui.icon("code", size="lg").classes("text-cyan-400")
                        ui.label("Developer").classes("text-lg font-medium text-center")


@ui.page("/sessions")
def sessions_page() -> None:
    apply_global_theme()
    if not is_logged_in():
        ui.navigate.to("/")
        return

    render_auth_header("Sessions", agents_page=False)

    cfg = load_rca_from_storage()
    sessions_arch: list[dict[str, str]] = cfg["arch"]

    @ui.refreshable
    def sessions_arch_section() -> None:
        _arch_editor(sessions_arch, sessions_arch_section)

    @ui.refreshable
    def previous_sessions_list() -> None:
        ps = ensure_past_sessions()
        if not ps:
            ui.label("No previous sessions yet.").classes("text-sm text-gray-500")
            return
        for item in ps:
            if not isinstance(item, dict):
                continue
            sid = item.get("sessionId", "—")
            with ui.card().classes("w-full p-3 mb-2 bg-gray-800 border border-gray-700"):
                ui.label(str(sid)).classes("text-sm font-mono font-medium text-indigo-200")

    with ui.column().classes("w-full flex-1 items-center p-6 bg-gray-950 min-h-[calc(100vh-3.5rem)]"):
        with ui.card().classes(
            "w-full max-w-3xl p-6 bg-gray-900 border border-gray-700 rca-sessions-center"
        ):
            with ui.tabs().classes("w-full rca-sessions-tabs") as stabs:
                tab_new = ui.tab("new_session", label="New session", icon="add_circle_outline").classes(
                    "rca-tab-new-session"
                )
                tab_prev = ui.tab("previous_sessions", label="Previous Sessions", icon="history").classes(
                    "rca-tab-previous-sessions"
                )

            with ui.tab_panels(stabs, value=tab_new).classes("w-full mt-4"):
                with ui.tab_panel(tab_new).classes("rca-tab-panel-new-session"):
                    ui.label(f"Agent API: `{get_agent_api_base()}` (configure on home **Settings**)").classes(
                        "text-xs text-gray-500 mb-2 w-full"
                    )
                    prompt_in = ui.textarea("Prompt", value=cfg["prompt"]).props("rows=4").classes("w-full")
                    session_in = ui.input("Session ID", value=cfg["session_id"]).classes("w-full")
                    ui.label("Repository (optional)").classes("text-xs text-gray-500 mt-1 mb-1")
                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        owner_in = (
                            ui.input("Owner ID", value=cfg["owner"])
                            .props(f"placeholder={DEFAULT_REPO['ownerId']} dense")
                            .classes("flex-1 min-w-[8rem]")
                        )
                        repo_in = (
                            ui.input("Repo name", value=cfg["repo_name"])
                            .props(f"placeholder={DEFAULT_REPO['name']} dense")
                            .classes("flex-1 min-w-[8rem]")
                        )
                    ui.label("Architecture (arch)").classes("text-xs text-gray-500 mt-2 mb-1")
                    sessions_arch_section()

                    def save_and_open_agents() -> None:
                        for row in sessions_arch:
                            if not row.get("value", "").strip():
                                ui.notify("Each arch row needs a value", type="warning")
                                return
                        save_rca_to_storage(
                            prompt=prompt_in.value or "",
                            session_id=session_in.value or str(uuid.uuid4())[:8],
                            owner=owner_in.value or "",
                            repo_name=repo_in.value or "",
                            arch=sessions_arch,
                        )
                        app.storage.user[STORAGE_AUTO_START_RUN] = True
                        ui.navigate.to("/agents")

                    ui.button("Start analysis", on_click=save_and_open_agents, icon="play_arrow").classes(
                        "w-full mt-4"
                    )

                with ui.tab_panel(tab_prev).classes("rca-tab-panel-previous-sessions"):
                    previous_sessions_list()


@ui.page("/agents")
async def agents_page() -> None:
    apply_global_theme()
    ui.add_css(MERMAID_CSS)
    if not is_logged_in():
        ui.navigate.to("/")
        return

    agents_header_title_lbl = render_agents_auth_header()

    def set_agents_header_title(text: str) -> None:
        agents_header_title_lbl.text = text
        agents_header_title_lbl.classes(remove="rca-agents-title-loading")

    hook_ctx = AgentHookContext(set_agents_header_title=set_agents_header_title)

    stored: dict[str, Any] = {}

    @ui.refreshable
    def summary_bar() -> None:
        if not stored:
            ui.label(
                "Configure your run on **Sessions** (New session), then click **Start analysis**."
            ).classes("text-gray-500 text-sm")
            return
        with ui.card().classes("w-full p-4 bg-gray-800 border border-gray-700"):
            with ui.row().classes("w-full items-center justify-between gap-2 mb-2"):
                ui.label("Current run").classes("text-xs uppercase tracking-wide text-gray-500")
                ui.button("Edit in Sessions", icon="edit", on_click=lambda: ui.navigate.to("/sessions")).props(
                    "flat dense sm"
                )
            with ui.row().classes("w-full items-start justify-between gap-4 flex-wrap"):
                with ui.column().classes("gap-1 flex-1 min-w-[16rem]"):
                    ui.label(f"API: `{stored.get('api', '')}`").classes("text-sm")
                    ui.label(f"Session: `{stored.get('session', '')}`").classes("text-sm")
                    ui.label(f"Repo: `{stored.get('repo', '(not set)')}`").classes("text-sm")
                with ui.column().classes("gap-1 flex-1 min-w-[16rem]"):
                    ui.label("Prompt").classes("text-xs text-gray-500")
                    ui.label(stored.get("prompt", "")).classes("text-sm whitespace-pre-wrap break-words")
                with ui.column().classes("gap-1 min-w-[12rem]"):
                    ui.label("Architecture").classes("text-xs text-gray-500")
                    for row in stored.get("arch", []) or []:
                        ui.label(
                            f"• [{row.get('type', '')}] {row.get('value', '')} — {row.get('desc', '')}"
                        ).classes("text-xs font-mono text-gray-300")

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-2"):
        run_slot: dict[str, Any] = {}
        summary_bar()
        with ui.row().classes("w-full gap-2 items-center my-2"):
            run_slot["btn"] = ui.button("Run analysis", icon="play_arrow").classes("rca-agents-run-btn")

        _exec_nodes = load_graph_md()
        _exec_known_ids = {n["id"] for n in _exec_nodes}
        _exec_mermaid_ref: dict[str, Any] = {}

        def update_execution_graph(active_node_id: str | None, show_spinner: bool) -> None:
            el = _exec_mermaid_ref.get("mermaid")
            if el is None:
                return
            aid = active_node_id if show_spinner and active_node_id else None
            src = build_mermaid_source(_exec_nodes, active_node_id=aid, highlight_active=bool(aid))
            el.set_content(src)

        with ui.card().classes("w-full p-4 bg-gray-800 border border-gray-700 mb-2"):
            ui.label("Execution graph").classes("text-xs uppercase tracking-wide text-gray-500 mb-2")
            if not _exec_nodes:
                ui.label("No nodes found in graph.md.").classes("text-sm text-gray-500")
            else:
                with ui.row().classes("w-full flex justify-center overflow-x-auto"):
                    mm = ui.mermaid(
                        build_mermaid_source(_exec_nodes, active_node_id=None, highlight_active=False)
                    ).classes("rca-mermaid max-w-full min-h-[8rem] bg-gray-900/80 rounded-lg p-1")
                    _exec_mermaid_ref["mermaid"] = mm

        refs: dict[str, Any] = {}
        status_label = ui.label("").classes("text-sm text-gray-500 mt-2")

        with ui.card().classes("w-full p-3 bg-gray-800/90 border border-indigo-900/40 mb-2"):
            with ui.row().classes("w-full items-center gap-4 flex-nowrap"):
                refs["activity_spinner"] = ui.spinner("dots", size="2.75em", color="primary").classes("shrink-0")
                refs["activity_spinner"].visible = False
                with ui.column().classes("flex-1 min-w-0 gap-0"):
                    ui.label("Live agent & tool").classes("text-[10px] uppercase text-gray-500")
                    refs["activity_agent"] = ui.label("—").classes("text-lg font-medium text-indigo-200")
                    refs["activity_tool"] = ui.label("").classes("text-sm text-gray-400 break-words")

        with ui.tabs().classes("w-full mt-2 rca-tabs") as tabs:
            tab_reasoning = ui.tab("reasoning", label="Reasoning", icon="psychology").classes(
                "rca-tab rca-tab-reasoning"
            )
            tab_activity = ui.tab("activity", label="Activity", icon="hub").classes("rca-tab rca-tab-activity")
            tab_raw_sse = ui.tab("raw_sse", label="Raw SSE", icon="terminal").classes(
                "rca-tab rca-tab-raw-sse"
            )

        with ui.tab_panels(tabs, value=tab_reasoning).classes("w-full rca-tab-panels"):
            with ui.tab_panel(tab_reasoning).classes("rca-tab-panel rca-tab-panel-reasoning"):
                refs["reasoning_md"] = ui.markdown(
                    "Submit the form to stream **reasoning / assistant text** here (tools appear under Activity only)."
                ).classes(
                    "rca-panel rca-panel-reasoning w-full p-4 rounded-lg bg-gray-900 border border-gray-700 min-h-[16rem]"
                )
            with ui.tab_panel(tab_activity).classes("rca-tab-panel rca-tab-panel-activity"):
                with ui.column().classes("rca-panel rca-panel-activity w-full gap-2"):
                    ui.label("Agent actions, tool calls, and token usage (live status is above).").classes(
                        "rca-panel-activity-intro text-xs text-gray-500"
                    )
                    refs["activity_log"] = ui.log(max_lines=500).classes(
                        "rca-panel-activity-log w-full h-[28rem] text-xs font-mono bg-gray-950 rounded-lg border border-gray-700 p-2"
                    )
            with ui.tab_panel(tab_raw_sse).classes("rca-tab-panel rca-tab-panel-raw-sse"):
                refs["raw_log"] = ui.log(max_lines=500).classes(
                    "rca-panel rca-panel-raw-sse w-full h-[32rem] text-xs font-mono opacity-90"
                )

        ui.separator().classes("w-full my-6")

        with ui.card().classes("w-full p-5 bg-gray-900 border border-emerald-900/40 rounded-lg"):
            ui.label("Final RCA report").classes("text-sm font-semibold text-emerald-300 mb-3")
            refs["final_report_md"] = ui.markdown(
                "_Complete after each successful run — assistant narrative and any structured output._"
            ).classes("w-full text-gray-200")

        async def run_agent() -> None:
            cfg_run = load_rca_from_storage()
            base = (cfg_run.get("api_base") or DEFAULT_API_BASE).rstrip("/")
            url = f"{base}/invocations"
            raw_arch = cfg_run.get("arch")
            arch_rows: list[dict[str, str]] = []
            if isinstance(raw_arch, list):
                arch_rows = [copy.deepcopy(x) for x in raw_arch if isinstance(x, dict)]
            if not arch_rows:
                ui.notify("Add at least one architecture row in Sessions.", type="warning")
                return
            for row in arch_rows:
                if not row.get("value", "").strip():
                    ui.notify("Each arch row needs a value — edit in Sessions.", type="warning")
                    return

            ro = (cfg_run.get("owner") or "").strip()
            rn = (cfg_run.get("repo_name") or "").strip()
            session_id = (cfg_run.get("session_id") or "").strip() or str(uuid.uuid4())[:8]
            prompt = cfg_run.get("prompt") or ""

            save_rca_to_storage(
                prompt=prompt,
                session_id=session_id,
                owner=ro,
                repo_name=rn,
                arch=arch_rows,
            )

            payload = build_payload(
                prompt,
                session_id,
                [{k: row[k] for k in ("type", "value", "desc")} for row in arch_rows],
                ro,
                rn,
            )

            stored.clear()
            stored.update(
                {
                    "api": base,
                    "session": session_id,
                    "repo": f"{ro}/{rn}".strip("/") if (ro or rn) else "(not set)",
                    "prompt": prompt,
                    "arch": [{k: r[k] for k in ("type", "value", "desc")} for r in arch_rows],
                }
            )
            summary_bar.refresh()

            btn = run_slot.get("btn")
            if btn:
                btn.disable()

            reasoning_md = refs["reasoning_md"]
            activity_log = refs["activity_log"]
            raw_log = refs["raw_log"]
            final_report_md = refs["final_report_md"]
            spin = refs["activity_spinner"]
            agent_lbl = refs["activity_agent"]
            tool_lbl = refs["activity_tool"]

            assistant_buffer = ""
            structured_snapshot: str | None = None
            reasoning_md.content = "_Waiting for agent text…_"
            final_report_md.content = "_Report will appear when the run completes._"
            activity_log.clear()
            raw_log.clear()
            status_label.text = "Streaming…"
            spin.visible = True
            agent_lbl.text = "Starting…"
            tool_lbl.text = ""
            update_execution_graph(None, False)
            agents_header_title_lbl.text = AGENTS_HEADER_LOADING
            agents_header_title_lbl.classes(add="rca-agents-title-loading")

            def apply_status(st: dict[str, Any]) -> None:
                node = st.get("node") or ""
                tool = st.get("tool")
                detail = (st.get("tool_detail") or "").strip()
                phase = str(st.get("phase", ""))
                if phase == "idle":
                    spin.visible = False
                    agent_lbl.text = "Idle"
                    tool_lbl.text = ""
                    return
                spin.visible = True
                if phase.startswith("stop:") and phase not in ("stop:tool_use",):
                    spin.visible = False
                if node:
                    agent_lbl.text = pretty_node(str(node))
                if tool:
                    line = str(tool)
                    if detail and phase == "tool_stream":
                        line = f"{tool} — {detail[-160:]}" if len(detail) > 160 else f"{tool} — {detail}"
                    tool_lbl.text = line
                elif phase == "generating":
                    tool_lbl.text = "Generating response…"
                elif phase in ("node_stop", "structured"):
                    tool_lbl.text = tool_lbl.text or ""
                elif phase == "tool_result":
                    tool_lbl.text = f"Tool finished ({st.get('result', 'ok')})"
                    spin.visible = False

                g_node = node if node in _exec_known_ids else None
                update_execution_graph(g_node, bool(g_node) and spin.visible)

            try:
                timeout = httpx.Timeout(600.0, connect=30.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, json=payload, headers={"Accept": "text/event-stream"}) as resp:
                        if resp.status_code >= 400:
                            body = (await resp.aread()).decode("utf-8", errors="replace")
                            reasoning_md.content = f"**HTTP {resp.status_code}**\n\n```\n{body[:8000]}\n```"
                            final_report_md.content = "_Run failed — see Reasoning tab._"
                            status_label.text = f"Error {resp.status_code}"
                            spin.visible = False
                            update_execution_graph(None, False)
                            ui.notify(f"HTTP {resp.status_code}", type="negative")
                            return

                        sse_asm = SseDataAssembler()

                        def dispatch_event(obj: dict[str, Any]) -> None:
                            nonlocal assistant_buffer, structured_snapshot
                            for channel, payload in fold_event(obj):
                                if channel == "text":
                                    assistant_buffer += str(payload)
                                    reasoning_md.content = assistant_buffer or "…"
                                elif channel == "assistant_message":
                                    assistant_buffer = str(payload)
                                    reasoning_md.content = assistant_buffer
                                elif channel == "structured":
                                    structured_snapshot = str(payload)
                                    activity_log.push("── structured output ──")
                                    activity_log.push(str(payload))
                                elif channel == "activity":
                                    activity_log.push(str(payload)[:2000])
                                elif channel == "raw":
                                    activity_log.push(f"(non-JSON) {str(payload)[:500]}")
                                elif channel == "status":
                                    if isinstance(payload, dict):
                                        apply_status(payload)
                                elif channel == "hook":
                                    dispatch_agent_hook(payload, ctx=hook_ctx)

                        async for raw_line in resp.aiter_lines():
                            raw_log.push(raw_line[:2000])
                            for completed_payload in sse_asm.feed_line(raw_line):
                                ev = parse_sse_data_value(completed_payload)
                                if ev is not None:
                                    dispatch_event(ev)
                        tail = sse_asm.flush()
                        if tail is not None:
                            ev = parse_sse_data_value(tail)
                            if ev is not None:
                                dispatch_event(ev)

                        status_label.text = "Done"
                        spin.visible = False
                        tool_lbl.text = ""
                        update_execution_graph(None, False)
                        final_report_md.content = compile_final_rca_report(assistant_buffer, structured_snapshot)
                        append_past_session(session_id)
                        ui.notify("Run finished", type="positive")
            except httpx.ConnectError as e:
                reasoning_md.content = f"**Cannot connect**\n\n`{e}`\n\nIs the agent running on `{base}`?"
                final_report_md.content = "_Report unavailable — connection error._"
                status_label.text = "Connection failed"
                spin.visible = False
                update_execution_graph(None, False)
                ui.notify("Connection refused — start the agent API", type="negative")
            except Exception as e:  # noqa: BLE001
                reasoning_md.content = f"**Error**\n\n```\n{e!r}\n```"
                final_report_md.content = "_Report unavailable — see Reasoning tab._"
                status_label.text = "Error"
                spin.visible = False
                update_execution_graph(None, False)
                ui.notify(str(e), type="negative")
            finally:
                if btn:
                    btn.enable()

        run_slot["btn"].on_click(run_agent)

        with ui.expansion("SSE reference", icon="info").classes("w-full mt-6"):
            ui.markdown(
                """
Events use **SSE**: strip the `data:` field, join multi-line `data:` chunks, parse JSON. The Activity tab
shows **agent** (`node_id`) and **tool** from `contentBlockStart` / streaming `toolUse` deltas.

API default: `http://localhost:8080` — AgentCore or local agent.
"""
            )

        await context.client.connected()
        if app.storage.user.pop(STORAGE_AUTO_START_RUN, False):
            await run_agent()


def main() -> None:
    load_dotenv()
    storage_secret = (os.environ.get("STORAGE_SECRET") or "").strip()
    if not storage_secret:
        raise SystemExit(
            "STORAGE_SECRET is not set. Copy .env.example to .env and set STORAGE_SECRET "
            "to a long random value (e.g. openssl rand -hex 32)."
        )
    ui.run(
        title="RCA Agent",
        port=8081,
        reload=True,
        favicon="🧩",
        storage_secret=storage_secret,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
