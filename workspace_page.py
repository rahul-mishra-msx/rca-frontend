"""Unified enterprise chat workspace (single-page RCA UI)."""

from __future__ import annotations

import copy
import uuid
from typing import Any

import httpx
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


def register_workspace_page(
    *,
    default_api_base: str,
    default_prompt: str,
    default_repo: dict[str, str],
    agents_header_loading: str,
    storage_auto_start_run: str,
    append_past_session: Any,
    apply_global_theme: Any,
    build_payload: Any,
    ensure_past_sessions: Any,
    get_agent_api_base: Any,
    is_logged_in: Any,
    load_rca_from_storage: Any,
    logout: Any,
    open_settings_dialog: Any,
    save_rca_to_storage: Any,
    _arch_editor: Any,
    mermaid_css: str,
    workspace_shell_css: str,
) -> None:
    AGENTS_HEADER_LOADING = agents_header_loading

    @ui.page("/workspace")
    async def workspace_page() -> None:  # noqa: C901, PLR0915
        apply_global_theme()
        ui.add_css(mermaid_css)
        ui.add_css(workspace_shell_css)
        if not is_logged_in():
            ui.navigate.to("/")
            return

        cfg = load_rca_from_storage()
        sessions_arch: list[dict[str, str]] = cfg["arch"]
        stored: dict[str, Any] = {}

        def set_title(text: str) -> None:
            title_lbl.text = text
            title_lbl.classes(remove="rca-agents-title-loading")

        hook_ctx = AgentHookContext(set_agents_header_title=set_title)

        @ui.refreshable
        def sidebar_arch_section() -> None:
            _arch_editor(sessions_arch, sidebar_arch_section)

        @ui.refreshable
        def sidebar_past_list() -> None:
            ps = ensure_past_sessions()
            if not ps:
                ui.label("No history yet.").classes("text-xs text-gray-500")
                return
            for item in ps[:16]:
                if isinstance(item, dict):
                    ui.label(str(item.get("sessionId", "—"))).classes(
                        "text-xs font-mono text-gray-400 truncate w-full"
                    )

        @ui.refreshable
        def sidebar_run_card() -> None:
            if not stored:
                ui.label("Send a message to see run metadata.").classes("text-xs text-gray-500")
                return
            ui.label(f"Session `{stored.get('session', '')}`").classes("text-xs font-mono text-gray-300")
            ui.label(stored.get("prompt", "")[:200] + ("…" if len(stored.get("prompt", "")) > 200 else "")).classes(
                "text-xs text-gray-400 whitespace-pre-wrap"
            )

        reset_chat_ref: dict[str, Any] = {"fn": lambda: None}
        conversation_started: list[bool] = [False]

        with ui.column().classes("w-full min-h-screen bg-gray-950 flex flex-col"):
            with ui.row().classes(
                "w-full items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700 text-gray-100 shrink-0"
            ):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("hub", size="sm").classes("text-indigo-400")
                    title_lbl = ui.label(AGENTS_HEADER_LOADING).classes(
                        "text-lg font-medium text-indigo-200 rca-header-title rca-agents-title-loading"
                    )
                with ui.row().classes("items-center gap-1"):
                    ui.button(icon="refresh", on_click=lambda: reset_chat_ref["fn"]()).props(
                        "flat round dense"
                    ).tooltip("Clear conversation")
                    ui.button(icon="settings", on_click=open_settings_dialog).props("flat round dense").tooltip(
                        "Agent connection"
                    )
                    uname = (app.storage.user.get("username") or "User").strip()
                    with ui.button().props("flat no-caps dense"):
                        ui.icon("account_circle").classes("text-2xl")
                        ui.label(uname).classes("text-sm ml-1")
                        with ui.menu():
                            ui.menu_item("Logout", on_click=logout)

            with ui.row().classes(
                "w-full flex-1 flex-nowrap items-stretch gap-0 min-h-0 max-w-[100vw] rca-workspace-root"
            ):
                # Left rail
                with ui.column().classes(
                    "w-80 min-w-[17rem] shrink-0 border-r border-gray-800 bg-gray-950 p-4 gap-3 overflow-y-auto "
                    "max-h-[calc(100vh-3.5rem)]"
                ):
                    ui.label("Run context").classes(
                        "text-xs font-semibold uppercase tracking-wide text-gray-500"
                    )
                    ui.label(f"API `{get_agent_api_base()}`").classes("text-xs text-gray-500 break-all")
                    owner_in = (
                        ui.input("Owner ID", value=cfg["owner"])
                        .props(f"placeholder={default_repo['ownerId']} dense outlined")
                        .classes("w-full")
                    )
                    repo_in = (
                        ui.input("Repo name", value=cfg["repo_name"])
                        .props(f"placeholder={default_repo['name']} dense outlined")
                        .classes("w-full")
                    )
                    ui.label("Architecture").classes("text-xs text-gray-500 mt-1")
                    sidebar_arch_section()
                    ui.separator().classes("my-2")
                    ui.label("Last run").classes("text-xs font-semibold text-gray-500")
                    sidebar_run_card()
                    ui.separator().classes("my-2")
                    ui.label("Recent sessions").classes("text-xs font-semibold text-gray-500")
                    sidebar_past_list()

                # Center: chat
                with ui.column().classes("flex-1 min-w-0 flex flex-col min-h-0 bg-gray-950 border-r border-gray-800"):
                    chat_scroll = ui.scroll_area().classes("flex-1 w-full min-h-0 rca-chat-scroll")
                    with chat_scroll:
                        chat_col = ui.column().classes("w-full max-w-3xl mx-auto gap-4 py-4 px-3")
                        with chat_col:
                            ui.markdown(
                                "**RCA Assistant** — Set **Owner**, **Repo**, and **Architecture** on the left, "
                                "then describe your incident below and click **Send**."
                            ).classes("text-gray-500 text-sm text-center px-4")

                    with ui.row().classes(
                        "w-full max-w-3xl mx-auto p-3 gap-2 border-t border-gray-800 items-end shrink-0 "
                        "bg-gray-900/90 backdrop-blur"
                    ):
                        composer = (
                            ui.textarea(
                                placeholder="Describe symptoms, errors, or what to analyze…",
                                value=cfg.get("prompt") or default_prompt,
                            )
                            .props("outlined autogrow")
                            .classes("flex-1")
                        )
                        send_btn = ui.button(icon="send").props("round color=primary").tooltip("Send")

                # Right: graph + details
                with ui.column().classes(
                    "w-[22rem] min-w-[19rem] shrink-0 overflow-y-auto max-h-[calc(100vh-3.5rem)] "
                    "p-3 gap-2 bg-gray-950"
                ):
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

                    with ui.card().classes("w-full p-3 bg-gray-800 border border-gray-700"):
                        ui.label("Execution graph").classes("text-[10px] uppercase text-gray-500 mb-2")
                        if not _exec_nodes:
                            ui.label("No graph.md nodes.").classes("text-xs text-gray-500")
                        else:
                            mm = ui.mermaid(
                                build_mermaid_source(_exec_nodes, active_node_id=None, highlight_active=False)
                            ).classes("rca-mermaid max-w-full min-h-[6rem] bg-gray-900/80 rounded-lg p-1")
                            _exec_mermaid_ref["mermaid"] = mm

                    refs: dict[str, Any] = {}
                    status_label = ui.label("").classes("text-xs text-gray-500")

                    with ui.card().classes("w-full p-2 bg-gray-800/90 border border-indigo-900/30"):
                        with ui.row().classes("w-full items-center gap-2"):
                            refs["activity_spinner"] = ui.spinner("dots", size="lg", color="primary").classes(
                                "shrink-0"
                            )
                            refs["activity_spinner"].visible = False
                            with ui.column().classes("flex-1 min-w-0 gap-0"):
                                ui.label("Agent / tool").classes("text-[9px] uppercase text-gray-500")
                                refs["activity_agent"] = ui.label("—").classes(
                                    "text-sm font-medium text-indigo-200"
                                )
                                refs["activity_tool"] = ui.label("").classes("text-xs text-gray-400 break-words")

                    with ui.tabs().classes("w-full") as tabs:
                        tab_activity = ui.tab("activity", label="Activity", icon="hub")
                        tab_raw_sse = ui.tab("raw_sse", label="Raw", icon="terminal")
                        tab_report = ui.tab("report", label="Report", icon="description")

                    with ui.tab_panels(tabs, value=tab_activity).classes("w-full"):
                        with ui.tab_panel(tab_activity):
                            refs["activity_log"] = ui.log(max_lines=400).classes(
                                "w-full h-52 text-xs font-mono bg-gray-900 rounded border border-gray-700 p-2"
                            )
                        with ui.tab_panel(tab_raw_sse):
                            refs["raw_log"] = ui.log(max_lines=400).classes(
                                "w-full h-52 text-xs font-mono opacity-90"
                            )
                        with ui.tab_panel(tab_report):
                            refs["final_report_md"] = ui.markdown(
                                "_Report appears when the run completes._"
                            ).classes("w-full text-sm text-gray-200")

                    with ui.expansion("SSE reference", icon="info").classes("w-full text-xs"):
                        ui.markdown(
                            f"Stream uses **SSE**; Activity shows agent node and tool events. "
                            f"Default API: `{default_api_base}`."
                        ).classes("text-xs")

        def reset_chat() -> None:
            conversation_started[0] = False
            chat_col.clear()
            with chat_col:
                ui.markdown(
                    "**RCA Assistant** — Set **Owner**, **Repo**, and **Architecture** on the left, "
                    "then describe your incident below and click **Send**."
                ).classes("text-gray-500 text-sm text-center px-4")
            composer.value = cfg.get("prompt") or default_prompt

        reset_chat_ref["fn"] = reset_chat

        async def run_agent() -> None:
            prompt = (composer.value or "").strip()
            if not prompt:
                ui.notify("Enter a message.", type="warning")
                return
            for row in sessions_arch:
                if not row.get("value", "").strip():
                    ui.notify("Each architecture row needs a value.", type="warning")
                    return

            ro = (owner_in.value or "").strip()
            rn = (repo_in.value or "").strip()
            session_id = str(uuid.uuid4())[:8]
            arch_rows = [copy.deepcopy(x) for x in sessions_arch]

            save_rca_to_storage(
                prompt=prompt,
                session_id=session_id,
                owner=ro,
                repo_name=rn,
                arch=arch_rows,
            )

            base = (get_agent_api_base() or default_api_base).rstrip("/")
            url = f"{base}/invocations"
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
            sidebar_run_card.refresh()

            if not conversation_started[0]:
                chat_col.clear()
                conversation_started[0] = True
            stream_md: Any
            with chat_col:
                ui.chat_message(text=prompt, sent=True, name="You")
                stream_md = ui.markdown("_Waiting for agent…_").classes(
                    "w-full max-w-2xl text-gray-100 bg-gray-800/60 rounded-xl px-4 py-3 border border-gray-700"
                )

            activity_log = refs["activity_log"]
            raw_log = refs["raw_log"]
            final_report_md = refs["final_report_md"]
            spin = refs["activity_spinner"]
            agent_lbl = refs["activity_agent"]
            tool_lbl = refs["activity_tool"]

            assistant_buffer = ""
            structured_snapshot: str | None = None
            final_report_md.content = "_Streaming…_"
            activity_log.clear()
            raw_log.clear()
            status_label.text = "Streaming…"
            spin.visible = True
            agent_lbl.text = "Starting…"
            tool_lbl.text = ""
            update_execution_graph(None, False)
            title_lbl.text = AGENTS_HEADER_LOADING
            title_lbl.classes(add="rca-agents-title-loading")
            send_btn.disable()

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
                        line = f"{tool} — {detail[-120:]}" if len(detail) > 120 else f"{tool} — {detail}"
                    tool_lbl.text = line
                elif phase == "generating":
                    tool_lbl.text = "Generating…"
                elif phase in ("node_stop", "structured"):
                    tool_lbl.text = tool_lbl.text or ""
                elif phase == "tool_result":
                    tool_lbl.text = f"Tool done ({st.get('result', 'ok')})"
                    spin.visible = False

                g_node = node if node in _exec_known_ids else None
                update_execution_graph(g_node, bool(g_node) and spin.visible)

            try:
                timeout = httpx.Timeout(600.0, connect=30.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", url, json=payload, headers={"Accept": "text/event-stream"}
                    ) as resp:
                        if resp.status_code >= 400:
                            body = (await resp.aread()).decode("utf-8", errors="replace")
                            stream_md.content = f"**HTTP {resp.status_code}**\n\n```\n{body[:8000]}\n```"
                            final_report_md.content = "_Run failed._"
                            status_label.text = f"Error {resp.status_code}"
                            spin.visible = False
                            update_execution_graph(None, False)
                            ui.notify(f"HTTP {resp.status_code}", type="negative")
                            return

                        sse_asm = SseDataAssembler()

                        def dispatch_event(obj: dict[str, Any]) -> None:
                            nonlocal assistant_buffer, structured_snapshot
                            for channel, pl in fold_event(obj):
                                if channel == "text":
                                    assistant_buffer += str(pl)
                                    stream_md.content = assistant_buffer or "…"
                                elif channel == "assistant_message":
                                    assistant_buffer = str(pl)
                                    stream_md.content = assistant_buffer
                                elif channel == "structured":
                                    structured_snapshot = str(pl)
                                    activity_log.push("── structured output ──")
                                    activity_log.push(str(pl))
                                elif channel == "activity":
                                    activity_log.push(str(pl)[:2000])
                                elif channel == "raw":
                                    activity_log.push(f"(non-JSON) {str(pl)[:500]}")
                                elif channel == "status":
                                    if isinstance(pl, dict):
                                        apply_status(pl)
                                elif channel == "hook":
                                    dispatch_agent_hook(pl, ctx=hook_ctx)

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
                        sidebar_past_list.refresh()
                        ui.notify("Run finished", type="positive")
            except httpx.ConnectError as e:
                stream_md.content = f"**Cannot connect**\n\n`{e}`\n\nIs the agent at `{base}`?"
                final_report_md.content = "_Unavailable._"
                status_label.text = "Connection failed"
                spin.visible = False
                update_execution_graph(None, False)
                ui.notify("Connection refused — start the agent API", type="negative")
            except Exception as e:  # noqa: BLE001
                stream_md.content = f"**Error**\n\n```\n{e!r}\n```"
                final_report_md.content = "_Error — see chat._"
                status_label.text = "Error"
                spin.visible = False
                update_execution_graph(None, False)
                ui.notify(str(e), type="negative")
            finally:
                send_btn.enable()

        send_btn.on_click(run_agent)

        await context.client.connected()
        if app.storage.user.pop(storage_auto_start_run, False):
            await run_agent()
