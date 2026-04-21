"""NiceGUI frontend for the AgentCore RCA agent (POST /invocations, SSE response)."""

from __future__ import annotations

import copy
import json
import os
import secrets
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import Request
from nicegui import app, ui

load_dotenv()

from cognito_auth import (
    build_authorize_url,
    claims_to_username,
    exchange_code_for_tokens,
    get_cognito_logout_redirect_url,
    load_cognito_settings,
    oauth_callback_path,
    verify_id_token,
)

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
STORAGE_OAUTH_STATE = "oauth_state"

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

WORKSPACE_SHELL_CSS = """
.rca-workspace-root { min-height: calc(100vh - 3.5rem); }
.rca-chat-scroll .q-scrollarea__content { display: flex; flex-direction: column; }
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
    cognito_out = None
    if app.storage.user.get("auth_provider") == "cognito":
        cognito_out = get_cognito_logout_redirect_url()
    app.storage.user.clear()
    if preserved is not None:
        app.storage.user[STORAGE_UI_SETTINGS] = preserved
    if cognito_out:
        ui.navigate.to(cognito_out)
    else:
        ui.navigate.to("/")


AGENTS_HEADER_LOADING = "···"


def open_settings_dialog() -> None:
    """Agent host/port and dark mode (shared by landing and workspace)."""
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


def build_payload(
    prompt: str,
    session_id: str,
    arch: list[dict[str, str]],
    repo_owner: str = "",
    repo_name: str = "",
    *,
    user_id: str = "",
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
    uid = (user_id or "").strip()
    if uid:
        body["userId"] = uid
    print("[RCA] /invocations payload:", json.dumps(body, indent=2, ensure_ascii=False))
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


async def process_oauth_callback(request: Request) -> None:
    """Complete Cognito authorization_code flow; on success navigates to /workspace."""
    err = request.query_params.get("error")
    if err:
        desc = request.query_params.get("error_description") or err
        with ui.column().classes("w-full min-h-screen items-center justify-center gap-4 p-8"):
            ui.label(f"Sign-in was cancelled or failed: {desc}").classes("text-red-300 text-center max-w-lg")
            ui.button("Back to login", on_click=lambda: ui.navigate.to("/"))
        return
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    expected = app.storage.user.get(STORAGE_OAUTH_STATE)
    if not code or not state or not expected or state != expected:
        app.storage.user.pop(STORAGE_OAUTH_STATE, None)
        with ui.column().classes("w-full min-h-screen items-center justify-center gap-4 p-8"):
            ui.label("Invalid or expired sign-in. Please try again from the home page.").classes(
                "text-amber-300 text-center max-w-lg"
            )
            ui.button("Back to login", on_click=lambda: ui.navigate.to("/"))
        return
    app.storage.user.pop(STORAGE_OAUTH_STATE, None)
    settings = load_cognito_settings()
    if not settings:
        with ui.column().classes("w-full min-h-screen items-center justify-center gap-4 p-8"):
            ui.label("Cognito is not configured on the server.").classes("text-red-300")
            ui.button("Back", on_click=lambda: ui.navigate.to("/"))
        return
    try:
        tokens = await exchange_code_for_tokens(settings, code)
        id_tok = tokens.get("id_token")
        if not id_tok or not isinstance(id_tok, str):
            raise RuntimeError("No id_token in token response")
        claims = verify_id_token(settings, id_tok)
        app.storage.user["username"] = claims_to_username(claims)
        sub = claims.get("sub")
        if isinstance(sub, str) and sub.strip():
            app.storage.user["user_id"] = sub.strip()
        app.storage.user["auth_provider"] = "cognito"
        app.storage.user["cognito_id_token"] = id_tok
        at = tokens.get("access_token")
        if at:
            app.storage.user["cognito_access_token"] = at
        rt = tokens.get("refresh_token")
        if rt:
            app.storage.user["cognito_refresh_token"] = rt
        ensure_past_sessions()
        ui.navigate.to("/workspace")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        with ui.column().classes("w-full min-h-screen items-center justify-center gap-4 p-8"):
            ui.label("Could not complete sign-in").classes("text-lg text-red-300")
            ui.label(msg).classes("text-red-400/90 text-sm whitespace-pre-wrap max-w-xl")
            ui.button("Back to login", on_click=lambda: ui.navigate.to("/"))


@ui.page("/")
async def landing_page(request: Request) -> None:
    apply_global_theme()

    if oauth_callback_path() == "/" and (
        request.query_params.get("code") is not None or request.query_params.get("error") is not None
    ):
        await process_oauth_callback(request)
        return

    def open_cognito() -> None:
        settings = load_cognito_settings()
        if not settings:
            ui.notify(
                "Cognito is not configured. Set COGNITO_USER_POOL_ID, COGNITO_CLIENT_ID, "
                "COGNITO_OAUTH_DOMAIN, and COGNITO_REDIRECT_URI (must match Cognito app client).",
                type="negative",
            )
            return
        st = secrets.token_urlsafe(32)
        app.storage.user[STORAGE_OAUTH_STATE] = st
        ui.navigate.to(build_authorize_url(settings, st))

    def open_developer_login() -> None:
        app.storage.user["username"] = "Developer"
        app.storage.user["user_id"] = "local:developer"
        ensure_past_sessions()
        ui.navigate.to("/workspace")

    with ui.column().classes("w-full min-h-screen relative bg-gray-950"):
        with ui.row().classes("absolute top-4 left-4 z-20"):
            ui.button(icon="settings", on_click=open_settings_dialog).props("flat round").classes(
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


_oauth_cb = oauth_callback_path()
if _oauth_cb != "/":

    @ui.page(_oauth_cb, response_timeout=45.0)
    async def cognito_oauth_callback(request: Request) -> None:
        apply_global_theme()
        await process_oauth_callback(request)


from workspace_page import register_workspace_page

register_workspace_page(
    default_api_base=DEFAULT_API_BASE,
    default_prompt=DEFAULT_PROMPT,
    default_repo=DEFAULT_REPO,
    agents_header_loading=AGENTS_HEADER_LOADING,
    storage_auto_start_run=STORAGE_AUTO_START_RUN,
    append_past_session=append_past_session,
    apply_global_theme=apply_global_theme,
    build_payload=build_payload,
    ensure_past_sessions=ensure_past_sessions,
    get_agent_api_base=get_agent_api_base,
    is_logged_in=is_logged_in,
    load_rca_from_storage=load_rca_from_storage,
    logout=logout,
    open_settings_dialog=open_settings_dialog,
    save_rca_to_storage=save_rca_to_storage,
    _arch_editor=_arch_editor,
    mermaid_css=MERMAID_CSS,
    workspace_shell_css=WORKSPACE_SHELL_CSS,
)


@ui.page("/sessions")
def legacy_sessions_redirect() -> None:
    apply_global_theme()
    if not is_logged_in():
        ui.navigate.to("/")
        return
    ui.navigate.to("/workspace")


@ui.page("/agents")
def legacy_agents_redirect() -> None:
    apply_global_theme()
    if not is_logged_in():
        ui.navigate.to("/")
        return
    ui.navigate.to("/workspace")


def main() -> None:
    storage_secret = (os.environ.get("STORAGE_SECRET") or "").strip()
    if not storage_secret:
        raise SystemExit(
            "STORAGE_SECRET is not set. Copy .env.example to .env and set STORAGE_SECRET "
            "to a long random value (e.g. openssl rand -hex 32)."
        )
    ui.run(
        title="RCA Workspace",
        port=int(os.environ.get("PORT", "8081")),
        reload=True,
        favicon="🧩",
        storage_secret=storage_secret,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
