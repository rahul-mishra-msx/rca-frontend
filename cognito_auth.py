"""Amazon Cognito OAuth 2.0 (hosted UI): authorize URL, token exchange, ID token verification."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
import jwt
from jwt import PyJWKClient


@dataclass(frozen=True)
class CognitoSettings:
    region: str
    user_pool_id: str
    client_id: str
    client_secret: str
    domain_base: str
    redirect_uri: str
    scopes: str


def _strip_env(value: str) -> str:
    return (value or "").strip().strip("`").strip()


def _auth_base_url(domain_base: str) -> str:
    p = urlparse(domain_base)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}".rstrip("/")
    return ""


def oauth_callback_path() -> str:
    """URL path for the OAuth redirect, derived from COGNITO_REDIRECT_URI (e.g. `/` or `/oauth/callback`)."""
    raw = _strip_env(os.environ.get("COGNITO_REDIRECT_URI") or "")
    if not raw:
        return "/oauth/callback"
    if not raw.startswith("http"):
        raw = "https://" + raw
    p = urlparse(raw)
    path = p.path or "/"
    return path if path else "/"


def load_cognito_settings() -> CognitoSettings | None:
    pool = _strip_env(os.environ.get("COGNITO_USER_POOL_ID") or "")
    cid = _strip_env(os.environ.get("COGNITO_CLIENT_ID") or "")
    secret = _strip_env(os.environ.get("COGNITO_CLIENT_SECRET") or "")
    domain = _strip_env(os.environ.get("COGNITO_OAUTH_DOMAIN") or "")
    if domain and not domain.startswith("http"):
        domain = "https://" + domain
    region = _strip_env(os.environ.get("AWS_REGION") or "us-west-2") or "us-west-2"
    try:
        port = int(_strip_env(os.environ.get("PORT") or "8081") or "8081")
    except ValueError:
        port = 8081
    redirect = _strip_env(os.environ.get("COGNITO_REDIRECT_URI") or "")
    if not redirect:
        redirect = f"http://localhost:{port}/oauth/callback"
    elif not redirect.startswith("http"):
        redirect = "https://" + redirect
    raw_scopes = _strip_env(os.environ.get("COGNITO_OAUTH_SCOPES") or "openid email")
    scopes = " ".join(
        part.strip("`").strip()
        for part in raw_scopes.replace(",", " ").split()
        if part.strip("`").strip()
    )
    if not scopes:
        scopes = "openid email"
    if not all([pool, cid, domain]):
        return None
    base = _auth_base_url(domain)
    if not base:
        return None
    return CognitoSettings(
        region=region,
        user_pool_id=pool,
        client_id=cid,
        client_secret=secret,
        domain_base=base,
        redirect_uri=redirect,
        scopes=scopes,
    )


def issuer_url(settings: CognitoSettings) -> str:
    return f"https://cognito-idp.{settings.region}.amazonaws.com/{settings.user_pool_id}"


def jwks_url(settings: CognitoSettings) -> str:
    return f"{issuer_url(settings)}/.well-known/jwks.json"


def build_authorize_url(settings: CognitoSettings, state: str) -> str:
    q = urlencode(
        {
            "client_id": settings.client_id,
            "response_type": "code",
            "scope": settings.scopes,
            "redirect_uri": settings.redirect_uri,
            "state": state,
        }
    )
    return f"{settings.domain_base}/oauth2/authorize?{q}"


async def exchange_code_for_tokens(settings: CognitoSettings, code: str) -> dict[str, Any]:
    token_url = f"{settings.domain_base}/oauth2/token"
    data: dict[str, str] = {
        "grant_type": "authorization_code",
        "client_id": settings.client_id,
        "code": code,
        "redirect_uri": settings.redirect_uri,
    }
    if settings.client_secret:
        data["client_secret"] = settings.client_secret
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code >= 400:
        detail = (resp.text or "").strip()
        try:
            err_json = resp.json()
            desc = err_json.get("error_description") or err_json.get("error")
            if desc:
                detail = str(desc)
        except Exception:
            pass
        raise RuntimeError(f"Token request failed (HTTP {resp.status_code}): {detail[:500]}")
    return resp.json()


def verify_id_token(settings: CognitoSettings, id_token: str) -> dict[str, Any]:
    jwks = PyJWKClient(jwks_url(settings), cache_keys=True)
    signing_key = jwks.get_signing_key_from_jwt(id_token)
    iss = issuer_url(settings)
    return jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256"],
        audience=settings.client_id,
        issuer=iss,
    )


def claims_to_username(claims: dict[str, Any]) -> str:
    for key in ("email", "preferred_username", "cognito:username", "sub"):
        v = claims.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "User"


def build_cognito_logout_url(settings: CognitoSettings, logout_redirect_uri: str) -> str:
    q = urlencode({"client_id": settings.client_id, "logout_uri": logout_redirect_uri})
    return f"{settings.domain_base}/logout?{q}"


def get_cognito_logout_redirect_url() -> str | None:
    if _strip_env(os.environ.get("COGNITO_LOGOUT_URI") or "") == "":
        return None
    settings = load_cognito_settings()
    if not settings:
        return None
    lou = _strip_env(os.environ.get("COGNITO_LOGOUT_URI") or "")
    return build_cognito_logout_url(settings, lou)
