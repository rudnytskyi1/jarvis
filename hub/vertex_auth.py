"""Credentials for Google Vertex AI image generation.

Владелец 2026-09-23: «для генерации картинок теперь используй vertexai api (у
меня бесплатные 300$ credits)». Vertex AI bills the Google Cloud project, so
the request no longer travels with one long-lived API key: it needs a project
and a credential that produces short-lived OAuth2 access tokens.

This module is the only place that knows how a Vertex request proves who it is,
and it supports every way the owner can plausibly have a credential:

* an Express-mode API key (``?key=``) - the shortest path, no service account;
* an access token that is already in the environment;
* a service-account JSON key - the file Google hands out for a robot account;
  the token is fetched with a signed JWT and cached until shortly before it
  expires, so the file is read once and the signature is computed once;
* the gcloud "application default credentials" file, which carries a refresh
  token instead of a private key.

Nothing here is billed, and secret material never reaches the log: a credential
file is named by its path only, and only its ``type`` is ever read for
diagnostics.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

#: Where Google exchanges a signed assertion for an access token.
DEFAULT_TOKEN_URI = 'https://oauth2.googleapis.com/token'
#: Vertex publishes models per region; the owner's config names one.
DEFAULT_LOCATION = 'us-central1'
VERTEX_HOST = 'aiplatform.googleapis.com'
CLOUD_PLATFORM_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
#: Renew before the token really expires: a request that starts with seconds
#: left would otherwise fail in the middle of a 16 MB upload.
TOKEN_MARGIN_S = 120.0
#: Shortest lifetime accepted from Google; a smaller number would mean a broken
#: answer, and caching it for less than a second would re-authenticate per image.
MIN_TOKEN_LIFETIME_S = 60.0
#: Searched when the config names no file: what ``gcloud auth application-default
#: login`` writes, and a file the owner drops into the repo by hand.
GCLOUD_ADC_RELATIVE = Path('gcloud') / 'application_default_credentials.json'
REPO_CREDENTIALS_RELATIVE = Path('data') / 'vertex-credentials.json'


class VertexAuthError(RuntimeError):
    """A Vertex credential cannot be produced; the text is shown to the owner."""


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def service_account_assertion(credentials: Mapping[str, Any], *, now: int | None = None,
                              token_uri: str = DEFAULT_TOKEN_URI) -> str:
    """The signed JWT Google accepts in exchange for an access token.

    RS256 over the service account's private key, exactly as Google documents:
    ``iss`` is the robot's address, ``scope`` is ``cloud-platform`` (the only
    scope Vertex needs), ``aud`` is the token endpoint, and the assertion lives
    an hour. The ``kid`` stays out when the file does not name a key id, because
    an empty ``kid`` is not the same thing as none at all.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    email = str(credentials.get('client_email') or '').strip()
    private_key = str(credentials.get('private_key') or '').strip()
    if not email or not private_key:
        raise VertexAuthError('The service-account file has no client_email or private_key.')
    issued = int(time.time() if now is None else now)
    header: dict[str, Any] = {'alg': 'RS256', 'typ': 'JWT'}
    if str(credentials.get('private_key_id') or '').strip():
        header['kid'] = str(credentials['private_key_id']).strip()
    claims = {'iss': email, 'scope': CLOUD_PLATFORM_SCOPE, 'aud': token_uri,
              'iat': issued, 'exp': issued + 3600}
    signing_input = (f'{_b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))}.'
                     f'{_b64url(json.dumps(claims, separators=(",", ":")).encode("utf-8"))}')
    try:
        key = serialization.load_pem_private_key(private_key.encode('utf-8'), password=None)
        signature = key.sign(signing_input.encode('ascii'), padding.PKCS1v15(), hashes.SHA256())
    except Exception as exc:  # noqa: BLE001 - any unusable key reads the same to the owner
        raise VertexAuthError(
            f'The service-account private key could not be used ({type(exc).__name__}).') from exc
    return f'{signing_input}.{_b64url(signature)}'


def _token_error(body: bytes) -> str:
    """Google's own sentence about a refused token, short and safe to print."""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return ''
    if not isinstance(parsed, dict):
        return ''
    text = parsed.get('error_description') or parsed.get('error')
    if isinstance(text, dict):
        text = text.get('message')
    return f': {str(text).strip()[:200]}' if str(text or '').strip() else ''


class VertexAuth:
    """Tokens (and the model URL) for one ``server.image_generation`` block."""

    def __init__(self, cfg: Any, *, transport: Any = None, environ: Any = None,
                 repo_root: Path | None = None) -> None:
        self.project = str(getattr(cfg, 'vertex_project', '') or '').strip()
        self.location = str(getattr(cfg, 'vertex_location', '') or '').strip() or DEFAULT_LOCATION
        self.credentials_path = str(getattr(cfg, 'vertex_credentials_path', '') or '').strip()
        self.credentials_env = str(getattr(cfg, 'vertex_credentials_env',
                                           'GOOGLE_APPLICATION_CREDENTIALS') or '').strip()
        self.access_token_env = str(getattr(cfg, 'vertex_access_token_env',
                                            'VERTEX_ACCESS_TOKEN') or '').strip()
        self.api_key_env = str(getattr(cfg, 'vertex_api_key_env', 'VERTEX_API_KEY') or '').strip()
        self.environ = os.environ if environ is None else environ
        self.repo_root = (Path(repo_root) if repo_root is not None
                          else Path(__file__).resolve().parents[1])
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0),
                                      transport=transport, follow_redirects=False)
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    # --- what is available ---------------------------------------------------

    def _env(self, name: str) -> str:
        if not name:
            return ''
        try:
            return str(self.environ.get(name) or '').strip()
        except Exception:  # noqa: BLE001 - a hostile mapping is simply "not set"
            return ''

    @property
    def api_key(self) -> str:
        """An Express-mode API key, when the owner uses one instead of a robot."""
        return self._env(self.api_key_env)

    @property
    def static_token(self) -> str:
        """A token somebody obtained outside Rowan (curl, gcloud, CI)."""
        return self._env(self.access_token_env)

    def candidate_paths(self) -> list[Path]:
        """Credential files to try, most explicit first."""
        candidates: list[Path] = []
        if self.credentials_path:
            candidates.append(Path(self.credentials_path).expanduser())
        from_env = self._env(self.credentials_env)
        if from_env:
            candidates.append(Path(from_env).expanduser())
        candidates.append(self.repo_root / REPO_CREDENTIALS_RELATIVE)
        appdata = self._env('APPDATA')
        if appdata:
            candidates.append(Path(appdata) / GCLOUD_ADC_RELATIVE)
        candidates.append(Path.home() / '.config' / GCLOUD_ADC_RELATIVE)
        return candidates

    def credentials_file(self) -> Path | None:
        for path in self.candidate_paths():
            try:
                if path.is_file():
                    return path
            except OSError:
                continue
        return None

    @property
    def ready(self) -> bool:
        """Whether a request could be signed right now, without touching the network.

        A credential file alone is not enough: a project id without a key and a
        key without a project are both unfinished setups, and ``/health`` has to
        say so before the person asks for a picture.
        """
        return self.missing_reason() == ''

    def missing_reason(self) -> str:
        """What still has to be set up, in one sentence; ``''`` when nothing has to."""
        if self.api_key:
            return ''
        if self.static_token:
            return '' if self.project else (
                'server.image_generation.vertex_project (the Google Cloud project id)')
        path = self.credentials_file()
        if path is None:
            return ('a credential: a service-account JSON key saved as '
                    'data/vertex-credentials.json, or its path in '
                    'server.image_generation.vertex_credentials_path, or a VERTEX_API_KEY')
        if not self.project:
            return 'server.image_generation.vertex_project (the Google Cloud project id)'
        return ''

    # --- one request ---------------------------------------------------------

    def model_url(self, model: str) -> str:
        """The ``generateContent`` endpoint for this project, region and model."""
        name = str(model or '').strip()
        if not name:
            raise VertexAuthError('Vertex AI image generation names no model.')
        if self.api_key:
            # Express mode: the key itself selects the project, so the URL
            # carries no project or region.
            return f'https://{VERTEX_HOST}/v1/publishers/google/models/{name}:generateContent'
        if not self.project:
            raise VertexAuthError(
                'Vertex AI needs server.image_generation.vertex_project - the Google Cloud '
                'project id that owns the credits.')
        host = VERTEX_HOST if self.location == 'global' else f'{self.location}-{VERTEX_HOST}'
        return (f'https://{host}/v1/projects/{self.project}/locations/{self.location}'
                f'/publishers/google/models/{name}:generateContent')

    async def authorization(self, model: str) -> tuple[str, dict[str, str], dict[str, str]]:
        """``(url, headers, query)`` for one metered image request."""
        url = self.model_url(model)
        if self.api_key:
            return url, {}, {'key': self.api_key}
        return url, {'Authorization': f'Bearer {await self.access_token()}'}, {}

    async def access_token(self) -> str:
        """A cached access token, fetching a new one when it is spent."""
        static = self.static_token
        if static:
            return static
        async with self._lock:
            if self._token and time.time() < self._expires_at - TOKEN_MARGIN_S:
                return self._token
            path = self.credentials_file()
            if path is None:
                raise VertexAuthError('Vertex AI image generation has no credential: ' +
                                      (self.missing_reason() or 'set one up first.'))
            payload = self._load(path)
            kind = str(payload.get('type') or '')
            if kind == 'authorized_user' or (
                    payload.get('refresh_token') and kind != 'service_account'):
                token, expires_at = await self._refresh_user_token(payload, path)
            elif kind == 'service_account' or payload.get('private_key'):
                token, expires_at = await self._service_account_token(payload, path)
            else:
                raise VertexAuthError(
                    f'{path.name} is not a Google credential file '
                    f'(type {kind or "unknown"!r}).')
            self._token, self._expires_at = token, expires_at
            return token

    def _load(self, path: Path) -> Mapping[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise VertexAuthError(f'The Vertex credential file {path} does not exist.') from exc
        except (OSError, ValueError) as exc:
            raise VertexAuthError(
                f'The Vertex credential file {path.name} could not be read '
                f'({type(exc).__name__}).') from exc
        if not isinstance(payload, Mapping):
            raise VertexAuthError(f'The Vertex credential file {path.name} is not a JSON object.')
        return payload

    async def _service_account_token(self, payload: Mapping[str, Any],
                                     path: Path) -> tuple[str, float]:
        token_uri = str(payload.get('token_uri') or DEFAULT_TOKEN_URI).strip() or DEFAULT_TOKEN_URI
        assertion = service_account_assertion(payload, token_uri=token_uri)
        return await self._post_token(token_uri, {
            'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
            'assertion': assertion}, path, 'service_account')

    async def _refresh_user_token(self, payload: Mapping[str, Any],
                                  path: Path) -> tuple[str, float]:
        """The ``gcloud auth application-default login`` file: a refresh token."""
        token_uri = str(payload.get('token_uri') or DEFAULT_TOKEN_URI).strip() or DEFAULT_TOKEN_URI
        missing = [name for name in ('client_id', 'client_secret', 'refresh_token')
                   if not str(payload.get(name) or '').strip()]
        if missing:
            raise VertexAuthError(
                f'The credential file {path.name} has no {", ".join(missing)}; run '
                '"gcloud auth application-default login" again, or use a service account.')
        return await self._post_token(token_uri, {
            'grant_type': 'refresh_token',
            'client_id': str(payload['client_id']).strip(),
            'client_secret': str(payload['client_secret']).strip(),
            'refresh_token': str(payload['refresh_token']).strip()}, path, 'authorized_user')

    async def _post_token(self, token_uri: str, form: dict[str, str], path: Path,
                          kind: str) -> tuple[str, float]:
        try:
            response = await self.http.post(token_uri, data=form)
        except Exception as exc:  # noqa: BLE001 - any transport failure reads the same
            raise VertexAuthError(
                'Google could not be reached for a Vertex access token '
                f'({type(exc).__name__}).') from exc
        if response.status_code != 200:
            raise VertexAuthError(
                f'Google refused the credentials in {path.name} '
                f'(HTTP {response.status_code}{_token_error(response.content)})')
        try:
            body = response.json()
            token = str(body['access_token']).strip()
            lifetime = float(body.get('expires_in') or 3600)
        except Exception as exc:  # noqa: BLE001 - a malformed answer is not a token
            raise VertexAuthError(
                f'Google returned no access token for {path.name} '
                f'({type(exc).__name__}).') from exc
        if not token:
            raise VertexAuthError(f'Google returned an empty access token for {path.name}.')
        log.info('Vertex access token obtained from %s (kind=%s, valid %.0f s)',
                 path.name, kind, lifetime)
        return token, time.time() + max(MIN_TOKEN_LIFETIME_S, lifetime)

    async def close(self) -> None:
        await self.http.aclose()


__all__ = ['DEFAULT_LOCATION', 'DEFAULT_TOKEN_URI', 'VERTEX_HOST', 'VertexAuth',
           'VertexAuthError', 'service_account_assertion']
