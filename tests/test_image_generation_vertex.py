"""Nano Banana through Vertex AI, paid by the owner's Google Cloud credits.

Владелец 2026-09-23: «для генерации картинок теперь используй vertexai api
(у меня бесплатные 300$ credits)». One config line switches the road, and these
tests keep the two roads apart: with ``provider: vertex`` the Gemini key is
never read, and a Vertex failure names its real cause instead of the old text
about AI Studio.
"""
from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from common.config import ImageGenerationConfig
from hub import vertex_auth
from hub.api_budget import CloudUnavailable
from hub.image_generation import ImageGenerator

TOKEN_HOST = 'oauth2.googleapis.com'
MODEL_HOST = 'us-central1-aiplatform.googleapis.com'
MODEL_PATH = ('/v1/projects/rowan-images/locations/us-central1/publishers/google/models/'
              'gemini-3.1-flash-image:generateContent')


def fixture_image() -> bytes:
    out = BytesIO()
    Image.new('RGB', (24, 16), 'blue').save(out, 'PNG')
    return out.getvalue()


def answer() -> dict:
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'inlineData': {'mimeType': 'image/png',
                        'data': base64.b64encode(fixture_image()).decode()}}]}}],
        'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 1120,
                          'candidatesTokensDetails': [{'modality': 'IMAGE',
                                                       'tokenCount': 1120}]}}


def write_credentials(folder: Path) -> None:
    (folder / 'vertex-credentials.json').write_text(json.dumps({
        'type': 'authorized_user', 'client_id': 'cid', 'client_secret': 'secret',
        'refresh_token': 'refresh', 'token_uri': 'https://oauth2.googleapis.com/token'}),
        encoding='utf-8')


def no_other_credentials(monkeypatch):
    """Hide the machine's own Google credentials from the generator."""
    for name in ('GEMINI_API_KEY', 'VERTEX_API_KEY', 'VERTEX_ACCESS_TOKEN', 'APPDATA'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vertex_auth, 'GCLOUD_ADC_RELATIVE', Path('gcloud/.absent-credentials.json'))
    monkeypatch.setattr(vertex_auth, 'REPO_CREDENTIALS_RELATIVE', Path('data/.absent-credentials.json'))


def vertex_generator(tmp_path, handler, monkeypatch, *, credentials: bool = True, **overrides):
    no_other_credentials(monkeypatch)
    if credentials:
        write_credentials(tmp_path)
        monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS',
                           str(tmp_path / 'vertex-credentials.json'))
    else:
        monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
    overrides.setdefault('vertex_project', 'rowan-images')
    cfg = ImageGenerationConfig(enabled=True, provider='vertex', **overrides)
    return ImageGenerator(cfg, ledger_path=tmp_path / 'usage.db', monthly_usd=0,
                          transport=httpx.MockTransport(handler))


def test_vertex_request_carries_a_bearer_token_and_never_the_gemini_key(tmp_path, monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == TOKEN_HOST:
            return httpx.Response(200, json={'access_token': 'ya29.vertex', 'expires_in': 3600})
        return httpx.Response(200, json=answer())

    async def run():
        client = vertex_generator(tmp_path, handler, monkeypatch)
        try:
            result = await client.generate('A blue rectangle')
            assert (result.width, result.height) == (24, 16)
            assert client.provider_label == 'Vertex AI' and client.ready
            model_call = seen[-1]
            assert model_call.url.host == MODEL_HOST and model_call.url.path == MODEL_PATH
            assert model_call.headers['Authorization'] == 'Bearer ya29.vertex'
            assert 'x-goog-api-key' not in {name.lower() for name in model_call.headers}
            assert dict(model_call.url.params) == {}
            assert client.budget.status()['accounted_usd'] == 0.06725
        finally:
            await client.close()

    asyncio.run(run())
    assert [request.url.host for request in seen] == [TOKEN_HOST, MODEL_HOST]


def test_a_vertex_402_is_a_billing_answer_and_not_an_invitation_to_rephrase(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == TOKEN_HOST:
            return httpx.Response(200, json={'access_token': 'ya29.vertex', 'expires_in': 3600})
        return httpx.Response(402, json={'error': {'message': 'Billing account closed.'}})

    async def run():
        client = vertex_generator(tmp_path, handler, monkeypatch)
        try:
            with pytest.raises(CloudUnavailable) as failure:
                await client.generate('A blue rectangle')
            text = str(failure.value)
            assert 'HTTP 402' in text and 'billing' in text.lower()
        finally:
            await client.close()

    asyncio.run(run())


def test_a_vertex_error_body_reaches_the_person(tmp_path, monkeypatch):
    """A wrong model or a disabled API is Google's sentence, not our guess."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == TOKEN_HOST:
            return httpx.Response(200, json={'access_token': 'ya29.vertex', 'expires_in': 3600})
        return httpx.Response(404, json={'error': {'message': 'Publisher Model not found.'}})

    async def run():
        client = vertex_generator(tmp_path, handler, monkeypatch)
        try:
            with pytest.raises(CloudUnavailable) as failure:
                await client.generate('A blue rectangle')
            assert 'HTTP 404' in str(failure.value)
            assert 'Publisher Model not found.' in str(failure.value)
        finally:
            await client.close()

    asyncio.run(run())


def test_a_refused_service_account_names_vertex_and_not_the_gemini_key(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == TOKEN_HOST:
            return httpx.Response(200, json={'access_token': 'ya29.vertex', 'expires_in': 3600})
        return httpx.Response(403, json={'error': {'message': 'Permission denied.'}})

    async def run():
        client = vertex_generator(tmp_path, handler, monkeypatch)
        try:
            with pytest.raises(CloudUnavailable) as failure:
                await client.generate('A blue rectangle')
            text = str(failure.value)
            assert 'Vertex' in text and 'GEMINI' not in text and 'AI Studio' not in text
        finally:
            await client.close()

    asyncio.run(run())


def test_without_a_project_or_a_credential_nothing_is_sent(tmp_path, monkeypatch):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=answer())

    async def run():
        client = vertex_generator(tmp_path, handler, monkeypatch, credentials=False)
        try:
            assert not client.ready
            with pytest.raises(CloudUnavailable) as failure:
                await client.generate('A blue rectangle')
            assert 'VERTEX_IMAGE_GENERATION' in str(failure.value)
            assert client.budget.status()['accounted_usd'] == 0
        finally:
            await client.close()

    asyncio.run(run())
    assert sent == [], 'a missing credential must not become a request'

    async def run_without_project():
        client = vertex_generator(tmp_path, handler, monkeypatch, vertex_project='')
        try:
            assert not client.ready
            with pytest.raises(CloudUnavailable, match='vertex_project'):
                await client.generate('A blue rectangle')
        finally:
            await client.close()

    asyncio.run(run_without_project())
    assert sent == [], 'a project id is required on the service-account road'


def test_an_express_api_key_is_sent_in_the_query_as_vertex_expects(tmp_path, monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=answer())

    async def run():
        client = vertex_generator(tmp_path, handler, monkeypatch, credentials=False,
                                  vertex_project='')
        monkeypatch.setenv('VERTEX_API_KEY', 'express-key')
        try:
            assert client.ready
            assert (await client.generate('A blue rectangle')).width == 24
        finally:
            await client.close()

    asyncio.run(run())
    assert len(seen) == 1
    assert seen[0].url.host == 'aiplatform.googleapis.com'
    assert seen[0].url.path == ('/v1/publishers/google/models/'
                                'gemini-3.1-flash-image:generateContent')
    assert dict(seen[0].url.params) == {'key': 'express-key'}
