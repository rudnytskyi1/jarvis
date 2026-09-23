"""Vertex AI credentials: the owner's $300 Google Cloud credits pay for images.

Everything here is offline: the token endpoint is an ``httpx.MockTransport``
handler and the service-account key is generated for the test, so no real
credential and no real request is involved. An autouse fixture hides whatever
credentials the machine itself has, so the tests say the same thing everywhere.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from common.config import ImageGenerationConfig
from hub import vertex_auth
from hub.vertex_auth import VertexAuth, VertexAuthError, service_account_assertion

TOKEN_URL = 'https://oauth2.googleapis.com/token'


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch):
    """The developer's own gcloud login must not decide what these tests see."""
    for name in ('GOOGLE_APPLICATION_CREDENTIALS', 'VERTEX_API_KEY', 'VERTEX_ACCESS_TOKEN',
                 'APPDATA'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vertex_auth, 'GCLOUD_ADC_RELATIVE', Path('gcloud/.absent-credentials.json'))


def service_account(folder: Path, **overrides) -> dict:
    """Write a service-account file the way Google hands one out."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    payload = {
        'type': 'service_account',
        'project_id': 'rowan-images',
        'private_key_id': 'key-1',
        'private_key': key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()).decode('utf-8'),
        'client_email': 'rowan-images@rowan-images.iam.gserviceaccount.com',
        'token_uri': TOKEN_URL,
        **overrides}
    (folder / 'vertex-credentials.json').write_text(json.dumps(payload), encoding='utf-8')
    return payload


def decode_segment(segment: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(segment + '=' * (-len(segment) % 4)))


def test_service_account_assertion_is_rs256_over_the_robot_address():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    credentials = {
        'client_email': 'rowan@example.iam.gserviceaccount.com',
        'private_key_id': 'abc123',
        'private_key': key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()).decode('utf-8')}
    header, claims, signature = service_account_assertion(
        credentials, now=1_700_000_000).split('.')
    assert decode_segment(header) == {'alg': 'RS256', 'typ': 'JWT', 'kid': 'abc123'}
    assert decode_segment(claims) == {
        'iss': 'rowan@example.iam.gserviceaccount.com',
        'scope': 'https://www.googleapis.com/auth/cloud-platform',
        'aud': TOKEN_URL, 'iat': 1_700_000_000, 'exp': 1_700_003_600}
    key.public_key().verify(base64.urlsafe_b64decode(signature + '=' * (-len(signature) % 4)),
                            f'{header}.{claims}'.encode('ascii'),
                            padding.PKCS1v15(), hashes.SHA256())
    # A file without a key id still signs, and simply carries no "kid".
    assert 'kid' not in decode_segment(service_account_assertion(
        {**credentials, 'private_key_id': ''}, now=1).split('.')[0])
    with pytest.raises(VertexAuthError, match='private_key'):
        service_account_assertion({'client_email': 'a@b'})


@pytest.mark.parametrize('location,host', [
    ('us-central1', 'us-central1-aiplatform.googleapis.com'),
    ('europe-west4', 'europe-west4-aiplatform.googleapis.com'),
    ('global', 'aiplatform.googleapis.com')])
def test_the_model_url_names_project_region_and_model(tmp_path, location, host):
    auth = VertexAuth(ImageGenerationConfig(provider='vertex', vertex_project='rowan-images',
                                            vertex_location=location),
                      environ={}, repo_root=tmp_path)
    assert auth.model_url('gemini-3.1-flash-image') == (
        f'https://{host}/v1/projects/rowan-images/locations/{location}'
        '/publishers/google/models/gemini-3.1-flash-image:generateContent')


def test_a_service_account_is_exchanged_once_and_the_token_is_reused(tmp_path):
    material = service_account(tmp_path)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        form = dict(httpx.QueryParams(request.content.decode()))
        assert form['grant_type'] == 'urn:ietf:params:oauth:grant-type:jwt-bearer'
        assert form['assertion'].count('.') == 2
        assert material['private_key'] not in request.content.decode()
        return httpx.Response(200, json={'access_token': 'ya29.test', 'expires_in': 3600})

    async def run():
        auth = VertexAuth(ImageGenerationConfig(
            provider='vertex', vertex_project='rowan-images',
            vertex_credentials_path=str(tmp_path / 'vertex-credentials.json')),
            transport=httpx.MockTransport(handler), environ={}, repo_root=tmp_path)
        try:
            url, headers, params = await auth.authorization('gemini-3.1-flash-image')
            assert headers == {'Authorization': 'Bearer ya29.test'} and params == {}
            assert url.startswith('https://us-central1-aiplatform.googleapis.com/v1/projects/rowan-images/')
            assert (await auth.authorization('gemini-3.1-flash-image'))[1] == headers
        finally:
            await auth.close()

    asyncio.run(run())
    # One token for two images, and the private key never reaches the model host.
    assert {request.url.host for request in calls} == {'oauth2.googleapis.com'}
    assert len(calls) == 1


def test_a_spent_token_is_fetched_again(tmp_path):
    service_account(tmp_path)
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={'access_token': f'ya29.{len(calls)}', 'expires_in': 3600})

    async def run():
        auth = VertexAuth(ImageGenerationConfig(
            provider='vertex', vertex_project='p',
            vertex_credentials_path=str(tmp_path / 'vertex-credentials.json')),
            transport=httpx.MockTransport(handler), environ={}, repo_root=tmp_path)
        try:
            assert await auth.access_token() == 'ya29.1'
            auth._expires_at = 0.0  # the hour is over
            assert await auth.access_token() == 'ya29.2'
        finally:
            await auth.close()

    asyncio.run(run())
    assert len(calls) == 2


def test_application_default_credentials_use_the_refresh_token(tmp_path):
    adc = tmp_path / 'adc.json'
    adc.write_text(json.dumps({'type': 'authorized_user', 'client_id': 'cid',
                               'client_secret': 'secret', 'refresh_token': 'refresh',
                               'token_uri': TOKEN_URL}), encoding='utf-8')
    forms: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        forms.append(dict(httpx.QueryParams(request.content.decode())))
        return httpx.Response(200, json={'access_token': 'ya29.adc', 'expires_in': 1800})

    async def run():
        auth = VertexAuth(ImageGenerationConfig(provider='vertex', vertex_project='p',
                                                vertex_credentials_path=str(adc)),
                          transport=httpx.MockTransport(handler), environ={}, repo_root=tmp_path)
        try:
            assert await auth.access_token() == 'ya29.adc'
        finally:
            await auth.close()

    asyncio.run(run())
    assert forms == [{'grant_type': 'refresh_token', 'client_id': 'cid',
                      'client_secret': 'secret', 'refresh_token': 'refresh'}]


def test_an_express_api_key_needs_no_project_and_travels_in_the_query(tmp_path):
    auth = VertexAuth(ImageGenerationConfig(provider='vertex', vertex_project=''),
                      environ={'VERTEX_API_KEY': 'express-key'}, repo_root=tmp_path)
    assert auth.ready and auth.missing_reason() == ''

    async def run():
        try:
            url, headers, params = await auth.authorization('gemini-3.1-flash-image')
            assert url == ('https://aiplatform.googleapis.com/v1/publishers/google/models/'
                           'gemini-3.1-flash-image:generateContent')
            assert headers == {} and params == {'key': 'express-key'}
            # An Express key replaces the token entirely; asking for one is a
            # mistake that says so instead of sending an unsigned request.
            with pytest.raises(VertexAuthError):
                await auth.access_token()
        finally:
            await auth.close()

    asyncio.run(run())


def test_a_static_access_token_is_used_as_it_is(tmp_path):
    auth = VertexAuth(ImageGenerationConfig(provider='vertex', vertex_project='rowan-images'),
                      environ={'VERTEX_ACCESS_TOKEN': 'ya29.manual'}, repo_root=tmp_path)
    assert auth.ready and auth.missing_reason() == ''

    async def run():
        try:
            assert (await auth.authorization('gemini-3.1-flash-image'))[1] == {
                'Authorization': 'Bearer ya29.manual'}
        finally:
            await auth.close()

    asyncio.run(run())


def test_a_missing_project_or_missing_file_says_exactly_what_is_missing(tmp_path):
    no_credentials = VertexAuth(ImageGenerationConfig(provider='vertex'),
                                environ={}, repo_root=tmp_path)
    assert not no_credentials.ready
    reason = no_credentials.missing_reason()
    assert 'data/vertex-credentials.json' in reason and 'VERTEX_API_KEY' in reason
    with pytest.raises(VertexAuthError, match='vertex_project'):
        no_credentials.model_url('gemini-3.1-flash-image')

    service_account(tmp_path)
    without_project = VertexAuth(ImageGenerationConfig(
        provider='vertex',
        vertex_credentials_path=str(tmp_path / 'vertex-credentials.json')),
                                 environ={}, repo_root=tmp_path)
    assert not without_project.ready
    assert 'vertex_project' in without_project.missing_reason()
    with pytest.raises(VertexAuthError, match='vertex_project'):
        without_project.model_url('gemini-3.1-flash-image')


def test_a_refused_credential_repeats_googles_own_sentence(tmp_path):
    service_account(tmp_path)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={'error': 'invalid_grant',
                                         'error_description': 'Invalid JWT Signature.'})

    async def run():
        auth = VertexAuth(ImageGenerationConfig(
            provider='vertex', vertex_project='p',
            vertex_credentials_path=str(tmp_path / 'vertex-credentials.json')),
            transport=httpx.MockTransport(handler), environ={}, repo_root=tmp_path)
        try:
            with pytest.raises(VertexAuthError) as failure:
                await auth.access_token()
            assert 'HTTP 400' in str(failure.value)
            assert 'Invalid JWT Signature.' in str(failure.value)
        finally:
            await auth.close()

    asyncio.run(run())


def test_a_broken_credential_file_names_the_file_and_not_its_contents(tmp_path):
    (tmp_path / 'vertex-credentials.json').write_text('{"type": "weird"}', encoding='utf-8')

    async def run():
        auth = VertexAuth(ImageGenerationConfig(
            provider='vertex', vertex_project='p',
            vertex_credentials_path=str(tmp_path / 'vertex-credentials.json')),
            environ={}, repo_root=tmp_path)
        try:
            with pytest.raises(VertexAuthError, match='not a Google credential file'):
                await auth.access_token()
        finally:
            await auth.close()

    asyncio.run(run())
