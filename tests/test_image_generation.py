import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from PIL import Image

from common.config import Config, ImageGenerationConfig, LLMConfig
from hub import app
from hub.api_budget import ApiBudget, BudgetExceeded, CloudUnavailable
from hub.image_generation import ImageGenerator, ImageStore, decode_image


def fixture_image():
    out = BytesIO()
    Image.new('RGB', (32, 24), 'green').save(out, 'PNG')
    return out.getvalue()


def response_image(**overrides):
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(fixture_image()).decode()}}]}}],
        'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 1120, 'thoughtsTokenCount': 50,
            'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': 1120}]}, **overrides}


def generator(tmp_path, monkeypatch, handler):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-key')
    return ImageGenerator(ImageGenerationConfig(enabled=True), ledger_path=tmp_path / 'usage.db',
                          monthly_usd=18, transport=httpx.MockTransport(handler))


def test_text_generation_and_lossless_followup_payload_metered_once(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        assert request.headers['x-goog-api-key'] == 'test-only-key'
        assert 'test-only-key' not in str(request.url) and b'test-only-key' not in request.content
        assert str(request.url) == 'https://generativelanguage.googleapis.com/v1/models/gemini-3.1-flash-image:generateContent'
        body = json.loads(request.content)
        assert 'safetySettings' not in body and 'tools' not in body
        assert body['generationConfig']['candidateCount'] == 1
        assert body['generationConfig']['imageConfig']['imageSize'] == '1K'
        calls.append(body)
        return httpx.Response(200, json=response_image())
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            result = await client.generate('A green square')
            assert (result.width, result.height) == (32, 24)
            assert len(calls[0]['contents'][0]['parts']) == 1
            assert client.budget.status()['accounted_usd'] == .0674
            await client.generate('Make the background blue', result.png, 'image/png')
            inline = calls[1]['contents'][0]['parts'][1]['inlineData']
            assert inline['mimeType'] == 'image/png'
            assert base64.b64decode(inline['data']) == result.png
            assert client.budget.status()['accounted_usd'] == .1348
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('data', [
    response_image(promptFeedback={'blockReason': 'SAFETY'}, candidates=[]),
    response_image(candidates=[{'finishReason': 'IMAGE_SAFETY'}]),
    response_image(candidates=[{'finishReason': 'STOP', 'content': {'parts': [{'text': 'I cannot edit this.'}]}}]),
    response_image(candidates=[{'finishReason': 'STOP', 'content': {'parts': [
        {'thought': True, 'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(fixture_image()).decode()}}]}}]),
    response_image(candidates=[{'finishReason': 'STOP', 'content': {'parts': [
        {'inlineData': {'mimeType': 'image/png', 'data': 'not base64'}}]}}]),
])
def test_refused_incomplete_text_only_thought_or_invalid_images_never_succeed(tmp_path, monkeypatch, data):
    handler = Mock(return_value=httpx.Response(200, json=data))
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            with pytest.raises(CloudUnavailable):
                await client.generate('Draw a square')
            assert handler.call_count == 1
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('usage,amount,pending', [
    ({}, .311296, 1),
    ({'promptTokenCount': 100, 'candidatesTokenCount': 1120, 'thoughtsTokenCount': 50}, .07025, 0),
    ({'promptTokenCount': 100, 'candidatesTokenCount': 1120,
      'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': 5}]}, .06725, 0),
    # Actual Nano Banana response: image details omit 390 output tokens.
    ({'promptTokenCount': 33, 'candidatesTokenCount': 1510, 'totalTokenCount': 1543,
      'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': 1120}]}, .090617, 0),
    ({'promptTokenCount': 100, 'candidatesTokenCount': 1120, 'thoughtsTokenCount': 50,
      'candidatesTokensDetails': [{'modality': 'TEXT', 'tokenCount': 100},
                                {'modality': 'IMAGE', 'tokenCount': 500}]}, .0617, 0),
    ({'promptTokenCount': 100, 'candidatesTokenCount': 1120,
      'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': 1121}]}, .311296, 1),
    ({'promptTokenCount': 100, 'candidatesTokenCount': 1120,
      'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': -1}]}, .311296, 1),
    ({'promptTokenCount': True, 'candidatesTokenCount': 1120}, .311296, 1),
])
def test_partial_usage_is_conservative_and_invalid_usage_keeps_reservation(tmp_path, monkeypatch, usage, amount, pending):
    async def run():
        client = generator(tmp_path, monkeypatch, lambda _: httpx.Response(200, json=response_image(usageMetadata=usage)))
        try:
            await client.generate('Draw a square')
            assert client.budget.status()['accounted_usd'] == amount
            assert client.budget.status()['unsettled_requests'] == pending
        finally:
            await client.close()
    asyncio.run(run())


def test_missing_key_and_exhausted_shared_budget_send_no_request(tmp_path, monkeypatch):
    handler = Mock()
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        monkeypatch.delenv('GEMINI_API_KEY')
        try:
            with pytest.raises(CloudUnavailable, match='key'):
                await client.generate('Draw a square')
            assert client.budget.status()['accounted_usd'] == 0
            monkeypatch.setenv('GEMINI_API_KEY', 'test-only-key')
            # Spending by chat counts against image generation's same allowance.
            chat = ApiBudget(tmp_path / 'usage.db', model='gpt-5.6-luna')
            chat.reserve(70_800_000, 0)  # conservative $17.70
            with pytest.raises(BudgetExceeded):
                await client.generate('Draw a square')
            handler.assert_not_called()
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('failure', [401, 403, 429, 500, 'timeout'])
def test_http_errors_and_timeouts_no_retries_retain_reservation(tmp_path, monkeypatch, failure):
    def handler(_):
        if failure == 'timeout':
            raise httpx.ReadTimeout('test timeout')
        return httpx.Response(failure)
    calls = Mock(side_effect=handler)
    async def run():
        client = generator(tmp_path, monkeypatch, calls)
        try:
            with pytest.raises(CloudUnavailable):
                await client.generate('Draw a square')
            assert calls.call_count == 1
            assert client.budget.status()['accounted_usd'] == .311296
            assert client.budget.status()['unsettled_requests'] == 1
        finally:
            await client.close()
    asyncio.run(run())


def test_cancel_stops_work_and_releases_busy_without_refunding(tmp_path, monkeypatch):
    async def run():
        started, forever = asyncio.Event(), asyncio.Event()
        async def handler(_):
            started.set()
            await forever.wait()
        client = generator(tmp_path, monkeypatch, handler)
        try:
            task = asyncio.create_task(client.generate('Draw a square'))
            await started.wait()
            with pytest.raises(CloudUnavailable, match='Another image'):
                await client.generate('Another square')
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not client._lock.locked()
            assert client.budget.status()['unsettled_requests'] == 1
        finally:
            await client.close()
    asyncio.run(run())


def test_store_is_personal_durable_and_does_not_overwrite(tmp_path):
    store = ImageStore(tmp_path)
    result = decode_image(fixture_image(), 'image/png')
    first = store.save('person:anton', result, 'gemini-3.1-flash-image')
    second = store.save('person:anton', result, 'gemini-3.1-flash-image')
    assert first != second and first.exists() and second.exists()
    restored = ImageStore(tmp_path)
    assert restored.last('person:anton')[0].png == result.png
    assert restored.last('person:theodric') is None
    restored.rename('Anton', 'Anthony')
    assert restored.last('person:anton') is None
    assert restored.last('person:anthony')[0].png == result.png


def connection():
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn._speaker_name, conn._speaker_role, conn._speaker_score = 'Anton', 'admin', .8
    conn.cfg.server.permissions_enabled = False
    conn._send_status = AsyncMock()
    conn._send_image_show = AsyncMock()
    conn._request_camera_frame_full = AsyncMock()
    conn._request_screenshot = AsyncMock()
    return conn


def test_tool_displays_saves_follows_up_and_does_not_mix_users_or_room_state(tmp_path, monkeypatch):
    async def run():
        client = generator(tmp_path, monkeypatch, lambda _: httpx.Response(200, json=response_image()))
        store = ImageStore(tmp_path / 'images')
        monkeypatch.setattr(app, '_image_generator', client)
        monkeypatch.setattr(app, '_generated_images', store)
        conn = connection()
        try:
            result = await conn._execute_tool('generate_image', {'prompt': 'Draw a green square', 'source': 'none'})
            assert result['ok'] and result['shown']
            conn._send_image_show.assert_awaited_once()
            conn._request_camera_frame_full.assert_not_awaited()
            assert conn._last_frames == {} and conn._last_annotated is None
            assert 'base64' not in json.dumps(conn._utterance_actions)
            assert not (await conn._execute_tool('generate_image', {'prompt': 'Again', 'source': 'last'}))['ok']
            before = client.budget.status()
            assert (await conn._run_show_photo({'which': 'generated'}))['ok']
            conn._run_client_action = AsyncMock(return_value={'ok': True, 'output': json.dumps({'saved': True, 'opened': True, 'path': 'Desktop/image.jpg'})})
            assert (await conn._run_save_photo({'source': 'generated'}))['opened']
            assert client.budget.status() == before
            saved = conn._run_client_action.call_args.args[1]
            assert base64.b64decode(saved['jpeg_base64']).startswith(b'\xff\xd8')
            conn._speaker_name = 'Theodric'
            assert not (await conn._run_show_photo({'which': 'generated'}))['ok']
            conn._image_generation_attempted = False
            assert not (await conn._run_generate_image({'prompt': 'Make it blue', 'source': 'last'}))['ok']
            assert client.budget.status() == before
            conn._speaker_name = 'Anton'
            conn._image_generation_attempted = False
            assert (await conn._run_generate_image({'prompt': 'Make it blue', 'source': 'last'}))['ok']
            assert client.budget.status()['accounted_usd'] > before['accounted_usd']
            assert connection()._anonymous_image_owner != conn._anonymous_image_owner
        finally:
            await client.close()
    asyncio.run(run())


def test_camera_requires_ready_provider_and_exact_frame_is_preserved(tmp_path, monkeypatch):
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response_image())
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        monkeypatch.setattr(app, '_image_generator', client)
        monkeypatch.setattr(app, '_generated_images', ImageStore(tmp_path / 'images'))
        conn = connection()
        try:
            monkeypatch.delenv('GEMINI_API_KEY')
            assert not (await conn._run_generate_image({'prompt': 'Make a comic', 'source': 'camera'}))['ok']
            conn._request_camera_frame_full.assert_not_awaited()
            monkeypatch.setenv('GEMINI_API_KEY', 'test-only-key')
            conn._image_generation_attempted = False
            assert not (await conn._run_generate_image({'prompt': 'Make a comic', 'source': 'camera', 'fresh': False}))['ok']
            conn._request_camera_frame_full.assert_not_awaited()
            conn._image_generation_attempted = False
            frame = SimpleNamespace(jpeg=decode_image(fixture_image(), 'image/png').jpeg)
            conn._last_frames['camera'] = frame
            assert (await conn._run_generate_image({'prompt': 'Make a comic', 'source': 'camera', 'fresh': False}))['ok']
            conn._request_camera_frame_full.assert_not_awaited()
            assert conn._last_frames['camera'] is frame
            assert sent[0]['contents'][0]['parts'][1]['inlineData']['mimeType'] == 'image/jpeg'
            conn._request_screenshot.assert_not_awaited()
        finally:
            await client.close()
    asyncio.run(run())


def test_budget_uses_original_image_rates_when_settled_by_text_client(tmp_path):
    images = ApiBudget(tmp_path / 'usage.db', model='gemini-3.1-flash-image')
    key = images.reserve(131072, 4096)
    text = ApiBudget(tmp_path / 'usage.db', model='gpt-5.6-luna')
    text.settle(key, 100, 1170, image_output_tokens=1120)
    assert images.status()['accounted_usd'] == .0674
    text.settle(key, 0, 0)
    assert images.status()['accounted_usd'] == .0674
    with pytest.raises(ValueError):
        LLMConfig(provider='openai_responses', model='gemini-3.1-flash-image')


def test_image_size_bounds_and_mime_mismatch():
    with pytest.raises(CloudUnavailable):
        decode_image(fixture_image(), 'image/jpeg')
    with pytest.raises(CloudUnavailable):
        decode_image(b'not an image', 'image/png')
    out = BytesIO()
    Image.new('RGB', (3000, 3000)).save(out, 'PNG')
    with pytest.raises(CloudUnavailable):
        decode_image(out.getvalue(), 'image/png')


def test_cancelled_tool_never_saves_or_shows_a_late_image(tmp_path, monkeypatch):
    async def run():
        started = asyncio.Event()
        async def handler(_):
            started.set()
            await asyncio.Event().wait()
        client = generator(tmp_path, monkeypatch, handler)
        store = ImageStore(tmp_path / 'images')
        monkeypatch.setattr(app, '_image_generator', client)
        monkeypatch.setattr(app, '_generated_images', store)
        conn = connection()
        try:
            task = asyncio.create_task(conn._run_generate_image({'source': 'none', 'prompt': 'Draw a square'}))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            conn._send_image_show.assert_not_awaited()
            assert store.last(conn._image_owner()) is None
            assert not list(store.folder.glob('*.png'))
        finally:
            await client.close()
    asyncio.run(run())


def test_provider_refusal_cannot_be_retried_by_model_or_display_an_old_image(tmp_path, monkeypatch):
    handler = Mock(return_value=httpx.Response(200, json=response_image(promptFeedback={'blockReason': 'SAFETY'}, candidates=[])))
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        monkeypatch.setattr(app, '_image_generator', client)
        store = ImageStore(tmp_path / 'images')
        monkeypatch.setattr(app, '_generated_images', store)
        conn = connection()
        original = store.save(conn._image_owner(), decode_image(fixture_image(), 'image/png'), client.cfg.model)
        try:
            result = await conn._run_generate_image({'source': 'last', 'prompt': 'A requested change'})
            assert not result['ok'] and 'declined' in result['error']
            result = await conn._run_generate_image({'source': 'last', 'prompt': 'Rephrased change'})
            assert not result['ok'] and 'one image' in result['error']
            assert handler.call_count == 1
            conn._send_image_show.assert_not_awaited()
            assert list(store.folder.glob('*.png')) == [original]
        finally:
            await client.close()
    asyncio.run(run())


def test_the_image_prompt_and_a_google_refusal_reach_the_owner_panel(tmp_path, monkeypatch):
    """The owner sees what went to Nano Banana and why Google said no."""
    from hub import migrations_runner, turn_trace
    conn = migrations_runner.connect(str(tmp_path / 'hub.db'))
    migrations_runner.migrate(conn)
    store = turn_trace.configure(conn)
    handler = Mock(return_value=httpx.Response(
        200, json=response_image(promptFeedback={'blockReason': 'PROHIBITED_CONTENT'}, candidates=[])))

    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            with turn_trace.turn('u-42', 'livingroom'):
                with pytest.raises(CloudUnavailable):
                    await client.generate('make them kiss')
        finally:
            await client.close()

    try:
        asyncio.run(run())
        steps = store.events('u-42')
    finally:
        turn_trace.configure(None)
    assert [step['kind'] for step in steps] == ['prompt', 'image']
    assert steps[0]['payload']['prompt'] == 'make them kiss'
    assert steps[0]['payload']['model'] == 'gemini-3.1-flash-image'
    assert steps[1]['ok'] is False
    assert steps[1]['payload']['reason'] == 'PROHIBITED_CONTENT'


def test_an_answer_with_no_picture_is_asked_once_more_and_the_second_one_is_used(tmp_path, monkeypatch):
    """The owner's "Отредактирцй фото чтобы он сидел на диване" heard "Google не смог".

    Google answers HTTP 200 with ``finishReason: NO_IMAGE`` and no text at all,
    and the SAME request produced the picture when it was repeated by hand a
    minute later. One repeat is therefore allowed - and only that one.
    """
    answers = [httpx.Response(200, json=response_image(candidates=[
                   {'finishReason': 'NO_IMAGE', 'content': {'parts': []}}])),
               httpx.Response(200, json=response_image())]
    handler = Mock(side_effect=answers)
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            result = await client.generate('Make him sit on the couch', fixture_image(), 'image/png')
            assert (result.width, result.height) == (32, 24)
            assert handler.call_count == 2
        finally:
            await client.close()
    asyncio.run(run())


def test_a_safety_answer_or_words_instead_of_a_picture_are_never_retried(tmp_path, monkeypatch):
    for data in (response_image(candidates=[{'finishReason': 'IMAGE_SAFETY'}]),
                 response_image(candidates=[{'finishReason': 'STOP', 'content': {
                     'parts': [{'text': 'I cannot edit this.'}]}}])):
        handler = Mock(return_value=httpx.Response(200, json=data))
        async def run(handler=handler):
            client = generator(tmp_path, monkeypatch, handler)
            try:
                with pytest.raises(CloudUnavailable) as failure:
                    await client.generate('Draw a square')
                assert handler.call_count == 1
                assert any(phrase in str(failure.value) for phrase in
                           ('without a picture', 'declined', 'returned no image'))
            finally:
                await client.close()
        asyncio.run(run())


def test_an_input_only_usage_report_does_not_hold_the_whole_reservation(tmp_path, monkeypatch):
    """A request with no picture still names its input tokens.

    Holding the whole reservation charged 131072 input tokens that were never
    sent: 18 such rows kept $3.92 of the owner's $18 monthly allowance, and
    image generation then looked broken although nothing was wrong with it.
    """
    handler = Mock(return_value=httpx.Response(200, json=response_image(
        candidates=[{'finishReason': 'STOP', 'content': {'parts': []}}],
        usageMetadata={'promptTokenCount': 100})))
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            with pytest.raises(CloudUnavailable):
                await client.generate('Draw a square')
            status = client.budget.status()
            assert status['unsettled_requests'] == 0
            # Exact input plus the FULL image output cap: never cheaper than the
            # real bill, and never the input limit that was never sent.
            assert status['accounted_usd'] == 0.24581
        finally:
            await client.close()
    asyncio.run(run())
