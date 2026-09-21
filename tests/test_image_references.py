"""Explicit named Nano Banana references; all provider requests are mocked."""

import asyncio
import base64
import json
from io import BytesIO
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from common.config import ImageGenerationConfig
from hub import image_generation as module
from hub.api_budget import CloudUnavailable


def picture(color='blue', image_format='PNG'):
    output = BytesIO()
    Image.new('RGB', (24, 16), color).save(output, format=image_format)
    return output.getvalue()


def named(name='Anton', *, color='green', kind='face', **extra):
    return {'name': name, 'image': picture(color), 'mime': 'image/png', 'kind': kind, **extra}


def response():
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(picture()).decode()}}]}}],
        'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 1120,
                          'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': 1120}]}}


def generator(tmp_path, monkeypatch, handler):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-key')
    return module.ImageGenerator(ImageGenerationConfig(enabled=True),
        ledger_path=tmp_path / 'usage.db', monthly_usd=18, transport=httpx.MockTransport(handler))


def test_primary_scene_then_individually_named_face_and_body_references(tmp_path, monkeypatch):
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        primary = picture('red')
        refs = [named('Anton', color='green'), named('Anton', color='yellow', kind='body'),
                named('Theodric', color='purple'), named('Theodric', color='orange', kind='body')]
        try:
            result = await client.generate('Place Theodric beside Anton.', primary, 'image/png', references=refs)
            assert result.width == 24
            assert len(sent) == 1
            parts = sent[0]['contents'][0]['parts']
            assert parts[0] == {'text': 'Place Theodric beside Anton.'}
            assert base64.b64decode(parts[1]['inlineData']['data']) == module.decode_image(primary, 'image/png').png
            assert len(parts) == 10
            for index, reference in enumerate(refs):
                label, inline = parts[2 + index * 2], parts[3 + index * 2]['inlineData']
                assert json.loads(label['text']) == {'name': reference['name'], 'kind': reference['kind']}
                assert inline['mimeType'] == 'image/png'
                assert base64.b64decode(inline['data']) == module.decode_image(reference['image'], 'image/png').png
            assert sent[0]['generationConfig']['candidateCount'] == 1
            assert 'safetySettings' not in sent[0]
            assert client.budget.status()['unsettled_requests'] == 0
            assert client.budget.status()['accounted_usd'] == .06725
        finally:
            await client.close()
    asyncio.run(run())


def test_references_without_primary_and_normalized_names_count_as_two_people(tmp_path, monkeypatch):
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        try:
            await client.generate('A group portrait.', references=[named('  Anton  '), named('ANTON', kind='body'), named('Theodric')])
            parts = sent[0]['contents'][0]['parts']
            assert len(parts) == 7
            assert '"Anton"' in parts[1]['text']
            assert 'inlineData' in parts[2]
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('references', [
    'not a sequence of records',
    {'name': 'Anton'},
    [None],
    [{}],
    [named(name='')],
    [named(name=' ')],
    [named(name='A' * 81)],
    [named(name='Anton\nSecond label')],
    [named(kind='unknown')],
    [named(image='C:/private/face.jpg')],
    [named(image=b'')],
    [named(mime='text/plain')],
    [named(mime='image/jpeg')],  # actual PNG, not JPEG
    [named(image=b'not an image')],
    [named(), named(image=b'bad final image')],
    [named('A'), named('B'), named('C')],
    [named()] * 5,
])
def test_invalid_references_never_reserve_budget_or_send_request(tmp_path, monkeypatch, references):
    handler = Mock()
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        reserve = Mock(wraps=client.budget.reserve)
        client.budget.reserve = reserve
        try:
            with pytest.raises(CloudUnavailable):
                await client.generate('A portrait.', picture(), 'image/png', references=references)
            handler.assert_not_called()
            reserve.assert_not_called()
            assert client.budget.status()['accounted_usd'] == 0
        finally:
            await client.close()
    asyncio.run(run())


def test_combined_raw_bytes_include_primary_and_named_images(tmp_path, monkeypatch):
    primary, ref = picture('red'), named()
    monkeypatch.setattr(module, 'MAX_REFERENCE_INPUT_BYTES', len(primary) + len(ref['image']) - 1)
    handler = Mock()
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            with pytest.raises(CloudUnavailable, match='combined'):
                await client.generate('A portrait.', primary, 'image/png', references=[ref])
            handler.assert_not_called()
            assert client.budget.status()['accounted_usd'] == 0
        finally:
            await client.close()
    asyncio.run(run())


def test_decoding_expansion_is_bounded_before_billing(tmp_path, monkeypatch):
    raw = picture()
    monkeypatch.setattr(module, 'MAX_REFERENCE_INPUT_BYTES', len(raw) * 3)
    monkeypatch.setattr(module, 'decode_image', lambda *_: module.GeneratedImage(raw * 2, raw * 2, 24, 16))
    handler = Mock()
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            with pytest.raises(CloudUnavailable, match='decoded'):
                await client.generate('A portrait.', raw, 'image/png', references=[named(image=raw)])
            handler.assert_not_called()
            assert client.budget.status()['accounted_usd'] == 0
        finally:
            await client.close()
    asyncio.run(run())


def test_single_reference_call_remains_backward_compatible(tmp_path, monkeypatch):
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        jpeg = picture('red', 'JPEG')
        try:
            await client.generate('A portrait.', jpeg)
            await client.generate('A portrait.', jpeg, references=[])
            assert sent[0] == sent[1]
            parts = sent[0]['contents'][0]['parts']
            assert len(parts) == 2
            assert parts[1]['inlineData']['mimeType'] == 'image/jpeg'
        finally:
            await client.close()
    asyncio.run(run())


def test_named_reference_labels_escape_quotes_as_data(tmp_path, monkeypatch):
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        try:
            name = 'Антон "AJ"'
            await client.generate('A portrait.', references=[named(name)])
            assert json.dumps(name, ensure_ascii=False) in sent[0]['contents'][0]['parts'][1]['text']
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['plain', 'camera', 'named_references'])
def test_user_prompt_reaches_http_exactly_without_creative_additions(tmp_path, monkeypatch, mode):
    sent = []
    prompt = '  \tPut a hat on my head.\n  '
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        try:
            scene = picture('red') if mode != 'plain' else None
            refs = [named('Антон "AJ"'), named('Theodric', kind='body')] if mode == 'named_references' else None
            await client.generate(prompt, scene, 'image/png', references=refs)
            assert len(sent) == 1
            payload = sent[0]
            assert set(payload) == {'contents', 'generationConfig'}
            assert len(payload['contents']) == 1
            message = payload['contents'][0]
            assert set(message) == {'role', 'parts'} and message['role'] == 'user'
            expected_text = [prompt]
            if refs:
                expected_text.extend(json.dumps({'name': ref['name'], 'kind': ref['kind']},
                                               ensure_ascii=False, separators=(',', ':')) for ref in refs)
            assert [part['text'] for part in message['parts'] if 'text' in part] == expected_text
            assert sum('inlineData' in part for part in message['parts']) == (0 if mode == 'plain' else 1 + len(refs or []))
            # Provider configuration controls I/O only: no hidden prompt or
            # style instruction may replace the exact user text above.
            assert set(payload['generationConfig']) == {
                'candidateCount', 'maxOutputTokens', 'responseModalities', 'imageConfig', 'thinkingConfig'}
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('prompt', ['', ' ', '\n\t  '])
def test_blank_prompt_is_rejected_before_provider_or_billing(tmp_path, monkeypatch, prompt):
    handle = Mock()
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        try:
            with pytest.raises(CloudUnavailable, match='nonempty'):
                await client.generate(prompt)
            handle.assert_not_called()
            assert client.budget.status()['accounted_usd'] == 0
        finally:
            await client.close()
    asyncio.run(run())


def test_scene_subject_is_technical_metadata_immediately_before_primary_image(tmp_path, monkeypatch):
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        subject = {'name': 'Антон "AJ"', 'face_box': [.1, .2, .4, .5], 'is_requester': True}
        prompt = '\tput a hat on my head\n'
        primary = picture('red')
        try:
            await client.generate(prompt, primary, 'image/png', references=[named('John')], scene_subject=subject)
            parts = sent[0]['contents'][0]['parts']
            assert parts[0] == {'text': prompt}
            assert parts[1] == {'text': json.dumps(subject, ensure_ascii=False, separators=(',', ':'))}
            assert base64.b64decode(parts[2]['inlineData']['data']) == module.decode_image(primary, 'image/png').png
            assert json.loads(parts[3]['text']) == {'name': 'John', 'kind': 'face'}
            assert 'inlineData' in parts[4]
            assert len(parts) == 5
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('subject', [
    'Anton', {}, {'name': 'Anton', 'face_box': [.1, .2, .3, .4]},
    {'name': 'Anton', 'face_box': [.1, .2, .3, .4], 'is_requester': True, 'style': 'emoji'},
    *({'name': name, 'face_box': [.1, .2, .3, .4], 'is_requester': True}
      for name in ['', ' ', 'Anton\nextra text', 'A' * 81, 1]),
    *({'name': 'Anton', 'face_box': box, 'is_requester': True}
      for box in [[.1, .2, .3], '0,0,1,1', [0, 0, True, 1], [0, 0, '1', 1],
                  [0, 0, float('nan'), 1], [0, 0, float('inf'), 1], [-.1, 0, 1, 1],
                  [0, 0, 1.1, 1], [.5, .2, .5, .4], [.5, .2, .4, .4], [.1, .5, .3, .4]]),
    *({'name': 'Anton', 'face_box': [.1, .2, .3, .4], 'is_requester': requester}
      for requester in [1, 'true', None]),
])
def test_invalid_scene_subject_is_rejected_before_provider_or_billing(tmp_path, monkeypatch, subject):
    handle = Mock()
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        try:
            with pytest.raises(CloudUnavailable, match='Scene subject'):
                await client.generate('put a hat on my head', picture(), 'image/png', scene_subject=subject)
            handle.assert_not_called()
            assert client.budget.status()['accounted_usd'] == 0
        finally:
            await client.close()
    asyncio.run(run())


def test_scene_subject_without_primary_image_is_rejected(tmp_path, monkeypatch):
    handle = Mock()
    async def run():
        client = generator(tmp_path, monkeypatch, handle)
        try:
            with pytest.raises(CloudUnavailable, match='primary image'):
                await client.generate('put a hat on my head', references=[named()],
                    scene_subject={'name': 'Anton', 'face_box': [.1, .2, .3, .4], 'is_requester': True})
            handle.assert_not_called()
            assert client.budget.status()['accounted_usd'] == 0
        finally:
            await client.close()
    asyncio.run(run())


def test_provider_refusal_with_references_is_not_retried(tmp_path, monkeypatch):
    handler = Mock(return_value=httpx.Response(200, json={
        **response(), 'promptFeedback': {'blockReason': 'SAFETY'}, 'candidates': []}))
    async def run():
        client = generator(tmp_path, monkeypatch, handler)
        try:
            with pytest.raises(CloudUnavailable, match='declined'):
                await client.generate('A portrait.', picture(), 'image/png', references=[named()])
            assert handler.call_count == 1
            assert client.budget.status()['accounted_usd'] == .06725
        finally:
            await client.close()
    asyncio.run(run())
