"""Primary-scene aspect selection; provider calls are always mocked."""
import asyncio
import base64
import json
from io import BytesIO

import httpx
import pytest
from PIL import Image

from common.config import ImageGenerationConfig
from hub.image_generation import ImageGenerator, decode_image


def picture(size, color='green', *, orientation=None):
    output = BytesIO()
    image = Image.new('RGB', size, color)
    if orientation is None:
        image.save(output, 'PNG')
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(output, 'PNG', exif=exif)
    return output.getvalue()


def request_payload(tmp_path, monkeypatch, primary=None, *, references=None, prompt='Add Anton beside him.'):
    sent = []
    output = picture((16, 9))

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
                {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(output).decode()}}]}}],
            'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 1120,
                              'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': 1120}]}})

    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-key')

    async def run():
        generator = ImageGenerator(ImageGenerationConfig(enabled=True),
            ledger_path=tmp_path / 'usage.sqlite3', monthly_usd=18, transport=httpx.MockTransport(handle))
        try:
            await generator.generate(prompt, primary, 'image/png', references=references)
            assert generator.budget.status()['unsettled_requests'] == 0
        finally:
            await generator.close()

    asyncio.run(run())
    assert len(sent) == 1
    return sent[0]


def test_portrait_face_and_body_references_cannot_replace_landscape_scene_geometry(tmp_path, monkeypatch):
    # Dimensions from the reported camera edit, which previously returned
    # 688x1537: the same aspect as its final 375x838 body-reference crop.
    primary = picture((1920, 1080), 'red')
    refs = [
        {'name': 'Anton', 'kind': 'face', 'mime': 'image/png', 'image': picture((102, 137))},
        {'name': 'Anton', 'kind': 'body', 'mime': 'image/png', 'image': picture((375, 838), 'blue')},
    ]
    prompt = '  Can you make Anton sit next to him and display this photo on the screen?\n'
    payload = request_payload(tmp_path, monkeypatch, primary, references=refs, prompt=prompt)
    assert payload['generationConfig']['imageConfig'] == {'imageSize': '1K', 'aspectRatio': '16:9'}
    parts = payload['contents'][0]['parts']
    assert parts[0] == {'text': prompt}
    images = [base64.b64decode(part['inlineData']['data']) for part in parts if 'inlineData' in part]
    assert images == [decode_image(raw, 'image/png').png for raw in [primary, *(ref['image'] for ref in refs)]]
    assert [json.loads(part['text']) for part in parts[1:] if 'text' in part] == [
        {'name': 'Anton', 'kind': 'face'}, {'name': 'Anton', 'kind': 'body'}]


@pytest.mark.parametrize('size,expected', [((1080, 1920), '9:16'), ((1200, 1200), '1:1'),
                                        ((1600, 1200), '4:3'), ((1200, 1600), '3:4'),
                                        ((1365, 768), '16:9'), ((400, 1600), '1:4')])
def test_scene_dimensions_select_nearest_supported_ratio(tmp_path, monkeypatch, size, expected):
    payload = request_payload(tmp_path, monkeypatch, picture(size))
    assert payload['generationConfig']['imageConfig']['aspectRatio'] == expected


def test_primary_exif_orientation_is_applied_before_aspect_selection(tmp_path, monkeypatch):
    payload = request_payload(tmp_path, monkeypatch, picture((1600, 900), orientation=6))
    assert payload['generationConfig']['imageConfig']['aspectRatio'] == '9:16'


@pytest.mark.parametrize('references', [None, [
    {'name': 'Anton', 'kind': 'face', 'mime': 'image/png', 'image': picture((100, 200))}]])
def test_absent_primary_does_not_promote_identity_crop_to_scene(tmp_path, monkeypatch, references):
    payload = request_payload(tmp_path, monkeypatch, references=references)
    assert payload['generationConfig']['imageConfig'] == {'imageSize': '1K'}


def test_explicit_supported_ratio_overrides_default_without_rewriting_prompt(tmp_path, monkeypatch):
    prompt = 'Add Anton beside him and change the aspect ratio to 9:16.'
    payload = request_payload(tmp_path, monkeypatch, picture((1920, 1080)), prompt=prompt)
    assert payload['generationConfig']['imageConfig']['aspectRatio'] == '9:16'
    assert payload['contents'][0]['parts'][0] == {'text': prompt}


@pytest.mark.parametrize('prompt', ['Add Anton, but do not change the photo to 9:16.',
                                   'Add the caption "9:16" beside Anton.',
                                   'Добавь Антона, не делай формат 9:16.'])
def test_negated_and_caption_ratios_do_not_override_scene(tmp_path, monkeypatch, prompt):
    payload = request_payload(tmp_path, monkeypatch, picture((1920, 1080)), prompt=prompt)
    assert payload['generationConfig']['imageConfig']['aspectRatio'] == '16:9'
    assert payload['contents'][0]['parts'][0] == {'text': prompt}


def test_multiple_format_mentions_do_not_force_a_conflicting_default(tmp_path, monkeypatch):
    prompt = 'Change this 16:9 image into 9:16.'
    payload = request_payload(tmp_path, monkeypatch, picture((1920, 1080)), prompt=prompt)
    assert 'aspectRatio' not in payload['generationConfig']['imageConfig']
    assert payload['contents'][0]['parts'][0] == {'text': prompt}


@pytest.mark.parametrize('prompt', [
    'Make this image portrait.', 'Make this image vertical.', 'Make this image square.',
    'Crop the photo to a square.', 'Turn it into landscape.', 'Use portrait orientation.',
    'Make a square image with Anton.', 'Сделай это фото вертикальным.',
    'Сделай изображение квадратным.', 'Используй портретный формат.',
])
def test_explicit_worded_output_format_is_not_overridden_by_primary_geometry(tmp_path, monkeypatch, prompt):
    payload = request_payload(tmp_path, monkeypatch, picture((1920, 1080)), prompt=prompt)
    assert 'aspectRatio' not in payload['generationConfig']['imageConfig']
    assert payload['contents'][0]['parts'][0] == {'text': prompt}


@pytest.mark.parametrize('prompt', [
    'Make a portrait of Anton.', 'Draw a portrait of the person sitting on the couch.',
    'Make this image a portrait of Anton.', 'Add a square on the wall.',
    'Do not make this image vertical.', 'Keep the background, do not crop the photo to a square.',
    'Add the caption "make this image vertical".', 'Нарисуй портрет Антона.',
    'Не сделай фото вертикальным.',
])
def test_subject_geometry_negations_and_captions_keep_primary_aspect(tmp_path, monkeypatch, prompt):
    payload = request_payload(tmp_path, monkeypatch, picture((1920, 1080)), prompt=prompt)
    assert payload['generationConfig']['imageConfig']['aspectRatio'] == '16:9'
    assert payload['contents'][0]['parts'][0] == {'text': prompt}
