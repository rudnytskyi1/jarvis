"""The speech-to-image boundary must not trust a model's creative rewrite."""
import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from common.config import Config
from hub import app
from hub.image_generation import ImageStore, decode_image


def setup(tmp_path, monkeypatch):
    data = BytesIO()
    Image.new('RGB', (32, 24), 'blue').save(data, 'PNG')
    picture = decode_image(data.getvalue(), 'image/png')
    provider = SimpleNamespace(check_ready=Mock(), generate=AsyncMock(return_value=picture),
                               cfg=SimpleNamespace(model='test', timeout_s=5))
    monkeypatch.setattr(app, '_image_generator', provider)
    monkeypatch.setattr(app, '_generated_images', ImageStore(tmp_path / 'images'))
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn._speaker_name, conn._speaker_role, conn._speaker_score = 'Anton', 'admin', .8
    conn.cfg.server.permissions_enabled = False
    conn._send_status = AsyncMock()
    conn._send_image_show = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={'ok': True, 'output': json.dumps(
        {'applied': True, 'verified': True, 'path': 'room/wallpaper.png'})})
    conn._request_camera_frame_full = AsyncMock(return_value=SimpleNamespace(jpeg=picture.jpeg))
    return conn, provider, picture


def spoken(conn, text, args):
    async def run():
        token = app._recording_turn.set({'transcript': text})
        try:
            return await conn._run_generate_image(args)
        finally:
            app._recording_turn.reset(token)
    return asyncio.run(run())


@pytest.mark.parametrize('model_prompt', [
    'Add a surreal cartoon head-shaped prop, harmless and family-friendly.',
    'Turn him into a cinematic red-blue superhero with a purple emoji.',
    'Draw a desktop wallpaper mockup with space for icons.',
])
def test_original_hat_request_overrides_every_model_rewrite_and_wallpaper_suffix(tmp_path, monkeypatch, model_prompt):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    text = ('Rowan, can you take a picture and make me look like a Spider-Man '
            'and also put a hat on my head and make that picture a background picture on this computer?')
    result = spoken(conn, text, {'source': 'camera', 'prompt': model_prompt, 'target': 'wallpaper'})
    expected = 'make me look like a Spider-Man and also put a hat on my head'
    assert result['ok'] and result['wallpaper']['verified']
    assert provider.generate.call_args.args[0] == expected
    assert conn._utterance_actions[0]['submitted_prompt'] == expected


def test_negative_constraints_survive_model_omission(tmp_path, monkeypatch):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    text = 'Put a hat on my head, not an emoji. Do not change the background.'
    result = spoken(conn, text, {'source': 'none', 'prompt': 'Put an emoji on his head.'})
    assert result['ok']
    assert provider.generate.call_args.args[0] == text


def test_hallucinated_wallpaper_target_only_displays_requested_skyscraper(tmp_path, monkeypatch):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    result = spoken(conn, 'Hey Rowan, can you take a picture and make me stand on a skyscraper right now?',
        {'source': 'camera', 'prompt': 'Stand on a skyscraper', 'target': 'wallpaper'})
    assert result['ok'] and result['shown'] and not result['saved_on_client']
    assert 'wallpaper' not in result
    conn._run_client_action.assert_not_awaited()
    assert conn._utterance_actions[0]['ignored_unrequested_wallpaper']


def test_direct_wallpaper_tool_is_guarded_by_current_request(tmp_path, monkeypatch):
    conn, _, picture = setup(tmp_path, monkeypatch)
    async def run():
        token = app._recording_turn.set({'transcript': 'Make me a superhero'})
        try:
            result = await conn._apply_wallpaper(picture.png)
            assert not result['ok'] and not result['applied']
            conn._run_client_action.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_unavailable_original_speech_stops_before_capture_or_api(tmp_path, monkeypatch):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    result = spoken(conn, '', {'source': 'camera', 'prompt': 'Invented instructions'})
    assert not result['ok']
    conn._request_camera_frame_full.assert_not_awaited()
    provider.generate.assert_not_awaited()


def test_exact_current_face_metadata_stays_outside_literal_artwork(tmp_path, monkeypatch):
    conn, provider, picture = setup(tmp_path, monkeypatch)
    box = [.2, .1, .3, .25]
    conn._camera_frame_people = AsyncMock(return_value={'faces_in_frame': [{'name': 'Anton', 'face_box': box}]})
    result = spoken(conn, 'Put a hat on my head.',
        {'source': 'camera', 'prompt': 'Purple hat, comedic style', 'target_person': 'me'})
    assert result['ok']
    assert provider.generate.call_args.args[0] == 'Put a hat on my head.'
    assert provider.generate.call_args.args[1] == picture.jpeg
    assert provider.generate.call_args.kwargs == {'scene_subject': {'name': 'Anton', 'face_box': box, 'is_requester': True}}


def test_model_cannot_add_unrequested_person_reference(tmp_path, monkeypatch):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'John': [[1, 0]]}))
    conn.gallery = Mock()
    result = spoken(conn, 'Put a hat on my head.',
        {'source': 'camera', 'prompt': 'Add John too', 'reference_people': ['John']})
    assert not result['ok'] and 'not requested' in result['error']
    conn.gallery.references.assert_not_called()
    conn._request_camera_frame_full.assert_not_awaited()
    provider.generate.assert_not_awaited()


def test_draw_me_a_cat_does_not_authorize_uploading_speakers_portrait(tmp_path, monkeypatch):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'Anton': [[1, 0]]}))
    conn.gallery = Mock()
    result = spoken(conn, 'draw me a cat',
        {'source': 'none', 'prompt': 'Draw Anton as a cat', 'reference_people': ['Anton']})
    assert not result['ok'] and 'not requested' in result['error']
    conn.gallery.references.assert_not_called()
    conn._request_camera_frame_full.assert_not_awaited()
    provider.generate.assert_not_awaited()


def test_named_references_do_not_append_prose(tmp_path, monkeypatch):
    conn, provider, picture = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'John': [[1, 0]]}))
    conn.gallery = Mock()
    conn.gallery.references.return_value = [{'jpeg': picture.jpeg, 'kind': 'face', 'captured_at': 1, 'sample_id': 'j1'}]
    result = spoken(conn, 'Put John next to me.',
        {'source': 'camera', 'prompt': 'Put John on the left in a cinematic scene', 'reference_people': ['John']})
    assert result['ok']
    assert provider.generate.call_args.args[0] == 'Put John next to me.'
    assert provider.generate.call_args.kwargs['references'][0]['name'] == 'John'


def test_absent_requested_person_gets_saved_face_and_body_automatically(tmp_path, monkeypatch):
    conn, provider, picture = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'Anton': [[1, 0]]}))
    conn.gallery = Mock()
    conn.gallery.references.return_value = [
        {'jpeg': picture.jpeg, 'kind': kind, 'captured_at': 1, 'sample_id': kind}
        for kind in ('face', 'body')]
    conn._camera_frame_people = AsyncMock(return_value={'faces_in_frame': [
        {'name': None, 'face_box': [.2, .2, .4, .4]}]})
    result = spoken(conn, 'Put Anton next to the person on the couch.',
                    {'source': 'camera', 'prompt': 'Rewritten', 'target_person': 'Anton'})
    assert result['ok']
    assert provider.generate.call_args.args[0] == 'Put Anton next to the person on the couch.'
    assert provider.generate.call_args.args[1] == picture.jpeg
    kwargs = provider.generate.call_args.kwargs
    assert 'scene_subject' not in kwargs  # The bystander is never labeled Anton.
    assert [(row['name'], row['kind']) for row in kwargs['references']] == [('Anton', 'face'), ('Anton', 'body')]


@pytest.mark.parametrize('spoken_request', ['Set it as wallpaper.', 'Send it to Telegram.',
    'Rowan, could you save this picture please?', 'Роуэн, поставь это на обои.'])
def test_existing_picture_workflow_cannot_regenerate(tmp_path, monkeypatch, spoken_request):
    conn, provider, picture = setup(tmp_path, monkeypatch)
    app._generated_images.save(conn._image_owner(), picture, 'test')
    result = spoken(conn, spoken_request, {'source': 'last', 'prompt': 'A new picture', 'target': 'wallpaper'})
    assert not result['ok']
    provider.generate.assert_not_awaited()
    conn._request_camera_frame_full.assert_not_awaited()
    assert conn._latest_generated() is not None


@pytest.mark.parametrize('spoken_request', ["Put a hat on me, don't add John.",
    'Put a hat on me without John.', 'Put John next to me, actually do not add John.',
    'Надень на меня шляпу, не добавляй John.'])
def test_negated_name_never_uploads_saved_portraits(tmp_path, monkeypatch, spoken_request):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'John': [[1, 0]]}))
    conn.gallery = Mock()
    result = spoken(conn, spoken_request, {'source': 'camera', 'prompt': 'Add John', 'reference_people': ['John']})
    assert not result['ok'] and 'not requested' in result['error']
    conn.gallery.references.assert_not_called()
    provider.generate.assert_not_awaited()


def test_negated_target_name_never_selects_face(tmp_path, monkeypatch):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    conn._camera_frame_people = AsyncMock()
    result = spoken(conn, "Put a hat on me, don't change John.",
                    {'source': 'camera', 'prompt': 'Hat', 'target_person': 'John'})
    assert not result['ok']
    conn._camera_frame_people.assert_not_awaited()
    provider.generate.assert_not_awaited()


@pytest.mark.parametrize('text', ["Set it as wallpaper, actually don't.", 'Set it as wallpaper. Cancel that.'])
def test_cancelled_wallpaper_never_reaches_pc(tmp_path, monkeypatch, text):
    conn, _, picture = setup(tmp_path, monkeypatch)
    async def run():
        token = app._recording_turn.set({'transcript': text})
        try:
            assert not (await conn._apply_wallpaper(picture.png))['ok']
            conn._run_client_action.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_clarification_retains_only_same_person_user_words(tmp_path, monkeypatch):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    original = 'Put a hat on my head, not an emoji.'
    conn._remember_image_clarification(original, 'Which person in the photo?')
    result = spoken(conn, 'The person on the left.', {'source': 'camera', 'prompt': 'A purple cartoon hat on the left person'})
    assert result['ok']
    assert provider.generate.call_args.args[0] == original + '\nThe person on the left.'
    assert 'Which person' not in provider.generate.call_args.args[0]


@pytest.mark.parametrize('reason', ['other_person', 'expired', 'unrelated'])
def test_old_or_other_person_request_never_reappears(tmp_path, monkeypatch, reason):
    conn, provider, _ = setup(tmp_path, monkeypatch)
    conn._remember_image_clarification('Put a hat on my head.', 'Which person?')
    if reason == 'other_person':
        conn._speaker_name = 'John'
    elif reason == 'expired':
        conn._pending_image_wording['person:anton']['at'] -= 181
    else:
        conn._remember_image_clarification('What time is it?', 'It is noon.')
    result = spoken(conn, 'Draw a tree.', {'source': 'none', 'prompt': 'Draw a tree with a hat'})
    assert result['ok']
    assert provider.generate.call_args.args[0] == 'Draw a tree.'
