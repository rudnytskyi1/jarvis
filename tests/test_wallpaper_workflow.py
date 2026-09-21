import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from common.config import Config
from hub import app
from hub.image_generation import ImageStore, decode_image


def setup(monkeypatch, tmp_path):
    buffer = BytesIO()
    Image.new('RGB', (48, 32), 'blue').save(buffer, 'PNG')
    picture = decode_image(buffer.getvalue(), 'image/png')
    generator = SimpleNamespace(check_ready=Mock(), generate=AsyncMock(return_value=picture),
                                cfg=SimpleNamespace(model='test', timeout_s=10))
    store = ImageStore(tmp_path / 'images')
    monkeypatch.setattr(app, '_image_generator', generator)
    monkeypatch.setattr(app, '_generated_images', store)
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.cfg.server.permissions_enabled = False
    conn._speaker_name, conn._speaker_role = 'Anton', 'admin'
    conn._send_status = AsyncMock()
    conn._send_image_show = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={'ok': True, 'output': json.dumps(
        {'applied': True, 'verified': True, 'path': 'room-local/wallpaper.png'})})
    return conn, generator, store, picture


def test_generation_installs_pixels_and_reports_verified_room_result(monkeypatch, tmp_path):
    async def run():
        conn, generator, store, picture = setup(monkeypatch, tmp_path)
        result = await conn._execute_tool('generate_image',
            {'source': 'none', 'prompt': 'A red and blue superhero.', 'target': 'wallpaper'})
        assert result['ok'] and result['wallpaper']['applied'] and result['wallpaper']['verified']
        assert result['saved_on_client'] is True
        assert 'saved_on_brain' not in result and str(tmp_path) not in json.dumps(result)
        tool, args = conn._run_client_action.call_args.args
        assert tool == 'set_wallpaper_file'
        assert base64.b64decode(args['image_base64']) == picture.png
        assert generator.generate.call_args.args[0] == 'A red and blue superhero.'
        assert generator.generate.await_count == 1
    asyncio.run(run())


def test_wallpaper_failure_keeps_generated_result_and_retry_is_free(monkeypatch, tmp_path):
    async def run():
        conn, generator, store, picture = setup(monkeypatch, tmp_path)
        conn._run_client_action.return_value = {'ok': False, 'error': 'Windows refused'}
        result = await conn._run_generate_image({'source': 'none', 'prompt': 'Blue sky', 'target': 'wallpaper'})
        assert not result['ok'] and result['generated'] and result['shown']
        assert not result['wallpaper']['verified'] and not result['saved_on_client']
        conn._run_client_action.return_value = {'ok': True, 'output': json.dumps(
            {'applied': True, 'verified': True, 'path': 'room-local/wallpaper.png'})}
        assert (await conn._execute_tool('set_wallpaper', {'source': 'generated'}))['ok']
        assert generator.generate.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize('output', ['True', '[]', '{}', 'invalid',
    '{"applied":true,"verified":false,"path":"x"}',
    '{"applied":"true","verified":true,"path":"x"}',
    '{"applied":true,"verified":true}'])
def test_unverified_acknowledgement_never_claims_wallpaper_success(monkeypatch, tmp_path, output):
    async def run():
        conn, _, store, picture = setup(monkeypatch, tmp_path)
        store.save(conn._image_owner(), picture, 'test')
        conn._run_client_action.return_value = {'ok': True, 'output': output}
        result = await conn._run_set_wallpaper({'source': 'generated'})
        assert not result['ok'] and not result['verified']
    asyncio.run(run())


def test_missing_picture_never_uses_another_person_or_runs_action(monkeypatch, tmp_path):
    async def run():
        conn, _, store, picture = setup(monkeypatch, tmp_path)
        store.save('person:theodric', picture, 'test')
        assert not (await conn._run_set_wallpaper({'source': 'generated'}))['ok']
        conn._run_client_action.assert_not_awaited()
    asyncio.run(run())


def test_shown_guest_picture_survives_next_voice_match_only_in_open_room(monkeypatch, tmp_path):
    async def run():
        conn, _, store, picture = setup(monkeypatch, tmp_path)
        old = BytesIO()
        Image.new('RGB', (48, 32), 'red').save(old, 'PNG')
        older = decode_image(old.getvalue(), 'image/png')
        store.save('person:anton', older, 'test')
        conn._speaker_name = 'unknown'
        assert (await conn._run_generate_image({'source': 'none', 'prompt': 'Blue sky'}))['ok']
        conn._speaker_name = 'Anton'
        assert (await conn._run_set_wallpaper({'source': 'generated'}))['ok']
        assert base64.b64decode(conn._run_client_action.call_args.args[1]['image_base64']) == picture.png
        conn.cfg.server.permissions_enabled = True
        assert (await conn._run_set_wallpaper({'source': 'generated'}))['ok']
        assert base64.b64decode(conn._run_client_action.call_args.args[1]['image_base64']) == older.png
    asyncio.run(run())


def test_failed_edit_cannot_apply_save_or_show_an_older_image(monkeypatch, tmp_path):
    async def run():
        from hub.api_budget import CloudUnavailable
        conn, generator, store, picture = setup(monkeypatch, tmp_path)
        store.save(conn._image_owner(), picture, 'test')
        generator.generate.side_effect = CloudUnavailable('Provider unavailable')
        assert not (await conn._run_generate_image({'source': 'last', 'prompt': 'Make it red'}))['ok']
        assert not (await conn._run_set_wallpaper({'source': 'generated'}))['ok']
        assert not (await conn._run_save_photo({'source': 'generated'}))['ok']
        assert not (await conn._run_show_photo({'which': 'generated'}))['ok']
        conn._run_client_action.assert_not_awaited()
        conn._send_image_show.assert_not_awaited()
        # A separate new turn may explicitly use the older saved image.
        conn._image_generation_attempted = False
        assert (await conn._run_set_wallpaper({'source': 'generated'}))['ok']
    asyncio.run(run())


def test_existing_frame_is_used_without_new_capture(monkeypatch, tmp_path):
    async def run():
        conn, _, _, picture = setup(monkeypatch, tmp_path)
        conn._last_frames['camera'] = SimpleNamespace(jpeg=picture.jpeg)
        conn._request_camera_frame_full = AsyncMock()
        assert (await conn._run_set_wallpaper({'source': 'camera'}))['ok']
        conn._request_camera_frame_full.assert_not_awaited()
        assert base64.b64decode(conn._run_client_action.call_args.args[1]['image_base64']) == picture.jpeg
    asyncio.run(run())


def test_photo_saved_but_not_opened_is_partial_failure(monkeypatch, tmp_path):
    async def run():
        conn, _, store, picture = setup(monkeypatch, tmp_path)
        store.save(conn._image_owner(), picture, 'test')
        conn._run_client_action.return_value = {'ok': True, 'output': json.dumps(
            {'saved': True, 'opened': False, 'path': 'Desktop/image.jpg'})}
        result = await conn._run_save_photo({'source': 'generated', 'open': True})
        assert result['saved'] and not result['ok'] and not result['opened']
        assert (await conn._run_save_photo({'source': 'generated', 'open': False}))['ok']
    asyncio.run(run())
