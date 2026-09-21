import asyncio
import json
import logging
from io import BytesIO
from urllib import error

import pytest
from PIL import Image
from pydantic import ValidationError

from common.config import Config, TelegramConfig
from hub import telegram
from hub.telegram import TelegramError, TelegramProvider

TOKEN = '123456:' + 'synthetic_test_token_' * 2
CHAT = -1234567890


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', TOKEN)
    # An accidentally uninjected transport must never send a real test message.
    monkeypatch.setattr(telegram.request, 'build_opener', lambda *a, **k: pytest.fail('Network transport was not mocked'))


def cfg(**kwargs):
    return TelegramConfig(enabled=True, chat_id=CHAT, **kwargs)


def jpeg(size=(32, 24), format='JPEG'):
    result = BytesIO()
    Image.new('RGB', size, 'red').save(result, format=format)
    return result.getvalue()


class Transport:
    def __init__(self, response=None, failure=None):
        self.calls = []
        self.failure = failure
        self.response = response or {'ok': True, 'result': {'message_id': 7, 'chat': {'id': CHAT}}}

    async def __call__(self, method, payload, upload):
        self.calls.append((method, payload, upload))
        if self.failure is not None:
            raise self.failure
        return self.response


def test_defaults_disable_telegram_and_config_forbids_secret_or_arbitrary_recipient():
    assert Config().server.telegram == TelegramConfig()
    assert not TelegramProvider(Config().server.telegram).ready
    for value in (123, 0, '@someone', True):
        with pytest.raises(ValidationError):
            TelegramConfig(chat_id=value)
    with pytest.raises(ValidationError):
        TelegramConfig(bot_token='never put tokens in YAML')
    with pytest.raises(ValidationError):
        TelegramConfig(api_key_env='SECRET;bad')


def test_missing_or_malformed_token_never_makes_request(monkeypatch):
    for value in ('', 'a path / leaking_token', '123:x'):
        monkeypatch.setenv('TELEGRAM_BOT_TOKEN', value)
        transport = Transport()
        provider = TelegramProvider(cfg(), transport=transport)
        assert not provider.ready
        with pytest.raises(TelegramError):
            asyncio.run(provider.send_text('hello'))
        assert not transport.calls


def test_fixed_group_text_preserves_unicode_and_does_not_enable_parse_mode():
    transport = Transport()
    provider = TelegramProvider(cfg(), transport=transport)
    assert provider.ready
    result = asyncio.run(provider.send_text('Привет *everyone* <hello> 👋'))
    assert result == {'ok': True, 'chat_id': CHAT, 'message_id': 7, 'kind': 'text'}
    method, payload, upload = transport.calls[0]
    assert method == 'sendMessage'
    assert payload == {'chat_id': CHAT, 'text': 'Привет *everyone* <hello> 👋'}
    assert upload is None
    with pytest.raises(TypeError):
        provider.send_text('hello', chat_id=-999)


@pytest.mark.parametrize('text', ['', '  ', None, 'x' * 4097, '😀' * 2049, '\ud800'])
def test_oversized_or_invalid_text_is_rejected_without_splitting(text):
    transport = Transport()
    with pytest.raises(TelegramError):
        asyncio.run(TelegramProvider(cfg(), transport=transport).send_text(text))
    assert not transport.calls


def test_jpeg_sent_as_photo_with_original_bytes_and_safe_filename():
    data = jpeg()
    transport = Transport()
    result = asyncio.run(TelegramProvider(cfg(), transport=transport).send_image(
        data, 'image/jpeg', 'Picture *as requested*', '../../image"\r\nHeader: bad.png'))
    method, payload, upload = transport.calls[0]
    assert method == 'sendPhoto'
    assert payload == {'chat_id': CHAT, 'caption': 'Picture *as requested*'}
    assert upload.field == result['kind'] == 'photo'
    assert upload.data == data and upload.mime == 'image/jpeg'
    assert upload.filename.endswith('.jpg')
    assert all(c not in upload.filename for c in '\r\n"/')


@pytest.mark.parametrize('size,format,mime', [
    ((100, 2), 'JPEG', 'image/jpeg'),
    ((6000, 4010), 'PNG', 'image/png'),
    ((30, 30), 'WEBP', 'image/webp'),
])
def test_dimensions_or_format_choose_document_before_any_upload(size, format, mime):
    data = jpeg(size, format)
    transport = Transport()
    result = asyncio.run(TelegramProvider(cfg(), transport=transport).send_image(data, mime))
    assert result['kind'] == 'document'
    assert len(transport.calls) == 1 and transport.calls[0][0] == 'sendDocument'
    assert transport.calls[0][2].data == data


def test_large_photo_chooses_document_without_recompressing(monkeypatch):
    data = jpeg()
    monkeypatch.setattr(telegram, 'MAX_PHOTO_BYTES', len(data) - 1)
    transport = Transport()
    asyncio.run(TelegramProvider(cfg(), transport=transport).send_image(data, 'image/jpeg'))
    assert transport.calls[0][0] == 'sendDocument'
    assert transport.calls[0][2].data == data


@pytest.mark.parametrize('data,mime,caption', [
    (b'', 'image/png', ''), (b'not a photo', 'image/jpeg', ''),
    (jpeg(), 'image/png', ''), (jpeg(), 'text/html', ''),
    (jpeg(), 'image/jpeg', 'x' * 1025), (jpeg(), 'image/jpeg', '😀' * 513),
])
def test_invalid_images_and_caption_fail_before_upload(data, mime, caption):
    transport = Transport()
    with pytest.raises(TelegramError):
        asyncio.run(TelegramProvider(cfg(), transport=transport).send_image(data, mime, caption))
    assert not transport.calls


@pytest.mark.parametrize('response', [
    {'ok': False, 'error_code': 429, 'description': 'Too Many Requests', 'parameters': {'retry_after': 12}},
    {'ok': False, 'error_code': 403, 'description': 'Forbidden'},
    {'ok': False, 'error_code': 500, 'description': 'Server error'},
    {'ok': True, 'result': {'message_id': 8, 'chat': {'id': -999}}},
    {'ok': True, 'result': {}}, {'ok': 'true', 'result': {}},
])
def test_failed_or_unconfirmed_sends_never_retry(response):
    transport = Transport(response=response)
    with pytest.raises(TelegramError) as caught:
        asyncio.run(TelegramProvider(cfg(), transport=transport).send_image(jpeg(), 'image/jpeg'))
    assert len(transport.calls) == 1 and transport.calls[0][0] == 'sendPhoto'
    if response.get('error_code') == 429:
        assert caught.value.retry_after == 12
        assert not caught.value.uncertain
    elif response.get('error_code') == 500:
        assert caught.value.uncertain


def test_api_error_and_network_exception_never_expose_token_in_logs_or_text(caplog):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    caplog.set_level(logging.DEBUG)
    failures = [Transport(failure=OSError(url)),
                Transport(response={'ok': False, 'error_code': 400, 'description': f'Bad request {url} {TOKEN}'}),
                Transport(failure=TelegramError(url))]
    for transport in failures:
        with pytest.raises(TelegramError) as caught:
            asyncio.run(TelegramProvider(cfg(), transport=transport).send_text('hello'))
        assert TOKEN not in str(caught.value)
        assert url not in str(caught.value)
        assert len(transport.calls) == 1
    assert TOKEN not in caplog.text


def test_migration_is_reported_without_changing_destination_or_retry():
    transport = Transport(response={'ok': False, 'error_code': 400, 'description': 'migrated',
                                   'parameters': {'migrate_to_chat_id': -999999, 'retry_after': True}})
    with pytest.raises(TelegramError) as caught:
        asyncio.run(TelegramProvider(cfg(), transport=transport).send_text('hello'))
    assert caught.value.retry_after is None
    assert 'No destination was changed' in str(caught.value)
    assert len(transport.calls) == 1 and transport.calls[0][1]['chat_id'] == CHAT


def test_check_connection_only_reads_identity_and_group():
    calls = []
    async def transport(method, payload, upload):
        calls.append((method, payload, upload))
        return {'ok': True, 'result': {'id': 77, 'is_bot': True, 'username': 'testbot', 'first_name': 'Rowan'}
                if method == 'getMe' else {'id': CHAT, 'type': 'group', 'title': 'Friends', 'description': 'not returned'}}
    result = asyncio.run(TelegramProvider(cfg(), transport=transport).check_connection())
    assert [r[0] for r in calls] == ['getMe', 'getChat']
    assert calls[1][1] == {'chat_id': CHAT}
    assert result == {'ok': True, 'bot': {'id': 77, 'username': 'testbot', 'first_name': 'Rowan'},
                      'chat': {'id': CHAT, 'type': 'group', 'title': 'Friends'}}


def test_default_transport_uses_https_and_no_redirects_without_httpx_logging(monkeypatch):
    requests = []
    class Reply(BytesIO):
        status = 200
    class Opener:
        def open(self, req, timeout):
            requests.append(req)
            assert timeout == 30
            return Reply(json.dumps({'ok': True, 'result': {'message_id': 9, 'chat': {'id': CHAT}}}).encode())
    def build(*handlers):
        assert any(isinstance(handler, telegram._NoRedirect) for handler in handlers)
        assert any(isinstance(handler, telegram.request.HTTPSHandler) for handler in handlers)
        return Opener()
    monkeypatch.setattr(telegram.request, 'build_opener', build)
    result = asyncio.run(TelegramProvider(cfg()).send_text('hello'))
    assert result['message_id'] == 9
    assert requests[0].full_url.startswith('https://api.telegram.org/bot')
    assert json.loads(requests[0].data) == {'chat_id': CHAT, 'text': 'hello'}
    assert telegram._NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example') is None


def test_default_transport_parses_http_errors_safely(monkeypatch):
    class Opener:
        def open(self, req, timeout):
            body = json.dumps({'ok': False, 'error_code': 401, 'description': 'Unauthorized'}).encode()
            raise error.HTTPError(req.full_url, 401, req.full_url, {}, BytesIO(body))
    monkeypatch.setattr(telegram.request, 'build_opener', lambda *a: Opener())
    with pytest.raises(TelegramError) as caught:
        asyncio.run(TelegramProvider(cfg()).send_text('hello'))
    assert caught.value.code == 401
    assert TOKEN not in str(caught.value)


def test_multipart_preserves_binary_image_and_fixed_fields():
    upload = telegram.TelegramUpload('photo', 'image.jpg', 'image/jpeg', jpeg())
    body, content_type = telegram._multipart({'chat_id': CHAT, 'caption': 'Привет'}, upload)
    assert content_type.startswith('multipart/form-data; boundary=')
    assert upload.data in body and 'Привет'.encode() in body
    assert b'name="photo"; filename="image.jpg"' in body
    assert b'parse_mode' not in body


def test_cancellation_propagates_without_retry():
    transport = Transport(failure=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(TelegramProvider(cfg(), transport=transport).send_text('hello'))
    assert len(transport.calls) == 1


def test_polling_passes_explicit_offset_and_never_sends_a_message():
    updates = [{'update_id': 91, 'message': {'chat': {'id': CHAT}, 'text': '@rowan hi'}}]
    transport = Transport(response={'ok': True, 'result': updates})
    received = asyncio.run(TelegramProvider(cfg(), transport=transport).get_updates(offset=90, timeout=25))
    assert received == updates
    assert transport.calls == [('getUpdates', {'timeout': 25, 'limit': 100,
                                             'allowed_updates': ['message', 'callback_query'], 'offset': 90}, None)]


@pytest.mark.parametrize('result', [{}, ['bad update'], None])
def test_polling_rejects_invalid_updates_shape(result):
    transport = Transport(response={'ok': True, 'result': result})
    with pytest.raises(TelegramError):
        asyncio.run(TelegramProvider(cfg(), transport=transport).get_updates())
    assert len(transport.calls) == 1


def test_webhook_check_redacts_url_and_never_deletes_it():
    transport = Transport(response={'ok': True, 'result': {
        'url': 'https://hooks.example/secret-value', 'pending_update_count': 7}})
    result = asyncio.run(TelegramProvider(cfg(), transport=transport).get_webhook_info())
    assert result == {'url': '[configured webhook]', 'pending_update_count': 7}
    assert transport.calls == [('getWebhookInfo', {}, None)]


def test_reply_targets_message_in_fixed_group_for_text_and_photo():
    transport = Transport()
    provider = TelegramProvider(cfg(), transport=transport)
    asyncio.run(provider.send_text('hello', reply_to_message_id=42))
    asyncio.run(provider.send_image(jpeg(), 'image/jpeg', reply_to_message_id=43))
    for call, message in zip(transport.calls, [42, 43]):
        payload = call[1]
        assert payload['chat_id'] == CHAT
        assert payload['reply_parameters'] == {'message_id': message, 'allow_sending_without_reply': True}
    body, _ = telegram._multipart(transport.calls[1][1], transport.calls[1][2])
    assert b'{"message_id": 43, "allow_sending_without_reply": true}' in body


def test_photo_download_uses_only_telegram_file_metadata_and_validates_image(monkeypatch):
    raw = jpeg()
    transport = Transport(response={'ok': True, 'result': {'file_path': 'photos/file_7.jpg', 'file_size': len(raw)}})
    provider = TelegramProvider(cfg(), transport=transport)
    seen = []
    def download(file_path, max_bytes):
        seen.append((file_path, max_bytes))
        return raw
    monkeypatch.setattr(provider, '_download_sync', download)
    assert asyncio.run(provider.download_photo('telegram_file_id')) == (raw, 'image/jpeg')
    assert transport.calls == [('getFile', {'file_id': 'telegram_file_id'}, None)]
    assert seen == [('photos/file_7.jpg', 8_000_000)]


@pytest.mark.parametrize('metadata', [
    {'file_path': 'https://other.example/photo.jpg'},
    {'file_path': '/absolute/photo.jpg'},
    {'file_path': '../secrets.jpg'},
    {'file_path': 'photos/../../file.jpg'},
    {'file_path': 'photos/file.jpg', 'file_size': 8_000_001},
])
def test_photo_download_rejects_external_paths_and_large_files_before_network(metadata, monkeypatch):
    transport = Transport(response={'ok': True, 'result': metadata})
    provider = TelegramProvider(cfg(), transport=transport)
    monkeypatch.setattr(provider, '_download_sync', lambda *a: pytest.fail('Unsafe download started'))
    with pytest.raises(TelegramError):
        asyncio.run(provider.download_photo('file_id'))


def test_download_errors_sanitize_token_bearing_url(monkeypatch):
    transport = Transport(response={'ok': True, 'result': {'file_path': 'photos/file.jpg'}})
    provider = TelegramProvider(cfg(), transport=transport)
    def fail(*args):
        raise OSError(f'https://api.telegram.org/file/bot{TOKEN}/photos/file.jpg')
    monkeypatch.setattr(provider, '_download_sync', fail)
    with pytest.raises(TelegramError) as caught:
        asyncio.run(provider.download_photo('file_id'))
    assert TOKEN not in str(caught.value)
