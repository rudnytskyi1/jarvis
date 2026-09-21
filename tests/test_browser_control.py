import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from client.actions.browser import BrowserController, web_url
from hub.speaker import check_permission
from hub.tools import actions_from_tool_calls


@pytest.mark.parametrize('url', ['file:///secret', 'javascript:alert(1)', 'data:text/html,x', 'https://user:secret@example.com', 'chrome://settings', 'example.com'])
def test_browser_rejects_non_web_or_credential_urls(url):
    with pytest.raises(ValueError):
        web_url(url)


def test_browser_preserves_authorization_and_client_routing():
    assert check_permission('unknown', 'browser_control', {'command': 'read'})
    assert check_permission('user', 'browser_control', {'command': 'read'})
    assert check_permission('trusted', 'browser_control', {'command': 'read'}) is None
    items = actions_from_tool_calls([{'name': 'browser_control', 'arguments': {'command': 'navigate', 'url': 'https://example.com'}}])
    assert items[0]['tool'] == 'browser_control'


def test_stale_ref_never_clicks():
    async def scenario():
        browser = BrowserController()
        page = Mock(url='https://example.com/new', bring_to_front=AsyncMock())
        browser._page = page
        browser._ensure = AsyncMock()
        element = Mock(click=AsyncMock())
        browser._refs = {'1:0': (page, 'https://example.com/old', element, {})}
        with pytest.raises(ValueError, match='Stale'):
            await browser.execute({'command': 'click', 'ref': '1:0'})
        element.click.assert_not_called()
    asyncio.run(scenario())


def test_cancel_closes_pending_browser_actions():
    async def scenario():
        browser = BrowserController()
        started = asyncio.Event()
        async def navigate(*a, **kw):
            started.set()
            await asyncio.sleep(10)
        browser._page = Mock(goto=navigate, bring_to_front=AsyncMock())
        browser._ensure = AsyncMock()
        browser.close = AsyncMock()
        task = asyncio.create_task(browser.execute({'command': 'navigate', 'url': 'https://example.com'}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        browser.close.assert_awaited_once()
    asyncio.run(scenario())
