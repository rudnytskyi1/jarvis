import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from client.actions import browser_desktop
from client.actions.browser_desktop import DesktopBrowserController, _ordinary_process, _WindowsUIA


def test_dispatcher_uses_existing_desktop_browser_in_production():
    from client.actions.dispatcher import Dispatcher
    async def run():
        dispatcher = Dispatcher({}, None)
        try:
            assert isinstance(dispatcher.browser, DesktopBrowserController)
        finally:
            await dispatcher.browser.close()
    asyncio.run(run())


class FakeDesktop:
    def __init__(self):
        self.inventory = [{'hwnd': 10, 'pid': 20, 'name': 'Google Chrome',
                           'image': 'chrome.exe', 'title': 'YouTube'}]
        self.active = 10
        self.url = 'https://www.youtube.com'
        self.signature = ((42,), 'input', 'Search', False, (10, 10, 100, 40))
        self.events = []
        self.released = False
        self.element = object()
        #: Windows whose accessibility tree holds no page document at all.
        self.pageless = set()

    def windows(self):
        return self.inventory

    def foreground(self):
        return self.active

    def focus(self, window, stop):
        self.active = window['hwnd']

    def page(self, window, stop):
        if window['hwnd'] in self.pageless:
            return {'key': ((), '', window['title']), 'url': '', 'caption': window['title'],
                    'typed': '', 'blank': True, 'title': window['title'],
                    'document': None, 'root': object()}
        return {'key': ((1,), self.url, 'YouTube'), 'url': self.url, 'typed': '',
                'caption': 'YouTube - Google Chrome', 'blank': False, 'title': 'YouTube',
                'document': object(), 'root': object()}

    def elements(self, page, stop):
        if page.get('document') is None:
            return [], ''
        return [(self.element, {'role': 'input', 'text': 'Search', 'disabled': False,
            'bounds': (10, 10, 100, 40), 'signature': self.signature})], 'Videos'

    def validate(self, element, signature):
        if signature != self.signature:
            raise ValueError('The element changed. Read the page again.')

    def act(self, window, page, command, element, args, stop):
        self.events.append((command, element, args))

    def navigate(self, window, url, stop):
        self.events.append(('navigate', url))
        self.url = url

    def close(self):
        self.released = True


def test_reuses_existing_browser_and_close_only_releases_backend():
    async def run():
        backend = FakeDesktop()
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        data = json.loads(await browser.execute({'command': 'navigate', 'url': 'https://example.com'}))
        assert data['url'] == 'https://example.com'
        assert backend.events == [('navigate', 'https://example.com')]
        await browser.close()
        assert backend.released
        assert backend.inventory[0]['hwnd'] == 10
    asyncio.run(run())


def test_a_typed_address_that_never_loads_is_a_failure_not_a_success(monkeypatch):
    """Chrome keeps the typed text in the omnibox when Enter did nothing.

    The live hub read that text back as "the URL" and reported a New Tab page as
    an open YouTube, so the room heard a success that never happened.
    """
    monkeypatch.setattr(browser_desktop, 'NAVIGATE_WAIT_S', 0.2)

    async def run():
        backend = FakeDesktop()
        backend.url = ''
        backend.page = lambda window, stop: {
            'key': ((7,), '', 'New Tab'), 'url': '',
            'typed': 'youtube.com/results?search_query=MrBeast', 'blank': True,
            'title': 'New Tab', 'document': object(), 'root': object()}
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        with pytest.raises(ValueError, match='did not open youtube.com'):
            await browser.execute({'command': 'navigate',
                                   'url': 'https://www.youtube.com/results?search_query=MrBeast'})
        await browser.close()
    asyncio.run(run())


def test_a_caption_change_is_not_proof_the_address_loaded(monkeypatch):
    """Only a loaded document proves the address opened.

    Chrome re-creates its accessibility tree while a page loads, so the window
    element changes on every read. The live report (VE-01/VE-02 in
    ``data/live-eval/last.json``) came back as a success with ``url: ""`` and
    nothing but a window caption - "YouTube - Google Chrome", or "Search icon"
    for the browser's own new tab: the caption was read as a page that loaded,
    and the room heard "YouTube is open" about a window that was still showing
    what it had before.
    """
    monkeypatch.setattr(browser_desktop, 'NAVIGATE_WAIT_S', 0.2)

    async def run():
        backend = FakeDesktop()
        backend.url = 'https://example.com/'
        backend.navigate = lambda window, url, stop: backend.events.append(('navigate', url))
        reads = []

        def churning(window, stop):
            reads.append(1)
            return {'key': ((len(reads),), backend.url, 'Example Domain'),
                    'url': backend.url, 'typed': 'youtube.com', 'blank': False,
                    'caption': f'draft {len(reads)} - Google Chrome',
                    'title': 'Example Domain', 'document': object(), 'root': object()}

        backend.page = churning
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        with pytest.raises(ValueError, match='did not open youtube.com'):
            await browser.execute({'command': 'navigate', 'url': 'https://www.youtube.com'})
        assert len(reads) > 1  # the address was given every chance to load
        await browser.close()
    asyncio.run(run())


def test_a_window_that_shows_no_page_is_not_reported_as_a_loading_page():
    """A window with no document is a failure, not a page to wait for.

    VE-07...VE-11 answered the room from a Chrome window whose accessibility
    tree had no document at all (no url, no elements, caption "Search icon")
    while still reporting ``ok``; the model then told the owner "the page isn't
    loaded" and gave up on ``type MrBeast in the search box and press enter``.
    """
    async def run():
        backend = FakeDesktop()
        backend.inventory[0]['title'] = 'Search icon'
        backend.pageless = {10}
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        with pytest.raises(ValueError, match='not showing a readable page'):
            await browser.execute({'command': 'read'})
        with pytest.raises(ValueError, match='not showing a readable page'):
            await browser.execute({'command': 'scroll', 'direction': 'down'})
        assert backend.events == []  # nothing was scrolled in that window
        await browser.close()
    asyncio.run(run())


def test_the_window_that_really_shows_a_page_is_used_instead():
    """The page can be open next to a page-less window of the same browser."""
    async def run():
        backend = FakeDesktop()
        backend.inventory.insert(0, {'hwnd': 11, 'pid': 20, 'name': 'Google Chrome',
                                     'image': 'chrome.exe', 'title': 'Search icon'})
        backend.pageless = {11}
        backend.active = 11  # the page-less window is in the foreground
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        data = json.loads(await browser.execute({'command': 'read'}))
        assert data['url'] == 'https://www.youtube.com'
        assert data['title'] == 'YouTube' and data['elements']
        assert backend.active == 10  # the window that really shows the page
        await browser.close()
    asyncio.run(run())


def test_a_transient_com_hiccup_is_retried_but_a_real_error_is_not():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError(
                "COMError: (-2147220991, 'An event was unable to invoke any of the subscribers')")
        return 'page'

    assert browser_desktop.attempt(flaky, attempts=3, pause=0) == 'page'
    assert len(calls) == 3
    with pytest.raises(ValueError, match='no such element'):
        browser_desktop.attempt(lambda: (_ for _ in ()).throw(ValueError('no such element')),
                                attempts=3, pause=0)


@pytest.mark.parametrize('change', ['url', 'signature', 'window_pid', 'read_again'])
def test_stale_or_replaced_ref_cannot_trigger_a_click(change):
    async def run():
        backend = FakeDesktop()
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        data = json.loads(await browser.execute({'command': 'read'}))
        ref = data['elements'][0]['ref']
        if change == 'url':
            backend.url += '/results'
        elif change == 'signature':
            backend.signature = ((99,), *backend.signature[1:])
        elif change == 'window_pid':
            backend.inventory[0]['pid'] += 1
        else:
            await browser.execute({'command': 'read'})
        with pytest.raises(ValueError, match='Read the page again'):
            await browser.execute({'command': 'click', 'ref': ref})
        assert backend.events == []
        await browser.close()
    asyncio.run(run())


def test_press_without_ref_keeps_current_browser_control_and_fill_accepts_submit():
    async def run():
        backend = FakeDesktop()
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        data = json.loads(await browser.execute({'command': 'press', 'key': 'Enter'}))
        assert backend.events[0][0:2] == ('press', None)
        await browser.execute({'command': 'fill', 'ref': data['elements'][0]['ref'],
                               'text': 'MrBeast', 'submit': True})
        assert backend.events[1][0] == 'fill'
        assert backend.events[1][2]['submit'] is True
        await browser.close()
    asyncio.run(run())


def test_ambiguous_browser_does_not_type_into_foreground_other_app():
    async def run():
        backend = FakeDesktop()
        backend.inventory.append({'hwnd': 30, 'pid': 40, 'name': 'Microsoft Edge', 'title': 'Other'})
        backend.active = 999  # An unrelated foreground application.
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        data = json.loads(await browser.execute({'command': 'press', 'key': 'Enter'}))
        assert data['needs_choice'] is True
        assert backend.events == [] and backend.active == 999
        await browser.execute({'command': 'press', 'key': 'Enter',
                               'window_ref': data['choices'][0]['window_ref']})
        assert backend.active == 10
        assert backend.events[0][0] == 'press'
        await browser.close()
    asyncio.run(run())


def test_multiple_browser_apps_require_choice_even_when_one_is_foreground():
    async def run():
        backend = FakeDesktop()
        backend.inventory.append({'hwnd': 30, 'pid': 40, 'name': 'Microsoft Edge', 'title': 'Other'})
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        data = json.loads(await browser.execute({'command': 'press', 'key': 'Enter'}))
        assert data['needs_choice'] is True and backend.events == []
        await browser.close()
    asyncio.run(run())


def test_navigate_to_an_address_picks_a_window_instead_of_asking_which_one():
    """Two Chrome windows are open and the owner says "open YouTube".

    The live bench heard "Which Chrome window should I use: Example Domain or
    Outlook?" - an explicit address does not depend on the page that is already
    open, so the tool must take a window and load the site.
    """
    async def run():
        backend = FakeDesktop()
        backend.inventory.append({'hwnd': 30, 'pid': 40, 'name': 'Google Chrome',
                                  'image': 'chrome.exe', 'title': 'Outlook'})
        backend.active = 999  # The foreground application is not one of them.
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        data = json.loads(await browser.execute(
            {'command': 'navigate', 'url': 'https://www.youtube.com'}))
        assert data.get('needs_choice') is None
        assert backend.events == [('navigate', 'https://www.youtube.com')]
        assert backend.active == 10
        await browser.close()
    asyncio.run(run())


def test_no_browser_requires_existing_app_selection_instead_of_launching_another_profile():
    async def run():
        backend = FakeDesktop()
        backend.inventory = []
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        with pytest.raises(ValueError, match='No matching ordinary browser'):
            await browser.execute({'command': 'read'})
        assert backend.events == []
        with pytest.raises(ValueError, match='Use open_app'):
            await browser.execute({'command': 'navigate', 'browser': 'Google Chrome', 'url': 'https://example.com'})
        assert backend.events == []
        await browser.close()
    asyncio.run(run())


def test_cancel_waits_for_worker_stop_without_closing_user_browser():
    async def run():
        backend = FakeDesktop()
        started, stopped = threading.Event(), threading.Event()
        def pending_navigation(window, url, stop):
            started.set()
            stop.wait(5)
            stopped.set()
            raise InterruptedError('cancelled')
        backend.navigate = pending_navigation
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        task = asyncio.create_task(browser.execute({'command': 'navigate', 'url': 'https://example.com'}))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set() and not backend.released
        assert backend.inventory[0]['hwnd'] == 10
        await browser.close()
    asyncio.run(run())


@pytest.mark.parametrize('command_line,ordinary', [
    (['chrome.exe', '--user-data-dir=C:\\work\\data\\rowan-browser'], False),
    (['chrome.exe', '--user-data-dir', 'C:/work/data/rowan-browser/'], False),
    (['chrome.exe', '--user-data-dir=C:\\Users\\Anton\\My Chrome'], True),
    (['chrome.exe', 'https://example.com/rowan-browser'], True),
    (['chrome.exe'], True),
])
def test_only_old_rowan_profile_is_excluded(command_line, ordinary):
    assert _ordinary_process(command_line) is ordinary


def test_backend_refuses_input_after_another_app_takes_focus():
    backend = object.__new__(_WindowsUIA)
    backend.verify_window = Mock()
    backend.foreground = lambda: 999
    with pytest.raises(ValueError, match='Browser focus changed'):
        backend.guard({'hwnd': 10, 'pid': 20}, threading.Event())


def test_real_backend_fill_submits_in_same_worker_action_and_press_preserves_focus():
    backend = object.__new__(_WindowsUIA)
    backend.guard = Mock()
    backend._belongs = lambda element, root, stop: True
    field = SimpleNamespace(CurrentControlType=50004, CurrentIsEnabled=True,
                            CurrentIsPassword=False, SetFocus=Mock())
    backend._uia = SimpleNamespace(GetFocusedElement=lambda: field,
        CompareElements=lambda first, second: first is second)
    events = []
    backend.key = lambda window, key, stop: events.append(key)
    backend._type = lambda window, text, stop: events.append(('type', text))
    window, page, stop = {'hwnd': 10, 'pid': 20}, {'root': object()}, threading.Event()
    backend.act(window, page, 'fill', field, {'text': 'MrBeast', 'submit': True}, stop)
    assert events == ['ctrl+a', ('type', 'MrBeast'), 'enter']
    field.SetFocus.reset_mock()
    backend.act(window, page, 'press', None, {'key': 'Enter'}, stop)
    field.SetFocus.assert_not_called()
    assert events[-1] == 'enter'


def test_password_controls_are_omitted_without_reading_name_or_value():
    protected = SimpleNamespace(CurrentIsPassword=True)
    assert _WindowsUIA._info(protected) is None


def test_async_browser_focus_notifications_settle_before_typing():
    backend = object.__new__(_WindowsUIA)
    backend.guard = Mock()
    field = SimpleNamespace(CurrentIsPassword=False)
    other = SimpleNamespace(CurrentIsPassword=False)
    backend._uia = SimpleNamespace(GetFocusedElement=Mock(side_effect=[other, field]))
    backend._belongs = lambda element, target, stop: element is target
    assert backend._focused_within({}, field, threading.Event()) is field


def test_navigation_rejects_control_characters_before_desktop_input():
    async def run():
        backend = FakeDesktop()
        browser = DesktopBrowserController(backend_factory=lambda: backend)
        with pytest.raises(ValueError, match='control characters'):
            await browser.execute({'command': 'navigate', 'url': 'https://example.com/\nwrong'})
        assert not backend.events
        await browser.close()
    asyncio.run(run())
