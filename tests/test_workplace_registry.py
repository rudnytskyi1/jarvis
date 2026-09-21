from concurrent.futures import ThreadPoolExecutor

from hub.telegram_admin_state import TelegramAdminState


def test_simultaneous_hellos_preserve_all_workplaces_after_restart(tmp_path):
    path = tmp_path / 'admin.sqlite'
    state = TelegramAdminState(path, 123)
    def register(index):
        state.update_mapping_setting('workplaces', str(index), {'name': f'Room {index}'})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(register, range(32)))
    expected = {str(index): {'name': f'Room {index}'} for index in range(32)}
    assert state.get_setting('workplaces') == expected
    assert TelegramAdminState(path, 123).get_setting('workplaces') == expected
