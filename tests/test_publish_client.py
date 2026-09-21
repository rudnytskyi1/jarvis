import pytest

from scripts.publish_client import release_path


@pytest.mark.parametrize('name', ['.git/config', '.GIT/config', '.git./config', '.GIT /config',
                                  '../config.yaml', 'C:/config.yaml', 'a\\b', '/root', ''])
def test_manifest_paths_cannot_target_git_or_escape_checkout(tmp_path, name):
    with pytest.raises(ValueError):
        release_path(tmp_path, name)


def test_release_path_allows_regular_source_path(tmp_path):
    assert release_path(tmp_path, 'client/main.py') == tmp_path / 'client' / 'main.py'
