"""Publish the audited, byte-identical client release without private Git history."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_client import ROOT, RUNTIME_FILES, export_client

REPOSITORY = 'https://github.com/rudnytskyi1/rowanai.git'


def git(folder, *args):
    result = subprocess.run(['git', '-C', str(folder), *args], check=True,
                            capture_output=True, text=True, encoding='utf-8')
    return result.stdout.strip()


def release_path(root, name):
    relative = PurePosixPath(name)
    if (not relative.parts or relative.is_absolute() or '..' in relative.parts
            or '\\' in name or ':' in name
            or any(part.casefold().rstrip(' .') == '.git' or part.rstrip(' .') != part
                   for part in relative.parts)):
        raise ValueError('Invalid path in release manifest')
    target = root.joinpath(*relative.parts)
    for component in (target, *target.parents):
        if component == root:
            break
        if component.is_symlink() or (component.exists() and getattr(component.lstat(), 'st_file_attributes', 0)
                                     & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
            raise ValueError('Linked paths cannot be published')
    resolved = target.resolve()
    if (not resolved.is_relative_to(root.resolve())
            or any(part.casefold().rstrip(' .') == '.git'
                   for part in resolved.relative_to(root.resolve()).parts)):
        raise ValueError('Release path escapes the checkout')
    return target


def publish(*, push=False, message='Update shared Rowan client release'):
    folder = ROOT / 'data' / 'releases' / ('publish-' + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    release, archive, count = export_client(ROOT, folder / 'release')
    checkout = folder / 'repository'
    subprocess.run(['git', 'clone', '-c', 'core.autocrlf=false', REPOSITORY, str(checkout)], check=True)
    tracked = set(filter(None, git(checkout, 'ls-files').splitlines()))
    previous = checkout / 'release-manifest.json'
    old_files = json.loads(previous.read_text(encoding='utf-8'))['files'] if previous.exists() else {}
    if tracked - set(old_files) - {'release-manifest.json'}:
        raise ValueError('Unexpected tracked files: inspect the public repo before publishing.')
    manifest = json.loads((release / 'release-manifest.json').read_text(encoding='utf-8'))
    names = set(manifest['files']) | {'release-manifest.json'}
    for name in sorted(set(old_files) - names):
        release_path(checkout, name).unlink(missing_ok=True)
    for name in sorted(names):
        target = release_path(checkout, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release / name, target)
    for name, digest in manifest['files'].items():
        if hashlib.sha256(release_path(checkout, name).read_bytes()).hexdigest() != digest:
            raise ValueError('Release hash verification failed')
    for name in RUNTIME_FILES:
        if (checkout / name).read_bytes() != (ROOT / name).read_bytes():
            raise ValueError('Client sources changed while preparing the release. Rebuild.')
    git(checkout, 'add', '--all')
    staged = set(filter(None, git(checkout, 'ls-files').splitlines()))
    if staged != names:
        raise ValueError('Staged files do not match the audited release.')
    if git(checkout, 'status', '--porcelain'):
        git(checkout, 'commit', '-m', message)
    if not git(checkout, 'show-ref', '--heads'):
        raise ValueError('No release commit was created')
    # Empty target repositories may start with a local master branch.
    git(checkout, 'branch', '-M', 'main')
    if push:
        git(checkout, 'push', '-u', 'origin', 'main')
    print(json.dumps({'published': push, 'repository': REPOSITORY,
                      'commit': git(checkout, 'rev-parse', 'HEAD'), 'files': count,
                      'checkout': str(checkout), 'release': str(release), 'zip': str(archive)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', action='store_true', help='Commit and push the verified public release')
    parser.add_argument('--message', default='Update shared Rowan client release', help='Release commit message')
    arguments = parser.parse_args()
    try:
        publish(push=arguments.push, message=arguments.message)
    except subprocess.CalledProcessError as exc:
        print('Git operation failed (exit %s). Check repository access/authentication.' % exc.returncode, file=sys.stderr)
        sys.exit(1)
