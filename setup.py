"""Build hook of AIMMD-Lab: the git commit a built package comes from.

Everything else about the build is declared in ``pyproject.toml``; this file
only adds the build record.

A checkout of AIMMD-Lab knows its commit through git, and pip records the
commit of an install from git (``pip install git+...``) in the
distribution's ``direct_url.json``. A package built from a checkout and then
installed as a wheel, or installed from a folder without ``-e``, has
neither. So every build writes ``aimmd/_build_record.json`` into the built
package: the commit of the checkout, whether its ``aimmd/`` folder had
uncommitted changes, and when it was built. The C-GAIMMD shot sidecars read
it when git and pip's install record name no commit
(:func:`aimmd.cgaimmd.sidecar.git_provenance`).

- A wheel or a non-editable install built from a git checkout records that
  checkout's commit.
- A source distribution carries the record of the checkout it was made
  from, and a wheel built from it (``python -m build`` builds the wheel from
  the sdist) keeps that record.
- Editable installs write no record: the checkout answers through git.
- A build from a folder that is neither a checkout nor an unpacked source
  distribution records no commit, and says why.

The record is never written into the source tree (``.gitignore`` lists it).
"""

import datetime
import json
import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join('aimmd', '_build_record.json')
SCHEMA = 'aimmd.build_record/1'


def _git(root, *args):
    """``(stdout, None)`` of ``git <args>`` in `root`, or ``(None, why it failed)``."""
    try:
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True,
                                timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f'git could not be run ({type(exc).__name__}: {exc})'
    if result.returncode != 0:
        message = (result.stderr or result.stdout or '').strip().splitlines()
        return None, (f'git {" ".join(args)} failed: '
                      f'{message[0] if message else f"exit code {result.returncode}"}')
    return result.stdout.strip(), None


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def build_record(root=ROOT):
    """The build record of the source tree `root`.

    Returns
    -------
    dict
        ``schema``, ``git_commit`` (None if unknown), ``git_dirty`` (whether
        ``aimmd/`` had uncommitted changes, compiled bytecode aside; None if
        unknown), ``git_reason`` (None, or why the commit or the flag is
        unknown), ``built_at`` (UTC) and ``built_from`` (``'git checkout'``,
        ``'source distribution'`` or ``'folder'``).
    """
    # only a repository that tracks the package counts: a folder inside
    # another project's work tree must not be given that project's commit
    tracked, why = _git(root, 'ls-files', '--error-unmatch', 'aimmd/__init__.py')
    if tracked:
        commit, why = _git(root, 'rev-parse', 'HEAD')
        if commit:
            status, status_why = _git(root, 'status', '--porcelain', '--untracked-files=all',
                                      '--', 'aimmd', ':(exclude)*.pyc')
            return {'schema': SCHEMA, 'git_commit': commit,
                    'git_dirty': None if status is None else bool(status),
                    'git_reason': None if status is not None
                    else f'the dirty flag is unknown: {status_why}',
                    'built_at': _now(), 'built_from': 'git checkout'}
    shipped = os.path.join(root, RECORD)
    if os.path.isfile(shipped):
        # an unpacked source distribution: keep the record of its checkout
        with open(shipped) as handle:
            record = json.load(handle)
        record.update(built_at=_now(), built_from='source distribution',
                      sdist_built_at=record.get('built_at'))
        return record
    return {'schema': SCHEMA, 'git_commit': None, 'git_dirty': None,
            'git_reason': f'built from a folder that is not a git checkout of AIMMD-Lab ({why})',
            'built_at': _now(), 'built_from': 'folder'}


def write_record(fname, record):
    """Write `record` as JSON to `fname` (its folder must exist)."""
    with open(fname, 'w') as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write('\n')


class BuildPyWithRecord(build_py):
    """``build_py`` that adds ``aimmd/_build_record.json`` to the built package."""

    def run(self):
        super().run()
        if not getattr(self, 'editable_mode', False):
            write_record(os.path.join(self.build_lib, RECORD), build_record())

    def get_outputs(self, *args, **kwargs):
        outputs = super().get_outputs(*args, **kwargs)
        if not getattr(self, 'editable_mode', False):
            outputs.append(os.path.join(self.build_lib, RECORD))
        return outputs


class SdistWithRecord(sdist):
    """``sdist`` that puts the checkout's build record into the source distribution."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        target = os.path.join(base_dir, RECORD)
        if os.path.exists(target):                          # a hard link into the source tree
            os.remove(target)
        write_record(target, build_record())


if __name__ == '__main__':
    setup(cmdclass={'build_py': BuildPyWithRecord, 'sdist': SdistWithRecord})
