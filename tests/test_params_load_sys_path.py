"""What ``Params.load`` leaves in ``sys.path``.

``Params.load`` puts the params folder and ``''`` in front of ``sys.path``
while it executes the params file, and used to undo one of them with
``sys.path.pop(0)``. A params file that inserts a folder of its own (to import
a project package, for example) puts it at index 0 itself, so the pop removed
the file's entry instead of the loader's: every later lazy import from that
folder failed (it did, in every worker of the trypsin smoke run of 28 Oct
2026), and ``''`` stayed behind after every load.

The loader now removes its own entry by value. The params folder itself still
stays in ``sys.path`` after a load, as before (code may import modules of the
params folder lazily); that it is added again on every load is a separate,
open issue.
"""
import sys

import numpy as np
import pytest

import aimmd
from tests._helpers_unit import build_params_file, build_path


TRANSITION = np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32)


@pytest.fixture
def own_sys_path(monkeypatch):
    """A private copy of sys.path, restored after the test."""
    monkeypatch.setattr(sys, 'path', list(sys.path))
    return sys.path


def params_file(tmp_path, prologue=''):
    initial = build_path(tmp_path, stem='initial', positions=TRANSITION)
    fname = build_params_file(tmp_path, initial.fname)
    fname.write_text(prologue + fname.read_text())
    return fname


def test_entries_the_params_file_adds_stay(tmp_path, own_sys_path):
    extra = str(tmp_path / 'project_package_folder')
    fname = params_file(
        tmp_path, f'import sys\nsys.path.insert(0, {extra!r})\n')
    before = list(sys.path)

    aimmd.Params.load(fname, save=False)

    assert extra in sys.path, "the params file's own sys.path entry was removed"
    assert sys.path.count('') == before.count(''), "the loader's '' stayed behind"


def test_a_plain_params_file_leaves_only_its_folder(tmp_path, own_sys_path):
    """Unchanged behaviour: the params folder stays importable, nothing else."""
    fname = params_file(tmp_path)
    before = list(sys.path)

    aimmd.Params.load(fname, save=False)

    assert sys.path[1:] == before
    assert sys.path[0] == str(tmp_path.resolve())
