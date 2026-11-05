"""The test groups defined in tests/conftest.py.

Each case runs pytest in a subprocess on a small generated test module, next
to a copy of tests/conftest.py, so the real collection hooks decide what runs
and what is skipped. ``--strict-markers`` makes an unregistered marker an
error.
"""

import importlib.util
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CONFTEST = Path(__file__).with_name('conftest.py')

TEST_MODULE = '''
import pytest

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.graph
def test_graph():
    pass

@pytest.mark.bioemu2_weights
def test_weights():
    pass

@pytest.mark.gromacs
def test_gromacs(gmx_executable):
    assert gmx_executable.endswith('gmx')

@pytest.mark.bioemu2
def test_bioemu2():
    pass

@pytest.mark.gpu
def test_gpu():
    pass

def test_plain():
    pass
'''


def _run_groups(tmp_path, *flags, path):
    """Outcome of each test of TEST_MODULE: 'passed', 'failed' or the skip reason."""
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'conftest.py').write_text(CONFTEST.read_text())
    (tmp_path / 'test_groups.py').write_text(TEST_MODULE)
    report = tmp_path / 'report.xml'
    env = dict(os.environ, PATH=str(path))
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '--strict-markers',
         f'--junitxml={report}', 'test_groups.py', *flags],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    outcomes = {}
    for case in ET.parse(report).iter('testcase'):
        skipped = case.find('skipped')
        failed = case.find('failure') is not None or case.find('error') is not None
        outcomes[case.get('name')] = ('failed' if failed else
                                      skipped.get('message') if skipped is not None else
                                      'passed')
    return outcomes


def _expected_self_skipping():
    import torch
    bioemu2 = ('passed' if importlib.util.find_spec('bioemu2') is not None
               else 'bioemu2 is not installed')
    gpu = 'passed' if torch.cuda.is_available() else 'CUDA is not available'
    return {'test_bioemu2': bioemu2, 'test_gpu': gpu, 'test_plain': 'passed'}


def test_default_run_skips_opt_in_groups_and_missing_gromacs(tmp_path):
    empty = tmp_path / 'empty_path'
    empty.mkdir()
    outcomes = _run_groups(tmp_path, path=empty)
    assert outcomes == {
        'test_slow': 'need --runslow option to run',
        'test_graph': 'need --rungraph option to run',
        'test_weights': 'need --runweights option to run',
        'test_gromacs': 'GROMACS not found (no gmx or gmx_mpi on PATH)',
        **_expected_self_skipping(),
    }


def test_flags_and_gromacs_on_path_run_every_group(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    gmx = bin_dir / 'gmx'
    gmx.write_text('#!/bin/sh\nexit 0\n')
    gmx.chmod(0o755)
    outcomes = _run_groups(tmp_path, '--runslow', '--rungraph', '--runweights', path=bin_dir)
    assert outcomes == {
        'test_slow': 'passed',
        'test_graph': 'passed',
        'test_weights': 'passed',
        'test_gromacs': 'passed',
        **_expected_self_skipping(),
    }

