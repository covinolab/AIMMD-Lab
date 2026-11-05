"""Packaging of C-GAIMMD.

The C-GAIMMD tooling needs ``networkx`` (topology automorphisms, bond-graph
validity), imported lazily. It is the extra ``cgaimmd`` of ``pyproject.toml``,
part of the ``tests`` extra, and listed in the CI test environment
(``devtools/conda-envs/test_env.yaml``, which installs ``pip install -e .``
without extras).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
tomllib = pytest.importorskip('tomllib')


def extras():
    with open(ROOT / 'pyproject.toml', 'rb') as file:
        return tomllib.load(file)['project']['optional-dependencies']


def test_cgaimmd_extra():
    assert extras()['cgaimmd'] == ['networkx']
    assert 'flow' not in extras()                  # the old name was never released


def test_tests_extra_has_networkx():
    assert 'networkx' in extras()['tests']


def test_ci_env_has_networkx():
    text = (ROOT / 'devtools' / 'conda-envs' / 'test_env.yaml').read_text()
    assert re.search(r'^\s*-\s*networkx\b', text, flags=re.MULTILINE)
