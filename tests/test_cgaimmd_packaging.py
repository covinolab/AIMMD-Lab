"""Packaging of C-GAIMMD.

The C-GAIMMD tooling needs ``networkx`` (topology automorphisms, bond-graph
validity), imported lazily. It is the extra ``cgaimmd`` of ``pyproject.toml``,
part of the ``tests`` extra, and listed in the CI test environment
(``devtools/conda-envs/test_env.yaml``, which installs ``pip install -e .``
without extras). The BioEmu2 generator is the extra ``bioemu2`` (or
``bioemu2-cuda``), pinned to the release the adaptor is tested with; its
drifting dependencies are bounded by ``devtools/constraints-bioemu2.txt``.
``devtools/conda-envs/cgaimmd_env.yaml`` is the environment of the CI jobs
that run every optional part.
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


def _requirements(extra):
    requirements = pytest.importorskip('packaging.requirements')
    return {req.name: req for req in map(requirements.Requirement, extras()[extra])}


def test_bioemu2_extras_pin_the_tested_release():
    """bioemu2 0.1.0rc0 is the release the adaptor is tested with (Python >= 3.12 only)."""
    bioemu2 = _requirements('bioemu2')
    assert bioemu2['aimmd-lab'].extras == {'cgaimmd'}
    assert str(bioemu2['bioemu2'].specifier) == '==0.1.0rc0'
    assert bioemu2['bioemu2'].marker.evaluate({'python_version': '3.12'})
    assert not bioemu2['bioemu2'].marker.evaluate({'python_version': '3.11'})
    cuda = _requirements('bioemu2-cuda')
    assert cuda['aimmd-lab'].extras == {'bioemu2'}
    assert cuda['bioemu2'].extras == {'cuda'} and str(cuda['bioemu2'].specifier) == '==0.1.0rc0'
    assert not cuda['bioemu2'].marker.evaluate({'python_version': '3.13',
                                                'sys_platform': 'darwin'})


def test_graphs_extra_keeps_torch_geometric_below_2_8():
    """torch-geometric 2.8 builds radius graphs with pyg-lib (not on PyPI), not torch-cluster."""
    graphs = _requirements('graphs')
    assert set(graphs) == {'torch-geometric', 'torch-cluster', 'lz4'}
    assert not graphs['torch-geometric'].specifier.contains('2.8.1')
    assert graphs['torch-geometric'].specifier.contains('2.7.0')


def test_tests_extra_has_the_time_limit_plugin():
    assert 'pytest-timeout' in extras()['tests']


def test_bioemu2_constraints():
    """The constraints pin bioemu2 and the drifting packages, never torch or triton."""
    lines = [line.split('#')[0].strip() for line in
             (ROOT / 'devtools' / 'constraints-bioemu2.txt').read_text().splitlines()]
    pins = {re.split(r'[<>=!~]', line)[0].strip().lower(): line for line in lines if line}
    assert pins['bioemu2'] == 'bioemu2==0.1.0rc0'
    assert {'biotite', 'rdkit', 'numpy'} <= set(pins)
    assert 'torch' not in pins and 'triton' not in pins


def test_full_conda_env():
    """The env of the full CI job: CPU torch 2.13 (bioemu2 < 2.14), PyG < 2.8, the constraints."""
    text = (ROOT / 'devtools' / 'conda-envs' / 'cgaimmd_env.yaml').read_text()
    for pattern in (r'^\s*-\s*python=3\.13\b', r'^\s*-\s*pytorch=2\.13\.\*=cpu\*',
                    r'^\s*-\s*pytorch_cluster\b', r'^\s*-\s*pytorch_geometric<2\.8',
                    r'^\s*-\s*gromacs\b', r'^\s*-\s*pytest-timeout\b',
                    r'^\s*-\s*-c \.\./constraints-bioemu2\.txt', r'^\s*-\s*bioemu2==0\.1\.0rc0'):
        assert re.search(pattern, text, flags=re.MULTILINE), pattern
    assert (ROOT / 'devtools' / 'conda-envs' / '..' / 'constraints-bioemu2.txt').is_file()


def test_gitignore_keeps_the_build_record_out_of_the_source_tree():
    assert 'aimmd/_build_record.json' in (ROOT / '.gitignore').read_text().splitlines()


def test_python_floor_and_classifiers():
    """requires-python and the version classifiers say the same (3.10 or later)."""
    with open(ROOT / 'pyproject.toml', 'rb') as file:
        project = tomllib.load(file)['project']
    assert project['requires-python'] == '>=3.10'
    versions = sorted(c.rsplit(':: ', 1)[1] for c in project['classifiers']
                      if re.fullmatch(r'Programming Language :: Python :: 3\.\d+', c))
    assert versions[0] == '3.10' and '3.9' not in versions
