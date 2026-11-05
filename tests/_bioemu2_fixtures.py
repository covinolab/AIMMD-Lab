"""Fixtures of the BioEmu2 adaptor tests: the tiny BioEmu2 and the two small systems (DESIGN section 7.2).

Import them into a test module (``from ._bioemu2_fixtures import ...``); each
skips its tests when bioemu2 cannot be imported (``pytest.importorskip``), so
a broken install skips them as a missing one does.

- ``tiny_bioemu2_checkpoint`` (session): the vendored production-shaped config
  ``tests/data/bioemu2_tiny_production_shaped.yaml`` with seeded random
  weights (161,422 parameters), written as ``config.yaml`` + ``model.ckpt``;
  no download.
- ``tiny_bioemu2_model``: a fresh frozen model per test (``_api.load_model``),
  so that a hook a test leaves on it goes with the test.
- ``tiny_peptide_system`` / ``tiny_protein_ligand_system`` (session): system
  folders built with ``build_system`` from ``tests/data/cgaimmd_bioemu2_peptide``
  (CLN025, 10 residues) and ``tests/data/cgaimmd_bioemu2_protein_ligand``
  (20 trypsin residues + benzamidinium; core = the 20 CA atoms, as the
  goldens' old constants). Tests must not change them.
"""

import pytest

from ._cgaimmd_bioemu2_golden import PEPTIDE, PROTEIN_LIGAND, write_tiny_checkpoint

LIGAND_CORE_RESIDUES = tuple(range(1, 21))
"""The core of the protein-ligand system: the CA atoms of its 20 residues (as the goldens' old constants)."""


@pytest.fixture(scope='session')
def tiny_bioemu2_checkpoint(tmp_path_factory):
    pytest.importorskip('bioemu2')
    return write_tiny_checkpoint(tmp_path_factory.mktemp('tiny_bioemu2'))


@pytest.fixture
def tiny_bioemu2_model(tiny_bioemu2_checkpoint):
    from aimmd.cgaimmd.bioemu2 import _api
    return _api.load_model(checkpoint_path=tiny_bioemu2_checkpoint, device='cpu')


def _build(folder, data, checkpoint, **kwargs):
    from aimmd.cgaimmd.bioemu2 import _api, build_system
    model = _api.load_model(checkpoint_path=checkpoint, device='cpu')
    return build_system(data / 'bioemu2_system.yaml', data / 'md.gro', data / 'md.top', folder, model=model,
                        checkpoint_path=checkpoint, **kwargs)


@pytest.fixture(scope='session')
def tiny_peptide_system(tiny_bioemu2_checkpoint, tmp_path_factory):
    return _build(tmp_path_factory.mktemp('peptide_system') / 'system', PEPTIDE, tiny_bioemu2_checkpoint)


@pytest.fixture(scope='session')
def tiny_protein_ligand_system(tiny_bioemu2_checkpoint, tmp_path_factory):
    return _build(tmp_path_factory.mktemp('protein_ligand_system') / 'system', PROTEIN_LIGAND,
                  tiny_bioemu2_checkpoint, core_residues=LIGAND_CORE_RESIDUES)
