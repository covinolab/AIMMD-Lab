"""The import guard of aimmd.cgaimmd.bioemu2 (DESIGN sections 5.1 and 5.4).

- ``import aimmd.cgaimmd.bioemu2`` imports nothing of bioemu2; the names that
  need bioemu2 raise one ImportError on first use that says how to install it
  (and names the 'bioemu2' extra for the generator registry);
- the system maps (``BioEmu2System``, ``MDAtoms``), the build step's MD
  reader, the command line and the network with its head and adapters
  import without bioemu2; a network is built from a
  system folder and loads its weights without bioemu2, and its first BioEmu2
  pass says how to install it; the generator ``'bioemu2_edm'`` names the
  'bioemu2' extra;
- inside the adaptor, only ``_api.py`` imports the bioemu2 stack.

The subprocess tests block the bioemu2 stack, so they run with and without
bioemu2 installed.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aimmd

from ._bioemu2_fixtures import tiny_bioemu2_checkpoint, tiny_peptide_system  # noqa: F401

ROOT = Path(aimmd.__file__).resolve().parents[1]
ADAPTOR = ROOT / 'aimmd' / 'cgaimmd' / 'bioemu2'
BIOEMU2_ONLY = ('bioemu2', 'openmm', 'biotite', 'huggingface_hub', 'cuequivariance', 'cuequivariance_torch',
                'hydra', 'omegaconf', 'einops', 'pdbecif', 'rdkit')
HINT = ("C-GAIMMD's BioEmu2 adaptor needs bioemu2 0.1.0rc0 (Python >= 3.12): pip install "
        "'aimmd-lab[bioemu2]' (CUDA: 'aimmd-lab[bioemu2-cuda]')")


def _run(code):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    proc = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_without_bioemu2_the_first_real_use_says_how_to_install_it():
    out = _run(f'''
        import json, sys
        BLOCKED = {BIOEMU2_ONLY!r}

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in BLOCKED:
                    raise ModuleNotFoundError(f'No module named {{name!r}} (blocked)', name=name)
                return None

        import aimmd
        for name in [n for n in sys.modules if n.split('.')[0] in BLOCKED]:
            del sys.modules[name]
        sys.meta_path.insert(0, Block())
        import aimmd.cgaimmd.bioemu2 as adaptor
        result = {{'all': adaptor.__all__}}
        from aimmd.cgaimmd.bioemu2 import BioEmu2System, MDAtoms, read_md_atoms
        import aimmd.cgaimmd.bioemu2.__main__
        from aimmd.cgaimmd.bioemu2 import BioEmu2Network, CommittorHead, TrunkAdapter, load_archived_network
        errors = {{}}
        for name in ('BioEmu2Features', 'BioEmu2Generator'):
            try:
                getattr(adaptor, name)
            except ImportError as exc:
                errors[name] = [str(exc), exc.name, type(exc.__cause__).__name__]
        from aimmd.cgaimmd.generation import make_generator
        try:
            make_generator('bioemu2_edm')
        except ImportError as exc:
            result['generator'] = str(exc)
        for name, call in (('build_system', lambda: adaptor.build_system('s.yaml', 'md.gro', 'md.top', 'out')),
                           ('BioEmu2System.build', lambda: BioEmu2System.build({{'entities': []}}, None))):
            try:
                call()
            except ImportError as exc:
                errors[name] = [str(exc), exc.name, type(exc.__cause__).__name__]
        result['errors'] = errors
        result['loaded'] = sorted(n for n in sys.modules if n.split('.')[0] in BLOCKED)
        print(json.dumps(result))
    ''')
    assert out['all'] == ['BioEmu2Features', 'BioEmu2Generator', 'BioEmu2Network', 'BioEmu2System',
                          'CommittorHead', 'ConditionedDenoiser', 'FeatureMLP', 'GuidanceSchedule', 'LAMBDA_CLIP',
                          'MDAtoms', 'SIGMA_REF', 'TrunkAdapter', 'build_system', 'fit_scaler',
                          'load_archived_network', 'parse_guidance_schedule', 'read_md_atoms']
    assert set(out['errors']) == {'BioEmu2Features', 'BioEmu2Generator', 'build_system', 'BioEmu2System.build'}
    for message, name, cause in out['errors'].values():
        assert message.startswith(HINT) and name == 'bioemu2' and cause == 'ModuleNotFoundError'
    assert "generation method 'bioemu2_edm'" in out['generator']
    assert "it needs the 'bioemu2' extra (pip install 'aimmd-lab[bioemu2]')" in out['generator']
    assert out['loaded'] == []


def test_only_the_api_module_imports_the_bioemu2_stack():
    found = []
    for fname in sorted(ADAPTOR.rglob('*.py')):
        if fname.name == '_api.py':
            continue
        for node in ast.walk(ast.parse(fname.read_text(), filename=str(fname))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module]
            else:
                continue
            found += [f'{fname.relative_to(ROOT)}:{node.lineno} imports {module}'
                      for module in modules if module.split('.')[0] in BIOEMU2_ONLY]
    assert found == []
    assert (ADAPTOR / '_api.py').exists()


@pytest.mark.bioemu2
def test_without_bioemu2_a_network_loads_and_its_first_pass_says_how_to_install(tiny_peptide_system, tmp_path):
    """Analysis elsewhere: the params file builds the network from its folder and loads the weights."""
    import torch
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    weights = tmp_path / 'networkARB.h5'
    torch.save(BioEmu2Network(tiny_peptide_system, hidden=(16,)).state_dict(), weights)
    out = _run(f'''
        import json, sys
        BLOCKED = {BIOEMU2_ONLY!r}

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in BLOCKED:
                    raise ModuleNotFoundError(f'No module named {{name!r}} (blocked)', name=name)
                return None

        import aimmd
        for name in [n for n in sys.modules if n.split('.')[0] in BLOCKED]:
            del sys.modules[name]
        sys.meta_path.insert(0, Block())
        import torch
        from aimmd.cgaimmd.bioemu2 import BioEmu2Network
        network = BioEmu2Network({str(tiny_peptide_system.folder)!r}, hidden=(16,))
        network.load_state_dict(torch.load({str(weights)!r}, weights_only=True))
        result = {{'n_atoms': network.n_atoms, 'generator': network.default_generator()}}
        try:
            network(torch.zeros(1, 3 * network.n_atoms))
        except ImportError as exc:
            result['error'] = [str(exc), exc.name]
        result['loaded'] = sorted(n for n in sys.modules if n.split('.')[0] in BLOCKED)
        print(json.dumps(result))
    ''')
    assert out['n_atoms'] == 166 and out['generator'] == {'method': 'bioemu2_edm'}
    assert out['error'][0].startswith(HINT) and out['error'][1] == 'bioemu2'
    assert out['loaded'] == []
