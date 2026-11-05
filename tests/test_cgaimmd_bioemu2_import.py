"""The import guard of aimmd.cgaimmd.bioemu2 (DESIGN sections 5.1 and 5.4).

- ``import aimmd.cgaimmd.bioemu2`` imports nothing of bioemu2; the names that
  need bioemu2 raise one ImportError on first use that says how to install it
  (and names the 'bioemu2' extra for the generator registry);
- the system maps (``BioEmu2System``, ``MDAtoms``), the build step's MD
  reader and the command line import without bioemu2;
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

import aimmd

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
        errors = {{}}
        try:
            adaptor.BioEmu2Features
        except ImportError as exc:
            errors['BioEmu2Features'] = [str(exc), exc.name, type(exc.__cause__).__name__]
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
    assert out['all'] == ['BioEmu2Features', 'BioEmu2System', 'MDAtoms', 'SIGMA_REF', 'build_system',
                          'read_md_atoms']
    assert set(out['errors']) == {'BioEmu2Features', 'build_system', 'BioEmu2System.build'}
    for message, name, cause in out['errors'].values():
        assert message.startswith(HINT) and name == 'bioemu2' and cause == 'ModuleNotFoundError'
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
