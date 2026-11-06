"""Import-time rules, docs stubs and process state of ``aimmd.cgaimmd``.

- ``import aimmd`` and ``import aimmd.cgaimmd`` must stay light: no PyG, no
  mlcolvar, no networkx, nothing of the bioemu2 stack (bioemu2 and what it
  brings: hydra, omegaconf, openmm, cuequivariance, biotite, ...), no CUDA
  context; core AIMMD already imports scipy.optimize
  (``aimmd/analysis/utils.py``) and, when they are installed, rdkit and
  biopython (MDAnalysis imports them, ``aimmd/_init.py``), and C-GAIMMD code
  must not add any of them.
- Only the subpackage ``aimmd.cgaimmd.bioemu2`` may import the bioemu2 stack
  at all, also inside functions (DESIGN section 5.1); checked on the source.
- Every ``aimmd.cgaimmd`` module and the compatibility modules of the old names
  must import under the Read-the-Docs stubs of ``docs/source``.
- ``aimmd.cgaimmd._once`` is the only process-global memory of C-GAIMMD code
  besides the old-name warnings and the generator registry; the autouse
  fixture of ``tests/conftest.py`` resets them (with the trainer registry and
  ``aimmd._config.WORKER_TASK``) around every test.

The subprocess tests run the interpreter of this test session on the checkout
that holds the imported ``aimmd`` package (cwd and ``PYTHONPATH``), so they
check this tree even when another checkout is installed in the environment.
"""

import ast
import json
import os
import pkgutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

import aimmd


ROOT = Path(aimmd.__file__).resolve().parents[1]
# bioemu2 and the packages it brings that core AIMMD does not import
BIOEMU2_STACK = ('bioemu2', 'openmm', 'biotite', 'huggingface_hub', 'cuequivariance',
                 'cuequivariance_torch', 'hydra', 'omegaconf', 'einops', 'pdbecif', 'pydantic',
                 'numba')
HEAVY = ('networkx', 'torch_geometric', 'torch_cluster', 'mlcolvar', 'scipy.optimize',
         'rdkit', 'Bio') + BIOEMU2_STACK
# where core AIMMD imports the heavy modules it already imports (rdkit and biopython, which
# bioemu2 also needs, only when they are installed: MDAnalysis imports them)
PRE_EXISTING = {'scipy.optimize': 'aimmd/analysis/utils.py', 'rdkit': 'aimmd/_init.py',
                'Bio': 'aimmd/_init.py'}
# what only aimmd.cgaimmd.bioemu2 may import, anywhere in its modules (DESIGN section 5.1)
BIOEMU2_ONLY = ('bioemu2', 'openmm', 'biotite', 'huggingface_hub', 'cuequivariance',
                'cuequivariance_torch', 'hydra', 'omegaconf', 'einops', 'pdbecif', 'rdkit')
# the compatibility modules of the old names
OLD_NAME_MODULES = ('aimmd.flow', 'aimmd.network.flow_fit', 'aimmd.network.joint_painn')


def _cgaimmd_module_names():
    """Every module of aimmd.cgaimmd (subpackages included), then the compatibility modules."""
    names = ['aimmd.cgaimmd']
    names += sorted(info.name for info in pkgutil.walk_packages(
        [str(ROOT / 'aimmd' / 'cgaimmd')], prefix='aimmd.cgaimmd.')
        if info.name.split('.')[2] != 'bioemu2')            # the bioemu2 extra: its own test
    return names + list(OLD_NAME_MODULES)


def _run(code):
    """Run `code` in a fresh interpreter on this checkout; return its last stdout line."""
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    proc = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('statement', [
    'import aimmd', 'import aimmd.cgaimmd',
    'from aimmd.cgaimmd import make_generator, protocol, values, ProtocolValuesFunction, '
    'CandidateContext',
    'from aimmd.cgaimmd import dataset, extract_dataset, TrainingDataset',
    'from aimmd.cgaimmd.generation import available_generators, make_generator; '
    'from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork; available_generators(); '
    'make_generator(None, TinyFlowNetwork())',
    'import aimmd.network.flow_fit as flow_fit; flow_fit.extract_flow_dataset',
])
def test_import_rules(statement):
    """No PyG, networkx, mlcolvar or bioemu2 stack, no CUDA context; the rest only from core."""
    out = _run(f'''
        import json, sys, traceback
        HEAVY = {HEAVY!r}
        ROOT = {str(ROOT)!r}
        origins = {{}}

        class Recorder:
            """Record which aimmd file triggers the first import of a heavy module."""
            def find_spec(self, name, path=None, target=None):
                if name in HEAVY and name not in origins:
                    frames = [f for f in traceback.extract_stack()
                              if f.filename.startswith(ROOT + '/aimmd/')]
                    origins[name] = (frames[-1].filename[len(ROOT) + 1:]
                                     if frames else None)
                return None

        sys.meta_path.insert(0, Recorder())
        {statement}
        import aimmd
        import torch
        print(json.dumps({{'present': [m for m in HEAVY if m in sys.modules],
                           'origins': origins,
                           'cuda': torch.cuda.is_initialized(),
                           'aimmd': aimmd.__file__}}))
    ''')
    assert out['aimmd'].startswith(str(ROOT))
    assert out['cuda'] is False
    for name in ('torch_geometric', 'torch_cluster', 'mlcolvar', 'networkx') + BIOEMU2_STACK:
        assert name not in out['present']
    for name in PRE_EXISTING:
        if name in out['present']:
            assert out['origins'].get(name) == PRE_EXISTING[name], out['origins']


def test_cgaimmd_modules_add_no_heavy_imports():
    """With the heavy modules dropped and blocked after `import aimmd`, C-GAIMMD still imports."""
    names = _cgaimmd_module_names()
    for name in ('aimmd.cgaimmd.protocol', 'aimmd.cgaimmd.generation',
                 'aimmd.cgaimmd.flow_matching.sampler', 'aimmd.cgaimmd._legacy'):
        assert name in names                       # the modules are found
    out = _run(f'''
        import importlib, json, sys
        HEAVY = {HEAVY!r}

        def heavy(name):
            return any(name == h or name.startswith(h + '.') for h in HEAVY)

        import aimmd
        for name in [n for n in sys.modules if heavy(n)]:
            del sys.modules[name]

        class Block:
            def find_spec(self, name, path=None, target=None):
                if heavy(name):
                    raise ImportError(f'{{name}} imported at C-GAIMMD import time')
                return None

        sys.meta_path.insert(0, Block())
        for name in {names!r}:
            importlib.import_module(name)
        import torch
        print(json.dumps({{'cuda': torch.cuda.is_initialized(),
                           'heavy': [n for n in sys.modules if heavy(n)]}}))
    ''')
    assert out == {'cuda': False, 'heavy': []}


def test_only_the_bioemu2_subpackage_imports_the_bioemu2_stack():
    """No module outside aimmd/cgaimmd/bioemu2/ imports the bioemu2 stack, not even lazily."""
    adaptor = ROOT / 'aimmd' / 'cgaimmd' / 'bioemu2'
    found = []
    for fname in sorted((ROOT / 'aimmd').rglob('*.py')):
        if adaptor in fname.parents:
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


def test_package_all_is_computed():
    """aimmd.cgaimmd.__all__: public submodules and names; importing the package imports none."""
    modules = sorted(info.name for info in pkgutil.iter_modules([str(ROOT / 'aimmd' / 'cgaimmd')])
                     if not info.name.startswith('_') and info.name != 'bioemu2')
    assert {'types', 'protocol', 'generation', 'devices', 'io', 'values', 'dataset',
            'flow_matching', 'source', 'pool', 'lambda_law', 'sidecar', 'restart', 'plumed',
            'preflight'} <= set(modules)
    names = {'Generator', 'register_generator', 'make_generator', 'check_generator',
             'available_generators', 'check_network_protocol', 'check_network_numerics',
             'is_cgaimmd_network', 'ProtocolValuesFunction', 'protocol_values_function',
             'CandidateContext', 'ValidityResult', 'PreparedFrame', 'ShootingPoint',
             'NetworkInfo', 'TrainingDataset', 'extract_dataset', 'GenerativeShootingPointSource',
             'make_cgaimmd_source', 'prepare_candidate', 'candidate_context', 'PreparedCandidate',
             'read_shot_sidecar', 'shot_table', 'summarise_shots', 'check_cgaimmd_restart',
             'derive_run', 'migrate_descriptors', 'run_preflight'}
    out = _run("""
        import json, sys
        import aimmd.cgaimmd
        before = sorted(m for m in sys.modules if m.startswith('aimmd.cgaimmd.'))
        from aimmd.cgaimmd import make_generator, CandidateContext
        print(json.dumps({'all': aimmd.cgaimmd.__all__, 'loaded': before,
                          'after': sorted(m for m in sys.modules
                                          if m.startswith('aimmd.cgaimmd.')),
                          'generator': make_generator.__module__,
                          'context': CandidateContext.__module__}))
    """)
    assert out['all'] == sorted({*modules, *names})
    assert out['loaded'] == []
    assert out['after'] == ['aimmd.cgaimmd.generation', 'aimmd.cgaimmd.types']
    assert out['generator'] == 'aimmd.cgaimmd.generation'
    assert out['context'] == 'aimmd.cgaimmd.types'
    flow_matching = sorted(
        info.name for info in pkgutil.iter_modules([str(ROOT / 'aimmd' / 'cgaimmd' /
                                                         'flow_matching')])
        if not info.name.startswith('_'))
    import aimmd.cgaimmd.flow_matching
    assert aimmd.cgaimmd.flow_matching.__all__ == flow_matching
    assert {'sampler', 'toy'} <= set(flow_matching)


def test_import_aimmd_imports_no_cgaimmd_module():
    """Ordinary params (RFPS, TPS) import no C-GAIMMD code."""
    out = _run("""
        import json, sys
        import aimmd
        print(json.dumps(sorted(m for m in sys.modules
                                if m.startswith(('aimmd.cgaimmd', 'aimmd.flow')))))
    """)
    assert out == []


def test_light_modules_import_stdlib_numpy_only():
    """io, _once, _legacy and generation import only the standard library and numpy."""
    allowed_relative = {'io', '_once'}
    for name in ('io', '_once', '_legacy', 'generation'):
        fname = ROOT / 'aimmd' / 'cgaimmd' / f'{name}.py'
        tree = ast.parse(fname.read_text())
        # module level = statements not nested in a function or class body
        nodes, stack = [], list(tree.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                nodes.append(node)
            stack.extend(child for child in ast.iter_child_nodes(node)
                         if isinstance(child, ast.stmt))
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1 and node.module in allowed_relative, (
                    f'{fname.name}: from {"." * node.level}{node.module or ""} import ...')
                continue
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module])
            for module in modules:
                top = module.split('.')[0]
                assert top == 'numpy' or top == '__future__' or top in sys.stdlib_module_names, (
                    f'{fname.name} imports {module} at module level')


def test_cgaimmd_modules_import_under_doc_stubs():
    """Every C-GAIMMD module imports with the heavy dependencies replaced by the docs stubs."""
    blocked = ('torch', 'torch_geometric', 'torch_cluster', 'MDAnalysis', 'mdtraj',
               'matplotlib', 'tqdm', 'filelock', 'psutil', 'mlcolvar', 'networkx', 'lz4',
               'rdkit') + BIOEMU2_STACK
    names = _cgaimmd_module_names()
    out = _run(f'''
        import importlib, json, runpy, sys
        BLOCKED = {blocked!r}

        class Block:
            """Make the packages Read the Docs does not install unimportable."""
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in BLOCKED:
                    raise ModuleNotFoundError(f'No module named {{name!r}} (blocked)')
                return None

        sys.meta_path.insert(0, Block())
        runpy.run_path({str(ROOT / 'docs' / 'source' / 'conf.py')!r})
        import torch
        assert getattr(torch, '__file__', None) is None, 'torch is not the docs stub'
        for name in {names!r}:
            importlib.import_module(name)
        print(json.dumps({{'imported': {names!r}}}))
    ''')
    assert out['imported'] == names


def test_once_and_reset():
    """`once(key)` is True the first time a key is seen in the process; `_reset` forgets."""
    from aimmd.cgaimmd import _once
    assert _once.once('warning A') is True
    assert _once.once('warning A') is False
    assert _once.once(('warning', 'B', 1)) is True
    assert _once.once(('warning', 'B', 1)) is False
    _once._reset()
    assert _once.once('warning A') is True
    # left set on purpose: the fixture must clear it before the next test
    _once.once('left set by test_once_and_reset')


def test_conftest_fixture_resets_process_state(request):
    """The autouse fixture clears the process state of C-GAIMMD code around each test."""
    import aimmd._config as config
    from aimmd.cgaimmd import _legacy, _once, generation
    from tests.conftest import reset_cgaimmd_process_state

    assert '_reset_cgaimmd_process_state' in request.fixturenames
    # nothing of the previous test survives (test_once_and_reset left a key)
    assert _once.once('left set by test_once_and_reset') is True
    assert config.WORKER_TASK is None

    name = 'aimmd.cgaimmd.flow_matching.fit'
    fit = sys.modules.get(name)
    inserted = fit is None
    if inserted:                                    # the fit module, if not imported yet
        fit = types.ModuleType(name)
        fit._RUNS = {}
        sys.modules[name] = fit
    try:
        fit._RUNS['/some/state_dir'] = object()
        config.WORKER_TASK = 'train'
        _once.once('some flag')
        with pytest.warns(DeprecationWarning):
            assert _legacy.warn_renamed('old', 'new', key='some old name')
        reset_cgaimmd_process_state()
        assert fit._RUNS == {}
        assert config.WORKER_TASK is None
        assert _once.once('some flag') is True
        with pytest.warns(DeprecationWarning):                           # warns again
            assert _legacy.warn_renamed('old', 'new', key='some old name')
    finally:
        if inserted:
            del sys.modules[name]
    # a method registered here is gone in the next test
    generation.register_generator('left_by_test_conftest_fixture', lambda: None)


def test_conftest_fixture_restores_the_generator_registry():
    from aimmd.cgaimmd import generation
    assert 'left_by_test_conftest_fixture' not in generation.available_generators()


def test_cgaimmd_helpers(tmp_path, monkeypatch):
    """tests/_cgaimmd_helpers.py builds what the C-GAIMMD streams test against."""
    import numpy as np
    import torch
    from aimmd.cgaimmd import protocol
    from aimmd.cgaimmd.io import file_sha256
    from aimmd.cgaimmd.types import CandidateContext
    from tests._cgaimmd_helpers import (build_cgaimmd_run_dir, cgaimmd_params_namespace,
                                        make_candidate_context, write_fake_network_file)

    params = cgaimmd_params_namespace(cgaimmd_parameters={'n_targets': 4,
                                                          'lambda_window': [-2, 2]})
    assert params.chain_type == 'cgaimmd' and params.sorted_states == 'ARB'
    assert params.cgaimmd_parameters == {'n_targets': 4, 'lambda_window': (-2.0, 2.0)}
    assert params.cgaimmd_settings['n_targets'] == 4
    assert params.cgaimmd_settings['lambda_tolerance'] == 1.0
    assert params.cgaimmd_generator is None
    protocol.check_network_protocol(params.network)
    assert params.values_function(np.zeros((3, 3), dtype=np.float32)).shape == (3,)

    # a run folder AIMMD itself loads, with the shooting frame at time 0
    run = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=3, n_free=1, n_atoms=2)
    chains = params.shot_chains(run.run, 'R')
    assert [len(chain) for chain in chains] == [3, 3]
    for chain in chains:
        for path in chain:
            assert path.shooting_index == run.shooting_index
            assert path.states[run.shooting_index] == 'R'
            assert path.descriptors.shape == (7, 6)
            assert path.is_complete('R', 'ARB')
    types = sorted(str(path.type)[:3] for chain in chains for path in chain)
    assert set(types) == {'ARB', 'BRA'}
    ensemble = params.pathensemble(run.run)
    assert len(ensemble) >= 6 + 2
    assert len(run.free) == 2 and len(run.shots) == 6
    rows = np.array([path.descriptors[run.shooting_index] for path in run.shots])
    assert len(np.unique(rows, axis=0)) == len(rows)          # distinct shooting frames

    # a network override does not build (and so does not draw) the default TinyFlowNetwork
    state = torch.get_rng_state()
    assert cgaimmd_params_namespace(network=params.network).network is params.network
    assert torch.equal(torch.get_rng_state(), state)

    # network files as the trainer writes them: the .json appears before the .h5 (D.7)
    replaced, real_replace = [], os.replace
    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', lambda src, dst: (
            replaced.append(os.path.basename(dst)), real_replace(src, dst))[1])
        sha = write_fake_network_file(run.run, params.network, 'ARB', round=3,
                                      n_trained_shots=12)
    assert replaced == ['networkARB.json', 'networkARB.h5']
    assert sha == file_sha256(f'{run.run}/networkARB.h5')
    meta = json.loads(Path(f'{run.run}/networkARB.json').read_text())
    assert meta['sha256'] == sha and meta['round'] == 3 and meta['n_trained_shots'] == 12
    state = torch.load(f'{run.run}/networkARB.h5', weights_only=True)
    params.network.load_state_dict(state)

    # a candidate context
    context = make_candidate_context(tmp_path / 'ctx', lambda_tolerance=2)
    assert isinstance(context, CandidateContext)
    assert context.settings['lambda_tolerance'] == 2.0
    assert context.params.cgaimmd_settings['lambda_tolerance'] == 2.0
    assert Path(context.scratch).is_dir() and context.folder.endswith('run/chainR0')
    assert context.scratch.endswith('chainR0/cgaimmd_scratch')
    context.log('hello')
    assert context.log.messages == ['hello'] and context.must_stop() is False
    assert context.rng.integers(10**6) == np.random.default_rng(0).integers(10**6)
