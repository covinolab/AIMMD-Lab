"""The Params side of C-GAIMMD: ``chain_type='cgaimmd'``, the hook fields,
``cgaimmd_generator``, ``initial_network``, the validation rules and the
default values function of C-GAIMMD networks.

Most tests load a real params file written into ``tmp_path`` (the toy engine,
one atom, one end-state file per basin), because the rules run in
``Params._process_and_check`` at load and at assignment, and the network's
class must be defined in the params file (rule 9).
"""

import ast
import importlib.util
import os
import re
import textwrap
import types

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd import _once, generation
from aimmd.cgaimmd.flow_matching.sampler import ODEGenerator
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.values import ProtocolValuesFunction
from aimmd.network.rescalable import Rescalable
from tests._helpers_unit import write_trajectory


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SEED_X = {'A': [-1.0, -1.1, -0.9, -1.2], 'B': [1.0, 1.1, 0.9, 1.2],
          'AB': [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]}

PARAMS_HEAD = '''\
import numpy as np
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork

engine = 'toy'
states = 'ARB'
chain_type = 'cgaimmd'
initial_paths = ['seeds_A.xtc', 'seeds_B.xtc']
round_shooting_index = True
gen_temperature = 300.0
cgaimmd_parameters = {'placement': 'none', 'lambda_clip': 10.0, 'window_after_trained_shots': 5}

def toy_mdrun(ts):
    ts.positions = ts.positions + 0.01

def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R'))

def descriptors_function(trajectory):
    frames = [ts.positions.copy() for ts in trajectory]
    return np.array(frames, dtype=np.float32).reshape(len(frames), -1)

class Network(TinyFlowNetwork):
    pass

network = Network()
'''


def write_seeds(folder, kinds=('A', 'B'), n_atoms=1):
    """One trajectory per kind (``seeds_<kind>.xtc``), atom 0 at the x of SEED_X."""
    names = []
    for kind in kinds:
        x = np.asarray(SEED_X[kind], dtype=np.float32)
        positions = np.zeros((len(x), n_atoms, 3), dtype=np.float32)
        positions[:, :, 0] = x[:, None]
        positions[:, :, 1] = 1.5 * np.arange(n_atoms)[None, :]
        names.append(os.path.basename(write_trajectory(folder, stem=f'seeds_{kind}',
                                                       positions=positions)))
    return names


def write_params(folder, extra='', name='params.py', seeds=('A', 'B'), n_atoms=1):
    """The C-GAIMMD params file PARAMS_HEAD plus `extra` (later assignments win)."""
    folder = str(folder)
    os.makedirs(folder, exist_ok=True)
    write_seeds(folder, kinds=seeds, n_atoms=n_atoms)
    fname = os.path.join(folder, name)
    with open(fname, 'w') as file:
        file.write(PARAMS_HEAD + '\n' + textwrap.dedent(extra))
    return fname


def load(folder, extra='', **kwargs):
    kwargs.setdefault('save', False)
    return aimmd.Params.load(write_params(folder, extra), **kwargs)


class LocalFlowNetwork(TinyFlowNetwork):
    """A C-GAIMMD network whose class is not in the aimmd package (rule 9)."""


class PlainNetwork(torch.nn.Module):
    """An RFPS network: forward only."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(torch.flatten(torch.as_tensor(x, dtype=torch.float32), 1)[:, :1])


class RescalableFlowNetwork(Rescalable, TinyFlowNetwork):
    """Passes the rescale_committor type check, so that rule 4 is reached."""


def x_descriptors(trajectory):
    frames = [ts.positions.copy() for ts in trajectory]
    return np.array(frames, dtype=np.float32).reshape(len(frames), -1)


def printed(capsys):
    captured = capsys.readouterr()
    return captured.out + captured.err


# ---------------------------------------------------------------------------
# chain_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    ('cgaimmd', 'cgaimmd'), ('CGAIMMD', 'cgaimmd'), ('cg', 'cgaimmd'),
    ('CG-AIMMD shooting', 'cgaimmd'), ('tps', 'tps'), ('rf', 'rfps')])
def test_chain_type_prefixes(tmp_path, value, expected):
    params = load(tmp_path, 'chain_type = "rfps"')
    if expected == 'tps':
        params.selection_pool_size = 1
    params.chain_type = value
    assert params.chain_type == expected


def test_chain_type_error_names_cgaimmd(tmp_path):
    params = load(tmp_path)
    with pytest.raises(TypeError, match="'tps', 'rfps' or 'cgaimmd'"):
        params.chain_type = 'metadynamics'
    assert params.chain_type == 'cgaimmd'


def test_chain_type_description_mentions_cgaimmd():
    description = aimmd.Params.__dataclass_fields__['chain_type'].metadata['description']
    assert "'cgaimmd' : committor-conditioned Generative AIMMD (C-GAIMMD)" in description
    assert "'cg'" in description


# ---------------------------------------------------------------------------
# hook fields
# ---------------------------------------------------------------------------

HOOKS = ('cgaimmd_validity_function', 'cgaimmd_prepare_function', 'cgaimmd_minimize_function',
         'cgaimmd_lambda_sampler')


def test_hook_fields_declared_after_cgaimmd_parameters():
    names = list(aimmd.Params.__dataclass_fields__)
    start = names.index('retry_with_state_definition_glitches')
    assert names[start + 1:start + 7] == ['cgaimmd_parameters', 'cgaimmd_generator', *HOOKS]
    assert names[names.index('network_save_interval') + 1] == 'initial_network'
    for name in (*HOOKS, 'cgaimmd_generator', 'initial_network'):
        field = aimmd.Params.__dataclass_fields__[name]
        assert field.default is None
        assert field.metadata['description']


@pytest.mark.parametrize('name', HOOKS)
def test_hooks_must_be_functions(tmp_path, name):
    params = load(tmp_path, "cgaimmd_parameters = {'lambda_clip': 10.0, "
                            "'window_after_trained_shots': 5}")

    def hook(*args):
        return None

    class CallableObject:
        def __call__(self, *args):
            return None

        def method(self, *args):
            return None

    setattr(params, name, hook)
    assert getattr(params, name) is hook
    setattr(params, name, None)
    assert getattr(params, name) is None
    for bad in (CallableObject(), CallableObject().method, len, 'not a function'):
        with pytest.raises(TypeError, match=f"'{name}' must be a function defined in the "
                                            f"params file \\(wrap a callable object in "
                                            f"'def {name}\\(...\\)'\\)"):
            setattr(params, name, bad)
        assert getattr(params, name) is None


def test_hooks_warning_outside_cgaimmd(tmp_path, capsys):
    load(tmp_path, '''
        chain_type = 'rfps'
        def cgaimmd_minimize_function(frame, context):
            return frame
        ''')
    assert ("Warning: cgaimmd_* hooks are set but chain_type is 'rfps'; they are ignored"
            in printed(capsys))
    load(tmp_path / 'b', "chain_type = 'rfps'")
    assert 'cgaimmd_* hooks are set' not in printed(capsys)


def test_hooks_in_file_under_cgaimmd(tmp_path, capsys):
    params = load(tmp_path, '''
        def cgaimmd_validity_function(positions, context):
            return None
        def cgaimmd_lambda_sampler(n, context):
            return np.zeros(n)
        ''')
    assert isinstance(params.cgaimmd_validity_function, types.FunctionType)
    assert params.cgaimmd_lambda_sampler(3, None).shape == (3,)
    assert 'cgaimmd_* hooks are set' not in printed(capsys)


# ---------------------------------------------------------------------------
# initial_network
# ---------------------------------------------------------------------------

def test_initial_network_relative_on_save(tmp_path):
    folder = tmp_path / 'sub'
    (folder / 'pretrained').mkdir(parents=True)
    torch.save(TinyFlowNetwork().state_dict(), folder / 'pretrained' / 'net.h5')
    params = load(folder, "initial_network = 'pretrained/net.h5'")
    assert params.initial_network == 'pretrained/net.h5'
    assert (os.path.realpath(params._initial_network_fname())
            == os.path.realpath(folder / 'pretrained' / 'net.h5'))
    cwd = os.getcwd()
    os.chdir(tmp_path)                             # saving from another cwd keeps it right
    try:
        text = str(params.__str__(params.parent))
    finally:
        os.chdir(cwd)
    assert "initial_network = 'pretrained/net.h5'" in text
    # an absolute path is written relative to the params file
    params.initial_network = str(folder / 'pretrained' / 'net.h5')
    assert "initial_network = 'pretrained/net.h5'" in params.__str__(params.parent)


def test_initial_network_missing_file_rejected(tmp_path):
    with pytest.raises(TypeError, match='initial_network'):
        load(tmp_path, "initial_network = 'missing.h5'")
    params = load(tmp_path / 'rfps', "chain_type = 'rfps'")
    with pytest.raises(TypeError, match='does not exist'):
        params.initial_network = 'missing.h5'
    assert params.initial_network is None
    with pytest.raises(TypeError, match='initial_network'):
        params.initial_network = 3


def test_initial_network_accepts_pathlike(tmp_path):
    fname = tmp_path / 'net.h5'
    torch.save(TinyFlowNetwork().state_dict(), fname)
    params = load(tmp_path)
    params.initial_network = fname
    assert params.initial_network == str(fname)


def test_initial_network_with_multi_system_not_implemented(tmp_path):
    fname = tmp_path / 'net.h5'
    torch.save(PlainNetwork().state_dict(), fname)
    params = aimmd.Params.placeholder.copy()
    with pytest.raises(NotImplementedError, match='multi_system'):
        params.update(multi_system=True, initial_network=str(fname), save=False)


# ---------------------------------------------------------------------------
# save round trip
# ---------------------------------------------------------------------------

def test_save_roundtrip(tmp_path, monkeypatch):
    torch.save(TinyFlowNetwork().state_dict(), tmp_path / 'net.h5')
    params = load(tmp_path, '''
        initial_network = 'net.h5'
        cgaimmd_parameters = {'placement': 'none', 'lambda_clip': 10.0,
                              'window_after_trained_shots': 5, 'lambda_window': [-3, 3]}
        cgaimmd_generator = {'method': 'ode', 'n_steps': 4}
        def cgaimmd_validity_function(positions, context):
            return None
        ''')
    monkeypatch.chdir(params.parent)                  # as the Launcher saves, from the run's folder
    fname = params.save()
    text = open(os.path.join(params.parent, os.path.basename(fname))).read()
    assert 'values_function = None' in text                 # the default is not written
    assert "chain_type = 'cgaimmd'" in text
    assert "initial_network = 'net.h5'" in text
    assert 'import cgaimmd_validity_function' in text
    assert "cgaimmd_generator = {'method': 'ode', 'n_steps': 4}" in text
    # only new names are written (the old ones are read with a warning, never written)
    assigned = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            assigned |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            assigned |= {alias.asname or alias.name for alias in node.names}
    assert not {name for name in assigned if name.startswith('flow')}
    assert {'cgaimmd_parameters', 'cgaimmd_generator', *HOOKS} <= assigned
    reloaded = aimmd.Params.load(os.path.join(params.parent, os.path.basename(fname)),
                                 save=False)
    assert reloaded.chain_type == 'cgaimmd'
    assert reloaded.cgaimmd_parameters == params.cgaimmd_parameters
    assert reloaded.cgaimmd_parameters['lambda_window'] == (-3.0, 3.0)
    assert reloaded.cgaimmd_generator == {'method': 'ode', 'n_steps': 4}
    assert reloaded.initial_network == 'net.h5'
    assert callable(reloaded.cgaimmd_validity_function)
    assert isinstance(reloaded.values_function, ProtocolValuesFunction)


# ---------------------------------------------------------------------------
# validation rules
# ---------------------------------------------------------------------------

def test_cgaimmd_params_load(tmp_path, capsys):
    params = load(tmp_path)
    assert params.chain_type == 'cgaimmd'
    assert isinstance(params.values_function, ProtocolValuesFunction)
    out = printed(capsys)
    for name in ('seeds_A', 'seeds_B'):             # replaces the brute-force warning
        assert re.search(f"C-GAIMMD: initial path '[^']*{name}.xtc' has no 'ARB' transition; "
                         f"it seeds free simulations only", out)
    assert 'brute-force shooting' not in out


def test_multi_system_not_implemented(tmp_path):
    """The same type on load and on assignment (_setattr re-raises it unwrapped)."""
    with pytest.raises(NotImplementedError,
                       match="chain_type='cgaimmd' supports single systems"):
        load(tmp_path, 'multi_system = True')
    params = load(tmp_path / 'b')
    with pytest.raises(NotImplementedError, match='single systems only'):
        params.multi_system = True
    assert params.multi_system is False


def test_needs_descriptors_function(tmp_path):
    with pytest.raises(TypeError, match="chain_type='cgaimmd' needs a descriptors_function"):
        load(tmp_path, 'descriptors_function = None')


def test_needs_end_state_frames(tmp_path):
    folder = tmp_path / 'onlyA'
    write_params(folder, "initial_paths = ['seeds_A.xtc']", seeds=('A',))
    with pytest.raises(TypeError, match="at least one frame in 'A' and one in 'B'"):
        aimmd.Params.load(str(folder / 'params.py'), save=False)


def never_b(trajectory):
    """A states function under which the B seeds are in R: no frame in 'B'."""
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', 'R')


def test_needs_end_state_frames_on_assignment(tmp_path):
    """Rule 12 also runs when states_function or states is assigned."""
    params = load(tmp_path)
    old = params.states_function
    with pytest.raises(TypeError, match="at least one frame in 'A' and one in 'B'"):
        params.states_function = never_b
    assert params.states_function is old                             # the old value stays
    with pytest.raises(TypeError, match="at least one frame in 'A' and one in 'C'"):
        params.states = 'ARC'                                          # no frame is 'C'
    assert params.states == 'ARB'


def test_transition_path_counts_as_end_state_frames(tmp_path):
    folder = tmp_path / 'ab'
    write_params(folder, "initial_paths = ['seeds_AB.xtc']", seeds=('AB',))
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    assert len(params.initial_paths) == 1


def test_descriptor_width(tmp_path):
    with pytest.raises(TypeError, match=r'width 2 .* 3 x 1 = 3'):
        load(tmp_path, '''
            def descriptors_function(trajectory):
                x = np.array([ts.positions[0, 0] for ts in trajectory], dtype=np.float32)
                return np.stack([x, x ** 2], axis=1)
            ''')


WIDE_DESCRIPTORS = '''
    network = Network(n_atoms=2)
    def descriptors_function(trajectory):
        frames = [np.concatenate([ts.positions[0], ts.positions[0]]) for ts in trajectory]
        return np.array(frames, dtype=np.float32)
    '''


def test_prepare_needed_when_m_ne_n(tmp_path):
    """M = 2 generated atoms, N = 1 atom (from the first initial path: no topology)."""
    with pytest.raises(TypeError, match=r'the network generates 2 atoms but the system has 1: '
                                        r'set cgaimmd_prepare_function'):
        load(tmp_path, WIDE_DESCRIPTORS)
    params = load(tmp_path / 'prepared', WIDE_DESCRIPTORS + '''
    cgaimmd_parameters = {'lambda_clip': 10.0, 'window_after_trained_shots': 5}
    def cgaimmd_prepare_function(positions, context):
        return None
    ''')
    assert params.cgaimmd_prepare_function is not None


def test_prepare_needed_when_m_ne_n_with_topology(tmp_path):
    """N from params.topology when there is one.

    AIMMD already refuses initial paths whose atom count differs from the
    topology's, so N is the same either way; the test pins the rule with a
    topology present.
    """
    folder = tmp_path / 'top'
    folder.mkdir()
    (folder / 'conf.gro').write_text('one atom\n    1\n'
                                     '    1MOL      C    1   0.000   0.000   0.000\n'
                                     '   2.00000   2.00000   2.00000\n')
    with pytest.raises(TypeError, match=r'generates 2 atoms but the system has 1'):
        load(folder, WIDE_DESCRIPTORS + "topology = 'conf.gro'\n")


def test_cgaimmd_rejects_non_protocol_network(tmp_path):
    with pytest.raises(TypeError, match='n_atoms'):
        load(tmp_path, '''
            import torch
            class Network(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(1, 1)
                def forward(self, x):
                    return self.linear(x[:, :1])
            network = Network()
            ''')


def test_assignment_order(tmp_path):
    """network and descriptors_function before chain_type = 'cgaimmd'; the other order raises."""
    params = load(tmp_path, '''
        chain_type = 'rfps'
        descriptors_function = None
        import torch
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
            def forward(self, x):
                return self.linear(torch.flatten(x, 1)[:, :1])
        network = Network()
        ''')
    with pytest.raises(TypeError, match="can't update 'chain_type'"):
        params.chain_type = 'cgaimmd'
    assert params.chain_type == 'rfps'
    params.network = LocalFlowNetwork()
    with pytest.raises(TypeError, match='descriptors_function'):
        params.chain_type = 'cgaimmd'
    params.descriptors_function = x_descriptors
    params.chain_type = 'cgaimmd'
    assert params.chain_type == 'cgaimmd'


def test_rescale_committor_rejected(tmp_path):
    params = load(tmp_path)
    params.network = RescalableFlowNetwork()
    with pytest.raises(TypeError, match="rescale_committor is not supported with "
                                        "chain_type='cgaimmd'"):
        params.rescale_committor = True
    assert params.rescale_committor is False


@pytest.mark.parametrize('temperature', [-300.0, 0.0])
def test_negative_gen_temperature_rejected(tmp_path, temperature):
    with pytest.raises(TypeError, match='generated shooting points carry no velocities; set '
                                        'gen_temperature > 0'):
        load(tmp_path, f'gen_temperature = {temperature}')


def test_placement_none_with_prepare_rejected(tmp_path):
    with pytest.raises(TypeError, match="placement.*'none'.*cgaimmd_prepare_function"):
        load(tmp_path, '''
            def cgaimmd_prepare_function(positions, context):
                return None
            ''')


def test_lambda_clip_required_without_network_attribute(tmp_path):
    with pytest.raises(TypeError, match='lambda_clip'):
        load(tmp_path, "cgaimmd_parameters = {'placement': 'none', "
                       "'window_after_trained_shots': 5}")
    params = load(tmp_path / 'own', '''
        cgaimmd_parameters = {'placement': 'none', 'window_after_trained_shots': 5}
        network = Network(lambda_clip=8.0)
        ''')
    with pytest.raises(TypeError, match='differs'):
        params.cgaimmd_parameters = {'placement': 'none', 'lambda_clip': 5.0}
    assert params.cgaimmd_parameters == {'placement': 'none', 'window_after_trained_shots': 5}


def test_network_class_must_be_local(tmp_path):
    message = "define the network's class in the params file"
    with pytest.raises(TypeError, match=message):
        load(tmp_path, 'network = TinyFlowNetwork()')
    # a C-GAIMMD network under any chain type (the Launcher re-saves the params)
    with pytest.raises(TypeError, match=message):
        load(tmp_path / 'rfps', "chain_type = 'rfps'\nnetwork = TinyFlowNetwork()")
    params = load(tmp_path / 'ok', "chain_type = 'rfps'")
    with pytest.raises(TypeError, match=message):
        params.network = TinyFlowNetwork()
    params.network = LocalFlowNetwork()


def test_round_shooting_index_warning(tmp_path, capsys):
    load(tmp_path, 'round_shooting_index = False')
    out = printed(capsys)
    assert ("WARNING: chain_type='cgaimmd' with round_shooting_index=False: reloaded shots "
            "may take the frame before t = 0 as their shooting point (set "
            "round_shooting_index = True)") in out
    load(tmp_path / 'again', 'round_shooting_index = False')
    assert 'round_shooting_index=False' not in printed(capsys)       # once per process
    load(tmp_path / 'true')
    assert 'round_shooting_index' not in printed(capsys)


def test_window_after_default_warning(tmp_path, capsys):
    load(tmp_path)
    assert 'window_after_trained_shots is at its default' not in printed(capsys)
    load(tmp_path / 'default', "cgaimmd_parameters = {'placement': 'none', 'lambda_clip': 10.0}")
    assert ('WARNING: window_after_trained_shots is at its default 1000; this is highly '
            'system dependent (it was 8000 for the NANMA test system), set it '
            'explicitly') in printed(capsys)
    # a once-key of its own: the shooting worker's start-up still prints its copy into the
    # worker log (key 'window_after_trained_shots default')
    assert _once.once('window_after_trained_shots default')
    assert not _once.once('params: window_after_trained_shots default')


def test_initial_network_warning(tmp_path, capsys):
    load(tmp_path)
    assert ("Warning: chain_type='cgaimmd' without initial_network: the generator starts "
            "from params.network's constructed weights") in printed(capsys)
    load(tmp_path / 'again')
    assert 'without initial_network' not in printed(capsys)          # once per process
    _once._reset()
    torch.save(TinyFlowNetwork().state_dict(), tmp_path / 'net.h5')
    load(tmp_path / 'b', f"initial_network = {str(tmp_path / 'net.h5')!r}")
    assert 'without initial_network' not in printed(capsys)


def test_non_boundary_seeding_needs_transition(tmp_path):
    with pytest.raises(TypeError, match='free_seeding_position'):
        load(tmp_path, "free_seeding_position = 'middle'")
    # on assignment as well: the free workers would die at launch
    params = load(tmp_path / 'assigned')
    with pytest.raises(TypeError, match='free_seeding_position'):
        params.free_seeding_position = 'middle'
    with pytest.raises(TypeError, match="for state 'B'"):
        params.free_seeding_position = {'B': 'random'}
    assert params.free_seeding_position_for('A') == 1.0              # the old value stays
    params.free_seeding_position = 'boundary'
    # with a transition in the initial paths the other positions are allowed
    folder = tmp_path / 'ab'
    write_params(folder, "initial_paths = ['seeds_AB.xtc']\nfree_seeding_position = 'middle'",
                 seeds=('AB',))
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    assert params.free_seeding_position_for('A') == 0.5


# ---------------------------------------------------------------------------
# default values function
# ---------------------------------------------------------------------------

def test_default_values_function_follows_network(tmp_path):
    params = load(tmp_path, "chain_type = 'rfps'")
    function = params.values_function
    assert isinstance(function, ProtocolValuesFunction)                  # under 'rfps' too
    assert params._default_values_function
    assert function.params is params and function.network is params.network
    params.network = PlainNetwork()
    assert not isinstance(params.values_function, ProtocolValuesFunction)
    assert params._default_values_function
    params.network = LocalFlowNetwork()
    function = params.values_function
    assert isinstance(function, ProtocolValuesFunction)
    assert function.network is params.network
    params.chain_type = 'cgaimmd'                                    # the choice is the network's
    assert params.values_function is function
    params.chain_type = 'rfps'
    assert params.values_function is function


def test_default_protocol_values_function_rebuilt_for_a_copy(tmp_path):
    params = load(tmp_path)
    copy = params.copy()
    assert isinstance(copy.values_function, ProtocolValuesFunction)
    assert copy.values_function.params is copy
    assert params.values_function.params is params


def test_user_values_function_kept(tmp_path):
    params = load(tmp_path, '''
        def values_function(descriptors):
            return np.zeros(len(descriptors))
        ''')
    assert not isinstance(params.values_function, ProtocolValuesFunction)
    assert not params._default_values_function


def test_rfps_unaffected(tmp_path, capsys):
    """An RFPS params file with a plain network: no C-GAIMMD rule or warning, AIMMD's default."""
    params = load(tmp_path, '''
        chain_type = 'rfps'
        import torch
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
            def forward(self, x):
                return self.linear(torch.flatten(x, 1)[:, :1])
        network = Network()
        gen_temperature = -1.0
        round_shooting_index = False
        descriptors_function = None
        cgaimmd_parameters = None
        ''')
    assert params.chain_type == 'rfps'
    assert not isinstance(params.values_function, ProtocolValuesFunction)
    assert params.values_function.__name__ == 'values_function'
    out = printed(capsys)
    assert 'C-GAIMMD' not in out and "chain_type='cgaimmd'" not in out
    assert 'WARNING' not in out and 'cgaimmd_* hooks' not in out
    assert 'brute-force shooting' in out                             # the RFPS warning, unchanged


# ---------------------------------------------------------------------------
# cgaimmd_generator: how candidates are generated (aimmd.cgaimmd.generation)
# ---------------------------------------------------------------------------

COMMITTOR_ONLY_NETWORK = '''
    import torch
    class Network(torch.nn.Module):
        """A C-GAIMMD network with only a committor (no generation members)."""
        n_atoms = 1
        lambda_clip = 10.0
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
        def forward(self, x):
            return self.linear(torch.flatten(torch.as_tensor(x, dtype=torch.float32), 1)[:, :1])
    network = Network()
    '''

PARAMS_FILE_GENERATOR = '''
    import torch
    from aimmd.cgaimmd.generation import register_generator
    class ConstantGenerator:
        """Every candidate at one position: needs nothing from the network but n_atoms."""
        method = 'params_file_constant'
        def __init__(self, x=0.0):
            self.x = float(x)
        def check(self, network):
            return []
        def generate(self, network, lam, *, generator=None):
            return torch.full((len(lam), network.n_atoms, 3), self.x)
        def settings(self):
            return {'x': self.x}
    register_generator('params_file_constant', ConstantGenerator, replace=True)
    '''


def test_generator_default_is_the_network_default(tmp_path):
    params = load(tmp_path)
    assert params.cgaimmd_generator is None
    assert generation.make_generator(params.cgaimmd_generator, params.network) == ODEGenerator()


def test_a_committor_only_network_needs_a_generator(tmp_path):
    """No default_generator and no cgaimmd_generator: the error names the field."""
    with pytest.raises(TypeError, match='params.cgaimmd_generator must name the generation '
                                        'method'):
        load(tmp_path, COMMITTOR_ONLY_NETWORK)
    # the 'ode' method names what the network lacks
    with pytest.raises(TypeError, match=r"generator 'ode': needs velocity\(x_t, t, lam"):
        load(tmp_path / 'ode', COMMITTOR_ONLY_NETWORK + "cgaimmd_generator = 'ode'\n")


def test_a_generator_registered_in_the_params_file(tmp_path, capsys):
    """A committor-only network with a method the params file registers: it loads and generates."""
    params = load(tmp_path, COMMITTOR_ONLY_NETWORK + PARAMS_FILE_GENERATOR + '''
    cgaimmd_generator = {'method': 'params_file_constant', 'x': -1.5}
    ''')
    assert params.cgaimmd_generator == {'method': 'params_file_constant', 'x': -1.5}
    generator = generation.make_generator(params.cgaimmd_generator, params.network)
    assert generator.method == 'params_file_constant' and generator.settings() == {'x': -1.5}
    out = generator.generate(params.network, torch.zeros(3))
    assert out.shape == (3, 1, 3) and torch.all(out == -1.5)
    # under chain_type='cgaimmd' its values function is the protocol one, without a
    # default_generator (which an RFPS run would need to pick it)
    assert isinstance(params.values_function, ProtocolValuesFunction)
    params.chain_type = 'rfps'
    assert not isinstance(params.values_function, ProtocolValuesFunction)
    params.chain_type = 'cgaimmd'
    assert isinstance(params.values_function, ProtocolValuesFunction)
    # a wrong setting is an error when the params are loaded
    with pytest.raises(TypeError, match="'params_file_constant'.*'y'"):
        params.cgaimmd_generator = {'method': 'params_file_constant', 'y': 1.0}
    assert params.cgaimmd_generator == {'method': 'params_file_constant', 'x': -1.5}


def test_generator_validated_on_assignment(tmp_path):
    params = load(tmp_path)
    params.cgaimmd_generator = 'ode'
    params.cgaimmd_generator = {'method': 'ode', 'n_steps': 4, 'sampler': 'rk4'}
    params.cgaimmd_generator = {'chunk': 8}                    # settings of the default method
    assert generation.make_generator(params.cgaimmd_generator, params.network) == \
        ODEGenerator(chunk=8)
    params.cgaimmd_generator = None
    for value, message in [('no_such_method', 'unknown generation method'),
                           (3, 'must be a method name'),
                           ({'method': 'no_such_method'}, 'unknown generation method'),
                           ({'method': 'ode', 'n_steps': 0}, "'ode'.*n_steps"),
                           ({'method': 'ode', 'bogus': 1}, "'ode'.*bogus"),
                           ({'method': 'ode', 'chunk': object()}, 'JSON-serializable'),
                           ({1: 2}, 'keys of a generator settings dict')]:
        with pytest.raises(TypeError, match=message):
            params.cgaimmd_generator = value
        assert params.cgaimmd_generator is None
    # lists become tuples (as in the other settings dicts)
    params.chain_type = 'rfps'                       # a setting the 'ode' method does not take
    params.cgaimmd_generator = {'chunk': 8, 'note': [1, 2]}
    assert params.cgaimmd_generator == {'chunk': 8, 'note': (1, 2)}


def test_generator_object_saved(tmp_path, monkeypatch):
    """A generator object is kept; it is saved as its settings (aimmd class) or imported."""
    params = load(tmp_path, PARAMS_FILE_GENERATOR + '''
    cgaimmd_generator = ConstantGenerator(x=2.0)
    ''')
    assert params.cgaimmd_generator.method == 'params_file_constant'
    monkeypatch.chdir(params.parent)
    text = open(params.save()).read()
    assert 'from params import cgaimmd_generator' in text
    params.cgaimmd_generator = ODEGenerator(n_steps=3)
    assert isinstance(params.cgaimmd_generator, ODEGenerator)
    saved = params.save()
    text = open(saved).read()
    assert ("cgaimmd_generator = {'method': 'ode', 'sampler': 'midpoint', 'n_steps': 3, "
            "'chunk': 128}") in text
    reloaded = aimmd.Params.load(saved, save=False)
    assert generation.make_generator(reloaded.cgaimmd_generator) == ODEGenerator(n_steps=3)


def test_generator_bioemu2_without_the_extra(tmp_path):
    """A method whose module needs an extra is accepted by name; building it says what is missing."""
    params = load(tmp_path)
    params.chain_type = 'rfps'                       # the cgaimmd rules would build it now
    params.cgaimmd_generator = 'bioemu2_edm'
    assert params.cgaimmd_generator == 'bioemu2_edm'
    if importlib.util.find_spec('aimmd.cgaimmd.bioemu2') is None:
        # on assignment the ImportError is wrapped like every failed check
        with pytest.raises(TypeError, match="needs the 'bioemu2' extra"):
            params.chain_type = 'cgaimmd'
        assert params.chain_type == 'rfps'
        with pytest.raises(ImportError, match="needs the 'bioemu2' extra"):
            load(tmp_path / 'load', "cgaimmd_generator = 'bioemu2_edm'")


# ---------------------------------------------------------------------------
# rule 11 reads the run's descriptors series
# ---------------------------------------------------------------------------

def test_descriptor_width_reads_the_named_series(tmp_path):
    params = load(tmp_path, "descriptors_series = 'descriptors-cg'")
    path = params.initial_paths[0]
    assert 'descriptors-cg' in path.__dict__ and 'descriptors' not in path.__dict__
    params._check_cgaimmd_descriptor_width(path)
    # the stored series array is what is checked (not a recomputation)
    path.__dict__['descriptors-cg'] = np.zeros((len(path), 2), dtype=np.float32)
    with pytest.raises(TypeError, match=r"width 2 .* series 'descriptors-cg'"):
        params._check_cgaimmd_descriptor_width(path)
    # without a stored array, the first frame is featurized
    del path.__dict__['descriptors-cg']
    params._check_cgaimmd_descriptor_width(path)


def test_descriptor_width_error_under_a_named_series(tmp_path):
    with pytest.raises(TypeError, match=r"width 2 .* 3 x 1 = 3.*series 'descriptors-two'"):
        load(tmp_path, '''
            descriptors_series = 'descriptors-two'
            def descriptors_function(trajectory):
                x = np.array([ts.positions[0, 0] for ts in trajectory], dtype=np.float32)
                return np.stack([x, x ** 2], axis=1)
            ''')
