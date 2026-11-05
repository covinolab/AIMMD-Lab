"""Device roles of C-GAIMMD code, ``aimmd.cgaimmd.devices``."""

from types import SimpleNamespace

import pytest
import torch

import aimmd
import aimmd._config as config
from aimmd.cgaimmd import devices
from aimmd.params.utils import CGAIMMD_PARAMETER_DEFAULTS, is_device_spec


def _no_cuda(monkeypatch, count=0):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: count > 0)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: count)


def test_auto_is_cpu_without_cuda(monkeypatch):
    _no_cuda(monkeypatch)
    assert devices.resolve_device('auto') == torch.device('cpu')
    # the default lane of the suite really has no visible GPU
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        monkeypatch.undo()
        assert devices.resolve_device('auto') == torch.device('cpu')


def test_explicit_specs(monkeypatch):
    _no_cuda(monkeypatch, count=2)
    assert devices.resolve_device('cpu') == torch.device('cpu')
    assert devices.resolve_device('auto') == torch.device('cuda', 0)
    assert devices.resolve_device('cuda') == torch.device('cuda', 0)
    assert devices.resolve_device('cuda:1') == torch.device('cuda', 1)
    with pytest.raises(RuntimeError, match='cuda:2'):
        devices.resolve_device('cuda:2')
    _no_cuda(monkeypatch, count=0)
    for spec in ('cuda', 'cuda:0'):
        with pytest.raises(RuntimeError, match='no CUDA device'):
            devices.resolve_device(spec)


@pytest.mark.parametrize('spec', ['gpu', 'cuda:', 'CPU', ' cpu', 'cuda:-1', 'mps', '', None, 0,
                                  torch.device('cpu')])
def test_bad_spec(spec):
    assert not is_device_spec(spec)
    with pytest.raises(ValueError):
        devices.resolve_device(spec)


def test_is_device_spec():
    for spec in ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:12'):
        assert is_device_spec(spec)


def _recording_resolver(monkeypatch):
    monkeypatch.setattr(devices, 'resolve_device', lambda spec: f'resolved {spec}')


def test_values_role_in_train_task(monkeypatch):
    _recording_resolver(monkeypatch)
    params = aimmd.Params.placeholder.copy()
    params.__dict__['cgaimmd_parameters'] = {'generation_device': 'cuda:1', 'values_device': 'cpu',
                                          'training_device': 'cuda:0'}
    for task in ('train', 'kinetics_convergence'):                  # the tasks that fit: their value passes
        monkeypatch.setattr(config, 'WORKER_TASK', task)
        assert devices.current_task() == task
        assert devices.device_for('values', params) == 'resolved cuda:0'
        assert devices.device_for('training', params) == 'resolved cuda:0'
        assert devices.device_for('generation', params) == 'resolved cuda:1'
    for task in ('shoot', 'free'):
        monkeypatch.setattr(config, 'WORKER_TASK', task)
        assert devices.device_for('values', params) == 'resolved cpu'
        assert devices.device_for('training', params) == 'resolved cuda:0'
        assert devices.device_for('generation', params) == 'resolved cuda:1'
    with pytest.raises(ValueError, match='role'):
        devices.device_for('value', params)


def test_none_outside_task_without_cuda_query(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('torch.cuda queried outside a worker task')

    for name in ('is_available', 'device_count', 'current_device', 'init', 'is_initialized',
                 'get_device_name', 'mem_get_info'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(config, 'WORKER_TASK', None)
    assert devices.current_task() is None
    params = aimmd.Params.placeholder.copy()
    params.__dict__['cgaimmd_parameters'] = {'values_device': 'cuda:3'}
    for role in devices.ROLES:
        assert devices.device_for(role, params) is None
        assert devices.device_for(role, SimpleNamespace()) is None
    # an unknown role is an error outside a task too
    with pytest.raises(ValueError, match='role'):
        devices.device_for('value', params)


def test_defaults_for_namespace_params(monkeypatch):
    _recording_resolver(monkeypatch)
    monkeypatch.setattr(config, 'WORKER_TASK', 'shoot')
    assert devices.ROLES == ('generation', 'values', 'training')
    # a plain namespace without cgaimmd_settings: CGAIMMD_PARAMETER_DEFAULTS
    for role in devices.ROLES:
        assert devices.device_for(role, SimpleNamespace()) == (
            f"resolved {CGAIMMD_PARAMETER_DEFAULTS[f'{role}_device']}")
    # ... updated with its cgaimmd_parameters, if it has them
    namespace = SimpleNamespace(cgaimmd_parameters={'values_device': 'cpu'})
    assert devices.device_for('values', namespace) == 'resolved cpu'
    assert devices.device_for('generation', namespace) == 'resolved auto'
    assert devices.device_for('values', SimpleNamespace(cgaimmd_parameters=None)) == 'resolved auto'


def test_worker_task_default_and_resolution_per_call(monkeypatch):
    """WORKER_TASK starts as None; devices are resolved at every call, never cached."""
    assert config.WORKER_TASK is None
    _no_cuda(monkeypatch, count=0)
    params = aimmd.Params.placeholder.copy()
    params.__dict__['cgaimmd_parameters'] = {'training_device': 'cpu', 'values_device': 'auto'}
    monkeypatch.setattr(config, 'WORKER_TASK', 'train')
    assert devices.device_for('values', params) == torch.device('cpu')
    monkeypatch.setattr(config, 'WORKER_TASK', 'shoot')
    _no_cuda(monkeypatch, count=1)
    assert devices.device_for('values', params) == torch.device('cuda', 0)
    monkeypatch.setattr(config, 'WORKER_TASK', None)
    assert devices.device_for('values', params) is None
