"""Unit tests for `flexible_sampling` features that branch did not cover, plus
the backward-compatibility / signature invariants established when merging it
with `multi_system_shared_committor`.

These are deliberately lightweight (no GROMACS, no real MD): they pin down the
public API surface and the value-processing rules that guarantee old params
files and old job scripts keep working.
"""
import inspect
from math import inf

import numpy as np
import pytest

import aimmd
from tests._helpers_unit import build_params_file, build_path


# ---------------------------------------------------------------------------
# Backward-compatible launcher API (merge Decision 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["create_job", "run"])
def test_nchains_per_worker_is_last_launcher_arg(method):
    """`nchains_per_worker` must be the LAST parameter of the public launcher
    entry points so that old positional/keyword calls (which never passed it)
    keep binding nsteps/nframes/walltime/... exactly as before.
    """
    sig = inspect.signature(getattr(aimmd.Launcher, method))
    names = [p for p in sig.parameters if p != "self"]
    assert names[-1] == "nchains_per_worker", (
        f"{method}: nchains_per_worker must stay last, got {names}")
    assert sig.parameters["nchains_per_worker"].default == 1


def test_old_positional_create_job_call_binds_correctly():
    """An old-style positional create_job(...) call (pre-nchains) must still map
    its positional nsteps/nframes onto the right parameters."""
    sig = inspect.signature(aimmd.Launcher.create_job)
    # bind the historical positional signature (filename, n, n1, n2, modes,
    # nsteps, nframes) exactly as an old create_jobscript.py would have called it
    bound = sig.bind_partial(
        object(), "job.sh", 5, 1, 1, "chain", "free", "free", 10000, 25000)
    bound.apply_defaults()
    assert bound.arguments["nsteps"] == 10000
    assert bound.arguments["nframes"] == 25000
    assert bound.arguments["nchains_per_worker"] == 1


# ---------------------------------------------------------------------------
# Merged Worker.shoot / _shoot signature (both branches' args, fixed order)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["shoot", "_shoot"])
def test_shoot_signature_has_sweep_target_then_nchains(name):
    """The merged shooting entry points must carry BOTH branches' added
    arguments in the order the launcher builds the run_args tuple:
    (target_state, k, sweep, sweep_target, nchains_per_worker).
    """
    sig = inspect.signature(getattr(aimmd.Worker, name))
    names = [p for p in sig.parameters if p != "self"]
    assert names == ["target_state", "k", "sweep",
                     "sweep_target", "nchains_per_worker"], names
    assert sig.parameters["nchains_per_worker"].default == 1
    assert sig.parameters["sweep_target"].default == inf


# ---------------------------------------------------------------------------
# Merged worker.utils helper signatures relied on by the merged _shoot
# ---------------------------------------------------------------------------

def test_selection_helpers_keep_both_branches_kwargs():
    """update_selection_pool keeps flexible's `boundaries`; select_shooting_point
    keeps flexible's `shooting_chains`; register_path keeps multi's
    `bias_function` — the merged _shoot depends on all three.
    """
    from aimmd.worker.utils import (update_selection_pool,
                                     select_shooting_point, register_path)
    assert "boundaries" in inspect.signature(update_selection_pool).parameters
    assert "shooting_chains" in inspect.signature(select_shooting_point).parameters
    assert "bias_function" in inspect.signature(register_path).parameters


# ---------------------------------------------------------------------------
# fit.py in-state anchor selection (merge Decision 2)
# ---------------------------------------------------------------------------

def test_category_specs_uses_broad_in_state_anchors():
    """Decision 2: the in-state anchor categories select ALL in-a / in-b frames
    (t==a / t==b) regardless of path history, so aa*/bb* frames anchor the
    basins. Reactive (free/shot) categories keep their (i,t,f,s) masks.
    """
    from aimmd.network.fit import _category_specs

    a, r, b = "A", "R", "B"
    # frame 0: aa* (started in A, currently in A) -> only the broad mask keeps it
    i = np.array(["A", "R", "B", "A"])
    t = np.array(["A", "A", "B", "R"])
    f = np.array(["A", "A", "B", "A"])
    s = np.array(["A", "R", "B", "A"])
    specs = {name: mask for name, mask, _ in _category_specs(i, t, f, s, a, r, b)}

    np.testing.assert_array_equal(specs["in1"], (t == a))
    np.testing.assert_array_equal(specs["in2"], (t == b))
    # the aa* frame (index 0: i='A', t='A') must be included as an in-A anchor
    assert specs["in1"][0]
    # in-state anchors carry no committor 'values' (so rates are unaffected)
    has_values = {name: hv for name, _m, hv in _category_specs(i, t, f, s, a, r, b)}
    assert has_values["in1"] is False and has_values["in2"] is False


# ---------------------------------------------------------------------------
# New Params fields from flexible_sampling: defaults + backward-compat
# ---------------------------------------------------------------------------

def _toy_params(tmp_path):
    initial = build_path(
        tmp_path, stem="initial",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                           dtype=np.float32),
    )
    params_file = build_params_file(tmp_path, initial.fname)
    return aimmd.Params.load(params_file, save=False)


def test_new_flexible_fields_have_backward_compatible_defaults(tmp_path):
    """An old params file (which does not set these) loads with defaults that
    reproduce the previous behaviour.
    """
    params = _toy_params(tmp_path)
    assert params.always_select_inside_the_bins is False
    assert params.shared_density_adjustment is False
    # density_adjustment default = inf (correct over all recent selection points)
    assert params.density_adjustment == inf


@pytest.mark.parametrize("given, expected", [
    (True, inf),     # legacy bool True  -> inf (all selection points)
    (False, 0),      # legacy bool False -> 0   (disabled)
    (3, 3),          # explicit count kept
    (3.4, 3),        # rounded
    (inf, inf),      # inf kept
])
def test_density_adjustment_bool_to_number_conversion(tmp_path, given, expected):
    """`density_adjustment` changed type bool -> Number; old bool configs must be
    converted so behaviour is preserved (True -> inf, False -> 0). Guards the
    `isinstance(name, bool)` -> `isinstance(value, bool)` bugfix: without it, an
    old `density_adjustment=True` silently degrades to ``round(True) == 1``.
    """
    params = _toy_params(tmp_path)
    params.update(density_adjustment=given, save=False)
    assert params.density_adjustment == expected


def test_always_select_inside_the_bins_accepted(tmp_path):
    """The flag is a real Params field that can be flipped on without error."""
    params = _toy_params(tmp_path)
    params.update(always_select_inside_the_bins=True, save=False)
    assert params.always_select_inside_the_bins is True
