from pathlib import Path
import shutil

import numpy as np

import aimmd
from tests._helpers_unit import build_params_file, build_path, simple_descriptors_function


def test_placeholder_params_properties():
    """`Params.placeholder` should expose a coherent minimal configuration."""

    params = aimmd.Params.placeholder
    assert params.sorted_states == "ARB"
    assert params.compute_states_args[1] == "states"
    assert params.compute_values_args[1] == "values"
    assert len(params.pipeline) == 2


def test_params_load_save_update_and_paths(tmp_path):
    """Load, update, save, and re-read a tiny toy-engine parameter set."""

    initial = build_path(
        tmp_path,
        stem="initial",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    params_file = build_params_file(tmp_path, initial.fname)

    params = aimmd.Params.load(params_file, save=False)
    assert params.engine == "toy"
    assert len(params.initial_paths) == 1
    assert params.path.is_file()

    # Saving should emit a canonical params file that can serve as the run's
    # source of truth.
    # Save explicitly inside `tmp_path` so the test cannot leak a top-level
    # `params.py` into the repository when AIMMD resolves relative paths.
    saved = Path(params.save(tmp_path / "saved_params.py"))
    assert saved.exists()

    # `Params.update()` saves by default, so keep this update in-memory only to
    # avoid writing a stray top-level `params.py` during the test run.
    params.update(nbins=5, descriptors_function=simple_descriptors_function, save=False)
    assert params.nbins == 5
    assert params.compute_descriptors_args[1] == "descriptors"

    # `Params.pathensemble` expects an AIMMD-style run folder, so we create the
    # minimal chain layout it knows how to scan.
    run_dir = tmp_path / "run"
    chain_dir = run_dir / "chainR0"
    chain_dir.mkdir(parents=True)
    shutil.copy2(initial.fname, chain_dir / "path000001.xtc")
    ensemble = params.pathensemble(run_dir)
    assert len(ensemble) >= 1


def test_params_validation_rejects_bad_network(tmp_path):
    """Network assignment should fail if the runtime interface is incomplete."""

    initial = build_path(
        tmp_path,
        stem="initial2",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    params_file = build_params_file(tmp_path, initial.fname)
    params = aimmd.Params.load(params_file, save=False)

    class BadNetwork:
        pass

    try:
        params.update(network=BadNetwork())
    except TypeError:
        pass
    else:
        raise AssertionError("Expected network validation failure")
