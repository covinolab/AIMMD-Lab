"""A failed `Params.load` leaves the interpreter's module registry usable.

`Params.load` executes the params file with its folder's local modules
temporarily renamed in `sys.modules`, and restores the previous modules when
anything fails. It used to do so by rebinding ``sys.modules`` to a copy. The
import system keeps using the dict it was started with, so the two diverged:
imports that ran afterwards in the same process could fail half way, e.g.
``import torch_geometric`` with "cannot import name 'distributed' from
partially initialized module". A params file with an invalid value (or a
typo) in a long-lived session (notebook, test suite) was enough.

Only the modules of the params folder are dropped after a failure. Packages
the params file imported for the first time stay imported: some cannot be
executed twice in one process (torch_geometric registers names in torch), so
deleting them made the corrected params file fail to load in the same
session.
"""
import sys

import pytest

import aimmd


def _write_failing_params(folder, error="raise RuntimeError('broken params')"):
    (folder / "local_helper_for_load_failure.py").write_text("VALUE = 1\n")
    (folder / "params.py").write_text(
        "import local_helper_for_load_failure\n"
        f"{error}\n")
    return folder / "params.py"


def test_failed_load_restores_sys_modules_in_place(tmp_path):
    modules = sys.modules
    before = dict(sys.modules)

    with pytest.raises(RuntimeError, match="broken params"):
        aimmd.Params.load(str(_write_failing_params(tmp_path)), save=False)

    assert sys.modules is modules
    # the modules of the params folder are gone, the others are untouched
    assert not [name for name in sys.modules
                if name.endswith("local_helper_for_load_failure")
                or name == "params"]
    assert all(sys.modules.get(name) is module
               for name, module in before.items())


def test_failed_validation_restores_sys_modules_in_place(tmp_path):
    modules = sys.modules
    params_file = _write_failing_params(
        tmp_path, error="def states_function(trajectory):\n"
                        "    return None\n"
                        "descriptors_series = 'not-a-series'")

    with pytest.raises(ValueError, match="descriptors_series"):
        aimmd.Params.load(str(params_file), save=False)

    assert sys.modules is modules


def _write_package(folder):
    """A package whose subpackage imports itself while being initialized,
    as torch_geometric.distributed does."""
    package = folder / "pkg_for_load_failure"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("from . import sub\n")
    (package / "sub" / "__init__.py").write_text("from .part import VALUE\n")
    (package / "sub" / "part.py").write_text(
        "import pkg_for_load_failure.sub as sub\nVALUE = 1\n")


def test_imports_work_after_a_failed_load(tmp_path, monkeypatch):
    _write_package(tmp_path / "site")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    (tmp_path / "run").mkdir()
    with pytest.raises(RuntimeError):
        aimmd.Params.load(str(_write_failing_params(tmp_path / "run")),
                          save=False)

    import pkg_for_load_failure
    assert pkg_for_load_failure.sub.VALUE == 1
    for name in [name for name in sys.modules
                 if name.startswith("pkg_for_load_failure")]:
        del sys.modules[name]


def _write_registering_module(folder):
    """A module that registers a name, at import, in the registry of an
    already imported module and refuses a second registration, as
    torch_geometric does in torch's DataPipe registry ("Unable to add
    DataPipe function name batch_graphs as it is already taken")."""
    folder.mkdir()
    (folder / "registry_for_load_failure.py").write_text("NAMES = set()\n")
    (folder / "registrant_for_load_failure.py").write_text(
        "import registry_for_load_failure as registry\n"
        "if 'batch_graphs' in registry.NAMES:\n"
        "    raise RuntimeError('batch_graphs is already taken')\n"
        "registry.NAMES.add('batch_graphs')\n")


def test_modules_first_imported_by_a_failed_load_stay(tmp_path, monkeypatch):
    """A package the failed params file imported first (e.g. torch_geometric)
    cannot be executed a second time in the same process: it stays imported,
    so that the corrected params file loads (and imports it) again."""
    _write_registering_module(tmp_path / "site")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    import registry_for_load_failure  # noqa: F401  (imported before, as torch)
    (tmp_path / "run").mkdir()
    params_file = _write_failing_params(
        tmp_path / "run", error="import registrant_for_load_failure\n"
                                "raise RuntimeError('broken params')")
    try:
        with pytest.raises(RuntimeError, match="broken params"):
            aimmd.Params.load(str(params_file), save=False)
        registrant = sys.modules.get("registrant_for_load_failure")
        assert registrant is not None

        # the retry gets as far as the params file's own error again
        with pytest.raises(RuntimeError, match="broken params"):
            aimmd.Params.load(str(params_file), save=False)
        assert sys.modules["registrant_for_load_failure"] is registrant
        import registrant_for_load_failure  # noqa: F401
        # the modules of the params folder are still dropped
        assert not [name for name in sys.modules
                    if name.endswith("local_helper_for_load_failure")
                    or name == "params"]
    finally:
        for name in ("registry_for_load_failure",
                     "registrant_for_load_failure"):
            sys.modules.pop(name, None)
