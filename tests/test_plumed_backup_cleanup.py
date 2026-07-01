"""Regression test for the PLUMED backup pile-up crash.

PLUMED rotates any existing output file to ``bck.N.<file>`` and hard-aborts once
there are 100 backups. Free (freeA/freeB) simulations share one accumulating
``COLVAR`` per ``free{t}`` folder and PLUMED reopens it non-restart at each new
trajectory, so ``bck.N.COLVAR`` piled up to 100 and crashed the run; shooting
chains never hit this because they rename ``COLVAR`` away every segment.

``run_simulation`` must therefore clear ``bck.*`` (and ``PLUMED.OUT``) in the
deffnm directory before every mdrun, while leaving the live ``COLVAR`` and
unrelated files untouched.
"""
import pytest

import aimmd


def _gmx_params():
    params = aimmd.Params.placeholder.copy()
    # engine='gromacs' selects the mdrun path; record_bias=False skips the
    # COLVAR rename/slice so this test isolates the bck.* cleanup.
    params.__dict__.update(
        engine="gromacs",
        gmx_mdrun="gmx mdrun",
        trajectory_extension=".xtc",
        record_bias=False,
    )
    return params


@pytest.mark.parametrize("noappend", [True, False])  # free and shoot deffnm dirs
def test_run_simulation_clears_plumed_backups(tmp_path, monkeypatch, noappend):
    # do not launch gmx; just capture that mdrun was invoked
    calls = {}

    def fake_execute(command, **kwargs):
        calls["cmd"] = command
        return 0

    monkeypatch.setattr("aimmd.params._methods.execute_command", fake_execute)

    folder = tmp_path / "freeA"
    folder.mkdir()
    # an accumulated PLUMED mess (mixed backup targets) + files that must survive
    for name in ("bck.0.COLVAR", "bck.1.COLVAR", "bck.99.COLVAR",
                 "bck.0.KERNELS", "PLUMED.OUT"):
        (folder / name).write_text("x")
    (folder / "COLVAR").write_text("live colvar")   # live file: must survive
    (folder / "traj000001.tpr").write_text("tpr")    # unrelated: must survive

    params = _gmx_params()
    params.run_simulation(str(folder / "traj000001"), noappend=noappend)

    # every PLUMED backup and the rotating log are gone
    assert not list(folder.glob("bck.*")), "bck.* backups were not cleared"
    assert not (folder / "PLUMED.OUT").exists()
    # the live COLVAR and unrelated files are untouched
    assert (folder / "COLVAR").read_text() == "live colvar"
    assert (folder / "traj000001.tpr").exists()
    # cleanup happened as part of a real mdrun invocation
    assert "mdrun" in calls["cmd"]
