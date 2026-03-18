from pathlib import Path

import numpy as np

from aimmd.engines.toy import ToyEngine
from tests._helpers_unit import write_trajectory


def test_toy_engine_append_and_noappend(tmp_path):
    """The toy engine should support both append and part-file continuation."""

    deffnm = str(tmp_path / "toy")
    write_trajectory(tmp_path, stem="toy", positions=np.array([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32))

    def mdrun(ts):
        ts.positions[:] = ts.positions + 1.0
        ts.time = ts.time + 1.0

    engine = ToyEngine(mdrun=mdrun, slowdown=0.0)
    code = engine(deffnm, backup=False, stop_condition=lambda: True, raise_if_failure=False, log_file=None)
    # The current toy engine exits without an explicit return code on the normal
    # stop path, so `None` is the behavior to preserve here.
    assert code is None
    assert Path(f"{deffnm}.xtc").exists()

    part0 = tmp_path / "toy.part0000.xtc"
    Path(f"{deffnm}.xtc").rename(part0)
    code = engine(deffnm, backup=False, noappend=True, stop_condition=lambda: True, raise_if_failure=False, log_file=None)
    assert code is None
