"""Extra coverage for projection/report mixins on small synthetic ensembles."""

from io import StringIO

import numpy as np
import pytest

import aimmd
from aimmd.pathensemble._report import PathEnsembleReport
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import build_path, write_trajectory


def test_project_respects_where_and_value_filters(tmp_path):
    """Projection should drop boundary frames and honor value-range filters.

    The helper is used heavily in density estimation, so we pin down two pieces
    of logic here:
    - `where="internal"` excludes end-state boundary frames.
    - `vmin`/`vmax` turn into a per-frame mask before histogramming.
    """

    path = build_path(
        tmp_path,
        stem="project",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
        values=np.array([-1.0, 0.0, 1.0]),
    )
    ensemble = aimmd.PathEnsemble(path)

    histogram = ensemble.project(
        bins=[-2.0, -0.5, 0.5, 2.0],
        source="values",
        where="internal",
        vmin=-0.1,
        vmax=0.1,
    )
    # Only the middle reactive frame survives both the "internal" slicing and
    # the value-range filter, so exactly one count should land in the center bin.
    np.testing.assert_allclose(histogram, np.array([0.0, 1.0, 0.0]))


def test_report_builds_histograms_and_summary(tmp_path):
    """The text report should summarize path types and per-path histograms."""

    path1 = build_path(
        tmp_path,
        stem="report1",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
        values=np.array([-1.0, 0.0, 1.0]),
    )
    path2 = build_path(
        tmp_path,
        stem="report2",
        positions=np.array([[[1, 0, 0]], [[0, 0, 0]], [[-1, 0, 0]]], dtype=np.float32),
        values=np.array([1.0, 0.0, -1.0]),
    )
    ensemble = aimmd.PathEnsemble(path1, path2)

    text, histograms = ensemble.report(bins=[-2.0, 0.0, 2.0], summary=True)
    assert "Summary" in text
    assert "Filenames" in text
    assert histograms.shape == (2, 2)


def test_print_report_currently_raises_due_to_undefined_log_file(tmp_path):
    """The current implementation has a known `log_file` bug.

    The goal is not to bless the bug, but to document the current behavior so a
    future fix will be explicit and intentional.
    """

    path = build_path(tmp_path, stem="print_report")
    ensemble = aimmd.PathEnsemble(path)
    with pytest.raises(UnboundLocalError):
        ensemble.print_report()


class DummyReport(PathEnsembleReport):
    """Small fake host that only implements `shooting_results`."""

    def shooting_results(self, states, sweep_size):
        return [(1, 3), (3, 1)]


def test_report_shooting_results_prints_committor_table(capsys):
    """Shooting-result reporting should include both probability and logit data."""

    DummyReport().report_shooting_results(states="ARB", sweep_size=0, alpha=0.95)
    output = capsys.readouterr().out
    assert "committor" in output
    assert "logit" in output


def test_project_internal_trim_uses_path_length_not_values_length(tmp_path):
    """`where="internal"` must trim in the path's index space, not the cache's.

    `.values.npy` is one element shorter than the trajectory by construction:
    values are computed only on the reactive region (see the `compute_condition`
    in `worker/_train.py`), so a path that ends in a state never has its final
    frame scored and the cache file stops one short.

    Deriving the end-of-path trim from ``len(input_data)`` therefore removes the
    last *real* internal frame, believing it to be the terminal state frame --
    which was never in the array at all. The trim must come from the segment
    length, which is what `Path._range("internal")` and `PathEnsemble.n_frames`
    already use.
    """

    # five frames: A R R R B  (x <= -0.5 -> A, x >= 0.5 -> B, else R)
    positions = np.array(
        [[[-1.0, 0, 0]], [[0.0, 0, 0]], [[0.1, 0, 0]], [[0.2, 0, 0]], [[1.0, 0, 0]]],
        dtype=np.float32,
    )
    # values exist for frames 0..3 only; frame 4 is state B and was never scored
    path = build_path(
        tmp_path,
        stem="short_values",
        positions=positions,
        values=np.array([0.0, -1.0, -0.5, 0.5]),
    )
    assert len(path) == 5
    # the FILE is one element short. Note `path.values` reports 5 because
    # `Path._get` zero-pads to the path length, while `_extract` -- which is what
    # `project` reads -- does not.
    on_disk = np.load(get_cache_fname(path._fnames[0], "values"))
    assert len(on_disk) == 4, "the cache file must be one element short"
    assert len(np.asarray(path._extract(0, "values"))) == 4
    assert path.n_frames == 3, "frames 1, 2 and 3 are internal"

    ensemble = aimmd.PathEnsemble(path)
    histogram = ensemble.project(
        bins=[-2.0, 2.0], source="values", where="internal"
    )
    assert histogram.sum() == path.n_frames, (
        f"project() binned {histogram.sum()} frames but the path has "
        f"{path.n_frames} internal frames"
    )


def test_project_keeps_single_frame_excursions(tmp_path):
    """A three-frame excursion has one internal frame and must not vanish.

    With ``stop = len(values) - 1`` the slice collapses to ``values[1:1]`` and the
    whole path contributes nothing, silently. These smallest excursions are the
    most common shape in a shooting run and carry the largest weights, so losing
    them biases the shooting-point selection density they feed.
    """

    # three frames: A R A
    positions = np.array(
        [[[-1.0, 0, 0]], [[0.0, 0, 0]], [[-1.0, 0, 0]]], dtype=np.float32
    )
    path = build_path(
        tmp_path,
        stem="tiny_excursion",
        positions=positions,
        values=np.array([0.0, -0.5]),
    )
    assert path.n_frames == 1
    ensemble = aimmd.PathEnsemble(path)
    histogram = ensemble.project(
        bins=[-2.0, 2.0], source="values", where="internal"
    )
    assert histogram.sum() == 1, "the single internal frame was dropped"


def test_project_internal_trim_only_at_path_ends_for_multi_file_paths(tmp_path):
    """For a multi-segment path the trims belong to the outer segments only.

    A segment whose own last frame is still reactive was fully scored, so only the
    final segment's cache is short -- which is exactly where the trailing trim
    fires. The middle of the path must not be trimmed at all.
    """

    seg0 = write_trajectory(
        tmp_path,
        stem="multi0",
        positions=np.array([[[-1.0, 0, 0]], [[0.0, 0, 0]], [[0.1, 0, 0]]],
                           dtype=np.float32),
    )
    seg1 = write_trajectory(
        tmp_path,
        stem="multi1",
        positions=np.array([[[0.2, 0, 0]], [[0.3, 0, 0]], [[1.0, 0, 0]]],
                           dtype=np.float32),
    )
    # seg0 ends reactive -> fully scored (3 values); seg1 ends in B -> one short
    save_npy(get_cache_fname(seg0, "states"), np.array(list("ARR"), dtype="<U1"))
    save_npy(get_cache_fname(seg0, "values"), np.array([0.0, -1.0, -0.5]))
    save_npy(get_cache_fname(seg1, "states"), np.array(list("RRB"), dtype="<U1"))
    save_npy(get_cache_fname(seg1, "values"), np.array([0.25, 0.5]))

    path = aimmd.Path([seg0, seg1])
    assert path.n_files == 2
    assert len(path) == 6
    assert path.n_frames == 4, "frames 1..4 of the concatenated path"

    ensemble = aimmd.PathEnsemble(path)
    histogram = ensemble.project(
        bins=[-2.0, 2.0], source="values", where="internal"
    )
    assert histogram.sum() == path.n_frames
