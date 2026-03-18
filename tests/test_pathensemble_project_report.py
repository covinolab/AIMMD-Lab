"""Extra coverage for projection/report mixins on small synthetic ensembles."""

from io import StringIO

import numpy as np
import pytest

import aimmd
from aimmd.pathensemble._report import PathEnsembleReport
from tests._helpers_unit import build_path


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
