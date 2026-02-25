"""
aimmd.pathensemble._report
=========================

Human-readable reporting utilities for :class:`aimmd.pathensemble.PathEnsemble`.

This module defines :class:`PathEnsembleReport`, a mixin that prints and returns
compact summaries of an ensemble. It is intended for interactive inspection
during sampling (debugging, monitoring convergence, checking path types and
lengths) rather than for structured downstream analysis.

Provided functionality
----------------------
report
    Build a text table with one row per path (filenames, type, length) and,
    optionally, per-path histograms of per-frame values.

print_report
    Convenience wrapper around :meth:`report` that prints the report text.

report_shooting_results
    Print per-shooting-point committor estimates with confidence intervals.

Data model assumptions
----------------------
The host ensemble is assumed to provide:

- iteration over Path objects (``for path in self``)
- ``__len__`` and ``lengths``
- ``types()`` returning a per-path "type" label (typically a 3-state code)
- ``values`` returning a per-path per-frame numeric stream (or caller supplies it)
- ``shooting_results(states, sweep_size)`` used by :meth:`report_shooting_results`

In addition, Path objects are assumed to expose ``path._fnames`` (list of
segment filenames). This is used only for display.

Histogram reporting
-------------------
If ``bins`` is provided, :meth:`report` computes, for each path, a 1D histogram
of the per-frame values ``v`` using ``np.histogram(v, bins)``. The histogram is
displayed in the table and also returned as a 2D float array of shape:

    (n_paths, len(bins) - 1)

If ``bins`` is empty (or coerces to an empty array), histogram columns are
omitted from the table and the returned histogram array is empty.

Notes
-----
- The report output is formatted for terminal display; widths are computed from
  filename length (capped) and the maximum number of digits in lengths.
- In the table, repeated filename strings are suppressed by replacing them with
  the literal string ``'""'`` when consecutive paths share the same joined name.
- This module uses the AIMMD-configured ``print`` from :mod:`aimmd._config`.
"""

# external
import sys
import numpy as np
from abc import ABC
from scipy.special import logit

# aimmd imports
from .._config import print
from ..analysis.utils import binomial_mean_and_confidence_interval

class PathEnsembleReport(ABC):
    def report(self, bins=[], summary=True, values=None):
        """
        Create a human-readable report for the ensemble.

        The report is returned as a formatted multi-line string. Optionally, a
        per-path histogram of `values` is computed and appended as aligned
        columns. The function also returns the per-path histogram matrix as a
        numeric array for programmatic use.

        Parameters
        ----------
        bins : array-like or int, default=[]
            Binning specification for histogram columns.

            This implementation expects `bins` to be a sequence of bin edges.
            If an empty sequence is provided (or it becomes empty after coercion),
            histogram columns are omitted.

            The docstring in the original code mentions "predefined values or
            n_bins, then call self.bins", but this implementation currently only
            handles explicit bin edges. Any transformation from an integer number
            of bins to bin edges must therefore happen outside this method.

        summary : bool, default=True
            If True, append a summary block at the end of the report, including:
            - total number of displayed paths,
            - total number of frames (sum of lengths),
            - a type-count table (counts and percentage per type),
            - and, if `bins` is provided, the aggregated histogram.

        values : sequence, optional
            Per-path values used for histogramming. If None, uses ``self.values``.

            Expected structure is one entry per path, where each entry is a
            1D array-like of per-frame scalars.

        Returns
        -------
        (str, numpy.ndarray)
            (report_text, histograms)

            report_text
                Formatted table as a single string.

            histograms
                2D float array with shape (n_paths, len(bins) - 1) if bins are
                provided, otherwise an empty array (or an array converted from
                an empty list).

        Notes
        -----
        - If the ensemble is empty, returns ``str(self)`` (as a string) without
          histogram output.
        - Path types are truncated to 3 characters and marked with a leading '*'
          if all three characters are distinct. This is a display convention used
          to highlight "transitions" vs repeated-state paths.
        - The function uses ``path._fnames`` to display the list of trajectory
          segments contributing to each path.
        """
        if not len(self):
            return str(self)
        types = self.types()
        if values is None:
            values = self.values
        filenames = []
        histograms = []
        for path in self:
            fnames = ", ".join(path._fnames)
            filenames.append(fnames)
        filenames = np.array(filenames)
        for i in range(len(filenames) - 1, 0, -1):
            if filenames[i] == filenames[i - 1]:
                filenames[i] = '""'
        lengths = self.lengths
        bins = np.asarray(bins)
        if not bins.shape:
            bins = np.array([])
            h = ''
        if len(filenames):
            w0 = max(min(int(filenames.dtype.str.split('<U')[-1]), 32), 10)
            w1 = max(int(np.ceil(np.log10(lengths.max()))) + 1, 6)
        else:
            w0 = 10
            w1 = 6
        b = [f'{b:>+9.2e}' for b in bins]
        if len(bins):
            result = [
            f'{"         ":{w0}}       '
            f'{"Bins->" if len(bins) else "":{w1}} {" ".join(b[:-1])}\n']
        else:
            result = []
        result.append(
            f'{"Filenames":{w0}} Type  {"Length":>{w1}} {" ".join(b[+1:])}\n')
        types_ = {}
        frames = 0
        if len(bins):
            histogram = np.zeros(len(bins) - 1, dtype=int)
        for f, s, l, v in zip(filenames, types, lengths, values):
            s = s[:3]
            if len(set(s)) == 3:
                s = f'*{s}'
            try:
                types_[s] += 1
            except:
                types_[s] = 1
            frames += l
            if len(bins):
                h = np.histogram(v, bins)[0]
                histogram += h
                histograms.append(h)
                h = ''.join([f'{h:10}' if h else f'{".":>10}' for h in h])
            result.append(f'{f[:w0]:{w0}} {s:>4}  {l:{w1}}{h}\n')
        if summary:
            if len(bins):
                h = ''.join([f'{h:10}' if h else
                             f'{".":>10}' for h in histogram])
            n = len(result) - 1 - (len(bins) > 1)
            if len(bins):
                result.append(
                f'{"         ":{w0}}       '
                f'{"Bins->" if len(bins) else "":{w1}} '
                f'{" ".join(b[:-1])}\n')
            result.append(
                f"Summary" + f"_" * (w0+w1) + f' {" ".join(b[+1:])}\n')
            if n:
                w2 = max(int(np.ceil(np.log10(frames))) + 2, 7)
            else:
                w2 = w1
            result.append(f'Total {n:{w0-6}}'
                          f'{"":{6-(w2-w1)}} {frames:{w2}}{h}\n')
            for s, m in sorted(types_.items(), key=lambda item: -item[1]):
                result.append(f'{m:{w0}} {s:>4} {m/n*100:{w1+1}.2f}%\n')
        return ''.join(result), np.array(histograms).astype(float)
                  
    def print_report(self, *args, **kwargs):
        """
        Print the report text to a log file or stdout.

        This is a convenience wrapper around :meth:`report`. It prints only the
        text component (index 0 of the tuple returned by :meth:`report`).

        Notes
        -----
        The current implementation references a variable ``log_file`` that is
        not defined in the function scope. In practice, callers likely intended
        to provide a `log_file` object via surrounding scope or to have this
        function retrieve it from ``kwargs``. The implementation is kept as-is
        by design (documentation-only pass).
        """
        if not log_file or log_file == 'stdout':
            log_file = sys.stdout
        print(self.report(*args, **kwargs)[0], file=log_file)

    def report_shooting_results(self, states='ARB', sweep_size=0, alpha=0.95):
        """
        Print committor estimates from shooting results.

        For each shooting point index `i`, the host method
        ``self.shooting_results(states, sweep_size)`` must yield a pair:

            (n_to_states[0], n_to_states[-1])

        interpreted as binomial counts of trajectories committed to the first and
        last state labels in `states`.

        The output includes:
        - point index,
        - counts to the two terminal states,
        - mean committor and confidence interval,
        - logit(committor) and confidence interval in logit space.

        Parameters
        ----------
        states : str, default='ARB'
            State label string. Only the first and last characters are used as
            the two terminal states in the printed table.
        sweep_size : int, default=0
            Passed through to ``self.shooting_results``.
        alpha : float, default=0.95
            Confidence level used by
            :func:`aimmd.analysis.utils.binomial_mean_and_confidence_interval`.

        Notes
        -----
        The logit transform diverges for p=0 or p=1. The confidence interval
        computation should therefore avoid returning exact 0/1 bounds or the
        resulting logit values will be ±inf.
        """
        result = [f'Point   n{states[0]}   n{states[-1]}  '
                  f'committor  conf. min  conf. max  '
                  f' logit     min     max']
        for i, r in enumerate(self.shooting_results(states, sweep_size)):
            p, p_min, p_max = binomial_mean_and_confidence_interval(*r, alpha)
            q, q_min, q_max = logit(p), logit(p_min), logit(p_max)
            result.append(f'{i:5g}{r[0]:5g}{r[1]:5g}  '
                          f'{p:9.3e}  {p_min:9.3e}  {p_max:9.3e}  '
                          f'{q:+6.3f}  {q_min:+6.3f}  {q_max:+6.3f}')
        print('\n'.join(result))
