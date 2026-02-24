"""
...
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
        """Print useful info.
        bins: predefined values or n_bins, then call self.bins
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
        if not log_file or log_file == 'stdout':
            log_file = sys.stdout
        print(self.report(*args, **kwargs)[0], file=log_file)

    def report_shooting_results(self, states='ARB', sweep_size=0, alpha=0.95):
        """alpha: alpha-confidence interval"""
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
