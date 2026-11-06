"""Figures of :mod:`aimmd.cgaimmd.analysis.validate`: the model card, the leaderboard and the parity figure.

Private: called by :func:`~aimmd.cgaimmd.analysis.validate.validate`,
:func:`~aimmd.cgaimmd.analysis.validate.leaderboard` and
:func:`~aimmd.cgaimmd.analysis.validate.parity_figure`. matplotlib and scipy
are imported inside the functions.
"""

import os
import textwrap

import numpy as np

# categorical slots (fixed order), ink and grid
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                                       '#e87ba4', '#008300', '#4a3aa7')
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET)
INK, INK2, MUTED, GRID = '#222222', '#555555', '#8a8a8a', '#e6e6e6'
FAILURES = (('bond graph', ('non_finite', 'bond_count', 'not_isomorphic', 'vf2_timeout'), ORANGE),
            ('chirality', ('chirality',), YELLOW), ('clash', ('clash',), MAGENTA))


def _fmt(value, digits=3, signed=False):
    if value is None:
        return 'n/a'
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    text = f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'
    return text.replace('-', '−')


def _pct(value):
    return 'n/a' if value is None else f'{100 * value:.0f} %'


def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -50, 50)))


def _reliability(lam, n_first, n_last, n_bins=10):
    p = _sigmoid(lam)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        sel = b == k
        n = (n_first[sel] + n_last[sel]).sum()
        if n > 0:
            rows.append((float(p[sel].mean()), float(n_last[sel].sum() / n), float(n)))
    return np.array(rows).reshape(-1, 3)


def _panel_validity(ax, report):
    rows = report['m1']['levels']
    x = np.arange(len(rows))
    valid = np.array([r['valid'] for r in rows])
    ax.bar(x, valid, width=0.66, color=BLUE, label='valid (raw)', zorder=2)
    bottom = valid.copy()
    for name, codes, colour in FAILURES:
        share = np.array([sum(r['reasons'].get(c, 0.0) for c in codes) for r in rows])
        if share.any():
            ax.bar(x, share, width=0.66, bottom=bottom, color=colour, label=f'rejected: {name}',
                   zorder=2, edgecolor='white', linewidth=0.8)
            bottom = bottom + share
    other = 1.0 - bottom
    if np.any(other > 1e-9):
        ax.bar(x, np.clip(other, 0, None), width=0.66, bottom=bottom, color='#cfcfcf',
               label='rejected: other', zorder=2, edgecolor='white', linewidth=0.8)
    polished = [r.get('polish') for r in rows]
    if any(p is not None for p in polished):
        ax.plot(x, [p['valid_after'] if p else np.nan for p in polished], 'o-', color=INK,
                ms=5, lw=1.5, zorder=4,
                label=f'valid after EM polish (first {max(p["n"] for p in polished if p)} per level)')
    waters = [r.get('waters_intact') for r in rows]
    if any(w is not None for w in waters):
        ax.plot(x, [np.nan if w is None else w for w in waters], 'D--', color=GREEN, ms=4,
                lw=1.2, label='all waters intact (raw)', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([_fmt(r['lambda'], 0, signed=True).replace('+0', '0') for r in rows])
    ax.set_ylim(0, 1.04)
    ax.set_xlabel('requested committor logit λ', color=INK2, fontsize=9)
    ax.set_ylabel(f'fraction of {rows[0]["n"]} draws per level', color=INK2, fontsize=9)
    _style(ax)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=7.2, loc='lower center', frameon=True, framealpha=0.9, ncol=2)
    checked = report['m1'].get('checked')
    s = report['m1']['summary']
    title = 'a   M1 validity of the generated samples'
    ax.set_title(title, loc='left', fontsize=10, fontweight='bold', color=INK, pad=17)
    note = (f'yield {_pct(s["yield"])}; lowest level {_pct(s["valid_min"])}; '
            f'transition region (|λ| ≤ {report["settings"]["transition_level"]:g}) '
            f'{_pct(s["valid_transition_mean"])}' if checked else
            'no validity function given: every sample counted as valid')
    ax.text(0, 1.012, note, transform=ax.transAxes, fontsize=7.8, color=INK2, va='bottom')


def _metric_lines(m2):
    """The parity numbers of one reference, one per line."""
    return [f'log-lik/endpoint {_fmt(m2["loglik_per_endpoint"])}',
            f'  in-sample ceiling {_fmt(m2["loglik_saturated"])}',
            f'  constant model {_fmt(m2["loglik_constant"])}',
            f'calibration q = σ(a + bλ): a {_fmt(m2["calibration_intercept"], 2, True)}, '
            f'b {_fmt(m2["calibration_slope"], 2)}',
            f'Spearman {_fmt(m2.get("spearman"), 2)}',
            f'0.1<q<0.9: RMSE {_fmt(m2["rmse_transition"])}, bias '
            f'{_fmt(m2["bias_transition"], 3, True)}',
            f'  saturated {m2["n_saturated_transition"]} of {m2["n_transition"]}',
            f'R² {_fmt(m2["r2"])} (noise ceiling {_fmt(m2.get("r2_noise_ceiling"))})']


def _binomial_band(a, b):
    """The 68 % Clopper-Pearson interval of q = b / (a + b) (b of a + b endpoints in the last state)."""
    from scipy.stats import beta
    lo = np.where(b > 0, beta.ppf(0.16, b, a + 1), 0.0)
    hi = np.where(a > 0, beta.ppf(0.84, b + 1, a), 1.0)
    return lo, hi


def _parity_axes(ax, reference, lam, colour):
    """Scatter (brute force) or binned reliability (one or two endpoints per entry) on one axis."""
    include = np.ones(len(lam), dtype=bool) if reference.include is None else reference.include
    a, b = reference.n_first[include], reference.n_last[include]
    p = _sigmoid(lam[include])
    ax.plot([0, 1], [0, 1], '--', color=MUTED, lw=1, zorder=1)
    if reference.kind == 'brute_force':
        n = a + b
        keep = n > 0
        a, b, p, n = a[keep], b[keep], p[keep], n[keep]
        q = b / n
        lo, hi = _binomial_band(a, b)
        ax.errorbar(p, q, yerr=[q - lo, hi - q], fmt='o', ms=3.6, color=colour, ecolor=colour,
                    elinewidth=0.6, alpha=0.75, zorder=2)
        return f'brute-force committor q ({int(n.min())}–{int(n.max())} endpoints)'
    rel = _reliability(lam[include], a, b)
    ax.plot(rel[:, 0], rel[:, 1], '-', color=colour, lw=1.2, zorder=2)
    ax.scatter(rel[:, 0], rel[:, 1], s=15 + 200 * rel[:, 2] / rel[:, 2].max(), color=colour,
               edgecolor='white', lw=0.8, zorder=3)
    return 'observed fraction of endpoints in the last state'


def parity_panels(entries, stem, *, title=None):
    """Side-by-side committor parity of several models; returns ``(png, pdf)`` paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .validate import committor_metrics
    entries = [(str(label), reference, np.asarray(lam, dtype=float).reshape(-1))
               for label, reference, lam in entries]
    if not entries:
        raise ValueError('parity_figure needs at least one (label, reference, logits) entry')
    n = len(entries)
    fig = plt.figure(figsize=(4.4 * max(n, 2), 8.6))
    gs = fig.add_gridspec(2, max(n, 2), left=0.07, right=0.98, top=0.9, bottom=0.07,
                          wspace=0.3, hspace=0.32)
    for k, (label, reference, lam) in enumerate(entries):
        ax = fig.add_subplot(gs[0, k])
        include = np.ones(len(lam), dtype=bool) if reference.include is None else reference.include
        m2 = committor_metrics(lam[include], reference.n_first[include],
                               reference.n_last[include])
        ylabel = _parity_axes(ax, reference, lam, SERIES[k % len(SERIES)])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_xlabel('model committor p = σ(λ)', color=INK2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
        _style(ax)
        ax.set_title(textwrap.shorten(label, width=48, placeholder=' ...'), loc='left',
                     fontsize=10, fontweight='bold', color=INK, pad=17)
        ax.text(0, 1.012, textwrap.shorten(f'{reference.name}; {m2["n_entries"]} configurations',
                                           width=64, placeholder=' ...'),
                transform=ax.transAxes, fontsize=7.8, color=INK2, va='bottom')
        ax.text(0.03, 0.97, '\n'.join(_metric_lines(m2)), transform=ax.transAxes, fontsize=7.0,
                va='top', color=INK, bbox=dict(fc='white', ec=GRID, alpha=0.9,
                                               boxstyle='round,pad=0.35'), zorder=5)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot([0, 1], [0, 1], '--', color=MUTED, lw=1, zorder=1)
    for k, (label, reference, lam) in enumerate(entries):
        include = np.ones(len(lam), dtype=bool) if reference.include is None else reference.include
        rel = _reliability(lam[include], reference.n_first[include], reference.n_last[include])
        colour = SERIES[k % len(SERIES)]
        ax.plot(rel[:, 0], rel[:, 1], '-', color=colour, lw=1.4, zorder=2,
                label=textwrap.shorten(label, width=40, placeholder=' ...'))
        ax.scatter(rel[:, 0], rel[:, 1], s=12 + 150 * rel[:, 2] / rel[:, 2].max(), color=colour,
                   edgecolor='white', lw=0.8, zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('model committor p = σ(λ), bin mean', color=INK2, fontsize=9)
    ax.set_ylabel('observed fraction of endpoints in the last state', color=INK2, fontsize=8.5)
    _style(ax)
    ax.legend(fontsize=7.2, loc='upper left', frameon=False)
    ax.set_title('Reliability: observed against predicted', loc='left', fontsize=10,
                 fontweight='bold', color=INK, pad=17)
    ax.text(0, 1.012, '10 equal bins in p; marker area ∝ endpoints in the bin',
            transform=ax.transAxes, fontsize=7.8, color=INK2, va='bottom')
    ax = fig.add_subplot(gs[1, 1])
    every = np.concatenate([lam for _, _, lam in entries])
    edges = np.linspace(np.floor(every.min()) - 0.5, np.ceil(every.max()) + 0.5, 41)
    for k, (label, reference, lam) in enumerate(entries):
        ax.hist(lam, bins=edges, histtype='step', lw=1.5, color=SERIES[k % len(SERIES)],
                label=textwrap.shorten(label, width=40, placeholder=' ...'))
    ax.set_xlabel('model committor logit λ = ln(p / (1 − p))', color=INK2, fontsize=9)
    ax.set_ylabel('configurations', color=INK2, fontsize=9)
    _style(ax)
    ax.legend(fontsize=7.2, loc='upper left', frameon=False)
    ax.set_title('Where the models put the configurations', loc='left', fontsize=10,
                 fontweight='bold', color=INK, pad=17)
    ax.text(0, 1.012, 'histogram of the logits of the reference configurations',
            transform=ax.transAxes, fontsize=7.8, color=INK2, va='bottom')
    if title:
        fig.suptitle(textwrap.shorten(title, width=110, placeholder=' ...'), x=0.07, ha='left',
                     fontsize=13, fontweight='bold', color=INK)
    png, pdf = stem + '.png', stem + '.pdf'
    os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _panel_parity(ax, report, references):
    if not references:
        ax.axis('off')
        ax.text(0.5, 0.5, 'M2: no committor reference given', ha='center', va='center',
                color=INK2, fontsize=10, transform=ax.transAxes)
        return
    reference, lam = references[0]
    m2 = report['m2'][reference.name]
    for lo in (0.0, 0.95):
        ax.axvspan(lo - 0.02 if lo == 0 else lo, lo + 0.05 if lo == 0 else 1.02, color='#f0f0f0',
                   zorder=0)
    ylabel = _parity_axes(ax, reference, lam, BLUE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('model committor p = σ(λ)', color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    _style(ax)
    kind = {'brute_force': 'brute-force set',
            'heldout_chains': 'held-out chains, binned (marker area ∝ shot halves)',
            'free_md_forward': 'free MD, binned (marker area ∝ frames)'}.get(reference.kind,
                                                                             reference.kind)
    ax.set_title(f'b   M2 committor model: {reference.name}', loc='left', fontsize=10,
                 fontweight='bold', color=INK, pad=17)
    ax.text(0, 1.012, f'{kind}; {m2["n_scored"]} configurations scored', transform=ax.transAxes,
            fontsize=7.8, color=INK2, va='bottom')
    ax.text(0.03, 0.97, '\n'.join(_metric_lines(m2)), transform=ax.transAxes, fontsize=7.4,
            va='top', color=INK, bbox=dict(fc='white', ec=GRID, alpha=0.9,
                                           boxstyle='round,pad=0.35'), zorder=5)


def _panel_reliability(ax, report, references):
    if not references:
        ax.axis('off')
        return
    ax.plot([0, 1], [0, 1], '--', color=MUTED, lw=1, zorder=1)
    for k, (reference, lam) in enumerate(references[:len(SERIES)]):
        include = np.ones(len(lam), dtype=bool) if reference.include is None else reference.include
        rel = _reliability(lam[include], reference.n_first[include], reference.n_last[include])
        colour = SERIES[k]
        ax.plot(rel[:, 0], rel[:, 1], '-', color=colour, lw=1.4, zorder=2, label=reference.name)
        ax.scatter(rel[:, 0], rel[:, 1], s=12 + 150 * rel[:, 2] / rel[:, 2].max(), color=colour,
                   edgecolor='white', lw=0.8, zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('model committor p = σ(λ), bin mean', color=INK2, fontsize=9)
    ax.set_ylabel('observed fraction of endpoints in the last state', color=INK2, fontsize=8.5)
    _style(ax)
    ax.legend(fontsize=7.2, loc='upper left', frameon=False)
    ax.set_title('c   M2 reliability, every reference', loc='left', fontsize=10,
                 fontweight='bold', color=INK, pad=17)
    ax.text(0, 1.012, '10 equal bins in p; marker area ∝ endpoints in the bin',
            transform=ax.transAxes, fontsize=7.8, color=INK2, va='bottom')


def _panel_consistency(ax, report, data):
    levels = np.asarray(data['levels'], dtype=float)
    ok = data['ok']
    tol = report['settings']['tolerance']
    span = max(abs(levels.min()), abs(levels.max())) + 2.5
    grid = np.linspace(-span, span, 2)
    ax.fill_between(grid, grid - tol, grid + tol, color='#ececec', zorder=0,
                    label=f'±{tol:g} around the request')
    ax.plot(grid, grid, '--', color=MUTED, lw=1, zorder=1)
    series = [('own', 'own committor model', BLUE, data['own_lambda'], -0.18)]
    m3 = report['m3']
    if m3.get('independent') is not None:
        series.append(('independent', m3.get('independent_name') or 'independent committor',
                       ORANGE, data['independent_lambda'], 0.18))
    for key, name, colour, lam, dx in series:
        stats = m3[key]
        med, q25, q75, xs = [], [], [], []
        for li, level in enumerate(levels):
            values = lam[li][ok[li]]
            values = values[np.isfinite(values)]
            if len(values):
                xs.append(level + dx)
                med.append(np.median(values))
                q25.append(np.percentile(values, 25))
                q75.append(np.percentile(values, 75))
        if not xs:
            continue
        xs, med, q25, q75 = map(np.asarray, (xs, med, q25, q75))
        clip = lambda v: np.clip(v, -span, span)                       # noqa: E731
        ax.vlines(xs, clip(q25), clip(q75), color=colour, lw=3.2, alpha=0.55, zorder=2)
        label = (f'{name}: Spearman {_fmt(stats["spearman"], 2)}, within ±{tol:g} '
                 f'{_pct(stats["within_tolerance"])}, median |Δλ| '
                 f'{_fmt(stats["median_abs_error"], 2)}')
        ax.plot(xs, clip(med), 'o', color=colour, ms=5.5, mec='white', mew=0.8, zorder=3,
                label=label)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect('equal')
    ax.set_xlabel('requested committor logit λ', color=INK2, fontsize=9)
    ax.set_ylabel('committor logit of the valid samples', color=INK2, fontsize=9)
    _style(ax)
    ax.legend(fontsize=6.8, loc='upper left', frameon=True, framealpha=0.92)
    ax.set_title('d   M3 does the generator hit the requested committor?', loc='left',
                 fontsize=10, fontweight='bold', color=INK, pad=17)
    ax.text(0, 1.012, 'dots: median; bars: interquartile range of the valid samples per level',
            transform=ax.transAxes, fontsize=7.8, color=INK2, va='bottom')


def _ecdf(ax, values, colour, label, ls='-'):
    v = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if len(v):
        ax.step(np.r_[v[0], v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where='post',
                color=colour, lw=1.6, ls=ls, label=label)


def _panel_guard(ax, report, data):
    guard = report.get('guard')
    ax.set_title('e   Guard: are samples copies of training frames?', loc='left',
                 fontsize=10, fontweight='bold', color=INK, pad=17)
    if guard is None:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no training frames given', ha='center',
                va='center', color=INK2, fontsize=9, transform=ax.transAxes)
        return
    sel = data['in_guard']
    if guard.get('training'):
        t = guard['training']
        _ecdf(ax, data['d_train'][sel], BLUE,
              f'sample → nearest training frame ({t["n_samples"]}; below p5: '
              f'{_pct(t["fraction_below_baseline_p5"])})')
    if data.get('baseline') is not None:
        b = guard['baseline']
        _ecdf(ax, data['baseline'], INK2, f'held-out MD → nearest training frame ({b["n_samples"]})',
              ls='--')
        ax.axvline(b['p5'], color=INK2, lw=0.8, ls=':')
        ax.text(b['p5'], 0.02, ' baseline p5', color=INK2, fontsize=7, rotation=90, va='bottom')
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel(f'symmetry-minimized RMSD (Å) over {guard["atoms"]}', color=INK2, fontsize=8.5)
    ax.set_ylabel('cumulative fraction', color=INK2, fontsize=9)
    _style(ax)
    ax.legend(fontsize=6.8, loc='lower right', frameon=True, framealpha=0.92)
    ax.text(0, 1.012, f'symmetry: {guard["symmetry"]}; {guard["n_training_frames"]} training and '
            f'{guard["n_heldout_frames"]} held-out frames; copies sit far left',
            transform=ax.transAxes, fontsize=7.8, color=INK2, va='bottom')


def _panel_summary(ax, report):
    ax.axis('off')
    s = report['m1']['summary']
    lines = ['f   Summary', '']
    lines.append(f'M1 valid: mean {_pct(s["valid_mean"])}, lowest level {_pct(s["valid_min"])}, '
                 f'transition {_pct(s["valid_transition_mean"])}')
    if s.get('valid_after_polish_mean') is not None:
        lines.append(f'     after EM polish: mean {_pct(s["valid_after_polish_mean"])}, lowest '
                     f'{_pct(s["valid_after_polish_min"])}')
    if s.get('waters_intact_min') is not None:
        lines.append(f'     all waters intact (raw): lowest level {_pct(s["waters_intact_min"])}')
    for name, m2 in report['m2'].items():
        lines.append(f'M2 {name}:')
        lines.append(f'     log-lik {_fmt(m2["loglik_per_endpoint"])} (ceiling '
                     f'{_fmt(m2["loglik_saturated"])}, constant {_fmt(m2["loglik_constant"])}); '
                     f'b {_fmt(m2["calibration_slope"], 2)}; '
                     f'Spearman {_fmt(m2.get("spearman"), 2)}; '
                     f'R² {_fmt(m2["r2"])} (noise ceiling {_fmt(m2.get("r2_noise_ceiling"))})')
    m3 = report['m3']
    for key, name in (('own', 'own committor'), ('independent', m3.get('independent_name')),
                      ('own_after_polish', 'own committor, valid after polish'),
                      ('independent_after_polish',
                       f'{m3.get("independent_name")}, valid after polish')):
        block = m3.get(key)
        if block:
            tr = block['transition_region']
            lines.append(f'M3 {name}:')
            lines.append(f'     Spearman {_fmt(block["spearman"], 2)}, within ±'
                         f'{report["settings"]["tolerance"]:g} {_pct(block["within_tolerance"])} '
                         f'(transition {_pct(tr["within_tolerance"])}), median |Δλ| '
                         f'{_fmt(block["median_abs_error"], 2)}')
            adherence = m3.get(f'{key}_adherence') if key in ('own', 'independent') else None
            if adherence:
                lines.append(f'     resolution {_fmt(adherence.get("resolution_all"), 2)} request '
                             f'logits (band {_fmt(adherence.get("resolution_band"), 2)}); core '
                             f'slope {_fmt((adherence.get("median_slopes") or {}).get("core"), 2)}')
    guard = report.get('guard') or {}
    block = guard.get('training')
    if block:
        lines.append(f'Guard nearest training frame: median {_fmt(block["median"], 2)} Å, below '
                     f'the baseline p5 {_pct(block["fraction_below_baseline_p5"])}')
    if guard.get('baseline'):
        lines.append(f'     baseline (held-out MD): median {_fmt(guard["baseline"]["median"], 2)} Å, '
                     f'p5 {_fmt(guard["baseline"]["p5"], 2)} Å')
    timing = report.get('timing', {})
    lines += ['', f'runtime {timing.get("total_seconds", 0):.0f} s (generation '
              f'{timing.get("generate_seconds", 0):.0f} s), {report["environment"]["host"]}, '
              f'{report["environment"]["torch_threads"]} CPU threads',
              '', 'Reading: M1 and M3 use the samples the generator produced at each requested '
              'level; M2 scores the committor model alone on each reference (whether a reference '
              'is held out for this model is in its notes in the JSON). Resolution: the spread '
              'of the samples\' committor around a line through the requests, divided by its '
              'slope, in request units (smaller is better). "below the baseline p5": fraction '
              'of samples closer to a training frame than 95 % of held-out MD frames are to any '
              'training frame (about 5 % for new structures, near 100 % for copies).']
    wrapped = []
    for line in lines:            # leading spaces of a line are kept; continuations indented
        indent = '' if line.startswith('Reading') else ' ' * 7
        wrapped += textwrap.wrap(line, width=74, subsequent_indent=indent) if line.strip() else ['']
    ax.text(0.0, 1.0, '\n'.join(wrapped), transform=ax.transAxes, fontsize=7.6, va='top',
            color=INK, linespacing=1.45)


def model_card(report, data, stem):
    """Draw the one-page model card of a report; returns ``(png, pdf)`` paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.5, 11.0))
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.985, top=0.84, bottom=0.07, wspace=0.32,
                          hspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    _panel_validity(axes[0], report)
    _panel_parity(axes[1], report, data['references'])
    _panel_reliability(axes[2], report, data['references'])
    _panel_consistency(axes[3], report, data)
    _panel_guard(axes[4], report, data)
    _panel_summary(axes[5], report)
    meta = report.get('metadata') or {}
    checkpoint = meta.get('checkpoint') or {}
    if not isinstance(checkpoint, dict):                 # a bare path, its hash as a sibling key
        checkpoint = {'path': str(checkpoint), 'sha256': meta.get('checkpoint_sha256')}
    network = report['network']
    settings = report['settings']
    fig.suptitle(textwrap.shorten(f'Model card: {report["label"]}', width=110, placeholder=' ...'),
                 x=0.05, ha='left', y=0.985, fontsize=14, fontweight='bold', color=INK)
    sub = [f'system: {report["system"]}'] if report.get('system') else []
    generator = settings.get('generator') or {}
    generator_text = ', '.join(f'{k} {v}' for k, v in (generator.get('settings') or {}).items())
    sub += [f'network {network["class"].split(".")[-1]}'
            + (f', {network["n_atoms"]} generator atoms' if 'n_atoms' in network else '')
            + (f', λ clip {network["lambda_clip"]:g}' if 'lambda_clip' in network else ''),
            f'samples: {settings["n_per_level"]} per level at λ = '
            + ', '.join(f'{v:g}' for v in settings['levels'])
            + f'; generator {generator.get("method")}'
            + (f' ({generator_text})' if generator_text else '') + f', seed {settings["seed"]}']
    if checkpoint:
        sub.append(f'checkpoint {checkpoint.get("path", "?")}'
                   + (f'  (sha256 {str(checkpoint["sha256"])[:12]})' if checkpoint.get('sha256')
                      else ''))
    sub.append(f'aimmd.cgaimmd.analysis.validate, {report["created"]}, aimmd '
               f'{report["environment"]["aimmd_commit"]}')
    fig.text(0.05, 0.952, '\n'.join(sub), ha='left', va='top', fontsize=8.6, color=INK2,
             linespacing=1.45)
    png, pdf = stem + '.png', stem + '.pdf'
    fig.savefig(png, dpi=130)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------

def leaderboard_figure(reports, rows, out, *, title=None):
    """Dot plot of the headline numbers of many reports; returns ``(png, pdf)`` paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    references = []
    for report in reports:
        for name in report.get('m2') or {}:
            if name not in references:
                references.append(name)
    shared = [name for name in references if all(name in (r.get('m2') or {}) for r in reports)]
    first = [min(r['m2'], key=lambda k: (r['m2'][k].get('kind') != 'brute_force', k))
             if r.get('m2') else None for r in reports]       # a brute-force set where there is one
    rows = [dict(row) for row in rows]
    panels = [('m1:valid_transition_mean', 'M1 valid, transition levels', (0, 1), 1.0),
              ('m1:valid_min', 'M1 valid, lowest level', (0, 1), 1.0)]
    if shared or len(references) <= 3:
        for name in (shared or references)[:3]:
            panels.append((f'm2:{name}:loglik_per_endpoint', f'M2 log-lik/endpoint\n{name}', None,
                           None))
            panels.append((f'm2:{name}:calibration_slope', f'M2 calibration slope\n{name}', None,
                           1.0))
    else:                       # no reference in common (several systems): each model's main one
        for row, name in zip(rows, first):
            for key in ('loglik_per_endpoint', 'loglik_saturated', 'calibration_slope', 'r2'):
                row[f'm2:first:{key}'] = None if name is None else row.get(f'm2:{name}:{key}')
            if name is not None:
                row['label'] = f'{row["label"]} [M2: {name}]'
        panels += [('m2:first:loglik_per_endpoint', 'M2 log-lik/endpoint\neach model\'s main reference',
                    None, None),
                   ('m2:first:calibration_slope', 'M2 calibration slope\neach model\'s main reference',
                    None, 1.0),
                   ('m2:first:r2', 'M2 R²\neach model\'s main reference', None, 1.0)]
    panels += [('m3:own:within_tolerance', 'M3 own committor within ±1', (0, 1), 1.0),
               ('m3:independent:within_tolerance', 'M3 independent within ±1', (0, 1), 1.0),
               ('guard:training:fraction_below_baseline_p5',
                'Guard: closer to training\nthan held-out p5', (0, 1), 0.05)]
    labels = ['\n'.join(textwrap.wrap(str(r['label']), 48)) for r in rows]
    n = len(rows)
    ncols = min(6, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, grid = plt.subplots(nrows, ncols, figsize=(2.3 * ncols + 3.2, nrows * (1.75 + 0.45 * n)),
                             sharey=True, squeeze=False)
    axes = grid.reshape(-1)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    y = np.arange(n)[::-1]
    for ax, (key, name, limits, ideal) in zip(axes, panels):
        values = np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows])
        _style(ax)
        ax.grid(axis='y', visible=False)
        if ideal is not None:
            ax.axvline(ideal, color=MUTED, lw=1, ls='--', zorder=1)
        if key.endswith('loglik_per_endpoint') and not key.startswith('m2:first:'):
            ceiling = [r.get(key.replace('loglik_per_endpoint', 'loglik_saturated')) for r in rows]
            ceiling = [c for c in ceiling if c is not None]
            if ceiling:
                ax.axvline(float(np.median(ceiling)), color=MUTED, lw=1, ls='--', zorder=1)
        ax.scatter(values, y, s=42, color=BLUE, edgecolor='white', lw=0.8, zorder=3)
        for yi, v in zip(y, values):
            if np.isfinite(v):
                ax.text(v, yi + 0.22, f'{v:.3g}', ha='center', va='bottom', fontsize=7,
                        color=INK2)
        if limits is not None:
            ax.set_xlim(limits[0] - 0.05, limits[1] + 0.05)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(4))
        head, _, reference = name.partition('\n')
        wrapped = '\n'.join([head] + textwrap.wrap(reference, 30)) if reference else name
        ax.set_title(wrapped, fontsize=8.5, color=INK, pad=8)
        ax.set_ylim(-0.7, n - 0.3)
    for row in grid:
        row[0].set_yticks(y)
        row[0].set_yticklabels(labels, fontsize=8.5, color=INK)
    fig.suptitle(title or 'Validation leaderboard', x=0.01, ha='left', fontsize=12,
                 fontweight='bold', color=INK)
    fig.text(0.01, 0.005, 'Dashed lines: the ideal value (M1, M3: 1; calibration slope: 1; guard: '
             '0.05, the held-out rate) or, for the log-likelihood, the median in-sample ceiling. '
             'Empty: not measured for that model.', fontsize=7.5, color=INK2, ha='left',
             va='bottom')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    png, pdf = out + '.png', out + '.pdf'
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
