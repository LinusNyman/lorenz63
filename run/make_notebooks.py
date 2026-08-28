"""Generate the thin driver notebooks, one per topic.

Generated rather than hand-written so that all ten stay consistent: same imports, same order,
same closing `## Findings` cell. A notebook here reads artifacts and narrates; it never
retrains. `run/train_all.py` owns the compute, so the compute is one thing to keep in step
instead of ten.

A rerun does not overwrite the Findings cell. That cell is the one hand-written part of a
notebook, so a rerun preserves its contents and replaces only the machinery around it.
`--reset` discards it and restores the placeholder.

Run:  PYTHONPATH=. python run/make_notebooks.py            [--reset]
"""

from __future__ import annotations

import argparse

import nbformat as nbf

from l63 import ROOT

NB = ROOT / 'notebooks'
PLACEHOLDER = '## Findings\n\n_Written after reading the numbers above._\n\n- \n- \n- '

# `clean` turns an undefined ruler into None; without it `spread` on the ODE prints the
# string "nan" everywhere.
BOOT = '''\
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks'
                       else pathlib.Path.cwd()))
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt, torch
from l63 import ARTIFACTS, evaluate as E, plots as P
from l63.data import make_datasets
from run.report import clean, load_model, load_rows, summarise

gt = json.load(open(ARTIFACTS / 'ground_truth.json'))
S  = clean(summarise(load_rows()))
DATA = list(gt['datasets'])                      # 'ode', 'sde', 'sde015'

def num(v, w=6, p=2):
    """A ruler value, or an em dash where it is undefined."""
    if isinstance(v, dict):
        v = v.get('median')
    undefined = v is None or v != v          # None from clean(), bare NaN from the json
    return f'{"—":>{w}}' if undefined else f'{v:{w}.{p}f}'

def rng(v, p=2):
    if v is None or v.get('lo') is None or v['lo'] != v['lo']:
        return '—'
    return f"[{v['lo']:.{p}f}–{v['hi']:.{p}f}]"

print(len(S), 'model x dataset entries ·', len(DATA), 'datasets')\
'''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def show(topic: str, label: str) -> str:
    return f'''\
KEY = 'ode'      # any of DATA
s = S['{topic}_{label}_' + KEY]
ref = gt['datasets'][KEY]
d = make_datasets(seed=0, kind=ref['kind'], b=ref['b'])
m, hist = load_model('{topic}_{label}_' + KEY + '_s' + str(s['rep_seed']))

print(f"{{m.n_params:,}} parameters, history {{m.history}}, figures show seed {{s['rep_seed']}}")
print()
print(f"{{'ruler':16s}}{{'median':>8s}}   range over seeds")
for k in ('horizon', 'spread', 'climate', 'climate_vs_truth', 'chaos', 'alive', 'lobe'):
    v = s[k]
    p = 0 if k == 'horizon' else 2
    print(f"  {{k:14s}}{{num(v, 8, p)}}   {{rng(v, p)}} over {{v['n']}} seeds")
print(f"\\ntruth on this dataset:  climate {{ref['truth_climate']:.2f}}   "
      f"alive {{ref['truth_alive']:.2f}}   lobe {{ref['truth_lobe']:.2f}}   "
      f"ground truth usable {{ref['floor_steps']}} steps")
print(f"\\nfirst steps, ||u_hat_n - u_n|| in Lorenz units:")
for i, e in enumerate(s['early'][:6], 1):
    print(f"  n={{i}}  {{e:.3e}}")\
'''


TOPICS = {
    '00': ('Ground truth, and the standard results', None, '''\
The ground truth has to reproduce what the system is known to do before the machine learning
means anything. This notebook is that check, plus the frozen datasets everything downstream
inherits.

Two kinds of row below. C±, ∇·f and ρ_H are closed forms evaluated: they check the formulas
were transcribed right and cannot fail otherwise. The measured divergence, the spectrum, its
sum against ∇·f, Kaplan–Yorke, the symmetry residual and the Lorenz map's slope are numerical
and can fail, so those are the integrator check.

`run/ground_truth.py` produces everything here; this notebook reads its output.\
''', '''\
k = gt['known']
print("--- closed forms, transcription checks (cannot fail) ---")
print(f"C+                {k['C_plus']}          expected (8.485, 8.485, 27)")
print(f"div f (formula)   {k['divergence']:.4f}                expected -13.6667")
print(f"rho_H             {k['rho_hopf']:.4f}                 expected 24.7368")
print("--- numerical, these are the integrator check ---")
print(f"div f (measured)  {k['divergence_measured']:.6f}   max deviation {k['divergence_measured_max_dev']:.1e}")
print(f"symmetry residual {k['symmetry_residual']:.1e}                  expected 0")
print(f"spectrum          {[round(v,4) for v in k['spectrum']]}")
print(f"  sum             {k['spectrum_sum']:.4f}   vs div f {k['divergence']:.4f}   <- an identity")
print(f"Kaplan-Yorke      {k['kaplan_yorke']:.4f}                 Strogatz measured ~2.05")
print(f"Lorenz map |f'|   min {k['lorenz_map_min_slope']:.3f} over {k['lorenz_map_peaks']} maxima   must be > 1")
print(f"\\nlambda_1 of the true map at dt: {gt['lambda_true']:.4f} +- {gt['lambda_true_sd']:.4f}")

for key in DATA:
    r = gt['datasets'][key]
    w = f"   conditional width {r['conditional_width_pct']:.1f}% of scale" if r['kind'] == 'sde' else ''
    print(f"\\n{key:7s} b={r['b']:<5g} scale {r['attractor_scale']:.2f}   "
          f"same-cost solver usable {r['euler_bar_steps']} steps   "
          f"ground truth usable {r['floor_steps']} steps{w}")\
'''),

    '01': ('The four rulers', None, '''\
What each ruler measures, and what truth itself scores on it. The truth row is the
calibration: every model column is read against it.

Two of them do not read 1.00 at truth. `climate`'s denominator is two disjoint halves of one
finite pool, which pushes them apart, so an independent draw scores below 1;
`climate_vs_truth` divides that out. On the SDE the floor is realisation-against-realisation
noise, which is irreducible: no model and no solver gets under it.\
''', '''\
for key in DATA:
    r = gt['datasets'][key]
    print(f"--- {key.upper()}  ({r['kind']}, b={r['b']:g}) ---")
    print(f"  horizon   ground truth usable {r['floor_steps']} steps; same-cost solver {r['euler_bar_steps']}"
          + ("   (floor = realisation noise, not discretisation)" if r['kind'] == 'sde' else ""))
    if r['spread_lead'] > 0:
        print(f"  spread    read at lead {r['spread_lead']} steps; a second independent truth "
              f"ensemble scores {r['truth_spread']:.3f} against the first")
    else:
        print("  spread    not defined (truth's ensemble spread is identically zero)")
    print(f"  climate   truth scores {r['truth_climate']:.2f} over 5 independent draws, "
          f"range {r['truth_climate_lo']:.2f}-{r['truth_climate_hi']:.2f}  <- the resolution")
    print(f"  chaos     lambda_1 of the true map = {gt['lambda_true']:.4f}, so the ratio is 1.00 by definition")
    print(f"  alive     truth scores {r['truth_alive']:.2f}   ·   lobe {r['truth_lobe']:.3f} switches/tau")

c = json.load(open(ARTIFACTS / 'summary.json'))['controls']
print("\\n--- the k=1 control: what this suite cannot resolve ---")
print("same model, same loss, differing only by float summation order (~7e-7 in the weights)")
for name, cc in sorted(c.items()):
    bits = '  '.join(f"{k} {v['rel']*100:5.1f}%" for k, v in cc.items())
    print(f"  {name:12s} {bits}")\
'''),

    '02': ('MLP one-step predictor', 'mlp', '''\
`u_{n+1} = F(u_n)`, trained by mean squared error on every consecutive pair.

The fact that governs the whole ladder: the minimiser of MSE is the conditional mean
`E[u_{n+1} | u_n]`. On the ODE the conditional is a point mass, so that mean is the flow map.
On the SDE the conditional has width, so it is not. `l63.models.Predictor` has the derivation.\
''', None),

    '03': ('MLP with a rollout loss', 'rollout_k8', '''\
The same network, unrolled `k` steps inside the loss so the gradient sees the composed map.
`k = 1` is the control: it has to reproduce topic 02, or the k axis means nothing.

The `k` axis moves less than the seed spread does, so the ranges below carry more than the
medians do.\
''', '''\
for key in DATA:
    print(f"--- {key.upper()} ---")
    for k in (1, 4, 8, 16):
        s = S.get(f'03_rollout_k{k}_{key}')
        if s is None: continue
        print(f"  k={k:2d}  horizon {num(s['horizon'],4,0)} {rng(s['horizon'],0):>12s}   "
              f"chaos {num(s['chaos'])} {rng(s['chaos']):>14s}   "
              f"alive {num(s['alive'])} {rng(s['alive']):>14s}")
    s = S.get(f'02_mlp_{key}')
    if s: print(f"  02    horizon {num(s['horizon'],4,0)} {rng(s['horizon'],0):>12s}   "
                f"chaos {num(s['chaos'])} {rng(s['chaos']):>14s}   "
                f"alive {num(s['alive'])} {rng(s['alive']):>14s}   <- the k=1 control target")\
'''),

    '04': ('Lead-time predictor', 'leadtime', '''\
`u_{n+s} = F(u_n, s)` — the network conditioned on the forecast horizon.

The question this rung answers: does forecasting directly to lead time `s` beat `s` repeated
single steps? Every other column here describes the s=1 map iterated, which is not what the
rung was trained for.\
''', '''\
s = S['04_leadtime_ode']
print('  s   direct      autoregressive   ratio')
for i in (0, 1, 3, 7, 11, 15):
    dd, aa = s['direct'][i], s['autoregressive'][i]
    print(f"{i+1:3d}   {dd:.3e}   {aa:.3e}      {dd/aa:.2f}")
print(f"\\nnote: {s['note']}")\
'''),

    '05': ('Recurrent models — RNN and LSTM', 'lstm', '''\
The Lorenz state is fully observed and the flow is Markov, so there is nothing in the history
for the hidden state to carry. These models are the control on that.

`chaos` here is measured on the map these models iterate, `(u, h, c)` rather than `u` alone,
because the hidden state is carried through the whole rollout. Measuring the memoryless map
instead gave an LSTM that tracked truth for 440 steps and scored λ₁ ≈ 0 in the same row.\
''', '''\
for key in DATA:
    print(f"--- {key.upper()} ---")
    for name in (f'05_rnn_{key}', f'05_lstm_{key}', f'05t_transformer_{key}', f'02_mlp_{key}'):
        s = S.get(name)
        if s is None: continue
        print(f"  {name:24s} {s['n_params']:7d} params   horizon {num(s['horizon'],4,0)} "
              f"{rng(s['horizon'],0):>12s}   chaos {num(s['chaos'])} {rng(s['chaos']):>14s}")\
'''),

    '06': ('Gaussian predictor', 'gaussian', '''\
`u_{n+1} ~ N(mu(u_n), Sigma(u_n))`, trained by maximum likelihood.

If sigma were fixed, minimising the likelihood would be minimising MSE, so any difference from
topic 02 comes from the sigma head and from sampling. The mean-against-sampled test below
isolates those two, over all five seeds, since it is read on `alive` and `climate`.

`chaos` is absent from the control: the estimator switches sampling off itself, so the two
rows would carry the same number by construction.\
''', '''\
for key in DATA:
    s = S.get(f'06_gaussian_{key}')
    if s is None: continue
    print(f"--- {key.upper()} ---")
    for tag, r in (('sampled', s), ('mean only', s['mean_only'])):
        print(f"  {tag:10s} horizon {num(r['horizon'],4,0)}   climate {num(r['climate'])}   "
              f"spread {num(r['spread'])}   alive {num(r['alive'])}   {rng(r['alive'])}")
    sr = s.get('sigma_over_residual')
    if sr and sr.get('median'):
        print(f"  learned sigma is {sr['median']:.2f}x the model's own one-step error "
              f"{rng(sr)}")\
'''),

    '05t': ('Transformer', 'transformer', '''\
Causal self-attention over the same 8-state window the recurrent models use. The Lorenz flow
is Markov, so there is nothing in that window for attention to find; this is the third and
largest control on how much apparent improvement is architecture rather than noise.

It carries ~100k parameters against the MLP's 17k. The shared budget is in iterations rather
than in capacity, which makes this an architecture control and not a matched comparison.\
''', '''\
for key in DATA:
    print(f"--- {key.upper()} ---")
    for name in (f'05t_transformer_{key}', f'05_lstm_{key}', f'02_mlp_{key}'):
        s = S.get(name)
        if s is None: continue
        print(f"  {name:24s} {s['n_params']:7d} params   horizon {num(s['horizon'],4,0)} "
              f"{rng(s['horizon'],0):>12s}   chaos {num(s['chaos'])}   alive {num(s['alive'])}")\
'''),

    '06f': ('Flow matching', 'flow', '''\
A learned transport from `N(0, I)` onto `p(u_{n+1} | u_n)`, with no assumption about the shape
of the conditional, where the Gaussian rung assumes a diagonal normal. Trained by conditional
flow matching on straight-line paths; sampled by integrating the learned velocity field.

Untuned, and solving a harder problem on the same budget as the other rungs, so a poor number
here is a statement about this budget rather than about flow matching.\
''', '''\
for key in DATA:
    print(f"--- {key.upper()} ---")
    for name, tag in ((f'06f_flow_{key}', 'flow'), (f'06_gaussian_{key}', 'gaussian')):
        s = S.get(name)
        if s is None: continue
        print(f"  {tag:10s} horizon {num(s['horizon'],4,0)}   climate {num(s['climate'])}   "
              f"spread {num(s['spread'])}   alive {num(s['alive'])}   {rng(s['alive'])}")
    s = S.get(f'06f_flow_{key}')
    if s:
        m = s['mean_only']
        print(f"  {'flow x0=0':10s} horizon {num(m['horizon'],4,0)}   climate {num(m['climate'])}   "
              f"spread {num(m['spread'])}   alive {num(m['alive'])}   {rng(m['alive'])}")\
'''),

    '07': ('All seven models', None, '''\
Every model against truth, on every dataset. Every number is a median over five seeds and the
range beside it is the full spread; where that range covers [0, 1] the median is a coin flip
rather than a measurement.

`run/ground_truth.py` measures the truth row through the same functions the models go through.\
''', '''\
for key in DATA:
    r = gt['datasets'][key]
    print(f"\\n=== {key.upper()}  ({r['kind']}, b={r['b']:g}) ===")
    print(f"{'':22s}{'horizon':>18s}{'spread':>8s}{'climate':>9s}{'/truth':>8s}{'chaos':>8s}{'alive':>19s}")
    print(f"{'truth':22s}{r['floor_steps']:>10d}{'':8s}"
          f"{num(r.get('truth_spread'))}{num(r['truth_climate'],9)}{1.0:8.2f}{1.0:8.2f}"
          f"{r['truth_alive']:>10.2f}")
    for name, key2 in [('MLP one-step','02_mlp'), ('rollout k=8','03_rollout_k8'),
                       ('lead time','04_leadtime'), ('LSTM','05_lstm'),
                       ('transformer','05t_transformer'), ('Gaussian','06_gaussian'),
                       ('flow matching','06f_flow')]:
        s = S.get(f'{key2}_{key}')
        if s is None: continue
        print(f"{name:22s}{num(s['horizon'],10,0)} {rng(s['horizon'],0):>7s}"
              f"{num(s['spread'])}{num(s['climate'],9)}{num(s['climate_vs_truth'],8)}"
              f"{num(s['chaos'])}{num(s['alive'],10)} {rng(s['alive']):>8s}")\
'''),
}


SLUG = {'00': 'ground_truth', '01': 'rulers', '02': 'mlp', '03': 'rollout',
        '04': 'leadtime', '05': 'recurrent', '05t': 'transformer', '06': 'gaussian',
        '06f': 'flow', '07': 'head_to_head'}


def existing_findings(path) -> str | None:
    """The hand-written Findings cell from the existing notebook, so a rerun preserves it.

    Returns None if the notebook does not exist, has no Findings cell, or still holds the
    untouched placeholder.
    """
    if not path.exists():
        return None
    for c in nbf.read(path, as_version=4).cells:
        src = ''.join(c['source'])
        if c['cell_type'] == 'markdown' and src.lstrip().startswith('## Findings'):
            return None if src.strip() == PLACEHOLDER.strip() else src
    return None


def build(topic: str, title: str, label: str | None, intro: str, extra: str | None,
          reset: bool = False):
    path = NB / f'{topic}_{SLUG[topic]}.ipynb'
    kept = None if reset else existing_findings(path)

    nb = nbf.v4.new_notebook()
    nb.cells = [md(f'# {topic} — {title}\n\n{intro}'), code(BOOT)]
    if extra:
        nb.cells += [md('## Numbers'), code(extra)]
    if label:
        nb.cells += [md('## This model' if extra else '## Numbers'), code(show(topic, label)),
                     md('## Figures\n\nBanked by `run/report.py`; regenerated here from the '
                        'same checkpoint, so the notebook and the banked figure cannot disagree.'),
                     code('P.loss_figure(hist, None, n_val_traj=8); plt.show()\n'
                          'P.arch_figure(m.spec(), "", m.n_params, None); plt.show()\n'
                          "long = d.raw(m.forecast(d.eval[:gt['n_long'], :m.history], "
                          "gt['long_steps']))\n"
                          'P.lorenz_map_figure(long, d.raw(d.eval), None); plt.show()')]
    nb.cells += [md(kept or PLACEHOLDER)]

    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                  'name': 'python3'},
                   'language_info': {'name': 'python'}}
    nbf.write(nb, path)
    print(f'{path.relative_to(ROOT)}' + ('   (kept your Findings)' if kept else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reset', action='store_true', help='discard hand-written Findings cells')
    a = p.parse_args()

    NB.mkdir(exist_ok=True)
    for t, (title, label, intro, extra) in TOPICS.items():
        build(t, title, label, intro, extra, reset=a.reset)
