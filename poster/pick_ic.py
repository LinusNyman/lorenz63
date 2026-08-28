"""Re-pick the hook figure's initial condition for 02_mlp_ode.

Applies the criteria in `make_figures.hook`: a crossing near the median of the 128 evaluation
starts, a lobe-change disagreement with the truth inside the drawn window, and no basin
capture over a long rollout.

The window is an argument rather than a constant, because the lobe counts and the end-gap are
measured inside it: `python pick_ic.py 223` for a 5.0 tau window.
"""
import sys
import torch, numpy as np
import l63.evaluate as E
from l63.data import make_datasets
from run.report import load_model

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
LAM = 0.897

d = make_datasets(kind='ode', b=0.0)
scale = d.scale
m, _ = load_model('02_mlp_ode_s0')
m.eval()

print(f'window {STEPS} steps = {STEPS * d.dt * LAM:.2f} tau\n')

pred, err = E.rollout(m, d.eval, d.std, steps=900)
cross = np.array([int((e > scale).float().argmax()) if bool((e > scale).any()) else 901
                  for e in err])
med = np.median(cross)
print(f'02_mlp_ode_s0: crossing over 128 starts -> min {cross.min()}  median {med:.0f}  '
      f'max {cross.max()}   ({(cross == 901).sum()} never cross)\n')

tr = d.raw(d.eval[:, :STEPS + 1])
pr = d.raw(pred[:, :STEPS + 1])
sw = lambda a: (a[..., 0].sign()[:, 1:] != a[..., 0].sign()[:, :-1]).sum(1)
sw_t, sw_p = sw(tr), sw(pr)
gap_end = (tr[:, -1] - pr[:, -1]).norm(dim=-1)

rows = []
for i in range(128):
    pct = 100 * (cross < cross[i]).mean()
    if not (35 <= pct <= 65):                      # mid-range percentile of the 128 starts
        continue
    if cross[i] > STEPS:                           # the two must part inside the window
        continue
    if sw_t[i] < 4 or sw_p[i] < 4:                 # both must keep changing wing
        continue
    if abs(int(sw_t[i]) - int(sw_p[i])) < 1:       # the lobe counts must differ
        continue
    if gap_end[i] < 0.6 * scale:                   # and they must be far apart at the end
        continue
    rows.append((i, int(cross[i]), pct, int(sw_t[i]), int(sw_p[i]), float(gap_end[i])))

print(f'{"ic":>4} {"cross":>6} {"pct":>5} {"truth sw":>9} {"model sw":>9} {"gap@end":>8}')
for r in sorted(rows, key=lambda r: abs(r[2] - 50)):
    print(f'{r[0]:4d} {r[1]:6d} {r[2]:5.0f} {r[3]:9d} {r[4]:9d} {r[5]:8.1f}')

print('\nlong-run check (20 000 steps, spread of the final 5000; truth-like is ~14.4):')
for r in sorted(rows, key=lambda r: abs(r[2] - 50))[:6]:
    i = r[0]
    with torch.no_grad():
        long_ = d.raw(m.forecast(d.eval[i:i + 1, 0], 20000))[0]
    print(f'  ic {i:3d}   final-5000 spread {float(long_[-5000:].std(0).norm()):.2f}')
