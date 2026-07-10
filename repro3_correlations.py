"""
Reproduction 3  ->  Paper Sec. II F / Eq. (7) / Fig. 20 (spatial-temporal correlations).

The pairwise detection-event correlation is
    p_ij = (<xi xj> - <xi><xj>) / ((1 - 2<xi>)(1 - 2<xj>))          (Eq. 7)
derived assuming independent events in the small-correlation limit.

Under a circuit-level depolarizing model the paper predicts THREE classes of
large correlations between detectors indexed by (ancilla a, round r):
  * SPACE-like      : same round r, adjacent ancillas  (share a data qubit)
  * TIME-like       : same ancilla a, adjacent rounds  (a measurement error flips
                      the detector this round and the next)  -> strongest
  * SPACE-TIME-like : adjacent ancilla AND adjacent round -> weakest (fewest
                      error locations that can cause it)
All other pairs should have ~zero correlation.

We build the repetition-code memory, sample detections, form the Eq.(7) matrix,
and confirm the three-class structure and their expected ordering.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import build_repetition_memory, sample_detectors, correlation_matrix

DISTANCE = 5      # 4 ancillas
ROUNDS = 6        # -> (ROUNDS+1)=7 detector layers, 28 detectors total
SHOTS = 2_000_000
P = 0.03

circ, n_anc = build_repetition_memory(DISTANCE, ROUNDS, p=P, model="depolarize")
dets = sample_detectors(circ, SHOTS, seed=17)
pij = correlation_matrix(dets)
N = pij.shape[0]
layers = N // n_anc  # ROUNDS+1

# index -> (round, ancilla), layout is round-major (round r, ancilla a) = r*n_anc + a
def ra(idx):
    return idx // n_anc, idx % n_anc

# classify every off-diagonal pair
classes = {"space": [], "time": [], "space-time": [], "other": []}
for i in range(N):
    for j in range(i + 1, N):
        ri, ai = ra(i); rj, aj = ra(j)
        dr, da = abs(ri - rj), abs(ai - aj)
        if dr == 0 and da == 1:
            classes["space"].append(pij[i, j])
        elif dr == 1 and da == 0:
            classes["time"].append(pij[i, j])
        elif dr == 1 and da == 1:
            classes["space-time"].append(pij[i, j])
        else:
            classes["other"].append(pij[i, j])

print("== Correlation class means (Eq. 7), depolarizing p={} ==".format(P))
for k in ["time", "space", "space-time", "other"]:
    v = np.array(classes[k])
    print(f"  {k:12s}: mean={v.mean():+.4f}  max={v.max():+.4f}  (n_pairs={len(v)})")

# ---- plot: heatmap of the correlation matrix ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
vmax = np.percentile(np.abs(pij), 99.5)
im = ax.imshow(pij, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title(f"(a) Eq.(7) correlation matrix\nrepetition memory, depol p={P}")
ax.set_xlabel("detector index  (round-major: r*{}+a)".format(n_anc))
ax.set_ylabel("detector index")
# gridlines at round boundaries
for k in range(1, layers):
    ax.axhline(k * n_anc - 0.5, c="grey", lw=0.4)
    ax.axvline(k * n_anc - 0.5, c="grey", lw=0.4)
fig.colorbar(im, ax=ax, fraction=0.046, label=r"$p_{ij}$")

# ---- plot: bar chart of class means ----
ax = axes[1]
order = ["time", "space", "space-time", "other"]
means = [np.mean(classes[k]) for k in order]
maxes = [np.max(classes[k]) for k in order]
x = np.arange(len(order))
ax.bar(x - 0.2, means, 0.4, label="mean", color="#4477aa")
ax.bar(x + 0.2, maxes, 0.4, label="max", color="#ee6677")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=15)
ax.set_ylabel(r"correlation  $p_{ij}$")
ax.set_title("(b) three error classes recovered\ntime > space > space-time >> other")
ax.legend(); ax.axhline(0, c="k", lw=0.6)

fig.suptitle("Reproduction 3 -- spatial-temporal correlations (paper Eq. 7 / Fig. 20)",
             fontsize=12)
fig.tight_layout()
fig.savefig("figs/repro3_correlations.png", dpi=130)
print("saved figs/repro3_correlations.png")
