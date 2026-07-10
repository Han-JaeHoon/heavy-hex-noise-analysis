"""
Reproduction 2  ->  Paper Fig. 12 (repeated syndrome measurement).

The paper runs 16 cycles of repeated stabilizer measurement and looks at the
operator change (detection-event) rate per cycle.  Under a UNIFORM DEPOLARIZING
model the theory predicts:
    * a LOWER change rate on the first and last cycle (each is compared against a
      freshly prepared / freshly measured data qubit -> fewer error locations),
    * a FLAT, constant change rate across all intermediate cycles.

The experimental hardware instead shows change rates that DRIFT UPWARD across
intermediate cycles and lose the first/last dips -- a signature of noise beyond
uniform depolarizing (leakage, thermal relaxation, cross-talk, measurement bias).

Here we reproduce the DEPOLARIZING baseline (dips + flat middle) exactly in
simulation, then show two things the paper invokes:
  (i)  Z-biased noise shifts the level but keeps the flat-middle structure,
  (ii) an ad-hoc 'leakage-like' rising component (error rate growing with cycle)
       reproduces the experimentally observed upward drift -- i.e. the flat-middle
       prediction is what fails on hardware.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim
from common import (build_repetition_memory, sample_detectors,
                    per_cycle_change_rate, add_1q_noise, add_2q_noise)

DISTANCE = 5      # 5 data qubits -> 4 Z ancillas
ROUNDS = 16
SHOTS = 200_000

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# ---- (a) uniform depolarizing: expect dips + flat middle ----
circ, n_anc = build_repetition_memory(DISTANCE, ROUNDS, p=0.02, model="depolarize")
dets = sample_detectors(circ, SHOTS, seed=3)
rates = per_cycle_change_rate(dets, n_anc, ROUNDS + 1)
ax = axes[0]
for i in range(n_anc):
    ax.plot(range(ROUNDS + 1), rates[:, i], "-o", ms=3, label=f"anc {i}")
mid = rates[1:-1].mean()
ax.axhline(mid, ls="--", c="k", lw=1, label=f"mid-cycle mean={mid:.3f}")
ax.set_title("(a) uniform depolarizing p=0.02\nlow first/last cycle, FLAT middle")
ax.set_xlabel("measurement cycle"); ax.set_ylabel("operator change rate")
ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 0.5)

# ---- (b) Z-biased noise: level shifts, middle still flat ----
circ, n_anc = build_repetition_memory(DISTANCE, ROUNDS, p=0.03, model="biased", eta=6.5)
dets = sample_detectors(circ, SHOTS, seed=4)
rates_b = per_cycle_change_rate(dets, n_anc, ROUNDS + 1)
ax = axes[1]
for i in range(n_anc):
    ax.plot(range(ROUNDS + 1), rates_b[:, i], "-o", ms=3, label=f"anc {i}")
midb = rates_b[1:-1].mean()
ax.axhline(midb, ls="--", c="k", lw=1, label=f"mid-cycle mean={midb:.3f}")
ax.set_title("(b) Z-biased noise p=0.03, eta=6.5\n(paper's best-fit bias) still flat middle")
ax.set_xlabel("measurement cycle"); ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 0.5)


# ---- (c) time-DEPENDENT (leakage-like) noise -> upward drift like experiment ----
def build_repetition_rising(distance, rounds, p0, growth):
    """Same repetition memory but the depolarizing strength grows each cycle,
    p_r = p0 * (1 + growth * r), a crude stand-in for leakage / relaxation build-up."""
    d = distance
    data = [2 * i for i in range(d)]
    anc = [2 * i + 1 for i in range(d - 1)]
    n_anc = len(anc)
    c = stim.Circuit()
    all_q = data + anc
    c.append("R", all_q); add_1q_noise(c, all_q, "depolarize", p0)

    def one_round(pr):
        c.append("R", anc); add_1q_noise(c, anc, "depolarize", pr)
        for i, a in enumerate(anc):
            c.append("CX", [data[i], a]); add_2q_noise(c, (data[i], a), "depolarize", pr)
        for i, a in enumerate(anc):
            c.append("CX", [data[i + 1], a]); add_2q_noise(c, (data[i + 1], a), "depolarize", pr)
        add_1q_noise(c, data, "depolarize", pr)
        add_1q_noise(c, anc, "depolarize", pr)
        c.append("M", anc)

    pr = p0
    one_round(pr)
    for i in range(n_anc):
        c.append("DETECTOR", [stim.target_rec(-n_anc + i)], (i, 0))
    for r in range(1, rounds):
        pr = p0 * (1 + growth * r)
        one_round(pr)
        for i in range(n_anc):
            c.append("DETECTOR", [stim.target_rec(-n_anc + i),
                                  stim.target_rec(-2 * n_anc + i)], (i, r))
    add_1q_noise(c, data, "depolarize", pr)
    c.append("M", data)
    for i in range(n_anc):
        c.append("DETECTOR", [stim.target_rec(-d + i), stim.target_rec(-d + i + 1),
                              stim.target_rec(-d - n_anc + i)], (i, rounds))
    return c, n_anc

circ, n_anc = build_repetition_rising(DISTANCE, ROUNDS, p0=0.012, growth=0.12)
dets = sample_detectors(circ, SHOTS, seed=5)
rates_c = per_cycle_change_rate(dets, n_anc, ROUNDS + 1)
ax = axes[2]
for i in range(n_anc):
    ax.plot(range(ROUNDS + 1), rates_c[:, i], "-o", ms=3, label=f"anc {i}")
ax.set_title("(c) time-DEPENDENT (leakage-like) noise\nchange rate DRIFTS UP -> like experiment")
ax.set_xlabel("measurement cycle"); ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 0.5)

fig.suptitle("Reproduction 2 -- repeated syndrome measurement, per-cycle change rate "
             "(paper Fig. 12)", fontsize=12)
fig.tight_layout()
fig.savefig("figs/repro2_repeated.png", dpi=130)

# quantitative summary
print("Depolarizing: first={:.3f} mid_mean={:.3f} last={:.3f}  (flatness std over middle = {:.4f})"
      .format(rates[0].mean(), rates[1:-1].mean(), rates[-1].mean(), rates[1:-1].mean(axis=1).std()))
print("Biased      : first={:.3f} mid_mean={:.3f} last={:.3f}  (flatness std over middle = {:.4f})"
      .format(rates_b[0].mean(), rates_b[1:-1].mean(), rates_b[-1].mean(), rates_b[1:-1].mean(axis=1).std()))
print("Rising/leak : first={:.3f} mid_mean={:.3f} last={:.3f}  (flatness std over middle = {:.4f})"
      .format(rates_c[0].mean(), rates_c[1:-1].mean(), rates_c[-1].mean(), rates_c[1:-1].mean(axis=1).std()))
print("saved figs/repro2_repeated.png")
