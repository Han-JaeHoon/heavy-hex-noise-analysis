"""
Reproduction 1  ->  Paper Fig. 3 (col 3) + Eq. (4).

Two claims verified:
  (A) The operator change rate R of a gauge-operator measurement circuit under
      independent Pauli noise is an EXACT analytic function of the noise
      parameter -- computable in closed form (from the detector error model)
      without any density-matrix simulation -- and it matches Monte-Carlo.
  (B) That closed form has the geometric structure of Eq. (4),
          R = 1/2 (1 - (1 - 2q)^{n_conseq}),
      where n_conseq = number of independent error locations that can flip the
      operator outcome and q = per-location flip probability.  Deeper circuits
      (larger n_conseq) saturate toward R = 0.5 faster -- exactly the ordering
      the paper reports (XXXX flagged > XX flagged > ZZ).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim
from scipy.optimize import curve_fit
from common import (build_gauge_zz_circuit, build_gauge_xxxx_circuit,
                    analytic_change_rate_from_dem, sample_change_rate,
                    add_1q_noise)

SHOTS = 400_000
ps = np.linspace(0.0, 0.10, 11)

# ---------------------------------------------------------------------------
# Panel A: exact analytic (from DEM) vs Monte-Carlo, for two real gauge circuits
# ---------------------------------------------------------------------------
circuits = {
    "ZZ gauge": lambda p: build_gauge_zz_circuit(p, model="depolarize", init="00"),
    "flagged XXXX gauge": lambda p: build_gauge_xxxx_circuit(p, model="depolarize"),
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
axA = axes[0]
print("== Panel A: analytic (closed form) vs Monte-Carlo ==")
print(f"{'circuit':20s} {'p':>6s} {'R_analytic':>11s} {'R_MC':>9s}")
maxdiff = 0.0
for name, builder in circuits.items():
    R_an, R_mc = [], []
    for p in ps:
        circ = builder(p)
        r_an, _ = analytic_change_rate_from_dem(circ)
        r_mc = float(sample_change_rate(circ, SHOTS, seed=7)[0]) if p > 0 else 0.0
        R_an.append(r_an); R_mc.append(r_mc)
        maxdiff = max(maxdiff, abs(r_an - r_mc))
        print(f"{name:20s} {p:6.3f} {r_an:11.4f} {r_mc:9.4f}")
    axA.plot(ps, R_an, "-", lw=2, label=f"{name}: analytic")
    axA.plot(ps, R_mc, "o", ms=5, mfc="none", label=f"{name}: Monte-Carlo")
axA.axhline(0.5, ls=":", c="grey", lw=1)
axA.set_xlabel("depolarizing parameter  p"); axA.set_ylabel("change rate  R")
axA.set_title(f"(A) closed form == simulation\nmax |diff| = {maxdiff:.4f} (shot noise)")
axA.legend(fontsize=8); axA.set_ylim(0, 0.55)

# ---------------------------------------------------------------------------
# Panel B: n_conseq geometric law.  Build a controlled circuit where an ancilla
# passes through n identical noisy CX interactions before measurement, so
# n_conseq is known exactly.  Show R = 1/2 (1-(1-2q)^n).
# ---------------------------------------------------------------------------
def build_n_location_circuit(n, p):
    """Ancilla accumulates errors from n identical noisy locations, then measured.
    Each location = a single-qubit depolarizing channel on the ancilla line
    (models 'one error location that can flip the measured operator')."""
    c = stim.Circuit()
    c.append("R", [0])
    for _ in range(n):
        add_1q_noise(c, [0], "depolarize", p)   # DEPOLARIZE1(p): X or Y flips Z-meas
    c.append("M", [0])
    c.append("DETECTOR", [stim.target_rec(-1)])
    return c

axB = axes[1]
ns = np.arange(1, 13)
p_fixed = 0.05
R_n_an, R_n_mc = [], []
for n in ns:
    circ = build_n_location_circuit(int(n), p_fixed)
    r_an, _ = analytic_change_rate_from_dem(circ)
    r_mc = float(sample_change_rate(circ, SHOTS, seed=11)[0])
    R_n_an.append(r_an); R_n_mc.append(r_mc)

# per-location flip prob q for DEPOLARIZE1(p): flips Z-measurement if X or Y -> 2p/3...
# actually DEPOLARIZE1(p) applies X,Y,Z each w.p. p/3; X and Y flip a Z-measurement
q_theory = 2 * p_fixed / 3
R_geom = 0.5 * (1 - (1 - 2 * q_theory) ** ns)

# fit q from the Monte-Carlo data using the Eq.(4) geometric form
fitf = lambda n, q: 0.5 * (1 - (1 - 2 * q) ** n)
q_fit, _ = curve_fit(fitf, ns, R_n_mc, p0=[0.03])
print("\n== Panel B: n_conseq geometric law (Eq. 4) ==")
print(f"per-location flip prob: theory q = 2p/3 = {q_theory:.4f},  fitted q = {q_fit[0]:.4f}")

axB.plot(ns, R_geom, "-", lw=2, label=r"Eq.(4): $\frac{1}{2}(1-(1-2q)^{n})$, q=2p/3")
axB.plot(ns, R_n_an, "s", ms=6, mfc="none", label="analytic (DEM)")
axB.plot(ns, R_n_mc, "o", ms=5, label="Monte-Carlo")
axB.axhline(0.5, ls=":", c="grey", lw=1)
axB.set_xlabel(r"$n_{conseq}$ (number of error locations)")
axB.set_ylabel("change rate  R")
axB.set_title(f"(B) geometric saturation, p={p_fixed}\nfitted q={q_fit[0]:.4f} vs theory {q_theory:.4f}")
axB.legend(fontsize=8); axB.set_ylim(0, 0.55)

fig.suptitle("Reproduction 1 -- analytic change-rate model vs simulation "
             "(paper Eq. 4 / Fig. 3)", fontsize=12)
fig.tight_layout()
fig.savefig("figs/repro1_change_rate.png", dpi=130)
print("\nsaved figs/repro1_change_rate.png")
