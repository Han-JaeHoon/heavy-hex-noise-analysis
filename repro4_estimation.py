"""
Reproduction 4  ->  Paper Sec. II F final paragraph + Fig. 23.

Claim: given a circuit-level (inhomogeneous) Pauli-noise model, the individual
error-mechanism probabilities can be RECOVERED from the pairwise
detection-event correlations -- to arbitrary precision when error rates are
below ~1% and shot noise is suppressed -- because each graph-like error
mechanism flips a specific pair of detectors and

    p_ij  ~=  p_edge(i,j)                                  (small-p limit of Eq. 7)

This is the correlation-based error-rate estimation the paper builds on
(Spitz et al. / Google [50]).  We:
  1. Inject a KNOWN inhomogeneous depolarizing model into a repetition memory.
  2. Read the TRUE per-mechanism (edge) probabilities from stim's detector
     error model.
  3. Estimate them purely from sampled correlations via Eq. (7).
  4. Show recovered vs true, and how accuracy degrades as the error rate grows
     past ~1% (higher-order terms) -- exactly the paper's Fig. 23 message.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim
from common import (build_repetition_memory, sample_detectors,
                    correlation_matrix, add_1q_noise, add_2q_noise)


def build_inhomogeneous_memory(distance, rounds, base_p, rng_scales):
    """Repetition memory with per-qubit inhomogeneous depolarizing strengths.
    rng_scales: dict giving a multiplicative scale for each physical qubit index."""
    d = distance
    data = [2 * i for i in range(d)]
    anc = [2 * i + 1 for i in range(d - 1)]
    n_anc = len(anc)
    c = stim.Circuit()
    all_q = data + anc
    c.append("R", all_q)
    for q in all_q:
        add_1q_noise(c, [q], "depolarize", base_p * rng_scales[q])

    def one_round():
        c.append("R", anc)
        for a in anc:
            add_1q_noise(c, [a], "depolarize", base_p * rng_scales[a])
        for i, a in enumerate(anc):
            c.append("CX", [data[i], a])
            add_2q_noise(c, (data[i], a), "depolarize", base_p * rng_scales[a])
        for i, a in enumerate(anc):
            c.append("CX", [data[i + 1], a])
            add_2q_noise(c, (data[i + 1], a), "depolarize", base_p * rng_scales[a])
        for q in data:
            add_1q_noise(c, [q], "depolarize", base_p * rng_scales[q])
        for a in anc:                     # readout noise, inhomogeneous
            add_1q_noise(c, [a], "depolarize", base_p * rng_scales[a])
        c.append("M", anc)

    one_round()
    for i in range(n_anc):
        c.append("DETECTOR", [stim.target_rec(-n_anc + i)], (i, 0))
    for r in range(1, rounds):
        one_round()
        for i in range(n_anc):
            c.append("DETECTOR", [stim.target_rec(-n_anc + i),
                                  stim.target_rec(-2 * n_anc + i)], (i, r))
    for q in data:
        add_1q_noise(c, [q], "depolarize", base_p * rng_scales[q])
    c.append("M", data)
    for i in range(n_anc):
        c.append("DETECTOR", [stim.target_rec(-d + i), stim.target_rec(-d + i + 1),
                              stim.target_rec(-d - n_anc + i)], (i, rounds))
    return c, n_anc


def true_edge_probs(circuit):
    """From stim DEM: map frozenset(detector pair) -> true mechanism probability.
    We merge boundary edges (single detector) under key frozenset({i})."""
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True,
                                       approximate_disjoint_errors=True)
    edges = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dets = frozenset(t.val for t in inst.targets_copy()
                         if t.is_relative_detector_id())
        if len(dets) == 0 or len(dets) > 2:
            continue
        # combine independent mechanisms with same symptom: p_comb = p1(1-p2)+p2(1-p1)
        if dets in edges:
            p1 = edges[dets]
            edges[dets] = p1 * (1 - prob) + prob * (1 - p1)
        else:
            edges[dets] = prob
    return edges


def estimate_edges_from_correlations(pij, mean, n_anc):
    """Recover two-detector edge probabilities from Eq.(7) correlations.
    Returns dict frozenset(pair)->estimate for graph-like (adjacent) pairs."""
    N = pij.shape[0]
    est = {}
    def ra(idx): return idx // n_anc, idx % n_anc
    for i in range(N):
        for j in range(i + 1, N):
            ri, ai = ra(i); rj, aj = ra(j)
            dr, da = abs(ri - rj), abs(ai - aj)
            # keep graph-like edges: time (dr1,da0), space (dr0,da1), space-time(dr1,da1)
            if (dr, da) in [(1, 0), (0, 1), (1, 1)]:
                est[frozenset((i, j))] = pij[i, j]
    return est


DISTANCE, ROUNDS = 5, 6
np.random.seed(0)
scales = {}
for q in range(2 * DISTANCE):
    scales[q] = float(np.random.uniform(0.4, 1.8))  # inhomogeneity

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

# ---- Panel A: recovered vs true at low error rate ----
base_p = 0.004
circ, n_anc = build_inhomogeneous_memory(DISTANCE, ROUNDS, base_p, scales)
true = true_edge_probs(circ)
SHOTS = 4_000_000
dets = sample_detectors(circ, SHOTS, seed=21)
pij = correlation_matrix(dets)
est = estimate_edges_from_correlations(pij, dets.mean(0), n_anc)

pairs = [k for k in est if len(k) == 2 and k in true]
xt = np.array([true[k] for k in pairs])
xe = np.array([est[k] for k in pairs])
ax = axes[0]
ax.plot([0, xt.max() * 1.1], [0, xt.max() * 1.1], "k--", lw=1, label="ideal y=x")
ax.plot(xt, xe, "o", ms=5, mfc="none", color="#4477aa")
ax.set_xlabel("TRUE mechanism probability (from DEM)")
ax.set_ylabel("ESTIMATED from correlations, Eq.(7)")
rel = np.median(np.abs(xe - xt) / xt)
ax.set_title(f"(A) inhomogeneous params recovered\nbase_p={base_p}, {SHOTS//10**6}M shots, "
             f"median rel.err={rel*100:.1f}%")
ax.legend(fontsize=9)

# ---- Panel B: recovery error vs error-rate (Fig. 23 message) ----
ax = axes[1]
ps = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
med_err = []
SHOTS_B = 2_000_000
for bp in ps:
    circ, n_anc = build_inhomogeneous_memory(DISTANCE, ROUNDS, bp, scales)
    true = true_edge_probs(circ)
    dets = sample_detectors(circ, SHOTS_B, seed=31)
    pij = correlation_matrix(dets)
    est = estimate_edges_from_correlations(pij, dets.mean(0), n_anc)
    pairs = [k for k in est if len(k) == 2 and k in true and true[k] > 1e-4]
    xt = np.array([true[k] for k in pairs]); xe = np.array([est[k] for k in pairs])
    med_err.append(float(np.median(np.abs(xe - xt) / xt)))
    print(f"base_p={bp:6.3f}  median rel.err of recovered mechanism probs = {med_err[-1]*100:6.2f}%")

ax.plot(np.array(ps) * 100, np.array(med_err) * 100, "-o", color="#ee6677")
ax.axvline(1.0, ls=":", c="grey", label="~1% error rate")
ax.set_xlabel("depolarizing error rate  (%)")
ax.set_ylabel("median rel. error of recovered params (%)")
ax.set_title("(B) accuracy degrades above ~1%\n(higher-order terms break Eq.7 approx)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=9)

fig.suptitle("Reproduction 4 -- inhomogeneous noise-parameter estimation from "
             "correlations (paper Fig. 23)", fontsize=12)
fig.tight_layout()
fig.savefig("figs/repro4_estimation.png", dpi=130)
print("saved figs/repro4_estimation.png")
