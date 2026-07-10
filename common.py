"""
Common utilities for reproducing:
  Gicev, Hollenberg, Usman,
  "Quantum computer error structure probed by quantum error correction
   syndrome measurements", Phys. Rev. Research 6, 043249 (2024).

We reproduce the *simulation / theory* portions (hardware data needs IBM access).

Key objects:
  - build_gauge_zz_circuit / build_gauge_xxxx_circuit : single operator-measurement
    circuits used to verify the analytic change-rate formula (Eq. 4 / Eq. 6).
  - build_repetition_memory : repeated Z-stabilizer (bit-flip repetition code)
    circuit used for per-cycle change rates (Fig. 12) and correlations (Fig. 20).
  - Both depolarizing and Z-biased noise can be attached.
"""
import numpy as np
import stim


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------
def biased_1q(p, eta):
    """Single-qubit biased Pauli channel, paper Eq. (3).
    Total nontrivial-error prob = p. r1=r2=1/(2(eta+1)), r3=eta/(eta+1).
    Returns (px, py, pz)."""
    r1 = r2 = 1.0 / (2 * (eta + 1))
    r3 = eta / (eta + 1)
    return p * r1, p * r2, p * r3


def add_1q_noise(circ, qubits, model, p, eta=None):
    if p <= 0:
        return
    if model == "depolarize":
        circ.append("DEPOLARIZE1", qubits, p)
    elif model == "biased":
        px, py, pz = biased_1q(p, eta)
        circ.append("PAULI_CHANNEL_1", qubits, (px, py, pz))
    else:
        raise ValueError(model)


def add_2q_noise(circ, pair, model, p, eta=None):
    """Two-qubit noise after a CX. Depolarize2 for depolarizing; for the biased
    model we apply an independent biased single-qubit channel to each of the two
    qubits (a common tensor-product approximation)."""
    if p <= 0:
        return
    if model == "depolarize":
        circ.append("DEPOLARIZE2", pair, p)
    elif model == "biased":
        add_1q_noise(circ, list(pair), "biased", p, eta)
    else:
        raise ValueError(model)


# ---------------------------------------------------------------------------
# Single gauge-operator measurement circuits (for Eq. 4 / Eq. 6 verification)
# ---------------------------------------------------------------------------
def build_gauge_zz_circuit(p, model="depolarize", eta=None,
                           init="00", noisy_reset=True, noisy_meas=True):
    """Prepare 2 data qubits in a Z-basis product state, measure ZZ via one
    ancilla, place a DETECTOR on the ancilla vs the expected eigenvalue.

    Qubits: 0,1 = data ; 2 = ancilla.
    n_conseq counting (per paper): reset(anc)=1, after CX0->anc=1, after
    CX1->anc=1, measure(anc)=1  -> the Z-flipping error locations on the anc line.
    """
    c = stim.Circuit()
    data = [0, 1]
    anc = 2
    # prepare data in |init>
    c.append("R", data + [anc])
    if noisy_reset:
        add_1q_noise(c, [anc], model, p, eta)          # reset noise (anc)
    for q, b in zip(data, init):
        if b == "1":
            c.append("X", [q])
    # ---- ZZ measurement ----
    c.append("CX", [data[0], anc]); add_2q_noise(c, (data[0], anc), model, p, eta)
    c.append("CX", [data[1], anc]); add_2q_noise(c, (data[1], anc), model, p, eta)
    if noisy_meas:
        add_1q_noise(c, [anc], model, p, eta)          # readout noise (anc)
    c.append("M", [anc])
    # For the analytic-vs-MC verification we use even-parity inputs (00, 11) so
    # a noiseless run gives ancilla == 0; the detector then fires exactly on an
    # operator change.  (Under depolarizing noise the change rate is independent
    # of the input state anyway -- one of the paper's baseline facts.)
    assert init.count("1") % 2 == 0, "use an even-parity Z-basis input"
    c.append("DETECTOR", [stim.target_rec(-1)])
    return c


def build_gauge_xxxx_circuit(p, model="depolarize", eta=None):
    """Flagged XXXX gauge operator, distance-relevant. 4 data + 1 syndrome + 1 flag.
    We prepare |++++>, measure XXXX, detector on syndrome ancilla.
    This is the deepest of the three gauge circuits (largest n_conseq)."""
    c = stim.Circuit()
    data = [0, 1, 2, 3]
    syn = 4
    flag = 5
    c.append("R", data + [syn, flag])
    add_1q_noise(c, [syn, flag], model, p, eta)
    # prep |++++>
    c.append("H", data)
    # flagged XXXX: H syndrome, entangle with flag, CX chain in X basis
    c.append("H", [syn])
    c.append("CX", [syn, flag]); add_2q_noise(c, (syn, flag), model, p, eta)
    for d in data:
        c.append("CX", [syn, d]); add_2q_noise(c, (syn, d), model, p, eta)
    c.append("CX", [syn, flag]); add_2q_noise(c, (syn, flag), model, p, eta)
    c.append("H", [syn])
    add_1q_noise(c, [syn], model, p, eta)
    c.append("M", [syn])
    c.append("DETECTOR", [stim.target_rec(-1)])
    return c


def analytic_change_rate_from_dem(circuit, detector_index=0):
    """Change rate predicted analytically from stim's detector error model.
    For independent mechanisms with probs {p_i} that flip the detector,
        R = (1 - prod_i (1 - 2 p_i)) / 2 .
    When all p_i == p this equals 1/2 (1 - (1-2p)^n).  This IS the generalized
    form of paper Eq. (4)/(6)."""
    dem = circuit.detector_error_model(decompose_errors=False,
                                       flatten_loops=True,
                                       allow_gauge_detectors=True,
                                       approximate_disjoint_errors=True)
    prod = 1.0
    n = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        targets = inst.targets_copy()
        flips = any(t.is_relative_detector_id() and t.val == detector_index
                    for t in targets)
        if flips:
            prod *= (1 - 2 * prob)
            n += 1
    return (1 - prod) / 2, n


def sample_change_rate(circuit, shots, seed):
    """Direct Monte-Carlo change rate = fraction of shots where detector fired."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets = sampler.sample(shots=shots)
    return dets.mean(axis=0)  # per-detector fire rate


# ---------------------------------------------------------------------------
# Repeated Z-stabilizer memory (bit-flip repetition code) : Fig. 12 / Fig. 20
# ---------------------------------------------------------------------------
def build_repetition_memory(distance, rounds, p, model="depolarize", eta=None):
    """Bit-flip repetition code memory experiment.
    Data qubits at even indices 0,2,...,2(d-1); ancillas at odd indices.
    Each round: reset anc, CX(left->anc), CX(right->anc), measure anc.
    DETECTORs compare an ancilla to its value in the previous round (first round
    vs reset; final round vs data-qubit parity).  Detector firing == the Z
    'operator change' event of the paper.
    Returns (circuit, n_anc)."""
    d = distance
    data = [2 * i for i in range(d)]
    anc = [2 * i + 1 for i in range(d - 1)]
    n_anc = len(anc)
    c = stim.Circuit()
    all_q = data + anc
    c.append("R", all_q)
    add_1q_noise(c, all_q, model, p, eta)

    def one_round():
        c.append("R", anc)
        add_1q_noise(c, anc, model, p, eta)
        # left CX
        for i, a in enumerate(anc):
            c.append("CX", [data[i], a]); add_2q_noise(c, (data[i], a), model, p, eta)
        # right CX
        for i, a in enumerate(anc):
            c.append("CX", [data[i + 1], a]); add_2q_noise(c, (data[i + 1], a), model, p, eta)
        # idle noise on data during measurement
        add_1q_noise(c, data, model, p, eta)
        add_1q_noise(c, anc, model, p, eta)   # readout noise
        c.append("M", anc)

    # round 0
    one_round()
    for i in range(n_anc):
        c.append("DETECTOR", [stim.target_rec(-n_anc + i)], (i, 0))
    # rounds 1..R-1
    for r in range(1, rounds):
        one_round()
        for i in range(n_anc):
            c.append("DETECTOR",
                     [stim.target_rec(-n_anc + i),
                      stim.target_rec(-2 * n_anc + i)], (i, r))
    # final data measurement -> reconstruct last-round parity
    add_1q_noise(c, data, model, p, eta)
    c.append("M", data)
    for i in range(n_anc):
        # ancilla i checks parity of data[i], data[i+1]
        c.append("DETECTOR",
                 [stim.target_rec(-d + i), stim.target_rec(-d + i + 1),
                  stim.target_rec(-d - n_anc + i)], (i, rounds))
    return c, n_anc


def sample_detectors(circuit, shots, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(shots=shots).astype(np.int8)


def per_cycle_change_rate(dets, n_anc, rounds_plus1):
    """dets shape (shots, n_anc*(rounds+1)) laid out round-major as appended.
    Returns array (rounds+1, n_anc) of firing rates."""
    m = dets.mean(axis=0)
    return m.reshape(rounds_plus1, n_anc)


def correlation_matrix(dets):
    """Paper Eq. (7):  p_ij = (<xi xj> - <xi><xj>) / ((1-2<xi>)(1-2<xj>)),
    diagonal set to 0."""
    x = dets.astype(np.float64)
    mean = x.mean(axis=0)
    N = x.shape[1]
    cov = (x.T @ x) / x.shape[0] - np.outer(mean, mean)
    denom = np.outer(1 - 2 * mean, 1 - 2 * mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        pij = cov / denom
    pij[~np.isfinite(pij)] = 0.0
    np.fill_diagonal(pij, 0.0)
    return pij
