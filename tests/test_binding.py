"""Cross-validates the QuaSARQ python binding against stim.

    make binding && make binding-test PYTHON=/path/to/venv/bin/python

Needs stim, pymatching and numpy, and the built module on PYTHONPATH.
"""

import math
import sys
import threading
import time

import numpy as np
import quasarq
import stim

FAILURES = []


def check(ok, what, detail=""):
    print(f"  {what:<62} {'PASS' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(what)


def section(title):
    print(f"\n{title}")


def circuit(distance, rounds, p=0.002):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def sampler(c, seed=1):
    return quasarq.compile_detector_sampler(c, seed=seed)


def msampler(c, seed=1):
    return quasarq.compile_sampler(c, seed=seed)


def packbits(a):
    return np.packbits(a, axis=1, bitorder="little")


def sigmas(rate_a, rate_b, n):
    # Gap between two independent binomial rates in pooled standard errors, elementwise
    # over per-detector arrays or over plain scalars.
    spread = np.clip(rate_a * (1 - rate_a) + rate_b * (1 - rate_b), 1e-12, None)
    return np.abs(rate_a - rate_b) / np.sqrt(spread / n)


def both_samplers(c, shots, seed=1):
    q = sampler(c, seed).sample(shots, separate_observables=True)
    s = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return q, s


def matcher_for(c):
    import pymatching

    return pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True))


def logical_error_rate(matcher, dets, obs):
    return float(np.mean(matcher.decode_batch(dets)[:, 0] != obs[:, 0]))


# Detector and observable counts must match stim, and every sample() keyword combination
# must return the same shapes and dtypes stim returns.
def test_interface(c, shots):
    qs, ss = sampler(c), c.compile_detector_sampler(seed=1)
    check(qs.num_detectors == c.num_detectors, "num_detectors agrees with stim",
          f"{qs.num_detectors}")
    check(qs.num_observables == c.num_observables, "num_observables agrees with stim",
          f"{qs.num_observables}")
    for kwargs in (dict(),
                   dict(bit_packed=True),
                   dict(separate_observables=True),
                   dict(separate_observables=True, bit_packed=True)):
        q, s = qs.sample(shots, **kwargs), ss.sample(shots, **kwargs)
        if isinstance(s, tuple):
            ok = (len(q) == len(s)
                  and all(a.shape == b.shape and a.dtype == b.dtype
                          for a, b in zip(q, s)))
            shapes = ", ".join(f"{a.shape}/{a.dtype}" for a in q)
        else:
            ok = q.shape == s.shape and q.dtype == s.dtype
            shapes = f"{q.shape}/{q.dtype}"
        check(ok, f"sample({kwargs}) shape+dtype", shapes)


# Packed output must be exactly numpy.packbits(..., bitorder='little') of the unpacked
# output. Two fresh samplers on one seed agree on their first call, so the two forms
# describe the same bits.
def test_bit_packing(c, shots):
    d_un, o_un = sampler(c, 42).sample(shots, separate_observables=True)
    d_pk, o_pk = sampler(c, 42).sample(shots, separate_observables=True, bit_packed=True)
    check(d_un.shape[0] == shots, "unpacked row count")
    ref = packbits(d_un)
    check(ref.shape == d_pk.shape, "packed detector shape", f"{d_pk.shape}")
    check(np.array_equal(ref, d_pk), "packbits(unpacked) == packed detectors")
    check(np.array_equal(packbits(o_un), o_pk), "packbits(unpacked) == packed observables")


# Seeding must behave as stim's does: one seed reproduces, different seeds diverge,
# seed=None is nondeterministic, and repeated calls on one sampler advance the stream
# without drifting statistically.
def test_seeding(c, shots):
    a = sampler(c, 7).sample(4000)
    b = sampler(c, 7).sample(4000)
    check(np.array_equal(a, b), "same seed gives identical detection events")
    check(not np.array_equal(a, sampler(c, 8).sample(4000)),
          "different seed gives different detection events")

    reused = sampler(c, 7)
    r1, r2 = reused.sample(4000), reused.sample(4000)
    check(not np.array_equal(r1, r2),
          "consecutive sample() calls advance the stream (as stim does)")
    check(abs(r1.mean() - r2.mean()) < 0.005,
          "consecutive calls keep the same statistics",
          f"{r1.mean():.5f} vs {r2.mean():.5f}")

    check(not np.array_equal(quasarq.compile_detector_sampler(c).sample(2000),
                             quasarq.compile_detector_sampler(c).sample(2000)),
          "seed=None is nondeterministic (as stim does)")

    chain = sampler(c, 99)
    batches = [chain.sample(3000) for _ in range(6)]
    check(all(not np.array_equal(batches[i], batches[j])
              for i in range(len(batches)) for j in range(i + 1, len(batches))),
          "six consecutive batches are all distinct")
    rates = [x.mean() for x in batches]
    check(max(rates) - min(rates) < 0.01, "batch rates are stable",
          f"spread={max(rates) - min(rates):.5f}")


# Per-detector firing rates and the observable flip rate must agree with stim to within
# 5 pooled standard errors.
def test_rates_match_stim(c, shots):
    (q_dets, q_obs), (s_dets, s_obs) = both_samplers(c, shots)
    q_rate, s_rate = q_dets.mean(axis=0), s_dets.mean(axis=0)
    z = sigmas(q_rate, s_rate, shots)
    worst = int(np.argmax(z))
    check(float(np.max(z)) < 5.0, "all detectors within 5 sigma of stim",
          f"max z={np.max(z):.2f} at detector {worst} "
          f"(q={q_rate[worst]:.4f} s={s_rate[worst]:.4f})")
    check(abs(q_rate.mean() - s_rate.mean()) < 0.01, "mean detector rate close to stim",
          f"q={q_rate.mean():.4f} s={s_rate.mean():.4f}")
    qo, so = q_obs.mean(), s_obs.mean()
    check(sigmas(qo, so, shots) < 5.0, "observable flip rate within 5 sigma",
          f"q={qo:.4f} s={so:.4f}")


# Decoding with pymatching must show logical error rate falling with code distance, and
# must agree with stim's sampler when both are decoded the same way.
def test_decoding(c, shots):
    try:
        import pymatching  # noqa: F401
    except ImportError:
        print("  pymatching unavailable, skipping")
        return

    n = 30000
    rates = []
    for dist in (3, 5, 7, 9):
        cd = circuit(dist, dist)
        dets, obs = sampler(cd, 5).sample(n, separate_observables=True)
        rate = logical_error_rate(matcher_for(cd), dets, obs)
        rates.append(rate)
        print(f"    d={dist:<2} logical error rate {rate:.5f}  ({round(rate * n)}/{n})")
    check(all(rates[i] > rates[i + 1] for i in range(len(rates) - 1)),
          "logical error rate strictly decreasing in distance",
          " > ".join(f"{r:.5f}" for r in rates))

    matcher = matcher_for(c)
    (qd, qo), (sd, so) = both_samplers(c, n, seed=11)
    q_err = logical_error_rate(matcher, qd, qo)
    s_err = logical_error_rate(matcher, sd, so)
    check(sigmas(q_err, s_err, n) < 5.0, "decoded LER within 5 sigma of stim",
          f"quasarq={q_err:.5f} stim={s_err:.5f}")


# A sample big enough to be split into chunks internally must come back as one coherent
# array: right shape, no all-zero region, and both halves agreeing statistically.
def test_chunked_sampling(c, shots):
    big = 200000
    d_big = sampler(c, 3).sample(big, separate_observables=True)[0]
    check(d_big.shape == (big, c.num_detectors), "large sample shape", f"{d_big.shape}")
    halves = [d_big[:big // 2].mean(), d_big[big // 2:].mean()]
    check(abs(halves[0] - halves[1]) < 0.005, "first and second half rates agree",
          f"{halves[0]:.5f} vs {halves[1]:.5f}")
    check(d_big.mean() > 0, "large sample is not all zeros", f"mean={d_big.mean():.5f}")


# Shot counts must be honoured exactly, including counts far below the internal 64-shot
# word granularity, and circuits with nothing to report must not be special-cased away.
def test_edge_cases(c, shots):
    qs = sampler(c)
    check(qs.sample(0).shape == (0, c.num_detectors), "zero shots returns empty array",
          f"{qs.sample(0).shape}")

    counts = (1, 2, 3, 7, 13, 63, 64, 65, 100, 500, 999)
    exact = []
    for n in counts:
        dets, obs = qs.sample(n, separate_observables=True)
        packed = qs.sample(n, bit_packed=True)
        exact.append(dets.shape == (n, c.num_detectors)
                     and obs.shape == (n, c.num_observables)
                     and packed.shape == (n, (c.num_detectors + 7) // 8))
    check(all(exact), "exact shot counts honoured from 1 upward",
          f"{len(exact)} counts checked")

    singles = [sampler(c, k).sample(1) for k in range(24)]
    check(all(x.shape == (1, c.num_detectors) for x in singles),
          "1 shot returns one row")
    distinct = len({x.tobytes() for x in singles})
    check(distinct > 1, "1-shot results vary across seeds", f"{distinct} distinct of 24")

    ps = sampler(stim.Circuit("H 0\nM 0"))
    check(ps.num_detectors == 0 and ps.num_observables == 0,
          "circuit with no detectors reports zero")
    dets, obs = ps.sample(100, separate_observables=True)
    check(dets.shape == (100, 0) and obs.shape == (100, 0),
          "no-detector circuit returns empty columns", f"{dets.shape}")

    try:
        sampler(stim.Circuit())
        check(False, "empty circuit raises")
    except Exception as e:
        check(True, "empty circuit raises", type(e).__name__)
    try:
        qs.sample(10, append_observables=True)
        check(False, "append_observables rejected")
    except Exception as e:
        check(isinstance(e, (ValueError, TypeError)), "append_observables rejected",
              type(e).__name__)


# Circuit text must be accepted wherever a stim.Circuit is, and produce the same samples.
def test_circuit_text(c, shots):
    from_text = sampler(str(c))
    check(from_text.num_detectors == c.num_detectors, "text input parses identically")
    from_obj = sampler(c)
    check(np.array_equal(from_text.sample(2000), from_obj.sample(2000)),
          "text input samples identically to stim.Circuit input")


# Sampling must drop the GIL, so an unrelated python thread keeps running throughout.
def test_gil_released(c, shots):
    ticks = {"n": 0}
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks["n"] += 1
            time.sleep(0.001)

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    sampler(circuit(11, 11)).sample(100000)
    stop.set()
    thread.join(timeout=1)
    check(ticks["n"] > 10, "python thread ran during sampling", f"{ticks['n']} ticks")

DETERMINISTIC_ONE = """
    X 0
    X_ERROR({p}) 0
    M 0
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]
"""


def test_reference_frame(c, shots):
    dets, obs = sampler(circuit(3, 3, p=0)).sample(8192, separate_observables=True)
    check(not dets.any(), "noiseless surface code fires no detectors",
          f"{int(dets.sum())} events")
    check(not obs.any(), "noiseless surface code flips no observables",
          f"{int(obs.sum())} flips")

    quiet = stim.Circuit(DETERMINISTIC_ONE.format(p=0))
    dets, obs = sampler(quiet).sample(1024, separate_observables=True)
    check(not dets.any(), "detector over a deterministic 1 stays silent",
          f"{int(dets.sum())} events")
    check(not obs.any(), "observable over a deterministic 1 stays unflipped",
          f"{int(obs.sum())} flips")

    n = 200000
    (qd, qo), (sd, so) = both_samplers(stim.Circuit(DETERMINISTIC_ONE.format(p=0.25)), n, seed=7)
    check(sigmas(qd.mean(), sd.mean(), n) < 5.0, "noisy detector rate follows the reference",
          f"quasarq={qd.mean():.4f} stim={sd.mean():.4f} "
          f"(raw parity would be {1 - sd.mean():.4f})")
    check(sigmas(qo.mean(), so.mean(), n) < 5.0, "noisy observable rate follows the reference",
          f"quasarq={qo.mean():.4f} stim={so.mean():.4f}")


MEASURE_FLIP = """
    {reset} 0
    {gate}({p}) 0
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]
"""

RESET_FOR_BASIS = {"M": "R", "MR": "R", "MX": "RX", "MRX": "RX", "MY": "RY", "MRY": "RY"}

SPLIT_OBSERVABLE = """
    R 0 1
    X_ERROR(0.3) 0
    X_ERROR(0.1) 1
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
    M 1
    OBSERVABLE_INCLUDE(0) rec[-1]
"""

MIXED_BASIS_OBSERVABLES = """
    RX 0 1
    R 2
    Z_ERROR(0.2) 0
    Z_ERROR(0.05) 1
    X_ERROR(0.1) 2
    MX(0.02) 0 1
    M(0.02) 2
    DETECTOR rec[-3]
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-3]
    OBSERVABLE_INCLUDE(1) rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-2]
"""

def test_measurement_flip_noise(c, shots):
    n = 200000
    for gate, reset in RESET_FOR_BASIS.items():
        text = MEASURE_FLIP.format(reset=reset, gate=gate, p=0.25)
        (qd, _), (sd, _) = both_samplers(stim.Circuit(text), n, seed=3)
        q, s = qd.mean(), sd.mean()
        # Absolute bounds as well as agreement: a dropped argument gives 0.0 and a
        # wrong-basis reset gives 0.5, and both must fail here.
        check(sigmas(q, s, n) < 5.0 and 0.2 < q < 0.3,
              f"{gate}(p) flip rate agrees with stim and is ~p",
              f"quasarq={q:.4f} stim={s:.4f}")

    for gate, reset in (("MX", "RX"), ("M", "R")):
        zero = sampler(stim.Circuit(MEASURE_FLIP.format(reset=reset, gate=gate, p=0))).sample(20000)
        check(not zero.any(), f"{gate}(0) flips nothing", f"{int(zero.sum())} events")


def test_observable_merging(c, shots):
    n = 200000
    text = stim.Circuit(SPLIT_OBSERVABLE)
    qs = sampler(text)
    check(qs.num_observables == text.num_observables,
          "two lines sharing an id count as one observable",
          f"quasarq={qs.num_observables} stim={text.num_observables}")
    (_, qo), (_, so) = both_samplers(text, n, seed=5)
    check(sigmas(qo[:, 0].mean(), so[:, 0].mean(), n) < 5.0,
          "merged observable rate agrees with stim",
          f"quasarq={qo[:, 0].mean():.4f} stim={so[:, 0].mean():.4f}")

    check(abs(qo[:, 0].mean() - 0.5) > 0.05, "merged observable is not a coin flip",
          f"rate={qo[:, 0].mean():.4f}")

    mixed = stim.Circuit(MIXED_BASIS_OBSERVABLES)
    qs = sampler(mixed)
    check(qs.num_observables == mixed.num_observables,
          "interleaved ids across bases count correctly",
          f"quasarq={qs.num_observables} stim={mixed.num_observables}")
    (qd, qo), (sd, so) = both_samplers(mixed, n, seed=5)
    check(bool(np.all(sigmas(qd.mean(0), sd.mean(0), n) < 5.0)),
          "mixed-basis detector rates agree with stim",
          f"quasarq={np.round(qd.mean(0), 4)} stim={np.round(sd.mean(0), 4)}")
    check(bool(np.all(sigmas(qo.mean(0), so.mean(0), n) < 5.0)),
          "mixed-basis observable rates agree with stim",
          f"quasarq={np.round(qo.mean(0), 4)} stim={np.round(so.mean(0), 4)}")


DETERMINISTIC_RECORDS = """
    R 0 1 2
    X 0
    H 2
    M 0 1
    MX 2
"""

def test_measurement_interface(c, shots):
    qs, ss = msampler(c), c.compile_sampler(seed=1)
    check(qs.num_measurements == c.num_measurements, "num_measurements agrees with stim",
          f"{qs.num_measurements}")
    for kwargs in (dict(), dict(bit_packed=True)):
        q, s = qs.sample(shots, **kwargs), ss.sample(shots, **kwargs)
        check(q.shape == s.shape and q.dtype == s.dtype, f"sample({kwargs}) shape+dtype",
              f"{q.shape}/{q.dtype}")

    check(qs.sample(0).shape == (0, c.num_measurements), "zero shots returns empty array")
    exact = [qs.sample(n).shape == (n, c.num_measurements)
             for n in (1, 2, 3, 7, 13, 63, 64, 65, 500, 999)]
    check(all(exact), "exact shot counts honoured from 1 upward", f"{len(exact)} counts checked")

    quiet = msampler(stim.Circuit("H 0"))
    check(quiet.num_measurements == 0, "circuit with no measurements reports zero")
    check(quiet.sample(100).shape == (100, 0), "no-measurement circuit returns empty columns")

    det = msampler(stim.Circuit(DETERMINISTIC_RECORDS))
    outcomes = det.sample(1024)
    check(bool(outcomes[:, 0].all()), "X 0 then M 0 reads 1 every shot",
          f"{outcomes[:, 0].mean():.4f}")
    check(not outcomes[:, 1:].any(), "R 1/H 2 then M 1/MX 2 read 0 every shot",
          f"{int(outcomes[:, 1:].sum())} ones")

    d_un = msampler(c, 42).sample(4000)
    d_pk = msampler(c, 42).sample(4000, bit_packed=True)
    check(np.array_equal(packbits(d_un), d_pk), "packbits(unpacked) == packed measurements")

    a, b = msampler(c, 7).sample(2000), msampler(c, 7).sample(2000)
    check(np.array_equal(a, b), "same seed gives identical measurements")
    check(not np.array_equal(a, msampler(c, 8).sample(2000)),
          "different seed gives different measurements")
    try:
        qs.sample(10, separate_observables=True)
        check(False, "separate_observables rejected")
    except Exception as e:
        check(isinstance(e, TypeError), "separate_observables rejected", type(e).__name__)


def test_measurement_rates_match_stim(c, shots):
    n = 100000
    q = msampler(c, 13).sample(n)
    s = c.compile_sampler(seed=13).sample(n)
    z = sigmas(q.mean(axis=0), s.mean(axis=0), n)
    worst = int(np.argmax(z))
    check(float(np.max(z)) < 5.0, "all measurements within 5 sigma of stim",
          f"max z={np.max(z):.2f} at measurement {worst}")

    noisy = stim.Circuit(MIXED_BASIS_OBSERVABLES)
    q = msampler(noisy, 17).sample(n)
    s = noisy.compile_sampler(seed=17).sample(n)
    z = sigmas(q.mean(axis=0), s.mean(axis=0), n)
    check(float(np.max(z)) < 5.0, "mixed-basis M(p) outcomes within 5 sigma of stim",
          f"quasarq={np.round(q.mean(axis=0), 4)} stim={np.round(s.mean(axis=0), 4)}")


def test_m2d_round_trip(c, shots):
    n = 50000
    converter = c.compile_m2d_converter()
    dets, obs = converter.convert(measurements=msampler(c, 21).sample(n),
                                  separate_observables=True)
    q_dets, q_obs = sampler(c, 23).sample(n, separate_observables=True)
    z = sigmas(dets.mean(axis=0), q_dets.mean(axis=0), n)
    check(float(np.max(z)) < 5.0, "m2d(measurements) detector rates match the detector sampler",
          f"max z={np.max(z):.2f}")
    z = sigmas(obs.mean(axis=0), q_obs.mean(axis=0), n)
    check(float(np.max(z)) < 5.0, "m2d(measurements) observable rates match the detector sampler",
          f"max z={np.max(z):.2f}")


# Splitting a request across GPU chunks must not change a single bit of the result, for either
# sampler: the seed stream and the destination row offsets both have to survive the split.
def test_chunk_invariance(c, shots):
    n = 20000
    whole_m = msampler(c, 31).sample(n)
    whole_d, whole_o = sampler(c, 31).sample(n, separate_observables=True)
    for chunk in (64, 1000, 4096):
        quasarq.set_chunk_shots(chunk)
        try:
            m = msampler(c, 31).sample(n)
            d, o = sampler(c, 31).sample(n, separate_observables=True)
        finally:
            quasarq.set_chunk_shots(0)
        check(np.array_equal(m, whole_m), f"chunk={chunk} measurements match unchunked")
        check(np.array_equal(d, whole_d) and np.array_equal(o, whole_o),
              f"chunk={chunk} detection events match unchunked")
    check(quasarq.get_chunk_shots() == 0, "chunk size restored to automatic")


# Circuits whose measurement outcomes are forced, so a single deterministic run must reproduce
# them exactly rather than statistically.
FORCED_MEASUREMENTS = (
    ("R 0 1\nX 0\nM 0 1", [True, False]),
    ("R 0 1\nX 0 1\nM 0 1", [True, True]),
    ("R 0\nH 0\nH 0\nM 0", [False]),
    ("R 0\nX 0\nMR 0\nM 0", [True, False]),
    ("RX 0\nMX 0", [False]),
)

# Measurement-free Clifford circuits, so the state is deterministic and can be compared against
# stim without the two disagreeing merely because a random outcome differed.
CLIFFORD_STATES = (
    ("H 0", 1),
    ("S 0", 1),
    ("H 0\nCX 0 1", 2),
    ("X 0\nH 1\nCX 1 2\nS 2", 3),
    ("H 0\nH 1\nCZ 0 1\nS 1\nH 0", 2),
    ("H 0\nCX 0 1\nCX 1 2\nS 2\nH 1\nCZ 0 2", 3),
)


def stim_inverse_rows(text, n):
    # stim writes identity as '_' where quasarq writes 'I'.
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    sim.do_circuit(stim.Circuit(text))
    inverse = sim.current_inverse_tableau()
    rows = [str(inverse.x_output(q)) for q in range(n)]
    rows += [str(inverse.z_output(q)) for q in range(n)]
    return [row.replace("_", "I") for row in rows]


# The simulation path runs a circuit once on the tableau, with no sampling, and reports what it
# measured and the state it ended in.
def test_simulation(c, shots):
    sim = quasarq.simulate("R 0 1 2\nX 0\nM 0 1 2")
    check(sim.num_qubits == 3, "num_qubits agrees with the circuit", f"{sim.num_qubits}")
    check(sim.num_measurements == 3, "num_measurements agrees with the circuit",
          f"{sim.num_measurements}")
    record = sim.measurements()
    check(record.shape == (3,) and record.dtype == np.bool_, "measurements shape+dtype",
          f"{record.shape}/{record.dtype}")

    exact = []
    for text, expected in FORCED_MEASUREMENTS:
        got = quasarq.simulate(text).measurements().tolist()
        exact.append(got == expected)
        if got != expected:
            check(False, f"forced outcome {text!r}", f"got {got}, want {expected}")
    check(all(exact), "forced measurement outcomes are exact", f"{len(exact)} circuits checked")

    quiet = quasarq.simulate("H 0\nCX 0 1")
    check(quiet.num_measurements == 0, "circuit with no measurements reports zero")
    check(quiet.measurements().shape == (0,), "no-measurement circuit returns an empty record",
          f"{quiet.measurements().shape}")

    # The rows are the inverse tableau: x_output for every qubit, then z_output.
    agreed = []
    for text, n in CLIFFORD_STATES:
        rows = quasarq.simulate(text).paulis()
        want = stim_inverse_rows(text, n)
        agreed.append(rows == want)
        if rows != want:
            check(False, f"pauli rows {text!r}", f"got {rows}, want {want}")
    check(all(agreed), "pauli rows match stim's inverse tableau",
          f"{len(agreed)} circuits, {sum(len(stim_inverse_rows(t, n)) for t, n in CLIFFORD_STATES)} rows")

    rows = quasarq.simulate("X 0\nH 1\nM 0 1").paulis()
    check(len(rows) == 4 and all(len(r) == 3 for r in rows),
          "an extended tableau lists destabilizers then stabilizers", f"{rows}")
    check(all(r[0] in "+-" and set(r[1:]) <= set("IXYZ") for r in rows),
          "every pauli row is a sign followed by IXYZ")

    try:
        quasarq.simulate(stim.Circuit())
        check(False, "empty circuit raises")
    except Exception as e:
        check(True, "empty circuit raises", type(e).__name__)


# Clifford identities, so the verdict is known independently of any simulator.
EQUIVALENT_PAIRS = (
    ("H 0\nH 0", "I 0"),
    ("S 0\nS 0", "Z 0"),
    ("X 0\nX 0", "I 0"),
    ("H 0\nS 0\nH 0\nS 0\nH 0\nS 0", "I 0"),
    ("CX 0 1", "H 1\nCZ 0 1\nH 1"),
    ("H 0\nH 1\nCX 0 1\nH 0\nH 1", "CX 1 0"),
    ("S 0\nS 0\nS 0\nS 0", "I 0"),
)

INEQUIVALENT_PAIRS = (
    ("X 0", "Z 0"),
    ("CX 0 1", "CX 1 0"),
    ("S 0", "Z 0"),
    ("H 0", "I 0"),
    ("CX 0 1", "CZ 0 1"),
)

# Random Clifford circuits, checked against stim's own tableau comparison so the oracle is
# independent of QuaSARQ rather than another QuaSARQ run.
def random_clifford_text(rng, qubits, gates):
    ops = []
    for _ in range(gates):
        pick = rng.integers(0, 5)
        if pick < 3:
            ops.append(f"{['H', 'S', 'X'][pick]} {rng.integers(0, qubits)}")
        else:
            a = int(rng.integers(0, qubits))
            b = int((a + 1 + rng.integers(0, qubits - 1)) % qubits)
            ops.append(f"{'CX' if pick == 3 else 'CZ'} {a} {b}")
    return "\n".join(ops)


# Equivalence checking answers whether two circuits realise the same Clifford operation.
def test_equivalence(c, shots):
    same = [quasarq.equivalent(a, b) for a, b in EQUIVALENT_PAIRS]
    check(all(same), "known-equivalent circuits are equivalent",
          f"{sum(same)}/{len(same)} pairs")
    differ = [quasarq.equivalent(a, b) for a, b in INEQUIVALENT_PAIRS]
    check(not any(differ), "known-different circuits are not equivalent",
          f"{len(differ) - sum(differ)}/{len(differ)} pairs rejected")

    rng = np.random.default_rng(7)
    agreed, compared = True, 0
    for _ in range(12):
        qubits = int(rng.integers(2, 6))
        a = random_clifford_text(rng, qubits, int(rng.integers(4, 14)))
        # Half the time compare against a genuinely different circuit, half against a rewrite
        # that must stay equivalent.
        b = a + "\nH 0\nH 0" if rng.integers(0, 2) else random_clifford_text(rng, qubits, 6)
        pad = f"\nI {qubits - 1}"
        want = stim.Tableau.from_circuit(stim.Circuit(a + pad)) == \
               stim.Tableau.from_circuit(stim.Circuit(b + pad))
        got = quasarq.equivalent(a + pad, b + pad)
        compared += 1
        if got != want:
            check(False, "random circuit verdict matches stim",
                  f"quasarq={got} stim={want} for {a!r} vs {b!r}")
            agreed = False
    check(agreed, "random circuit verdicts match stim's tableau comparison",
          f"{compared} pairs")

    # Every form a circuit can take is accepted, since they all reach the same text.
    forms = (quasarq.Circuit("CX 0 1"), "CX 0 1", stim.Circuit("CX 0 1"))
    accepted = [quasarq.equivalent(f, "H 1\nCZ 0 1\nH 1") for f in forms]
    check(all(accepted), "accepts Circuit, text and stim.Circuit", f"{len(accepted)} forms")

    # A tableau comparison cannot see measurements, so a circuit containing one has to be
    # refused rather than silently compared as though it were absent.
    for a, b, what in (("H 0\nM 0", "H 0", "first"), ("H 0", "H 0\nM 0", "second")):
        try:
            quasarq.equivalent(a, b)
            check(False, f"measurements in the {what} circuit are refused", "no error raised")
        except Exception as e:
            check("measurement" in str(e), f"measurements in the {what} circuit are refused",
                  type(e).__name__)

    same_text = quasarq.Circuit("H 0\nH 0")
    check(not (same_text == quasarq.Circuit("H 0\nH 0")),
          "== is identity, not equivalence", "equivalence lives in quasarq.equivalent")


TESTS = (
    ("interface matches stim.CompiledDetectorSampler", test_interface),
    ("bit packing matches numpy.packbits(bitorder='little')", test_bit_packing),
    ("seeding semantics match stim", test_seeding),
    ("detection events are relative to the reference sample", test_reference_frame),
    ("measurement flip probability M(p)/MX(p) is applied", test_measurement_flip_noise),
    ("observables merge by id, not by line", test_observable_merging),
    ("detector and observable rates agree with stim", test_rates_match_stim),
    ("interface matches stim.CompiledMeasurementSampler", test_measurement_interface),
    ("raw measurement outcomes agree with stim", test_measurement_rates_match_stim),
    ("measurements convert to the same detection events", test_m2d_round_trip),
    ("decoded logical error rate falls with distance (pymatching)", test_decoding),
    ("chunked sampling gives the same statistics", test_chunked_sampling),
    ("chunking does not change a single bit", test_chunk_invariance),
    ("edge cases", test_edge_cases),
    ("accepts circuit text as well as stim.Circuit", test_circuit_text),
    ("GIL is released during sampling", test_gil_released),
    ("simulation path reports measurements and final state", test_simulation),
    ("equivalence checking agrees with stim", test_equivalence),
)


def main():
    print(f"quasarq {quasarq.version()} on {quasarq.device_name()}")
    print(f"stim {stim.__version__}, numpy {np.__version__}")
    c = circuit(5, 5)
    shots = 20000
    for number, (title, test) in enumerate(TESTS, start=1):
        section(f"[{number}] {title}")
        test(c, shots)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + "; ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
