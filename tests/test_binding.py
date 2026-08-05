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

    check(not np.array_equal(quasarq.compile_sampler(c).sample(2000),
                             quasarq.compile_sampler(c).sample(2000)),
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


TESTS = (
    ("interface matches stim.CompiledDetectorSampler", test_interface),
    ("bit packing matches numpy.packbits(bitorder='little')", test_bit_packing),
    ("seeding semantics match stim", test_seeding),
    ("detector and observable rates agree with stim", test_rates_match_stim),
    ("decoded logical error rate falls with distance (pymatching)", test_decoding),
    ("chunked sampling gives the same statistics", test_chunked_sampling),
    ("edge cases", test_edge_cases),
    ("accepts circuit text as well as stim.Circuit", test_circuit_text),
    ("GIL is released during sampling", test_gil_released),
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
