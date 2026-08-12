"""Collects surface code logical error rates with sinter, sampling on the GPU with QuaSARQ.

Decoding costs roughly twice what sampling does, and pymatching holds the GIL while it runs, so
several worker processes are worth having: only separate processes decode in parallel. They can
share one GPU because the pool is sized per run from the circuit and the shot count.
"""

import time

import sinter
import stim

from quasarq import sinter as quasarq_sinter

DISTANCES = (3, 5, 7, 9, 11, 13)

def tasks():
    return [

        sinter.Task(

            circuit=stim.Circuit.generated(
                "surface_code:rotated_memory_z", distance=d, rounds=d,
                after_clifford_depolarization=0.004,
                before_measure_flip_probability=0.004,
                after_reset_flip_probability=0.004,
                before_round_data_depolarization=0.004),

            json_metadata={"d": d},
        )

        for d in DISTANCES
    ]

def run(n_workers=4, shots=400_000):
    print(f"Running {n_workers} workers, {shots} shots per distance, on distances {DISTANCES}\n")

    start = time.perf_counter()

    stats = sinter.collect(
        num_workers=n_workers,
        tasks=tasks(),
        max_shots=shots,
        max_errors=1_000_000,
        decoders=["quasarq"],
        custom_decoders={"quasarq": quasarq_sinter.QuaSARQSampler(seed=1)},
    )

    elapsed = time.perf_counter() - start
    stats = sorted(stats, key=lambda s: s.json_metadata["d"])
    rates = [s.errors / s.shots for s in stats]

    for s, rate in zip(stats, rates):
        print(f"d={s.json_metadata['d']:<3} {s.errors:>7}/{s.shots} = {rate:.5f}")

    shots = sum(s.shots for s in stats)

    print(f"\n{elapsed:.2f}s over {n_workers} workers  ({shots / elapsed / 1000:.0f}k shots/s)")

    assert all(a > b for a, b in zip(rates, rates[1:])), "the logical error rate must fall as the distance grows"

    print("")
    

if __name__ == "__main__":
    run(2)
    run(4)
    print("\nAll tests passed.")
