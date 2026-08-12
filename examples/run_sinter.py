"""Collects surface code logical error rates with sinter, sampling on the GPU with QuaSARQ."""

import time

import sinter
import stim

from quasarq import sinter as quasarq_sinter


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

        for d in (3, 5)
    ]


if __name__ == "__main__":
    start = time.perf_counter()

    stats = sinter.collect(
        # `num_workers` should stay at 1: QuaSARQ serialises sampling on one GPU, and each
        # extra worker is a separate process competing for the same device memory.
        num_workers=1,
        tasks=tasks(),
        max_shots=400_000,
        max_errors=1_000_000,
        decoders=["quasarq"],
        custom_decoders={"quasarq": quasarq_sinter.QuaSARQSampler(seed=1)},
    )

    elapsed = time.perf_counter() - start
    stats = sorted(stats, key=lambda s: s.json_metadata["d"])
    rates = [s.errors / s.shots for s in stats]

    for s, rate in zip(stats, rates):
        print(f"d={s.json_metadata['d']}  {s.errors}/{s.shots} = {rate:.5f}")
    print(f"\n{elapsed:.2f}s  ({sum(s.shots for s in stats) / elapsed / 1000:.0f}k shots/s)")

    assert rates[0] > rates[1], "error rate must fall with distance"
    print("All tests passed.")
