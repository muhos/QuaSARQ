"""GPU stabilizer sampling.

    import quasarq, stim

    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.001)
    dets, obs = quasarq.compile_detector_sampler(circuit, seed=1).sample(100000, separate_observables=True)

The sinter adapter lives in `quasarq.sinter` and is imported separately, as it needs sinter and pymatching installed.
"""

from ._quasarq import (
    CompiledDetectorSampler,
    CompiledMeasurementSampler,
    compile_detector_sampler,
    compile_sampler,
    device_name,
    get_chunk_shots,
    get_verbosity,
    set_chunk_shots,
    set_kernel_config,
    set_verbosity,
    version,
)

__all__ = [
    "CompiledDetectorSampler",
    "CompiledMeasurementSampler",
    "compile_detector_sampler",
    "compile_sampler",
    "device_name",
    "get_chunk_shots",
    "get_verbosity",
    "set_chunk_shots",
    "set_kernel_config",
    "set_verbosity",
    "version",
]
