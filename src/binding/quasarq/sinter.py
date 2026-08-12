"""sinter.Sampler adapter that samples detection events on the GPU with QuaSARQ.

    import sinter
    from quasarq import sinter as quasarq_sinter

    # The `__main__` guard is essential to prevent sinter from recursively starting workers
    if __name__ == '__main__':
        stats = sinter.collect(
            # Without `max_device_memory` the first worker reserves the whole device and the
            # rest fail, so `num_workers` must stay at 1. With a cap, small circuits fit
            # several workers on one GPU, which is worth having: decoding costs about twice
            # what sampling does and pymatching holds the GIL, so only separate processes
            # decode in parallel.
            num_workers=8,
            tasks=[sinter.Task(circuit=circuit, json_metadata={"d": 5})],
            max_shots=1_000_000,
            max_errors=1000,
            decoders=["quasarq"],
            custom_decoders={"quasarq": quasarq_sinter.QuaSARQSampler(seed=1, max_device_memory=2048)},
        )
"""

import numpy as np
import sinter


DEFAULT_MIN_BATCH_SHOTS = 8192


class QuaSARQCompiledSampler(sinter.CompiledSampler):

    def __init__(self, task, seed, decoder, min_batch_shots=DEFAULT_MIN_BATCH_SHOTS, max_device_memory=0):

        import pymatching
        import quasarq

        if max_device_memory:
            quasarq.set_max_device_memory(
                max_device_memory if isinstance(max_device_memory, str) else int(max_device_memory)
            )

        if decoder not in (None, "pymatching"):
            raise ValueError(f"only pymatching is supported, got {decoder!r}")
        dem = task.detector_error_model
        if dem is None:
            dem = task.circuit.detector_error_model(decompose_errors=True)
        self.matcher = pymatching.Matching.from_detector_error_model(dem)
        self.sampler = quasarq.compile_detector_sampler(task.circuit, seed=seed)
        self.num_observables = self.sampler.num_observables
        self.min_batch_shots = max(int(min_batch_shots or 0), 1)

    def sample(self, suggested_shots):
        shots = max(int(suggested_shots), self.min_batch_shots)

        dets, obs = self.sampler.sample(shots, separate_observables=True, bit_packed=True)

        if self.num_observables == 0:
            return sinter.AnonTaskStats(shots=shots, errors=0)
        
        predicted = self.matcher.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True)

        errors = int(np.count_nonzero(np.any(predicted != obs, axis=1)))
        return sinter.AnonTaskStats(shots=shots, errors=errors)

    def handles_throttling(self):
        return True


class QuaSARQSampler(sinter.Sampler):

    def __init__(self, *, seed=None, decoder="pymatching", min_batch_shots=DEFAULT_MIN_BATCH_SHOTS, max_device_memory=0):
        self.seed = seed
        self.decoder = decoder
        self.min_batch_shots = min_batch_shots
        self.max_device_memory = max_device_memory

    def compiled_sampler_for_task(self, task):
        return QuaSARQCompiledSampler(task, self.seed, self.decoder, self.min_batch_shots, self.max_device_memory)
