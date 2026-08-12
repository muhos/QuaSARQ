from typing import Any, overload

import numpy as np
import numpy.typing as npt

__all__ = [
    "CompiledDetectorSampler",
    "CompiledMeasurementSampler",
    "compile_detector_sampler",
    "compile_sampler",
    "device_name",
    "get_chunk_shots",
    "get_max_device_memory",
    "get_verbosity",
    "set_chunk_shots",
    "set_kernel_config",
    "set_max_device_memory",
    "set_verbosity",
    "version",
]

class CompiledDetectorSampler:
    """Samples detection events."""

    @property
    def num_detectors(self) -> int: ...

    @property
    def num_observables(self) -> int: ...

    @property
    def holds_device_memory(self) -> bool: ...
    def release(self) -> None:
        """Free the GPU pool this sampler is holding. The next sample() rebuilds it."""

    @overload
    def sample(self, shots: int, *,
        separate_observables: bool = False,
        bit_packed: bool = False,
        append_observables: bool = False,
        prepend_observables: bool = False,
    ) -> npt.NDArray[Any]: ...

    @overload
    def sample(self, shots: int, *,
        separate_observables: bool,
        bit_packed: bool = False,
        append_observables: bool = False,
        prepend_observables: bool = False,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...

class CompiledMeasurementSampler:
    """Samples raw measurement outcomes."""

    @property
    def num_measurements(self) -> int: ...

    @property
    def holds_device_memory(self) -> bool: ...
    def release(self) -> None:
        """Free the GPU pool this sampler is holding. The next sample() rebuilds it."""

    def sample(self, shots: int, *, bit_packed: bool = False) -> npt.NDArray[Any]: ...

def compile_detector_sampler(circuit: Any, *, seed: int | None = None) -> CompiledDetectorSampler:
    """Compile a detection-event sampler for a stim.Circuit or circuit text."""

def compile_sampler(circuit: Any, *, seed: int | None = None) -> CompiledMeasurementSampler:
    """Compile a measurement sampler for a stim.Circuit or circuit text."""

def set_verbosity(level: int) -> None: ...
def get_verbosity() -> int: ...
def set_chunk_shots(shots: int) -> None: ...
def get_chunk_shots() -> int: ...
def set_max_device_memory(megabytes: int) -> None: ...
def get_max_device_memory() -> int: ...
def set_kernel_config(path: str) -> None: ...
def version() -> str: ...
def device_name() -> str: ...
