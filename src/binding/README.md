# QuaSARQ Python Binding

GPU stabilizer simulator. Accepts stim-format circuits directly, so a `stim.Circuit` (or circuit text) goes straight in:

```python
import quasarq, stim

circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.001)

dets, obs = quasarq.compile_detector_sampler(circuit, seed=1).sample(100_000, separate_observables=True)

measurements = quasarq.compile_sampler(circuit, seed=1).sample(100_000)
```

## Installing

```bash
pip install quasarq
```

The budled wheels are for CPython 3.10 through 3.13. The wheel contains the CUDA
runtime and device code for every major architecture from Pascal onwards, so it asks for nothing
but a driver (CUDA toolkit is not needed):

| | requirement |
|---|---|
| GPU | Pascal (sm_60) or newer, up to Blackwell |
| driver | 525 or newer |
| glibc | 2.28 or newer (RHEL 8, Ubuntu 20.04, Debian 11, …) |

GPUs newer than the compiled set are covered by PTX, which the driver compiles on first use.

### Building from source instead

pip falls back to a source build wherever no wheel matches, and drives the whole thing itself:

```bash
pip install .            # or: pip install -e .   for a development install
```

`import quasarq` then works from any directory, with no `PYTHONPATH`. The build compiles the
CUDA core, cuarena and the extension through the project Makefiles, in parallel (`-j8` by
default), taking roughly half a minute. It targets only the building machine's GPU by default —
see `QUASARQ_CUDA_ARCH` below.

`make binding` still works if you would rather build without installing; put `src/binding` on
`PYTHONPATH` in that case. The published wheels are built by
[.github/workflows/wheels.yml](../../.github/workflows/wheels.yml) inside the image described in
[.github/docker/manylinux-cuda.Dockerfile](../../.github/docker/manylinux-cuda.Dockerfile).

### What must be present at build time

- The CUDA toolkit (`nvcc`). Set `CUDA_PATH` if it is not at `/usr/local/cuda`.
- `make` and `cmake`.
- [cuarena](https://github.com/muhos/cuarena), found in this order: `$CUARENA_DIR`, then
  `extern/cuarena`, then `~/cuarena`. Fetch the bundled copy with

      git submodule update --init --recursive

### Build options

| variable | default | meaning |
|---|---|---|
| `QUASARQ_CUDA_ARCH` | `native` | GPU target. `native` builds only for the machine's own GPU. Use `all-major` for a binary that runs on every architecture this nvcc supports, or a comma-separated list ordered lowest to highest, such as `sm_80,sm_90` — the last entry also gets PTX, so newer GPUs still run. |
| `QUASARQ_BUILD_JOBS` | `8` | parallel compile jobs |
| `QUASARQ_WORD_SIZE` | `64` | tableau word size |
| `CUARENA_DIR` | &ndash; | explicit path to cuarena |

The default `-arch=native` makes the result **not portable to another GPU architecture**. Build
a redistributable binary with:

```bash
QUASARQ_CUDA_ARCH=all pip wheel .
```

That takes considerably longer, since every kernel is compiled for every architecture.

## GPU memory

The pool is sized per run from the circuit and the shot count, which keeps a run from locking
the whole device against other processes. Override it if you need to:

```python
quasarq.set_max_device_memory("auto")   # default: size it from the circuit
quasarq.set_max_device_memory(512)      # fixed cap in MB
quasarq.set_max_device_memory(0)        # take whatever is free
```

Results never depend on the cap: a smaller pool only changes how many shots are simulated per
chunk. `auto` sizes for the whole request so it does not split shots, growing the pool if a later
request needs more.

## Contents

- `quasarq.compile_detector_sampler(circuit, *, seed=None)` → `CompiledDetectorSampler`
- `quasarq.compile_sampler(circuit, *, seed=None)` → `CompiledMeasurementSampler`
- `quasarq.sinter` — a `sinter.Sampler` adapter that decodes with pymatching
- `set_verbosity`, `set_chunk_shots`, `set_kernel_config`, `device_name`, `version`

`kernel.config` holds per-size kernel launch geometry and is copied next to the extension at
build time; the core locates it relative to the shared object.
