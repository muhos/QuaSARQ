# QuaSARQ Python Binding

GPU stabilizer sampling. Accepts stim-format circuits directly, so a `stim.Circuit` (or circuit text) goes straight in:

```python
import quasarq, stim

circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.001)

dets, obs = quasarq.compile_detector_sampler(circuit, seed=1).sample(100_000, separate_observables=True)

measurements = quasarq.compile_sampler(circuit, seed=1).sample(100_000)
```

## Installing

pip drives the whole build, so this is all it takes from the repository root:

```bash
pip install .            # or: pip install -e .   for a development install
```

`import quasarq` then works from any directory, with no `PYTHONPATH`. The build compiles the
CUDA core, cuarena and the extension through the project Makefiles, in parallel (`-j8` by
default), taking roughly half a minute.

`make binding` still works if you would rather build without installing; put `src/binding` on
`PYTHONPATH` in that case.

### What must be present at build time

- The CUDA toolkit (`nvcc`). Set `CUDA_PATH` if it is not at `/usr/local/cuda`.
- `make` and `cmake`.
- [cuarena](https://github.com/muhos/cuarena), found in this order: `$CUARENA_DIR`, then
  `extern/cuarena`, then `~/cuarena`. Fetch the bundled copy with

      git submodule update --init --recursive

### Build options

| variable | default | meaning |
|---|---|---|
| `QUASARQ_CUDA_ARCH` | `native` | GPU target. `native` builds only for the machine's own GPU. Use `all` for a binary that runs on every architecture this nvcc supports, or a list such as `sm_80,sm_89`. |
| `QUASARQ_BUILD_JOBS` | `8` | parallel compile jobs |
| `QUASARQ_WORD_SIZE` | `64` | tableau word size |
| `CUARENA_DIR` | &ndash; | explicit path to cuarena |

The default `-arch=native` makes the result **not portable to another GPU architecture**. Build
a redistributable binary with:

```bash
QUASARQ_CUDA_ARCH=all pip wheel .
```

That takes considerably longer, since every kernel is compiled for every architecture.

## Contents

- `quasarq.compile_detector_sampler(circuit, *, seed=None)` → `CompiledDetectorSampler`
- `quasarq.compile_sampler(circuit, *, seed=None)` → `CompiledMeasurementSampler`
- `quasarq.sinter` — a `sinter.Sampler` adapter that decodes with pymatching
- `set_verbosity`, `set_chunk_shots`, `set_kernel_config`, `device_name`, `version`

`kernel.config` holds per-size kernel launch geometry and is copied next to the extension at
build time; the core locates it relative to the shared object.
