# QuaSARQ command-line interface

The `quasarq` executable exposes everything the library does: single-shot simulation, many-shot
sampling, equivalence checking, and random circuit generation. This page is the full reference.
For the Python package see the [main README](../README.md) and
[src/binding/README.md](../src/binding/README.md).

## Requirements

The Python package needs none of these. They apply to the executable and the static library.

- NVIDIA GPU, Pascal (sm_60) or newer.
- CUDA Toolkit 12 or later.
- GCC/G++ with C++20 support.
- CMake 3.18 or later, used to build cuArena.
- [cuArena](https://github.com/muhos/cuArena), the GPU memory allocator used by QuaSARQ.

## Build

Install CUDA using the official NVIDIA instructions for your platform. On Ubuntu 24.04, one
possible setup is:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```

Clone cuArena, then build QuaSARQ:

```bash
git clone https://github.com/muhos/cuArena.git /path/to/cuArena
make CUARENA_DIR=/path/to/cuArena
```

The build creates:

- `build/quasarq`: the command-line executable.
- `build/libquasarq.a`: the static library.

Useful build variants:

```bash
make CUARENA_DIR=/path/to/cuArena assert=1
make CUARENA_DIR=/path/to/cuArena debug=1
make CUARENA_DIR=/path/to/cuArena word=32
make CUARENA_DIR=/path/to/cuArena nocolor=1
```

Run the test suite with:

```bash
make test
```

## Usage

```text
quasarq [<circuit.stim>] [<circuit2.stim>] [<option> ...]
```

The mode follows from the inputs:

| Input | Mode |
| --- | --- |
| One circuit file | Single-shot simulation |
| One circuit file and `--shots=<n>` | Many-shot sampling |
| Two circuit files | Equivalence checking |
| `--qubits=<n> --depth=<d>` | Random circuit generation |

Run a circuit:

```bash
./build/quasarq examples/deterministic_measurement.stim
```

Sample measurements:

```bash
./build/quasarq examples/deterministic_measurement.stim --shots=1024 --seed=1
```

Check two equivalent circuits:

```bash
./build/quasarq examples/equiv_a.stim examples/equiv_b.stim
```

```text
Circuits check                 : EQUIVALENT
```

Check two different circuits:

```bash
./build/quasarq examples/equiv_a.stim examples/not_equiv.stim
```

```text
Circuits check                 : NOT EQUIVALENT
Failed state                   : ...
```

Generate and simulate a random stabilizer circuit:

```bash
./build/quasarq --qubits=1000 --depth=500
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--shots=<n>` | Number of shots, enabling sampling mode when nonzero | `0` |
| `--seed=<n>` | Random seed for sampling | `0` |
| `--qubits=<n>` | Number of qubits for random generation | `1` |
| `--depth=<n>` | Circuit depth for random generation | `1` |
| `--initial=<0\|1\|2>` | Initial state: `0` = zero, `1` = plus, `2` = i | `0` |
| `--max-gpu-memory=<MB>` | Cap the GPU pool; `0` takes what is free | `0` |
| `--chunk-shots=<n>` | Shots simulated per chunk; `0` sizes it automatically | `0` |
| `--verbose=<0..3>` | Verbosity level | `1` |
| `--timeout=<s>` | Abort after this many seconds; `0` disables timeout | `0` |
| `--streams=<n>` | Number of GPU streams (4 to 32) | `6` |
| `--write-circuit=<0\|1\|2>` | Write generated circuits: `1` = Stim, `2` = CHP | `0` |
| `-ignore-ticks` | Ignore Stim `TICK` scheduling barriers | off |
| `--config-path=<path>` | Kernel configuration file | `src/kernel.config` |
| `--state-path=<path>` | Final-state output file | `build/<circuit>_paulis.qstate` |
| `--print-limit=<n>` | Size limit for writing the quantum state to the state file | `1000` |
| `--min-shots-write=<n>` | Write sampling output to file above this many shots | `100` |
| `--min-measures-write=<n>` | Write measurement output to file above this many measurements | `100` |
| `-report` / `-no-report` | Print final statistics | on |
| `-progress` / `-no-progress` | Print progress tables | on |
| `-q` | Quiet mode | off |
| `-force-report` | Print statistics even in quiet mode | off |

Random generation draws each gate from a tunable mix: every supported instruction has a
`--<NAME>=<probability>` option, such as `--CX=0.05`, `--H=0.1`, or `--DEPOLARIZE1=0.001`. The
defaults spread the mass evenly across the whole set. `--helpmore` lists them all.

Diagnostics:

| Option | Description |
| --- | --- |
| `-check-all` | Enable all available internal checkers |
| `-check-scheduler` | Check scheduled windows for duplicate gate inputs |
| `-check-tableau` | Check tableau operations |
| `-check-measurement` | Check measurement records, detectors, and observables where applicable |
| `-check-transpose` | Check the parallel transpose procedure |
| `-check-identity` | Check the parallel identity procedure |
| `-print-observable` | Print or write observable bitstrings |
| `-print-detector` | Print or write detection events |
| `-print-sample` | Print or write sampling outcomes, one shot per line |
| `-print-sample_qubits` | Print or write sampling outcomes, one qubit per line |
| `-print-record` | Print the measurement record |
| `-print-finalstate` | Write the final state as Pauli strings |
| `-color-results` | Color printed detector/observable bitstrings |
| `-profile` | Profile simulator components and report their share of the runtime |

For the complete option list:

```bash
./build/quasarq --help
./build/quasarq --helpmore
```

## GPU memory

By default the pool is sized per run from the circuit and the shot count, so a run does not lock
the whole device against other processes. `--max-gpu-memory=<MB>` caps it explicitly, which is
what lets several QuaSARQ jobs share one card. Results never depend on the cap: a smaller pool
only changes how many shots are simulated per chunk.

## Simulation

QuaSARQ simulates stabilizer circuits by applying Clifford gates directly to a GPU-resident
tableau. Projective measurements are handled with parallel pivot search and prefix-XOR based
updates, avoiding the sequential bottleneck that often dominates large tableaux.

For detector/observable checks on a surface-code circuit:

```bash
./build/quasarq tests/circuits/surface_code_d10_r3.stim -print-observable -print-detector --verbose=1
```

For quiet regression-style runs that still write detector and observable data when triggered:

```bash
./build/quasarq tests/circuits/surface_code_d10_r3.stim -print-observable -print-detector -q
```

## Sampling

Sampling mode uses GPU Pauli frames to amortize tableau collapse costs across many shots:

```bash
./build/quasarq tests/circuits/surface_code_d50_r10.stim --shots=4096 --seed=47
```

This path is intended for circuits with measurements. When observables are present, QuaSARQ
reports the raw logical-observable error rate across shots. Use `-print-sample`,
`-print-detector`, or `-print-observable` to inspect generated bitstrings, and
`-check-measurement` to validate measurement records in supported sampling paths.

## Equivalence checking

With two input circuits, QuaSARQ checks stabilizer-circuit equivalence:

```bash
./build/quasarq examples/equiv_a.stim examples/equiv_b.stim
./build/quasarq examples/equiv_a.stim examples/not_equiv.stim
```

For progress output:

```bash
./build/quasarq circuit_a.stim circuit_b.stim --verbose=1
```

Equivalence checking ignores operations that are outside the deterministic Clifford evolution
being compared, such as noise and measurement/reset operations. It reports whether the circuits
are equivalent, and if not, the initial state that exposed the mismatch.

## Kernel configuration

`src/kernel.config` holds per-size launch grid for each kernel, picked by qubit count. The
`-tune-*` options re-measure a kernel on the current GPU and are how that file was produced;
`--tune-initial-qubits` and `--tune-step-qubits` set the sweep range. Point the tool at a
different file with `--config-path`.
