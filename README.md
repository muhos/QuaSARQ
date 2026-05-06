[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://github.com/muhos/QuaSARQ/actions/workflows/test-build.yml/badge.svg)](https://github.com/muhos/QuaSARQ/actions/workflows/test-build.yml)
# QuaSARQ
QuaSARQ stands for Quantum Simulation and Automated Reasoning. 
It is a parallel simulator of quantum stabilizer circuits capable of harnessing NVIDIA CUDA-enabled GPUs to accelerate the simulation of stabilizer gates. 

---

## Requirements
- CUDA-capable GPU with a pre-installed NVIDIA driver
- [CUDA Toolkit](https://docs.nvidia.com/cuda/) v12 or later
- [cuArena](https://github.com/muhos/cuArena)  GPU memory allocator library
- CMake 3.18 or later (to build cuArena)
- GCC/G++ with C++20 support

---

## Build

### 1. Install CUDA
For Ubuntu 24.04:<br>

`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`<br>
`sudo dpkg -i cuda-keyring_1.1-1_all.deb`<br>
`sudo apt-get update`<br>
`sudo apt-get -y install cuda-toolkit-12-8`<br>

The source code is also platform-compatible with Windows and WSL2. To install CUDA on those platforms, follow the
installation guide in https://docs.nvidia.com/cuda/.

### 2. Install QuaSARQ

- Clone the cuArena library before building QuaSARQ:

```
git clone https://github.com/muhos/cuArena.git /path/to/cuArena
```

- Build the simulator by pointing it at the cuArena directory:

```
make CUARENA_DIR=/path/to/cuArena
```

Make will build cuArena first then the `quasarq` binary and the library `libquasarq.a` will be created by default in the `build` directory.<br>

### Debug and Testing
Add `assert=1` argument with the make command to enable assertions or `debug=1` to collect debugging information.<br>

```
make CUARENA_DIR=/path/to/cuArena assert=1
```

---

## Usage

```
quasarq [<circuit.stim|circuit.qasm>] [<circuit2.stim|circuit2.qasm>] [<option> ...]
```

QuaSARQ operates in four modes depending on the arguments given.

**Single-shot simulation**: simulate a circuit from a file and report the final stabilizer state:
```
quasarq circuit.stim
quasarq circuit.qasm --verbose=2
```

**Many-shot sampling**: sample measurement outcomes over many shots using GPU-based Pauli frames:
```
quasarq circuit.stim --shots=1024
quasarq circuit.stim --shots=10000 --seed=42
```

**Equivalence checking**: verify that two circuits produce identical stabilizer evolution:
```
quasarq circuit1.stim circuit2.stim
```
Prints `EQUIVALENT` or `NOT EQUIVALENT` (with the failing initial state). Accepts `.stim` and `.qasm` files in any combination.

**Random circuit generation** — generate and simulate a random stabilizer circuit without an input file:
```
quasarq --qubits=1000 --depth=500
quasarq --qubits=5000 --depth=1000 --shots=256
```

### Key options

| Option | Description | Default |
|--------|-------------|---------|
| `--shots=<n>` | Number of measurement shots (enables sampling mode) | 0 |
| `--qubits=<n>` | Number of qubits for random generation | 1 |
| `--depth=<n>` | Circuit depth for random generation | 1 |
| `--initial=<0\|1\|2>` | Initial state: 0 = \|0⟩, 1 = \|+⟩, 2 = \|i⟩ | 0 |
| `--timeout=<s>` | Abort after this many seconds | 0 (off) |
| `--verbose=<0..3>` | Verbosity level | 1 |
| `--seed=<n>` | Random seed for sampling | 0 |
| `--write-circuit=<1\|2>` | Write generated circuit to file (1: stim, 2: chp) | 0 |
| `-report` / `-no-report` | Print statistics after simulation | on |
| `-progress` / `-no-progress` | Show per-step progress | on |
| `-q` | Quiet mode (suppress all output) | off |

For the full option list run:
```
quasarq -h
quasarq --helpmore
```

---

## Simulation Benchmarking
QuaSARQ implements two GPU-accelerated simulation modes:
- **Single-shot simulation**: applies parallel Gaussian elimination via a three-pass prefix-XOR formulation to handle projective measurements, eliminating sequential dependencies present in CPU-based approaches like Stim.
- **Many-shot sampling**: uses GPU-based Pauli frames to amortize tableau collapse costs across thousands of shots in parallel without repeated Gaussian elimination.

Benchmarks were run on an RTX 4090 (24 GB) against Stim, Qiskit-Aer (CPU/GPU), Qibo, Cirq, and PennyLane, across two suites:
- **Light suite**: 100–10,000 qubits, depths ∈ {100, 500, 1000}
- **Heavy suite**: 1,000–180,000 qubits, depths ∈ {100, 500, 1000} (~130M gates at peak)

QuaSARQ completes **177 circuits within 72 hours** on the heavy suite, vs. Stim's 125 circuits in 132 hours, with up to **105× speedup** on tableau evolution and **over 80% energy reduction** on demanding instances. For 1,024-shot sampling, QuaSARQ's Pauli-frame sampler shows flat runtime across circuit sizes while Stim's cost scales steeply.

Check our paper on [arXiv](https://arxiv.org/abs/2603.14641) for full algorithmic details.

<table>
  <tr>
    <td><img src="graphs/light_runtime.png" alt="Light suite runtime (d=100)" width="400"></td>
    <td><img src="graphs/heavy_runtime.png" alt="Heavy suite runtime vs Stim (d=100)" width="400"></td>
  </tr>
</table>

---

## Equivalence Checking
QuaSARQ supports equivalence checking of two stabilizer circuits. For example, `quasarq C1.stim C2.stim` checks if `C1 == C2`. 
The outcome will be `EQUIVALENT` or otherwise `NOT EQUIVALENT`, indicating the failing initial state.
Check our paper in [TACAS'25](https://doi.org/10.1007/978-3-031-90660-2_6) for more insights.
The following plots compares the performance of QuaSARQ against CCEC (a Stim-based checker) and Quokka-Sharp (universal circuit simulator based on model counting).
Circuits have qubits in range of 1,000 to 500,000 qubits.

<table>
  <tr>
    <td><img src="graphs/time_vs_ccec.png" alt="Runtime for QuaSARQ vs CCEC" width="400"></td>
    <td><img src="graphs/time_vs_quokka.png" alt="Runtime for QuaSARQ vs Quokka-Sharp" width="400"></td>
  </tr>
</table>
<br>
