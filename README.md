[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://github.com/muhos/QuaSARQ/actions/workflows/test-build.yml/badge.svg)](https://github.com/muhos/QuaSARQ/actions/workflows/test-build.yml)
# QuaSARQ
QuaSARQ stands for Quantum Simulation and Automated Reasoning. 
It is a parallel simulator of quantum stabilizer circuits capable of harnessing NVIDIA CUDA-enabled GPUs to accelerate the simulation of stabilizer gates. 

# Build
To build the simulator, make sure you have a CUDA-capable GPU with pre-installed NVIDIA driver and CUDA toolkit.

For installing CUDA v12, run the following commands on Ubuntu 24.04:<br>

`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`<br>
`sudo dpkg -i cuda-keyring_1.1-1_all.deb`<br>
`sudo apt-get update`<br>
`sudo apt-get -y install cuda-toolkit-12-8`<br>

The source code is also platform-compatible with Windows and WSL2. To install CUDA on those platforms, follow the
installation guide in https://docs.nvidia.com/cuda/.

Now, the simulator can be built via the command `make`.<br>
The `quasarq` binary and the library `libquasarq.a` will be created by default in the build directory.<br>

## Debug and Testing
Add `assert=1` argument with the make command to enable assertions or `debug=1` to collect debugging information.<br>

# Usage
The simulator can be used via the command `quasarq [<circuit>.<stim>/<qasm>][<option> ...]`.<br>
For more options, type `quasarq -h` or `quasarq --helpmore`.

# Equivalence Checking
QuaSARQ supports equivalence checking of two stabilizer circuits. For example, `quasarq C1.stim C2.stim` checks if `C1 == C2`. 
The outcome will be `EQUIVALENT` or otherwise `NOT EQUIVALENT`, indicating the failing initial state.<br>
The following plots compares the performance of QuaSARQ against CCEC (a Stim-based checker) and Quokka-Sharp (universal circuit simulator based on model counting)

<table>
  <tr>
    <td><img src="graphs/time_vs_ccec.pdf" alt="Runtime for QuaSARQ vs CCEC" width="300"></td>
    <td><img src="graphs/time_vs_quokka.pdf" alt="Runtime for QuaSARQ vs Quokka-Sharp" width="300"></td>
  </tr>
</table>
