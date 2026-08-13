"""Builds the CUDA core, cuarena and the nanobind extension through the project Makefiles.
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from shutil import which

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

ROOT = Path(__file__).parent.resolve()
EXTENSION = f"_quasarq{sysconfig.get_config_var('EXT_SUFFIX')}"

CUARENA_HINT = (
    "cuarena sources were not found. Either set CUARENA_DIR=/path/to/cuarena, or vendor them "
    "at extern/cuarena (git submodule update --init --recursive)."
)


def find_cuarena():
    candidates = [os.environ.get("CUARENA_DIR"), ROOT / "extern" / "cuarena", Path.home() / "cuarena"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "CMakeLists.txt").is_file():
            return str(Path(candidate).resolve())
    raise SystemExit(CUARENA_HINT)


def require_tool(name, hint):
    if which(name) is None:
        raise SystemExit(f"'{name}' is required to build quasarq but is not on PATH. {hint}")


class BuildThroughMake(build_py):

    def run(self):
        require_tool("make", "Install build-essential.")
        require_tool("cmake", "cuarena is built with cmake.")
        cuda_home = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        if not (Path(cuda_home) / "bin" / "nvcc").is_file():
            require_tool("nvcc", f"No nvcc under {cuda_home}; set CUDA_PATH to the CUDA toolkit.")

        # 'native' keeps the install quick by targeting only this machine's GPU. Setting
        # QUASARQ_CUDA_ARCH=all makes the result portable at the cost of a much longer build.
        arch = os.environ.get("QUASARQ_CUDA_ARCH", "native")
        command = [
            "make", "-C", str(ROOT), "binding",
            f"PYTHON={sys.executable}",
            f"CUARENA_DIR={find_cuarena()}",
            f"GPU_ARCH={arch}",
            f"WORDSIZE={os.environ.get('QUASARQ_WORD_SIZE', '64')}",
        ]
        command.append(f"-j{os.environ.get('QUASARQ_BUILD_JOBS', '8')}")
        print(f"building quasarq for GPU_ARCH={arch}: {' '.join(command)}", flush=True)
        subprocess.check_call(command)
        super().run()


# The wheel carries a compiled extension, so it must not be tagged as pure python.
class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True


setup(
    cmdclass={"build_py": BuildThroughMake},
    distclass=BinaryDistribution,
    package_data={"quasarq": [EXTENSION, "kernel.config", "py.typed", "*.pyi"]},
)
