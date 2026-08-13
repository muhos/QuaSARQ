# Build image for redistributable quasarq wheels: a manylinux base plus a minimal CUDA toolkit.
#
#   docker build -t quasarq-manylinux-cuda:12.9 -f .github/docker/manylinux-cuda.Dockerfile .
#
# CUDA 13 builds too via:
#
#   docker build -t quasarq-manylinux-cuda:13.3 -f .github/docker/manylinux-cuda.Dockerfile \
#       --build-arg CUDA_VERSION=13-3 --build-arg CUDA_DISTRO=rhel9 \
#       --build-arg MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_34_x86_64:latest .

ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64:latest
FROM ${MANYLINUX_IMAGE}

# Only build what we need: the compiler, cudart (static, so the wheel carries its own runtime), 
# curand and cccl headers for the device code, and the NVML stub to link against on a machine with no driver.
ARG CUDA_VERSION=12-9
ARG CUDA_DISTRO=rhel8
RUN dnf -y install 'dnf-command(config-manager)' \
 && dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-${CUDA_DISTRO}.repo \
 && dnf -y install --setopt=install_weak_deps=False \
      cuda-nvcc-${CUDA_VERSION} \
      cuda-cudart-devel-${CUDA_VERSION} \
      cuda-nvml-devel-${CUDA_VERSION} \
      libcurand-devel-${CUDA_VERSION} \
      cuda-cccl-${CUDA_VERSION} \
      cmake \
 && dnf clean all \
 && rm -rf /var/cache/dnf

ENV CUDA_PATH=/usr/local/cuda
RUN test -d "$CUDA_PATH" || ln -s /usr/local/cuda-$(echo ${CUDA_VERSION} | tr '-' '.') "$CUDA_PATH"
RUN "$CUDA_PATH"/bin/nvcc --version && cmake --version | head -1
