CUDA_PATH ?= /usr/local/cuda
CXX       := g++
NVCC      := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
PYTHON    ?= python3

QUASARQ_DIR := ../..
CUARENA_DIR ?= $(HOME)/cuarena
CUARENA_LIB := $(CUARENA_DIR)/build/libcuarena.a
QUASARQ_LIB := $(QUASARQ_DIR)/build/libquasarq_pic.a

PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
NB_DIR     := $(shell $(PYTHON) -c "import nanobind, os; print(os.path.dirname(nanobind.__file__))" 2>/dev/null)

ifeq ($(filter clean,$(MAKECMDGOALS)),)
ifeq ($(NB_DIR),)
      $(error nanobind not found for $(PYTHON). Install it with: $(PYTHON) -m pip install nanobind)
endif
endif

INCLUDES  := -I. -I$(QUASARQ_DIR)/src -I$(CUARENA_DIR)/include -I$(CUDA_PATH)/include \
             -I$(PY_INCLUDE) -I$(NB_DIR)/include -I$(NB_DIR)/ext/robin_map/include
LIBS      := -L$(QUASARQ_DIR)/build -lquasarq_pic -L$(CUARENA_DIR)/build -lcuarena \
             -L$(CUDA_PATH)/lib64 -lcudart -lcurand -lnvidia-ml -ldl

CCFLAGS   := -std=c++20 -fPIC -fvisibility=hidden -fno-strict-aliasing -fdiagnostics-show-option
NVCCFLAGS := -m64 -std=c++20 -arch=native

ifeq ($(debug),1)
      NVCCFLAGS += -g -G -DDEBUG
      CCFLAGS   += -g -DDEBUG
else  ifeq ($(assert),1)
      NVCCFLAGS += -O3
      CCFLAGS   += -O3
else
      NVCCFLAGS += -O3 -DNDEBUG -diag-suppress 68 -diag-suppress 186 -diag-suppress 20091 -diag-suppress 20011
      CCFLAGS   += -O3 -DNDEBUG
endif

WORDSIZE ?= 64
ifeq ($(word),8)
      WORDSIZE := 8
else  ifeq ($(word),32)
      WORDSIZE := 32
else  ifeq ($(word),64)
      WORDSIZE := 64
endif

NVCCFLAGS += -DWORD_SIZE_$(WORDSIZE) -Xcompiler -fPIC -Xcompiler -fvisibility=hidden
CCFLAGS   += -DWORD_SIZE_$(WORDSIZE)
CUDA_FLAGS := $(NVCCFLAGS)

MODULE    := quasarq$(EXT_SUFFIX)
CUDA_OBJS := sampler.pic.cuda.o
HOST_OBJS := module.pic.o nb_combined.pic.o
OBJS      := $(CUDA_OBJS) $(HOST_OBJS)
HEADERS   := sampler.hpp module.hpp $(wildcard $(QUASARQ_DIR)/src/*.hpp) $(wildcard $(QUASARQ_DIR)/src/*.cuh)

all: $(MODULE)

sampler.pic.cuda.o: sampler.cu $(HEADERS)
	@echo -n "compiling $<..."
	@$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ -c $<
	@echo "done."

module.pic.o: module.cpp module.hpp sampler.hpp
	@echo -n "compiling $<..."
	@$(CXX) $(CCFLAGS) $(INCLUDES) -o $@ -c $<
	@echo "done."

nb_combined.pic.o: $(NB_DIR)/src/nb_combined.cpp
	@echo -n "compiling nanobind..."
	@$(CXX) $(CCFLAGS) $(INCLUDES) -o $@ -c $<
	@echo "done."

$(MODULE): $(OBJS) $(QUASARQ_LIB) $(CUARENA_LIB)
	@echo -n "linking $@..."
	@$(NVCC) $(CUDA_FLAGS) -shared -cudart=shared -o $@ $(OBJS) $(LIBS)
	@echo "done."

$(QUASARQ_LIB):
	@$(MAKE) -C $(QUASARQ_DIR) pic=1 CUARENA_DIR=$(CUARENA_DIR) word=$(WORDSIZE)

$(CUARENA_LIB):
	@$(MAKE) -C $(QUASARQ_DIR) $(CUARENA_LIB) CUARENA_DIR=$(CUARENA_DIR)

test: $(MODULE)
	@PYTHONPATH=$(CURDIR) $(PYTHON) $(QUASARQ_DIR)/tests/test_binding.py

clean:
	rm -f $(OBJS) $(MODULE)

.PHONY: all test clean
