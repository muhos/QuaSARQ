#############################################
# Muhammad Osama Mahmoud, 2025.
# LIACS, Leiden University, The Netherlands.
#############################################

# advance progress

ifndef PROGRESS
# MAKEFLAGS is cleared for this counting pass: inherited flags would make it try to join the
# parent's jobserver, which it cannot reach from $(shell), and warn on every invocation.
T := $(shell MAKEFLAGS= $(MAKE) $(MAKECMDGOALS) $(MAKEOVERRIDES) --no-print-directory -nrRf $(firstword $(MAKEFILE_LIST)) PROGRESS="COUNTME" | grep -c "COUNTME")
N := x
C = $(words $N)$(eval N := x $N)
# guard the divisor: a zero count must not turn every recipe into a shell error
PERC = `expr $C '*' 100 / $(if $(filter-out 0,$(T)),$(T),1)`
PROGRESS = printf " [ %3d%% ] compiling: %-20s\r" $(PERC)
ARCHIVE  = printf " [ 100%% ] compiling: %-30s\r\n building archive ( %-15s )..."
ENDING   = printf " building binary  ( %-15s )..."
INSTALL  = printf " installing       ( %-15s )..."
DONE     = printf "%-30s\n" " done."
endif

ifeq ($(MAKELEVEL),0)
ifeq ($(filter -j%,$(MAKEFLAGS)),)
MAKEFLAGS += -j8
endif
endif

# version
# The VERSION file is the single source: pyproject.toml reads it too.

VERSION := $(strip $(shell cat $(dir $(firstword $(MAKEFILE_LIST)))VERSION 2>/dev/null))
ifeq ($(VERSION),)
$(error cannot read the VERSION file)
endif

# CUDA path

CUDA_PATH ?= /usr/local/cuda

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(CUDA_PATH),NONE)
$(error cannot find CUDA local directory)
endif
endif

# device/host compilers (nvcc is the master)

CXX  := g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

# compiler flags

RELOC	  := -dc
CCFLAGS   := -std=c++20 -fdiagnostics-show-option
NVCCFLAGS := -m64 -std=c++20

# Common includes
INCLUDES  := -I../common/inc
EXTRALIB  :=

# cuarena library
CUARENA_DIR ?= $(HOME)/cuarena
CUARENA_LIB := $(CUARENA_DIR)/build/libcuarena.a
INCLUDES    += -I$(CUARENA_DIR)/include
EXTRALIB    += -L$(CUARENA_DIR)/build -lcuarena

# generated binaries

BIN     := quasarq
LIBNAME := quasarq
LIB     := lib$(LIBNAME).a

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(CCFLAGS),NONE)
$(error no host compiler flags have been specified)
endif
ifeq ($(NVCCFLAGS),NONE)
$(error no device compiler flags have been specified)
endif
ifeq ($(BIN),NONE)
$(error no binary name is given)
else ifeq ($(LIB),NONE)
$(error no library name is given)
endif
endif

# debug build flags
ifeq ($(debug),1)
      NVCCFLAGS         += -g -G -DDEBUG
      CUARENA_BUILD_TYPE := Debug
else  ifeq ($(assert),1)
      NVCCFLAGS         += -O3
      CUARENA_BUILD_TYPE := Release
else
      NVCCFLAGS         += -O3 -DNDEBUG -diag-suppress 68 -diag-suppress 186 -diag-suppress 20091 -diag-suppress 20011
      CUARENA_BUILD_TYPE := Release
endif

# no colors
ifeq ($(nocolor),1)
      NVCCFLAGS += -DNCOLORS
endif

# word sizes
WORDSIZE := 64
ifeq ($(word),8)
      WORDSIZE := 8
else  ifeq ($(word),32)
      WORDSIZE := 32
else  ifeq ($(word),64)
      WORDSIZE := 64
endif

NVCCFLAGS += -DWORD_SIZE_$(WORDSIZE) -DVERSION=\"$(VERSION)\"

# position-independent build, required to link the archive into a shared
# object such as the Python extension module. Objects and archive get their
# own names so a pic build and a normal build can coexist without a clean.
ifeq ($(pic),1)
      NVCCFLAGS += -Xcompiler -fPIC
      OBJINFIX  := pic.
      LIBNAME   := quasarq_pic
      LIB       := lib$(LIBNAME).a
endif

# combine all flags
ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# gencode arguments
include $(dir $(firstword $(MAKEFILE_LIST)))arch.mk

CUARENA_ARCH_STAMP := $(CUARENA_DIR)/build/.quasarq-arch

ifneq ($(MAKECMDGOALS),clean)
cuarena_cleaning := $(shell [ -f $(CUARENA_LIB) ] && \
    [ "$$(cat $(CUARENA_ARCH_STAMP) 2>/dev/null)" != "$(CUARENA_ARCH)" ] && rm -f $(CUARENA_LIB))
endif

# target rules

SRC_DIR   := src
BUILD_DIR := build
CPPOBJEXT := $(OBJINFIX)o
CUOBJEXT  := $(OBJINFIX)cuda.o
PTX_DIR   := ptx
PTXEXT    := ptx
ARCH_STAMP := $(BUILD_DIR)/.gpu-arch

ifneq ($(MAKECMDGOALS),clean)
RECORDED_ARCH := $(strip $(shell cat $(ARCH_STAMP) 2>/dev/null))
ifneq ($(RECORDED_ARCH),)
ifneq ($(RECORDED_ARCH),$(GPU_ARCH))
$(error $(BUILD_DIR) holds objects compiled for GPU_ARCH=$(RECORDED_ARCH). Run 'make clean' before building for $(GPU_ARCH))
endif
endif
endif

mainsrc  := main
cusrc    := $(sort $(wildcard $(SRC_DIR)/*.cu))
allcppsrc:= $(sort $(wildcard $(SRC_DIR)/*.cpp))
headers  := $(sort $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.hpp))
cppmain  := $(SRC_DIR)/$(mainsrc).cpp
cppsrc   := $(filter-out $(cppmain),$(allcppsrc))
mainobj  := $(SRC_DIR)/$(mainsrc).$(CPPOBJEXT)
cuobj    := $(patsubst $(SRC_DIR)/%.cu,$(SRC_DIR)/%.$(CUOBJEXT),$(cusrc))
cppobj   := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%.$(CPPOBJEXT),$(cppsrc))
ptxfiles := $(patsubst $(SRC_DIR)/%.cu,$(PTX_DIR)/%.$(PTXEXT),$(cusrc))

ifneq ($(MAKECMDGOALS),clean)
	ifeq ($(cusrc),)
		$(error no CUDA source files exist)
	endif
	ifeq ($(cppsrc),)
		$(error no C++ source files exist)
	endif
	ifeq ($(cuobj),)
		$(error no CUDA object files to generate)
	endif
	ifeq ($(cppobj),)
		$(error no C++ object files to generate)
	endif
endif

ifeq ($(ptx),1)
all: $(ptxfiles)
else
all: $(BUILD_DIR)/$(BIN)
endif


$(CUARENA_LIB):
	@[ -f $(CUARENA_DIR)/build/CMakeCache.txt ] && \
	    [ "$$(cat $(CUARENA_ARCH_STAMP) 2>/dev/null)" = "$(CUARENA_ARCH)" ] || \
	    cmake -B $(CUARENA_DIR)/build -S $(CUARENA_DIR) \
	    -DCMAKE_BUILD_TYPE=$(CUARENA_BUILD_TYPE) \
	    -DCMAKE_CUDA_ARCHITECTURES="$(CUARENA_ARCH)" \
	    -DCUARENA_BUILD_EXAMPLES=OFF \
	    -DCUARENA_BUILD_TESTS=OFF \
	    > /dev/null
	@MAKEFLAGS= cmake --build $(CUARENA_DIR)/build --target cuArena --parallel 8 > /dev/null
	@echo "$(CUARENA_ARCH)" > $(CUARENA_ARCH_STAMP)

$(BUILD_DIR)/$(LIB): $(cuobj) $(cppobj)
	@$(ARCHIVE) "done" $@
	@mkdir -p $(BUILD_DIR)
	@ar rc $@ $+
	@ranlib $@
	@echo "$(GPU_ARCH)" > $(ARCH_STAMP)
	@$(DONE)

$(BUILD_DIR)/$(BIN): $(mainobj) $(BUILD_DIR)/$(LIB) $(CUARENA_LIB)
	@$(ENDING) $@
	@$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $(mainobj) \
	    -L$(CUDA_PATH)/lib64 -lcudart -ldl \
	    -L$(BUILD_DIR) -l$(LIBNAME) $(EXTRALIB)
	@$(DONE)

$(mainobj): $(cppmain) $(headers)
	@$(PROGRESS) $<
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(SRC_DIR)/%.$(CUOBJEXT): $(SRC_DIR)/%.cu $(headers)
	@$(PROGRESS) $<
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(RELOC) -o $@ -c $<

$(SRC_DIR)/%.$(CPPOBJEXT): $(SRC_DIR)/%.cpp $(headers)
	@$(PROGRESS) $<
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(PTX_DIR):
	@mkdir -p $@

$(PTX_DIR)/%.$(PTXEXT): $(SRC_DIR)/%.cu | $(PTX_DIR)
	@$(PROGRESS) $<
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) --ptx -o $@ $<

test: $(BUILD_DIR)/$(BIN)
	@$(MAKE) -C tests run

PYTHON       ?= python3
BINDING_DIR  := $(SRC_DIR)/binding
BINDING_ARGS := CUARENA_DIR=$(CUARENA_DIR) PYTHON=$(PYTHON) WORDSIZE=$(WORDSIZE) GPU_ARCH=$(GPU_ARCH)

pic-archive:
	@$(MAKE) pic=1 $(BUILD_DIR)/libquasarq_pic.a $(CUARENA_LIB) CUARENA_DIR=$(CUARENA_DIR) word=$(WORDSIZE) GPU_ARCH=$(GPU_ARCH)

binding: pic-archive
	@$(MAKE) -C $(BINDING_DIR) $(BINDING_ARGS)

binding-test: pic-archive
	@$(MAKE) -C $(BINDING_DIR) test $(BINDING_ARGS)

clean:
	rm -f $(SRC_DIR)/*.o
	rm -rf $(BUILD_DIR) $(PTX_DIR)
	@echo -n "cleaning up cuarena... "
	@MAKEFLAGS= cmake --build $(CUARENA_DIR)/build --target clean -- --no-print-directory
	@echo "done"

.PHONY: all test clean binding binding-test pic-archive
