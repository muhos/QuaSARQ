#############################################
# Muhammad Osama Mahmoud, 2025.
# LIACS, Leiden University, The Netherlands.
#############################################

# advance progress

ifndef PROGRESS
T := $(shell $(MAKE) $(MAKECMDGOALS) --no-print-directory -nrRf $(firstword $(MAKEFILE_LIST)) PROGRESS="COUNTME" | grep -c "COUNTME")
N := x
C = $(words $N)$(eval N := x $N)
PERC = `expr $C '*' 100 / $T`
PROGRESS = printf " [ %3d%% ] compiling: %-20s\r" $(PERC)
ARCHIVE  = printf " [ 100%% ] compiling: %-30s\r\n building archive ( %-15s )..."
ENDING   = printf " building binary  ( %-15s )..."
INSTALL  = printf " installing       ( %-15s )..."
DONE     = printf "%-30s\n" " done."
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
INCLUDES  := -I../../common/inc 
EXTRALIB  := 

# generated binaries

BIN := quasarq
LIB := libquasarq.a

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
      NVCCFLAGS += -g -G -DDEBUG
else  ifeq ($(assert),1)
      NVCCFLAGS += -O3 
else
      NVCCFLAGS += -O3 -DNDEBUG -diag-suppress 68 -diag-suppress 186 -diag-suppress 20091
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

NVCCFLAGS += -DWORD_SIZE_$(WORDSIZE)

# combine all flags
ALL_CCFLAGS := 
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# gencode arguments

SMS ?= 75 80 89
ifeq ($(SMS),)
$(error no SM architectures have been specified)
endif

ifeq ($(GENCODE_FLAGS),)
  # generate SASS code for each SM architecture in $(SMS)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
  # generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
  HIGHEST_SM := $(lastword $(sort $(SMS)))
  ifneq ($(HIGHEST_SM),)
  GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
  endif
endif

# target rules

CPPOBJEXT := o
CUOBJEXT  := cuda.o 
BUILD_DIR := ../build

mainsrc   := main
cusrc     := $(sort $(wildcard *.cu))
allcppsrc := $(sort $(wildcard *.cpp))
cppsrc    := $(filter-out $(mainsrc).cpp,$(allcppsrc))
cuobj     := $(patsubst %.cu,%.$(CUOBJEXT),$(cusrc))
cppobj    := $(patsubst %.cpp,%.$(CPPOBJEXT),$(cppsrc))

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

all: $(BIN)
	
$(LIB): $(cuobj) $(cppobj)
	@$(ARCHIVE) "done" $@
	@ar rc $@ $+
	@ranlib $@
	@$(DONE)

$(BIN): $(mainsrc).$(CPPOBJEXT) $(LIB)
	@$(ENDING) $@
	@$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $(mainsrc).$(CPPOBJEXT) -L$(CUDA_PATH)/lib64 -lcudart -lnvidia-ml -L. -l$(BIN) $(EXTRALIB)
	@$(DONE)

$(mainsrc).$(CPPOBJEXT): $(mainsrc).cpp
	@$(PROGRESS) $<
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $< 

%.$(CUOBJEXT): %.cu
	@$(PROGRESS) $<
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(RELOC) -o $@ -c $< 
	
%.$(CPPOBJEXT): %.cpp
	@$(PROGRESS) $<
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

install: all
	@$(INSTALL) $(BIN)
	@mkdir -p $(BUILD_DIR)
	@cp $(BIN) $(BUILD_DIR)
	@cp $(LIB) $(BUILD_DIR)
	@$(DONE)

clean:
	rm -f *.$(CPPOBJEXT) *.$(CUOBJEXT) $(LIB) $(BIN) 
	rm -rf $(BUILD_DIR)
	
.PHONY: all clean