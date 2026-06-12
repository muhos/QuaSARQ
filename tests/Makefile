CUDA_PATH ?= /usr/local/cuda
CXX       := g++
NVCC      := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

QUASARQ_DIR := ..
CUARENA_DIR ?= $(HOME)/cuarena
CUARENA_LIB := $(CUARENA_DIR)/build/libcuarena.a
QUASARQ_LIB := $(QUASARQ_DIR)/build/libquasarq.a

INCLUDES  := -I$(QUASARQ_DIR)/src -I$(CUARENA_DIR)/include
LIBS      := -L$(QUASARQ_DIR)/build -lquasarq -L$(CUARENA_DIR)/build -lcuarena -L$(CUDA_PATH)/lib64 -lcudart -lcurand -lnvidia-ml
CCFLAGS   := -std=c++20 -fdiagnostics-show-option
NVCCFLAGS := -m64 -std=c++20 -arch=native

ifeq ($(debug),1)
      NVCCFLAGS += -g -G -DDEBUG
else  ifeq ($(assert),1)
      NVCCFLAGS += -O3
else
      NVCCFLAGS += -O3 -DNDEBUG -diag-suppress 68 -diag-suppress 186 -diag-suppress 20091 -diag-suppress 20011
endif

ifeq ($(nocolor),1)
      NVCCFLAGS += -DNCOLORS
endif

WORDSIZE := 64
ifeq ($(word),8)
      WORDSIZE := 8
else  ifeq ($(word),32)
      WORDSIZE := 32
else  ifeq ($(word),64)
      WORDSIZE := 64
endif

NVCCFLAGS += -DWORD_SIZE_$(WORDSIZE)
ALL_FLAGS := $(NVCCFLAGS) $(addprefix -Xcompiler ,$(CCFLAGS))
TEST_HEADERS := $(sort $(wildcard *.hpp) $(wildcard $(QUASARQ_DIR)/src/*.hpp) $(wildcard $(QUASARQ_DIR)/src/*.cuh))

PARSER_SRCS := test_parser.cpp test_stubs.cpp $(QUASARQ_DIR)/src/parser.cpp $(QUASARQ_DIR)/src/expansion.cpp
PARSER_BIN  := test_parser

SCHEDULER_SRCS := test_scheduler.cpp test_stubs.cpp $(QUASARQ_DIR)/src/parser.cpp $(QUASARQ_DIR)/src/expansion.cpp $(QUASARQ_DIR)/src/scheduler.cpp
SCHEDULER_BIN  := test_scheduler

SIMULATOR_SRCS := test_simulator.cpp
SIMULATOR_BIN  := test_simulator

SAMPLING_SRCS := test_sampling.cpp
SAMPLING_BIN  := test_sampling

EQUIVALENCE_SRCS := test_equivalence.cpp
EQUIVALENCE_BIN  := test_equivalence

all: $(PARSER_BIN) $(SCHEDULER_BIN) $(SIMULATOR_BIN) $(SAMPLING_BIN) $(EQUIVALENCE_BIN)

$(PARSER_BIN): $(PARSER_SRCS) $(TEST_HEADERS)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(PARSER_SRCS)
	@echo "done."

$(SCHEDULER_BIN): $(SCHEDULER_SRCS) $(TEST_HEADERS)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(SCHEDULER_SRCS)
	@echo "done."

parser: $(PARSER_BIN)
	@./$(PARSER_BIN)

scheduler: $(SCHEDULER_BIN)
	@./$(SCHEDULER_BIN)

FORCE:

$(QUASARQ_LIB): FORCE
	@$(MAKE) -C $(QUASARQ_DIR) build/libquasarq.a

$(CUARENA_LIB):
	@$(MAKE) -C $(QUASARQ_DIR) $(CUARENA_LIB)

$(SIMULATOR_BIN): $(SIMULATOR_SRCS) $(QUASARQ_LIB) $(CUARENA_LIB)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(SIMULATOR_SRCS) $(LIBS)
	@echo "done."

$(SAMPLING_BIN): $(SAMPLING_SRCS) $(QUASARQ_LIB) $(CUARENA_LIB)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(SAMPLING_SRCS) $(LIBS)
	@echo "done."

$(EQUIVALENCE_BIN): $(EQUIVALENCE_SRCS) $(QUASARQ_LIB) $(CUARENA_LIB)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(EQUIVALENCE_SRCS) $(LIBS)
	@echo "done."

run: all
	@./$(PARSER_BIN)
	@./$(SCHEDULER_BIN)
	@./$(SIMULATOR_BIN)
	@./$(SAMPLING_BIN)
	@./$(EQUIVALENCE_BIN)

clean:
	rm -f $(PARSER_BIN) $(SCHEDULER_BIN) $(SIMULATOR_BIN) $(SAMPLING_BIN) $(EQUIVALENCE_BIN)

.PHONY: all run clean FORCE parser scheduler
