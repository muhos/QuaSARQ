CUDA_PATH ?= /usr/local/cuda
CXX       := g++
NVCC      := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

QUASARQ_DIR := ..
CUARENA_DIR ?= $(HOME)/cuarena

INCLUDES  := -I$(QUASARQ_DIR)/src -I$(CUARENA_DIR)/include
CCFLAGS   := -std=c++20 -fdiagnostics-show-option
NVCCFLAGS := -m64 -std=c++20 -arch=native
ALL_FLAGS := $(NVCCFLAGS) $(addprefix -Xcompiler ,$(CCFLAGS))

PARSER_SRCS := test_parser.cpp $(QUASARQ_DIR)/src/parser.cpp $(QUASARQ_DIR)/src/expansion.cpp
PARSER_BIN  := test_parser

SCHEDULER_SRCS := test_scheduler.cpp test_stubs.cpp $(QUASARQ_DIR)/src/parser.cpp $(QUASARQ_DIR)/src/expansion.cpp $(QUASARQ_DIR)/src/scheduler.cpp
SCHEDULER_BIN  := test_scheduler

all: $(PARSER_BIN) $(SCHEDULER_BIN)

$(PARSER_BIN): $(PARSER_SRCS)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(PARSER_SRCS)
	@echo "done."

$(SCHEDULER_BIN): $(SCHEDULER_SRCS)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(SCHEDULER_SRCS)
	@echo "done."

run: all
	@./$(PARSER_BIN)
	@./$(SCHEDULER_BIN)

clean:
	rm -f $(PARSER_BIN) $(SCHEDULER_BIN)

.PHONY: all run clean
