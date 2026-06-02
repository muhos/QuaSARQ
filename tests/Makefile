CUDA_PATH ?= /usr/local/cuda
CXX       := g++
NVCC      := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

QUASARQ_DIR := ..
CUARENA_DIR ?= $(HOME)/cuarena

INCLUDES  := -I$(QUASARQ_DIR)/src -I$(CUARENA_DIR)/include
CCFLAGS   := -std=c++20 -fdiagnostics-show-option
NVCCFLAGS := -m64 -std=c++20 -arch=native
ALL_FLAGS := $(NVCCFLAGS) $(addprefix -Xcompiler ,$(CCFLAGS))

SRCS := test_parser.cpp $(QUASARQ_DIR)/src/parser.cpp $(QUASARQ_DIR)/src/expansion.cpp
BIN  := test_parser

all: $(BIN)

$(BIN): $(SRCS)
	@echo -n "Building $@..."
	@$(NVCC) $(ALL_FLAGS) $(INCLUDES) -o $@ $(SRCS)
	@echo "done."

run: $(BIN)
	@./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: all run clean
