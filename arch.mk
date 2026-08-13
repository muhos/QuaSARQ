#############################################
# GPU architecture selection, shared by the
# core and the Python extension builds.
#############################################

GPU_ARCH ?= native

ARCH_EMPTY :=
ARCH_SPACE := $(ARCH_EMPTY) $(ARCH_EMPTY)
ARCH_COMMA := ,
ARCH_LIST  := $(subst $(ARCH_COMMA),$(ARCH_SPACE),$(GPU_ARCH))
ARCH_NUMS  := $(patsubst sm_%,%,$(patsubst compute_%,%,$(ARCH_LIST)))
ARCH_TOP   := $(lastword $(ARCH_NUMS))

ifeq ($(words $(ARCH_LIST)),1)
      GENCODE_FLAGS := -arch=$(GPU_ARCH)
      CUARENA_ARCH  := $(GPU_ARCH)
else
      GENCODE_FLAGS := $(foreach a,$(ARCH_NUMS),-gencode arch=compute_$(a),code=sm_$(a)) \
                       -gencode arch=compute_$(ARCH_TOP),code=compute_$(ARCH_TOP)
      CUARENA_ARCH  := $(subst $(ARCH_SPACE),;,$(ARCH_NUMS))
endif
