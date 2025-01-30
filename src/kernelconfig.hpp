
#ifndef __KERNELCONFIG_H
#define __KERNELCONFIG_H

#include <cuda_runtime.h>

namespace QuaSARQ {

    #define FOREACH_CONFIG(CONFIG) \
		CONFIG(reset) \
		CONFIG(identity) \
		CONFIG(step) \
        CONFIG(transpose2r) \
        CONFIG(transpose2c) \
        CONFIG(transposebits) \
        CONFIG(transposeswap) \
        CONFIG(allpivots) \
        CONFIG(newpivots) \
        CONFIG(multdeterminate) \
        CONFIG(singdeterminate) \
        CONFIG(copyindeterminate) \
        CONFIG(phase1indeterminate) \
        CONFIG(phase2indeterminate) \

    #define FOREACH_CONFIG_INIT(CONFIG) \
        CONFIG(reset, 4, 1, 500, 1) \
        CONFIG(identity, 4, 1, 500, 1) \
        CONFIG(step, 2, 128, 103, 52) \
        CONFIG(transpose2r, 2, 128, 103, 52) \
        CONFIG(transpose2c, 2, 128, 103, 52) \
        CONFIG(transposebits, 0, 0, 0, 0) \
        CONFIG(transposeswap, 0, 0, 0, 0) \
        CONFIG(allpivots, 2, 128, 103, 52) \
        CONFIG(newpivots, 2, 128, 103, 52) \
        CONFIG(multdeterminate, 0, 0, 0, 0) \
        CONFIG(singdeterminate, 0, 0, 0, 0) \
        CONFIG(copyindeterminate, 4, 1, 500, 1) \
        CONFIG(phase1indeterminate, 2, 128, 103, 52) \
        CONFIG(phase2indeterminate, 4, 1, 500, 1) \

    #define CONFIG2EXTERN(NAME) \
        extern dim3 bestgrid ## NAME; \
		extern dim3 bestblock ## NAME; \

	#define CONFIG2INITIAL(CONFIG, BLOCKX, BLOCKY, GRIDX, GRIDY) \
        dim3 bestgrid ## CONFIG(GRIDX, GRIDY); \
		dim3 bestblock ## CONFIG(BLOCKX, BLOCKY); \

	#define CONFIG2RESET(NAME) \
        bestgrid ## NAME = dim3(); \
		bestblock ## NAME = dim3(); \

	// Kernel configuration parameters.
	// If they are set to default (1), tuner will be triggered.
	FOREACH_CONFIG(CONFIG2EXTERN);
}

#endif