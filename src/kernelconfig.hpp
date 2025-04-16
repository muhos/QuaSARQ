
#ifndef __KERNELCONFIG_H
#define __KERNELCONFIG_H

#include <cuda_runtime.h>

namespace QuaSARQ {

    #define FOREACH_CONFIG(CONFIG) \
        CONFIG(reset, 4, 1, 500, 1) \
        CONFIG(identity, 4, 1, 500, 1) \
        CONFIG(step, 2, 128, 103, 52) \
        CONFIG(transposebits, 64, 8, 50, 100) \
        CONFIG(transposeswap, 64, 8, 50, 100) \
        CONFIG(transposec2r, 32, 16, 50, 50) \
        CONFIG(transposer2c, 32, 16, 50, 50) \
        CONFIG(allpivots, 256, 2, 100, 16) \
        CONFIG(newpivots, 256, 1, 100, 1) \
        CONFIG(marking, 256, 1, 100, 1) \
        CONFIG(injectswap, 256, 1, 100, 1) \
        CONFIG(prefixprepare, 16, 64, 96, 28) \
        CONFIG(prefixsingle, 512, 2, 1, 10) \
        CONFIG(prefixfinal, 256, 2, 100, 16) \
        CONFIG(injectprepare, 256, 2, 100, 16) \
        CONFIG(injectfinal, 256, 2, 100, 16) \

    #define CONFIG2EXTERN(NAME, BLOCKX, BLOCKY, GRIDX, GRIDY) \
        extern dim3 bestgrid ## NAME; \
		extern dim3 bestblock ## NAME; \

	#define CONFIG2INITIAL(CONFIG, BLOCKX, BLOCKY, GRIDX, GRIDY) \
        dim3 bestgrid ## CONFIG(GRIDX, GRIDY); \
		dim3 bestblock ## CONFIG(BLOCKX, BLOCKY); \

	#define CONFIG2RESET(NAME, BLOCKX, BLOCKY, GRIDX, GRIDY) \
        if (options.tune_ ## NAME) { \
            bestgrid ## NAME = dim3(); \
            bestblock ## NAME = dim3(); \
        }

	// Kernel configuration parameters.
	// If they are set to default (1), tuner will be triggered.
	FOREACH_CONFIG(CONFIG2EXTERN);
}

#endif