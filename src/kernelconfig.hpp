
#ifndef __KERNELCONFIG_H
#define __KERNELCONFIG_H

#include <cuda_runtime.h>

namespace QuaSARQ {

    #define FOREACH_CONFIG(CONFIG) \
		CONFIG(reset) \
		CONFIG(identity) \
		CONFIG(step) \
        CONFIG(transposebits) \
        CONFIG(transposeswap) \
        CONFIG(transposerowmajor) \
        CONFIG(allpivots) \
        CONFIG(newpivots) \
        CONFIG(marking) \
        CONFIG(injectswap) \
        CONFIG(prefixprepare) \
        CONFIG(prefixsingle) \
        CONFIG(prefixfinal) \
        CONFIG(injectprepare) \
        CONFIG(injectfinal) \
        CONFIG(collapsetargets) \


    #define FOREACH_CONFIG_INIT(CONFIG) \
        CONFIG(reset, 512, 1, 120, 1) \
        CONFIG(identity, 128, 1, 120, 1) \
        CONFIG(step, 2, 64, 95, 71) \
        CONFIG(transposebits, 64, 8, 103, 97) \
        CONFIG(transposeswap, 64, 16, 60, 97) \
        CONFIG(transposerowmajor, 64, 4, 1, 1) \
        CONFIG(allpivots, 2, 32, 55, 55) \
        CONFIG(newpivots, 256, 1, 102, 1) \
        CONFIG(marking, 512, 1, 106, 1) \
        CONFIG(injectswap, 512, 1, 6, 1) \
        CONFIG(prefixprepare, 64, 8, 96, 28) \
        CONFIG(prefixsingle, 1024, 1, 1, 68) \
        CONFIG(prefixfinal, 64, 8, 96, 28) \
        CONFIG(injectprepare, 128, 2, 107, 52) \
        CONFIG(injectfinal, 32, 8, 55, 4) \
        CONFIG(collapsetargets, 8, 16, 8, 1) \

    #define CONFIG2EXTERN(NAME) \
        extern dim3 bestgrid ## NAME; \
		extern dim3 bestblock ## NAME; \

	#define CONFIG2INITIAL(CONFIG, BLOCKX, BLOCKY, GRIDX, GRIDY) \
        dim3 bestgrid ## CONFIG(GRIDX, GRIDY); \
		dim3 bestblock ## CONFIG(BLOCKX, BLOCKY); \

	#define CONFIG2RESET(NAME) \
        if (options.tune_ ## NAME) { \
            bestgrid ## NAME = dim3(); \
            bestblock ## NAME = dim3(); \
        }

	// Kernel configuration parameters.
	// If they are set to default (1), tuner will be triggered.
	FOREACH_CONFIG(CONFIG2EXTERN);
}

#endif