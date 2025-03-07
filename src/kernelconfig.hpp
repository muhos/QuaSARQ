
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
        CONFIG(allpivots) \
        CONFIG(newpivots) \
        CONFIG(marking) \
        CONFIG(injectswap) \
        CONFIG(prefixpass1) \
        CONFIG(prefixpass2) \
        CONFIG(injectpass1) \
        CONFIG(injectpass2) \
        CONFIG(collapsetargets) \


    #define FOREACH_CONFIG_INIT(CONFIG) \
        CONFIG(reset, 4, 1, 500, 1) \
        CONFIG(identity, 4, 1, 500, 1) \
        CONFIG(step, 2, 128, 103, 52) \
        CONFIG(transposebits, 64, 16, 33, 1) \
        CONFIG(transposeswap, 64, 16, 33, 1) \
        CONFIG(allpivots, 32, 1, 2, 1) \
        CONFIG(newpivots, 32, 1, 2, 1) \
        CONFIG(marking, 4, 1, 500, 1) \
        CONFIG(injectswap, 4, 1, 500, 1) \
        CONFIG(prefixpass1, 4, 1, 500, 1) \
        CONFIG(prefixpass2, 4, 1, 500, 1) \
        CONFIG(injectpass1, 4, 1, 500, 1) \
        CONFIG(injectpass2, 4, 1, 500, 1) \
        CONFIG(collapsetargets, 4, 1, 500, 1) \

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