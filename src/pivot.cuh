#ifndef __CU_PIVOT_H
#define __CU_PIVOT_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "logging.hpp"

namespace QuaSARQ {

    constexpr uint32 INVALID_PIVOT = UINT32_MAX;
    struct Pivot {

        uint32 determinate;
        uint32 indeterminate;

        INLINE_ALL 
        Pivot() : determinate(INVALID_PIVOT), indeterminate(INVALID_PIVOT) { }

        INLINE_ALL 
        void reset() {
            determinate = INVALID_PIVOT;
            indeterminate = INVALID_PIVOT;
        }

        INLINE_ALL
        void print(const bool& nonl = false) const {
            LOGGPU("(indet: %-6d, det: %-6d)", indeterminate, determinate);
            if (!nonl) LOGGPU("\n");
        }
 
    };

    #define min_pivot_load_shared(smem, val, init_val, tid, size) \
	{ \
		smem[tid] = (threadIdx.x < size) ? val : init_val; \
		__syncthreads(); \
	}

	#define min_pivot_shared(smem, val, tid) \
	{ \
    	if (blockDim.x >= 1024) { \
			if (threadIdx.x < 512) { \
                smem[tid] = val = MIN(val, smem[tid + 512]); \
            } \
			__syncthreads(); \
		} \
		if (blockDim.x >= 512) { \
			if (threadIdx.x < 256) { \
                smem[tid] = val = MIN(val, smem[tid + 256]); \
            } \
			__syncthreads(); \
		} \
		if (blockDim.x >= 256) { \
			if (threadIdx.x < 128) { \
                smem[tid] = val = MIN(val, smem[tid + 128]); \
            } \
			__syncthreads(); \
		} \
		if (blockDim.x >= 128) { \
			if (threadIdx.x < 64) { \
                smem[tid] = val = MIN(val, smem[tid + 64]); \
            } \
			__syncthreads(); \
		} \
        if (threadIdx.x < 32) { \
            if (blockDim.x >= 64) { \
                smem[tid] = val = MIN(val, smem[tid + 32]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 32) { \
                smem[tid] = val = MIN(val, smem[tid + 16]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 16) { \
                smem[tid] = val = MIN(val, smem[tid + 8]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 8) { \
                smem[tid] = val = MIN(val, smem[tid + 4]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 4) { \
                smem[tid] = val = MIN(val, smem[tid + 2]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 2) { \
                smem[tid] = val = MIN(val, smem[tid + 1]); \
                __syncthreads(); \
            } \
        } \
	}

}

#endif