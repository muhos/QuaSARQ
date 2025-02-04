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

    #define min_pivot_load_shared(smem, val, init_val, tid, tx, size) \
	{ \
		smem[tid] = (tx < size) ? val : init_val; \
		__syncthreads(); \
	}

	#define min_pivot_shared(smem, val, tid, BX, tx) \
	{ \
		if (BX >= 512) { \
			if (tx < 256) { \
                smem[tid] = val = MIN(val, smem[tid + 256]); \
            } \
			__syncthreads(); \
		} \
		if (BX >= 256) { \
			if (tx < 128) { \
                smem[tid] = val = MIN(val, smem[tid + 128]); \
            } \
			__syncthreads(); \
		} \
		if (BX >= 128) { \
			if (tx < 64) { \
                smem[tid] = val = MIN(val, smem[tid + 64]); \
            } \
			__syncthreads(); \
		} \
	}

	#define min_pivot_warp(smem, val, tid, BX, tx) \
    { \
        if (tx < 32) { \
            const grid_t mask = __activemask(); \
            if (BX >= 64) { \
                smem[tid] = val = MIN(val, smem[tid + 32]); \
            } \
            if (BX >= 32) val = MIN(val, __shfl_down_sync(mask, val, 16)); \
            if (BX >= 16) val = MIN(val, __shfl_down_sync(mask, val, 8)); \
            if (BX >= 8) val = MIN(val, __shfl_down_sync(mask, val, 4)); \
            if (BX >= 4) val = MIN(val, __shfl_down_sync(mask, val, 2)); \
            if (BX >= 2) val = MIN(val, __shfl_down_sync(mask, val, 1)); \
        } \
    }

}

#endif