
#ifndef __CU_SUM_H
#define __CU_SUM_H

#include "definitions.cuh"
#include "shared.cuh"

namespace QuaSARQ {

	#define FULL_WARP 0xffffffff

	#define USE_SHARED_IN_WARP 0

	#define load_shared(smem1, val1, smem2, val2, tid, tx, size) \
	{ \
		if (BX >= 64) { \
			smem1[tid] = (tx < size) ? val1 : 0; \
			smem2[tid] = (tx < size) ? val2 : 0; \
		} \
		__syncthreads(); \
	}

	#define sum_shared(smem1, val1, smem2, val2, tid, BX, tx) \
	{ \
		if (BX >= 512) { \
			if (tx < 256) { \
				smem1[tid] = val1 = val1 + smem1[tid + 256]; \
				smem2[tid] = val2 = val2 + smem2[tid + 256]; \
			} \
			__syncthreads(); \
		} \
		if (BX >= 256) { \
			if (tx < 128) { \
				smem1[tid] = val1 = val1 + smem1[tid + 128]; \
				smem2[tid] = val2 = val2 + smem2[tid + 128]; \
			} \
			__syncthreads(); \
		} \
		if (BX >= 128) { \
			if (tx < 64) { \
				smem1[tid] = val1 = val1 + smem1[tid + 64]; \
				smem2[tid] = val2 = val2 + smem2[tid + 64]; \
			} \
			__syncthreads(); \
		} \
	}

	#define sum_warp(smem1, val1, smem2, val2, tid, BX, tx) \
	{ \
		if (tx < 32) { \
			const grid_t mask = __activemask(); \
			if (BX >= 64) { \
				smem1[tid] = val1 = val1 + smem1[tid + 32]; \
				smem2[tid] = val2 = val2 + smem2[tid + 32]; \
			} \
			if (BX >= 32) { \
				val1 += __shfl_down_sync(mask, val1, 16); \
				val2 += __shfl_down_sync(mask, val2, 16); \
			} \
			if (BX >= 16) { \
				val1 += __shfl_down_sync(mask, val1, 8); \
				val2 += __shfl_down_sync(mask, val2, 8); \
			} \
			if (BX >= 8) { \
				val1 += __shfl_down_sync(mask, val1, 4); \
				val2 += __shfl_down_sync(mask, val2, 4); \
			} \
			if (BX >= 4) { \
				val1 += __shfl_down_sync(mask, val1, 2); \
				val2 += __shfl_down_sync(mask, val2, 2); \
			} \
			if (BX >= 2) { \
				val1 += __shfl_down_sync(mask, val1, 1); \
				val2 += __shfl_down_sync(mask, val2, 1); \
			} \
		} \
	}

}

#endif