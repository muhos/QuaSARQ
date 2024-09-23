
#ifndef __CU_REDUCE_H
#define __CU_REDUCE_H

#include "definitions.cuh"
#include "shared.cuh"

namespace QuaSARQ {

	#define FULL_WARP 0xffffffff

	#define USE_SHARED_IN_WARP 0

	#define collapse_load_shared(smem, val, tid, tx, size) \
	{ \
		if (BX >= 64) smem[tid] = (tx < size) ? val : 0; \
		__syncthreads(); \
	}

	#define collapse_shared(smem, val, tid, BX, tx) \
	{ \
		if (BX >= 512) { \
			if (tx < 256) smem[tid] = val = val ^ smem[tid + 256]; \
			__syncthreads(); \
		} \
		if (BX >= 256) { \
			if (tx < 128) smem[tid] = val = val ^ smem[tid + 128]; \
			__syncthreads(); \
		} \
		if (BX >= 128) { \
			if (tx < 64) smem[tid] = val = val ^ smem[tid + 64]; \
			__syncthreads(); \
		} \
	}

	#if USE_SHARED_IN_WARP
	#define collapse_warp(smem, val, tid, BX, tx) \
		{ \
			if (tx < 32) { \
				if (BX >= 64) smem[tid] = val = val ^ smem[tid + 32]; \
				if (BX >= 32) smem[tid] = val = val ^ smem[tid + 16]; \
				if (BX >= 16) smem[tid] = val = val ^ smem[tid + 8]; \
				if (BX >= 8) smem[tid] = val = val ^ smem[tid + 4]; \
				if (BX >= 4) smem[tid] = val = val ^ smem[tid + 2]; \
				if (BX >= 2) smem[tid] = val = val ^ smem[tid + 1]; \
			} \
		}
	#else
	#define collapse_warp(smem, val, tid, BX, tx) \
		{ \
			if (tx < 32) { \
				const grid_t mask = __activemask(); \
				if (BX >= 64) smem[tid] = val = val ^ smem[tid + 32]; \
				if (BX >= 32) val ^= __shfl_down_sync(mask, val, 16); \
				if (BX >= 16) val ^= __shfl_down_sync(mask, val, 8); \
				if (BX >= 8) val ^= __shfl_down_sync(mask, val, 4); \
				if (BX >= 4) val ^= __shfl_down_sync(mask, val, 2);\
				if (BX >= 2) val ^= __shfl_down_sync(mask, val, 1); \
			} \
		}
	#endif

	#define collapse_warp_only(val) \
	{ \
		const grid_t mask = __activemask(); \
		if (BX >= 32) val ^= __shfl_down_sync(mask, val, 16); \
		if (BX >= 16) val ^= __shfl_down_sync(mask, val, 8); \
		if (BX >= 8) val ^= __shfl_down_sync(mask, val, 4); \
		if (BX >= 4) val ^= __shfl_down_sync(mask, val, 2);\
		if (BX >= 2) val ^= __shfl_down_sync(mask, val, 1); \
	}

}

#endif