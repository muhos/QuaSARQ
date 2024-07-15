
#ifndef __CU_REDUCE_H
#define __CU_REDUCE_H

#include "definitions.cuh"
#include "shared.cuh"

namespace QuaSARQ {

	#define FULL_WARP 0xffffffff

	#define USE_SHARED_IN_WARP 0

	#define load_shared(smem, val, tid, tx, size) \
	{ \
		if (bx >= 64) { \
			smem[tid] = (tx < size) ? val : 0; \
			__syncthreads(); \
		} \
	}

	#define collapse_shared(smem, val, tid, bx, tx) \
	{ \
		if (bx >= 512) { \
			if (tx < 256) smem[tid] = val = val ^ smem[tid + 256]; \
			__syncthreads(); \
		} \
		if (bx >= 256) { \
			if (tx < 128) smem[tid] = val = val ^ smem[tid + 128]; \
			__syncthreads(); \
		} \
		if (bx >= 128) { \
			if (tx < 64) smem[tid] = val = val ^ smem[tid + 64]; \
			__syncthreads(); \
		} \
	}

	#if USE_SHARED_IN_WARP
	#define collapse_warp(smem, val, tid, bx, tx) \
		{ \
			if (tx < 32) { \
				if (bx >= 64) smem[tid] = val = val ^ smem[tid + 32]; \
				if (bx >= 32) smem[tid] = val = val ^ smem[tid + 16]; \
				if (bx >= 16) smem[tid] = val = val ^ smem[tid + 8]; \
				if (bx >= 8) smem[tid] = val = val ^ smem[tid + 4]; \
				if (bx >= 4) smem[tid] = val = val ^ smem[tid + 2]; \
				if (bx >= 2) smem[tid] = val = val ^ smem[tid + 1]; \
			} \
		}
	#else
	#define collapse_warp(smem, val, tid, bx, tx) \
		{ \
			if (tx < 32) { \
				const grid_t mask = __activemask(); \
				if (bx >= 64) smem[tid] = val = val ^ smem[tid + 32]; \
				if (bx >= 32) val ^= __shfl_down_sync(mask, val, 16); \
				if (bx >= 16) val ^= __shfl_down_sync(mask, val, 8); \
				if (bx >= 8) val ^= __shfl_down_sync(mask, val, 4); \
				if (bx >= 4) val ^= __shfl_down_sync(mask, val, 2);\
				if (bx >= 2) val ^= __shfl_down_sync(mask, val, 1); \
			} \
		}
	#endif

	#define collapse_warp_only(val) \
	{ \
		const grid_t mask = __activemask(); \
		if (bx >= 32) val ^= __shfl_down_sync(mask, val, 16); \
		if (bx >= 16) val ^= __shfl_down_sync(mask, val, 8); \
		if (bx >= 8) val ^= __shfl_down_sync(mask, val, 4); \
		if (bx >= 4) val ^= __shfl_down_sync(mask, val, 2);\
		if (bx >= 2) val ^= __shfl_down_sync(mask, val, 1); \
	}

}

#endif