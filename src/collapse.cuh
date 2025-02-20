
#ifndef __CU_COLLAPSE_H
#define __CU_COLLAPSE_H

#include "definitions.cuh"
#include "shared.cuh"

namespace QuaSARQ {

	#define FULL_WARP 0xffffffff

	#define USE_SHARED_IN_WARP 0

	#define collapse_load_shared(smem, val, tid, size) \
	{ \
		if (blockDim.x >= 64) smem[tid] = (threadIdx.x < size) ? val : 0; \
		__syncthreads(); \
	}

	#define collapse_shared(smem, val, tid) \
	{ \
		if (blockDim.x >= 1024) { \
			if (threadIdx.x < 512) smem[tid] = val = val ^ smem[tid + 512]; \
			__syncthreads(); \
		} \
		if (blockDim.x >= 512) { \
			if (threadIdx.x < 256) smem[tid] = val = val ^ smem[tid + 256]; \
			__syncthreads(); \
		} \
		if (blockDim.x >= 256) { \
			if (threadIdx.x < 128) smem[tid] = val = val ^ smem[tid + 128]; \
			__syncthreads(); \
		} \
		if (blockDim.x >= 128) { \
			if (threadIdx.x < 64) smem[tid] = val = val ^ smem[tid + 64]; \
			__syncthreads(); \
		} \
		if (blockDim.x >= 64 && threadIdx.x < 32) smem[tid] = val = val ^ smem[tid + 32]; \
	}

	#if USE_SHARED_IN_WARP
	#define collapse_warp(smem, val, tid) \
		{ \
			if (threadIdx.x < 32) { \
				if (blockDim.x >= 32) smem[tid] = val = val ^ smem[tid + 16]; \
				if (blockDim.x >= 16) smem[tid] = val = val ^ smem[tid + 8]; \
				if (blockDim.x >= 8) smem[tid] = val = val ^ smem[tid + 4]; \
				if (blockDim.x >= 4) smem[tid] = val = val ^ smem[tid + 2]; \
				if (blockDim.x >= 2) smem[tid] = val = val ^ smem[tid + 1]; \
			} \
		}
	#else
	#define collapse_warp(smem, val, tid) \
		{ \
			if (threadIdx.x < 32) { \
				const grid_t mask = __activemask(); \
				if (blockDim.x >= 32) val ^= __shfl_down_sync(mask, val, 16); \
				if (blockDim.x >= 16) val ^= __shfl_down_sync(mask, val, 8); \
				if (blockDim.x >= 8) val ^= __shfl_down_sync(mask, val, 4); \
				if (blockDim.x >= 4) val ^= __shfl_down_sync(mask, val, 2);\
				if (blockDim.x >= 2) val ^= __shfl_down_sync(mask, val, 1); \
			} \
		}
	#endif

	#define collapse_warp_only(val) \
	{ \
		const grid_t mask = __activemask(); \
		if (blockDim.x >= 32) val ^= __shfl_down_sync(mask, val, 16); \
		if (blockDim.x >= 16) val ^= __shfl_down_sync(mask, val, 8); \
		if (blockDim.x >= 8) val ^= __shfl_down_sync(mask, val, 4); \
		if (blockDim.x >= 4) val ^= __shfl_down_sync(mask, val, 2);\
		if (blockDim.x >= 2) val ^= __shfl_down_sync(mask, val, 1); \
	}

	#define collapse_load_shared_dual(smem1, val1, smem2, val2, tid, size) \
	{ \
		if (blockDim.x >= 64) { \
			smem1[tid] = (threadIdx.x < size) ? val1 : 0; \
			smem2[tid] = (threadIdx.x < size) ? val2 : 0; \
		} \
		__syncthreads(); \
	}

	#define collapse_shared_dual(smem1, val1, smem2, val2, tid) \
	{ \
		if (blockDim.x >= 1024) { \
			if (threadIdx.x < 512) { \
				smem1[tid] = val1 = val1 ^ smem1[tid + 512]; \
				smem2[tid] = val2 = val2 ^ smem2[tid + 512]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 512) { \
			if (threadIdx.x < 256) { \
				smem1[tid] = val1 = val1 ^ smem1[tid + 256]; \
				smem2[tid] = val2 = val2 ^ smem2[tid + 256]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 256) { \
			if (threadIdx.x < 128) { \
				smem1[tid] = val1 = val1 ^ smem1[tid + 128]; \
				smem2[tid] = val2 = val2 ^ smem2[tid + 128]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 128) { \
			if (threadIdx.x < 64) { \
				smem1[tid] = val1 = val1 ^ smem1[tid + 64]; \
				smem2[tid] = val2 = val2 ^ smem2[tid + 64]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 64) { \
			if (threadIdx.x < 32) { \
				smem1[tid] = val1 = val1 ^ smem1[tid + 32]; \
				smem2[tid] = val2 = val2 ^ smem2[tid + 32]; \
			}\
		} \
	}

	#define collapse_warp_dual(smem1, val1, smem2, val2, tid) \
	{ \
		if (threadIdx.x < 32) { \
			const grid_t mask = __activemask(); \
			if (blockDim.x >= 32) { \
				val1 ^= __shfl_down_sync(mask, val1, 16); \
				val2 ^= __shfl_down_sync(mask, val2, 16); \
			} \
			if (blockDim.x >= 16) { \
				val1 ^= __shfl_down_sync(mask, val1, 8); \
				val2 ^= __shfl_down_sync(mask, val2, 8); \
			} \
			if (blockDim.x >= 8) { \
				val1 ^= __shfl_down_sync(mask, val1, 4); \
				val2 ^= __shfl_down_sync(mask, val2, 4); \
			} \
			if (blockDim.x >= 4) { \
				val1 ^= __shfl_down_sync(mask, val1, 2); \
				val2 ^= __shfl_down_sync(mask, val2, 2); \
			} \
			if (blockDim.x >= 2) { \
				val1 ^= __shfl_down_sync(mask, val1, 1); \
				val2 ^= __shfl_down_sync(mask, val2, 1); \
			} \
		} \
	}

}

#endif