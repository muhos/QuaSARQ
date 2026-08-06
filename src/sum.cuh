#pragma once

#include "definitions.cuh"
#include "shared.cuh"

namespace QuaSARQ {

	#define USE_SHARED_IN_WARP 0

	#define load_shared(smem1, val1, smem2, val2, tid, size) \
	{ \
		if (blockDim.x >= 64) { \
			smem1[tid] = (threadIdx.x < size) ? val1 : 0; \
			smem2[tid] = (threadIdx.x < size) ? val2 : 0; \
		} \
		__syncthreads(); \
	}

	#define load_shared_single(smem, val, tid, size) \
	{ \
		if (blockDim.x >= 64) { \
			smem[tid] = (threadIdx.x < size) ? val : 0; \
		} \
		__syncthreads(); \
	}

	#define sum_shared(smem1, val1, smem2, val2, tid) \
	{ \
		if (blockDim.x >= 512) { \
			if (threadIdx.x < 256) { \
				smem1[tid] = val1 = val1 + smem1[tid + 256]; \
				smem2[tid] = val2 = val2 + smem2[tid + 256]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 256) { \
			if (threadIdx.x < 128) { \
				smem1[tid] = val1 = val1 + smem1[tid + 128]; \
				smem2[tid] = val2 = val2 + smem2[tid + 128]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 128) { \
			if (threadIdx.x < 64) { \
				smem1[tid] = val1 = val1 + smem1[tid + 64]; \
				smem2[tid] = val2 = val2 + smem2[tid + 64]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 64) { \
			if (threadIdx.x < 32) { \
				smem1[tid] = val1 = val1 + smem1[tid + 32]; \
				smem2[tid] = val2 = val2 + smem2[tid + 32]; \
			} \
		} \
	}

	#define sum_shared_single(smem, val, tid) \
	{ \
		if (blockDim.x >= 512) { \
			if (threadIdx.x < 256) { \
				smem[tid] = val = val + smem[tid + 256]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 256) { \
			if (threadIdx.x < 128) { \
				smem[tid] = val = val + smem[tid + 128]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 128) { \
			if (threadIdx.x < 64) { \
				smem[tid] = val = val + smem[tid + 64]; \
			} \
			__syncthreads(); \
		} \
		if (blockDim.x >= 64) { \
			if (threadIdx.x < 32) { \
				smem[tid] = val = val + smem[tid + 32]; \
			} \
		} \
	}

	#define sum_warp(smem1, val1, smem2, val2, tid) \
	{ \
		if (threadIdx.x < 32) { \
			const grid_t mask = __activemask(); \
			if (blockDim.x >= 32) { \
				val1 += __shfl_down_sync(mask, val1, 16); \
				val2 += __shfl_down_sync(mask, val2, 16); \
			} \
			if (blockDim.x >= 16) { \
				val1 += __shfl_down_sync(mask, val1, 8); \
				val2 += __shfl_down_sync(mask, val2, 8); \
			} \
			if (blockDim.x >= 8) { \
				val1 += __shfl_down_sync(mask, val1, 4); \
				val2 += __shfl_down_sync(mask, val2, 4); \
			} \
			if (blockDim.x >= 4) { \
				val1 += __shfl_down_sync(mask, val1, 2); \
				val2 += __shfl_down_sync(mask, val2, 2); \
			} \
			if (blockDim.x >= 2) { \
				val1 += __shfl_down_sync(mask, val1, 1); \
				val2 += __shfl_down_sync(mask, val2, 1); \
			} \
		} \
	}

	#define reduce_warp(OP, val) \
	{ \
		const grid_t mask = __activemask(); \
		for (unsigned off = 16; off; off >>= 1) { \
			val = OP(val, __shfl_down_sync(mask, val, off)); \
		} \
	}

	#define reduce_warps(OP, IDENTITY, val, warp_smem) \
	{ \
		const unsigned nwarps = blockDim.x >> 5; \
		reduce_warp(OP, val); \
		if (!(threadIdx.x & 31)) \
			warp_smem[threadIdx.x >> 5] = val; \
		__syncthreads(); \
		if (threadIdx.x < 32) { \
			val = (threadIdx.x < nwarps) ? warp_smem[threadIdx.x] : (IDENTITY); \
			reduce_warp(OP, val); \
		} \
	}

	#define sum_warp_single(smem, val, tid) \
	{ \
		if (threadIdx.x < 32) { \
			const grid_t mask = __activemask(); \
			if (blockDim.x >= 32) { \
				val += __shfl_down_sync(mask, val, 16); \
			} \
			if (blockDim.x >= 16) { \
				val += __shfl_down_sync(mask, val, 8); \
			} \
			if (blockDim.x >= 8) { \
				val += __shfl_down_sync(mask, val, 4); \
			} \
			if (blockDim.x >= 4) { \
				val += __shfl_down_sync(mask, val, 2); \
			} \
			if (blockDim.x >= 2) { \
				val += __shfl_down_sync(mask, val, 1); \
			} \
		} \
	}

}
