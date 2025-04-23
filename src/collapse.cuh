
#pragma once

#include "definitions.cuh"
#include "shared.cuh"

namespace QuaSARQ {

	#define collapse_load_shared(smem, val, tid, size) \
	{ \
		if (blockDim.x >= 64) smem[tid] = (tid < size) ? val : 0; \
		__syncthreads(); \
	}

	template <int B, typename T>
	INLINE_DEVICE
	void collapse_shared(T* smem, T &val, const int& tid) {
		if constexpr (B >= 1024) {
			if (tid < 512)
				smem[tid] = val ^ smem[tid + 512];
			__syncthreads();
		}
		if constexpr (B >= 512) {
			if (tid < 256)
				smem[tid] = val ^ smem[tid + 256];
			__syncthreads();
		}
		if constexpr (B >= 256) {
			if (tid < 128)
				smem[tid] = val ^ smem[tid + 128];
			__syncthreads();
		}
		if constexpr (B >= 128) {
			if (tid < 64)
				smem[tid] = val ^ smem[tid + 64];
			__syncthreads();
		}
		if constexpr (B >= 64) {
			if (tid < 32)
				smem[tid] = val ^ smem[tid + 32];
			// No __syncthreads() needed.
		}
	}

	template <int B, typename T>
	INLINE_DEVICE 
	void collapse_warp(T &val, const int& tid) {
		if (tid < 32) {
			unsigned mask = __activemask();
			if constexpr (B >= 32)
				val ^= __shfl_down_sync(mask, val, 16);
			if constexpr (B >= 16)
				val ^= __shfl_down_sync(mask, val, 8);
			if constexpr (B >= 8)
				val ^= __shfl_down_sync(mask, val, 4);
			if constexpr (B >= 4)
				val ^= __shfl_down_sync(mask, val, 2);
			if constexpr (B >= 2)
				val ^= __shfl_down_sync(mask, val, 1);
		}
	}

	#define collapse_load_shared_dual(smem1, val1, smem2, val2, tid, size) \
	{ \
		if (blockDim.x >= 64) { \
			smem1[tid] = (tid < size) ? val1 : 0; \
			smem2[tid] = (tid < size) ? val2 : 0; \
		} \
		__syncthreads(); \
	}

	template <int B, typename T>
	INLINE_DEVICE
	void collapse_shared_dual(
				T *		smem1, 
				T &		val1, 
				T *		smem2, 
				T &		val2, 
		const 	int& 	tid) {
		if constexpr (B >= 1024) {
			if (tid < 512) {
				smem1[tid] = val1 = val1 ^ smem1[tid + 512];
				smem2[tid] = val2 = val2 ^ smem2[tid + 512];
			}
			__syncthreads();
		}
		if constexpr (B >= 512) {
			if (tid < 256) {
				smem1[tid] = val1 = val1 ^ smem1[tid + 256];
				smem2[tid] = val2 = val2 ^ smem2[tid + 256];
			}
			__syncthreads();
		}
		if constexpr (B >= 256) {
			if (tid < 128) {
				smem1[tid] = val1 = val1 ^ smem1[tid + 128];
				smem2[tid] = val2 = val2 ^ smem2[tid + 128];
			}
			__syncthreads();
		}
		if constexpr (B >= 128) {
			if (tid < 64) {
				smem1[tid] = val1 = val1 ^ smem1[tid + 64];
				smem2[tid] = val2 = val2 ^ smem2[tid + 64];
			}
			__syncthreads();
		}
		if constexpr (B >= 64) {
			if (tid < 32) {
				smem1[tid] = val1 = val1 ^ smem1[tid + 32];
				smem2[tid] = val2 = val2 ^ smem2[tid + 32];
			}
			// No __syncthreads() required in a warp.
		}
	}

	template <int B, typename T>
	INLINE_DEVICE 
	void collapse_warp_dual(T &val1, T &val2, int tid) {
		if (tid < 32) {
			unsigned mask = __activemask();
			if constexpr (B >= 32) {
				val1 ^= __shfl_down_sync(mask, val1, 16);
				val2 ^= __shfl_down_sync(mask, val2, 16);
			}
			if constexpr (B >= 16) {
				val1 ^= __shfl_down_sync(mask, val1, 8);
				val2 ^= __shfl_down_sync(mask, val2, 8);
			}
			if constexpr (B >= 8) {
				val1 ^= __shfl_down_sync(mask, val1, 4);
				val2 ^= __shfl_down_sync(mask, val2, 4);
			}
			if constexpr (B >= 4) {
				val1 ^= __shfl_down_sync(mask, val1, 2);
				val2 ^= __shfl_down_sync(mask, val2, 2);
			}
			if constexpr (B >= 2) {
				val1 ^= __shfl_down_sync(mask, val1, 1);
				val2 ^= __shfl_down_sync(mask, val2, 1);
			}
		}
	}

}