#ifndef __CU_GRID_H
#define __CU_GRID_H

#include "definitions.cuh"
#include "datatypes.hpp"

namespace QuaSARQ {

	typedef size_t grid_t;

	// GPU capability.
	extern cudaDeviceProp devProp;
	extern grid_t maxGPUThreads;
	extern grid_t maxGPUBlocks;
	extern grid_t maxGPUBlocks2D;
	extern grid_t maxWarpSize;
	extern size_t maxGPUSharedMem;

	// Kernel configuration parameters.
	// If they are set to default (1), tuner will be triggered.
	extern dim3 bestBlockReset, bestGridReset;
	extern dim3 bestBlockIdentity, bestGridIdentity;
	extern dim3 bestBlockCheckIdentity, bestGridCheckIdentity;
	extern dim3 bestBlockStep, bestGridStep;
	extern dim3 bestBlockMeasure, bestGridMeasure;

	// x
	#define global_bx		((grid_t)blockDim.x * blockIdx.x)
	#define global_bx_off	(((grid_t)blockDim.x << 1) * blockIdx.x)
	#define global_tx		(global_bx + threadIdx.x)
	#define global_tx_off	(global_bx_off + threadIdx.x)
	#define stride_x        ((grid_t)blockDim.x * gridDim.x)
	#define stride_x_off	(((grid_t)blockDim.x << 1) * gridDim.x)
	// y
	#define global_by		((grid_t)blockDim.y * blockIdx.y)
	#define global_ty		(global_by + threadIdx.y)
	#define stride_y		((grid_t)blockDim.y * gridDim.y)

	#define for_parallel_x(IDX, SIZE) \
		for (grid_t IDX = global_tx, stride = stride_x, data_size = grid_t(SIZE); IDX < data_size; IDX += stride)

	#define for_parallel_y(IDX, SIZE) \
		for (grid_t IDX = global_ty, stride = stride_y, data_size = grid_t(SIZE); IDX < data_size; IDX += stride)

	#define for_parallel_y_off(IDX, OFF, SIZE) \
		for (grid_t IDX = global_ty + OFF, stride = stride_y, data_size = grid_t(SIZE); IDX < data_size; IDX += stride)

	// macros for blocks calculation
	#define ROUNDUPBLOCKS(DATALEN, NTHREADS) ((grid_t(DATALEN) + (NTHREADS) - 1) / (NTHREADS))

	#define OPTIMIZEBLOCKS(NBLOCKS, DATALEN, NTHREADS)           \
			assert(DATALEN);                                     \
			assert(NTHREADS);                                    \
			assert(maxGPUBlocks);                                \
			grid_t NBLOCKS = ROUNDUPBLOCKS(DATALEN, NTHREADS); 	 \
			NBLOCKS = MIN(NBLOCKS, maxGPUBlocks)  		 		 \

	#define OPTIMIZEBLOCKS2D(NBLOCKS, DATALEN, NTHREADS)         \
			assert(DATALEN);                                     \
			assert(NTHREADS);                                    \
			assert(maxGPUBlocks2D);                              \
			grid_t NBLOCKS = ROUNDUPBLOCKS(DATALEN, NTHREADS); 	 \
			NBLOCKS = MIN(NBLOCKS, maxGPUBlocks2D)  		     \

	// macros for shared memory calculation
    #define OPTIMIZESHARED(SMEMSIZE, NTHREADS, MINCAP)       \
            assert(MINCAP);                                  \
            assert(NTHREADS);                                \
            assert(maxGPUSharedMem);                         \
            const size_t SMEMSIZE = (NTHREADS) * (MINCAP);   \
            assert(maxGPUSharedMem >= SMEMSIZE)              \

}

#endif