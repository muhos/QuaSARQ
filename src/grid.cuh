#pragma once

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

	#define for_parallel_x_tiled(BIDX, SIZE) \
		for (grid_t BIDX = blockIdx.x, MAX_BLOCKS_X = ROUNDUP(SIZE, blockDim.x); BIDX < MAX_BLOCKS_X; BIDX += gridDim.x)

	#define for_parallel_y(IDX, SIZE) \
		for (grid_t IDX = global_ty, stride = stride_y, data_size = grid_t(SIZE); IDX < data_size; IDX += stride)

	#define for_parallel_y_off(IDX, OFF, SIZE) \
		for (grid_t IDX = global_ty + OFF, stride = stride_y, data_size = grid_t(SIZE); IDX < data_size; IDX += stride)

	#define for_parallel_y_tiled(BIDX, SIZE) \
		for (grid_t BIDX = blockIdx.y, MAX_BLOCKS_Y = ROUNDUP(SIZE, blockDim.y); BIDX < MAX_BLOCKS_Y; BIDX += gridDim.y)

	// macros for blocks calculation
	#define ROUNDUP(DATALEN, DIVISOR) ((grid_t(DATALEN) + (DIVISOR) - 1) / (DIVISOR))

	#define OPTIMIZEBLOCKS(NBLOCKS, DATALEN, NTHREADS)       \
			assert(DATALEN);                                 \
			assert(NTHREADS);                                \
			assert(maxGPUBlocks);                            \
			NBLOCKS = ROUNDUP(DATALEN, NTHREADS); 	 \
			NBLOCKS = MIN(NBLOCKS, maxGPUBlocks)  		 	 \

	#define OPTIMIZEBLOCKS2D(NBLOCKS, DATALEN, NTHREADS)     \
			assert(DATALEN);                                 \
			assert(NTHREADS);                                \
			assert(maxGPUBlocks2D);                          \
			NBLOCKS = ROUNDUP(DATALEN, NTHREADS); 	 \
			NBLOCKS = MIN(NBLOCKS, maxGPUBlocks2D)  		 \

	// macros for shared memory calculation
    #define OPTIMIZESHARED(SMEMSIZE, NTHREADS, MINCAP)       \
            assert(MINCAP);                                  \
            assert(NTHREADS);                                \
            assert(maxGPUSharedMem);                         \
            const size_t SMEMSIZE = (NTHREADS) * (MINCAP);   \
            assert(maxGPUSharedMem >= SMEMSIZE)              \

	#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
        #define TRIM_BLOCK_IN_DEBUG_MODE(BLOCK, GRID, DATALEN_X, DATALEN_Y) \
            if (BLOCK.x * BLOCK.y >= 512) { \
				if (BLOCK.y == 1) { \
					BLOCK.x = MIN(64, BLOCK.x); \
					OPTIMIZEBLOCKS(GRID.x, DATALEN_X, BLOCK.x); \
				} \
				else if (BLOCK.x <= 2) \
					BLOCK.y = MIN(64, BLOCK.y); \
				else { \
					BLOCK.x = MIN(16, BLOCK.x); \
					BLOCK.y = MIN(16, BLOCK.y); \
					OPTIMIZEBLOCKS2D(GRID.x, DATALEN_X, BLOCK.x); \
					OPTIMIZEBLOCKS2D(GRID.y, DATALEN_Y, BLOCK.y); \
				} \
            }
		#define TRIM_Y_BLOCK_IN_DEBUG_MODE(BLOCK, GRID, DATALEN_Y) \
            if (BLOCK.x * BLOCK.y >= 512) { \
				BLOCK.y = (512 / BLOCK.x) / 2; \
				BLOCK.y = MAX(1, BLOCK.y); \
				OPTIMIZEBLOCKS2D(GRID.y, DATALEN_Y, BLOCK.y); \
            }
    #else
        #define TRIM_BLOCK_IN_DEBUG_MODE(BLOCK, GRID, DATALEN_X, DATALEN_Y)
		#define TRIM_Y_BLOCK_IN_DEBUG_MODE(BLOCK, GRID, DATALEN_Y)
    #endif

    #define TRIM_GRID_IN_1D(DATALEN, DIM) \
        if (config_qubits > num_qubits) { \
            if (size_t(currentgrid.DIM) * size_t(currentblock.DIM) > (DATALEN)) { \
                OPTIMIZEBLOCKS(currentgrid.DIM, (DATALEN), currentblock.DIM); \
            } \
        }

	#define TRIM_GRID_IN_2D(BLOCK, GRID, DATALEN, DIM) \
        if (config_qubits > num_qubits) { \
            if (size_t(GRID.DIM) * size_t(BLOCK.DIM) > (DATALEN)) { \
                OPTIMIZEBLOCKS2D(GRID.DIM, (DATALEN), BLOCK.DIM); \
            } \
        }

    #define TRIM_GRID_IN_XY(DATALEN_X, DATALEN_Y) \
        if (config_qubits > num_qubits) { \
            if (size_t(currentgrid.x) * size_t(currentblock.x) > (DATALEN_X)) { \
                OPTIMIZEBLOCKS2D(currentgrid.x, (DATALEN_X), currentblock.x); \
            } \
            if (size_t(currentgrid.y) * size_t(currentblock.y) > (DATALEN_Y)) { \
                OPTIMIZEBLOCKS2D(currentgrid.y, (DATALEN_Y), currentblock.y); \
            } \
        }

	#define FORCE_TRIM_GRID_IN_XY(DATALEN_X, DATALEN_Y) \
		if (size_t(currentgrid.x) * size_t(currentblock.x) > (DATALEN_X)) { \
			OPTIMIZEBLOCKS2D(currentgrid.x, (DATALEN_X), currentblock.x); \
		} \
		if (size_t(currentgrid.y) * size_t(currentblock.y) > (DATALEN_Y)) { \
			OPTIMIZEBLOCKS2D(currentgrid.y, (DATALEN_Y), currentblock.y); \
		} \

}