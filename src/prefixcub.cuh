
#ifndef __CU_PREFIXCUB_H
#define __CU_PREFIXCUB_H

#include "prefix.cuh"
#include "access.cuh"
#include "cub/block/block_scan.cuh"

namespace QuaSARQ {

    template <int BLOCKX, int BLOCKY>
    __global__ 
    void scan_blocks_single_pass(
                word_std_t* block_intermediate_prefix_z, 
                word_std_t* block_intermediate_prefix_x,
        const   size_t      num_chunks,
        const   size_t      num_words_minor,
        const   size_t      max_blocks) {

        typedef cub::BlockScan<word_std_t, BLOCKX, cub::BLOCK_SCAN_RAKING> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_z[BLOCKY];
        __shared__ typename BlockScan::TempStorage temp_storage_x[BLOCKY];

        for_parallel_y_tiled(by, num_words_minor) {

            const grid_t w = threadIdx.y + by * blockDim.y;

            word_std_t z = 0;
            word_std_t x = 0;

            if (w < num_words_minor && threadIdx.x < num_chunks) {
                const size_t bid = w * max_blocks + threadIdx.x;
                z = block_intermediate_prefix_z[bid];
                x = block_intermediate_prefix_x[bid];
            }

            BlockScan(temp_storage_z[threadIdx.y]).ExclusiveScan(z, z, 0, XOROP());
            BlockScan(temp_storage_x[threadIdx.y]).ExclusiveScan(x, x, 0, XOROP());

            if (w < num_words_minor && threadIdx.x < num_chunks) {
                const size_t bid = w * max_blocks + threadIdx.x;
                block_intermediate_prefix_z[bid] = z;
                block_intermediate_prefix_x[bid] = x;
            }

        }
    }

    template <int BLOCKX, int BLOCKY>
    __global__ 
    void scan_blocks_pass_1(
		        word_std_t*     block_intermediate_prefix_z,
		        word_std_t*     block_intermediate_prefix_x,
		        word_std_t*     subblocks_prefix_z, 
		        word_std_t*     subblocks_prefix_x, 
		const   size_t          num_blocks,
		const   size_t          num_words_minor,
        const   size_t          max_blocks,
		const   size_t          max_sub_blocks
	)
	{
        typedef cub::BlockScan<word_std_t, BLOCKX, cub::BLOCK_SCAN_RAKING> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_z[BLOCKY];
        __shared__ typename BlockScan::TempStorage temp_storage_x[BLOCKY];

		for_parallel_y_tiled(by, num_words_minor) {

            const grid_t w = threadIdx.y + by * blockDim.y;
			
			for_parallel_x_tiled(bx, num_blocks) {

                const grid_t tid_x = threadIdx.x + bx * blockDim.x;
                
                word_std_t z = 0;
                word_std_t x = 0;

                if (w < num_words_minor && tid_x < num_blocks) {
                    const size_t bid = w * max_blocks + tid_x;
                    z = block_intermediate_prefix_z[bid];
                    x = block_intermediate_prefix_x[bid];
                }

                word_std_t blockSum_z;
                word_std_t blockSum_x;

                BlockScan(temp_storage_z[threadIdx.y]).ExclusiveScan(z, z, 0, XOROP(), blockSum_z);
                BlockScan(temp_storage_x[threadIdx.y]).ExclusiveScan(x, x, 0, XOROP(), blockSum_x);

                if (w < num_words_minor && tid_x < num_blocks) {
                    const size_t bid = w * max_blocks + tid_x;
                    block_intermediate_prefix_z[bid] = z;
                    block_intermediate_prefix_x[bid] = x;
                }

                if (w < num_words_minor && threadIdx.x == blockDim.x - 1) {
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    grid_t sub_bid = w * max_sub_blocks + bx;
                    subblocks_prefix_z[sub_bid] = blockSum_z;
                    subblocks_prefix_x[sub_bid] = blockSum_x;
                }
            }
		}
	}

    void call_single_pass_kernel(
                word_std_t *        intermediate_prefix_z,
                word_std_t *        intermediate_prefix_x,
        const   size_t              num_chunks,
        const   size_t              num_words_minor,
        const   size_t              max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);

    void call_scan_blocks_pass_1_kernel(
                word_std_t*     block_intermediate_prefix_z,
                word_std_t*     block_intermediate_prefix_x,
                word_std_t*     subblocks_prefix_z, 
                word_std_t*     subblocks_prefix_x, 
        const   size_t          num_blocks,
        const   size_t          num_words_minor,
        const   size_t          max_blocks,
        const   size_t          max_sub_blocks,
        const   dim3&           currentblock,
        const   dim3&           currentgrid,
        const   cudaStream_t&   stream);

    void tune_single_pass(
                dim3&       bestBlock, 
                dim3&       bestGrid,
        const   size_t&     shared_element_bytes, 
        const   size_t&     data_size_in_x, 
        const   size_t&     data_size_in_y,
                word_std_t* block_intermediate_prefix_z, 
                word_std_t* block_intermediate_prefix_x,
        const   size_t      num_chunks,
        const   size_t      num_words_minor,
        const   size_t      max_blocks);

    void tune_prefix_pass_1(
                dim3&       bestBlock, 
                dim3&       bestGrid,
        const   size_t&     shared_element_bytes, 
        const   size_t&     data_size_in_x, 
        const   size_t&     data_size_in_y,
                word_std_t* block_intermediate_prefix_z,
                word_std_t* block_intermediate_prefix_x,
                word_std_t* subblocks_prefix_z, 
                word_std_t* subblocks_prefix_x,
        const   size_t&     num_blocks,
        const   size_t&     num_words_minor,
        const   size_t&     max_blocks,
        const   size_t&     max_sub_blocks);

    #define CALL_SINGLE_PASS_FOR_BLOCK(X, Y) \
        scan_blocks_single_pass <X, Y> \
        <<<currentgrid, currentblock, 0, stream>>> ( \
            intermediate_prefix_z, \
            intermediate_prefix_x, \
            num_chunks, \
            num_words_minor, \
            max_blocks \
        )

    #define CALL_PREFIX_PASS_1_FOR_BLOCK(X, Y) \
        scan_blocks_pass_1 <X, Y> \
        <<<currentgrid, currentblock, 0, stream>>> ( \
            block_intermediate_prefix_z, \
            block_intermediate_prefix_x, \
            subblocks_prefix_z, \
            subblocks_prefix_x, \
            num_blocks, \
            num_words_minor, \
            max_blocks, \
            max_sub_blocks \
        )

}

#endif