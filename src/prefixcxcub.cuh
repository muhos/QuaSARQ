
#ifndef __CU_PREFIXCXCUB_H
#define __CU_PREFIXCXCUB_H

#include "prefix.cuh"
#include "access.cuh"
#include "cub/block/block_scan.cuh"

namespace QuaSARQ {

    template <int BLOCKX, int BLOCKY>
    __global__ 
    void scan_targets_pass_1(
                Table *             prefix_xs, 
                Table *             prefix_zs, 
                Table *             inv_xs, 
                Table *             inv_zs,
                word_std_t *        block_intermediate_prefix_z,
                word_std_t *        block_intermediate_prefix_x,
        const   Commutation *       commutations,
        const   uint32              pivot,
        const   size_t              total_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks) {
            
        typedef cub::BlockScan<word_std_t, BLOCKX, cub::BLOCK_SCAN_RAKING> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_z[BLOCKY];
        __shared__ typename BlockScan::TempStorage temp_storage_x[BLOCKY];

        for_parallel_y_tiled(by, num_words_minor) {
            const grid_t w = threadIdx.y + by * blockDim.y;

            for_parallel_x_tiled(bx, total_targets) {
                const grid_t tid_x = threadIdx.x + bx * blockDim.x;
                
                word_std_t z = 0;
                word_std_t x = 0;
                word_std_t init_z = 0;
                word_std_t init_x = 0;

                if (w < num_words_minor && tid_x < total_targets) {
                    const size_t t = tid_x + pivot + 1;
                    if (commutations[t].anti_commuting) {
                        const size_t t_destab = TABLEAU_INDEX(w, t);
                        z = (*inv_zs)[t_destab];
                        x = (*inv_xs)[t_destab];
                        const size_t c_destab = TABLEAU_INDEX(w, pivot);
                        init_z = (*inv_zs)[c_destab];
                        init_x = (*inv_xs)[c_destab];
                    }
                }

                word_std_t blockSum_z;
                word_std_t blockSum_x;

                BlockScan(temp_storage_z[threadIdx.y]).ExclusiveScan(z, z, 0, XOROP(), blockSum_z);
                BlockScan(temp_storage_x[threadIdx.y]).ExclusiveScan(x, x, 0, XOROP(), blockSum_x);

                if (w < num_words_minor && tid_x < total_targets) {
                    const size_t word_idx = PREFIX_TABLEAU_INDEX(w, tid_x);
                    assert(word_idx < prefix_zs->size());
                    assert(word_idx < prefix_xs->size());
                    (*prefix_zs)[word_idx] = init_z ^ z;
                    (*prefix_xs)[word_idx] = init_x ^ x;
                }

                if (w < num_words_minor && threadIdx.x == blockDim.x - 1) {
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    const size_t bid = w * max_blocks + bx;
                    block_intermediate_prefix_z[bid] = blockSum_z;
                    block_intermediate_prefix_x[bid] = blockSum_x;
                }
            }
        }
    }

    void call_injectcx_pass_1_kernel(
                Tableau<DeviceAllocator>& targets, 
                Tableau<DeviceAllocator>& input,
                word_std_t *        block_intermediate_prefix_z,
                word_std_t *        block_intermediate_prefix_x,
        const   Commutation *       commutations,
        const   uint32              pivot,
        const   size_t              total_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);

    void tune_inject_pass_1(
		        dim3&           bestBlock, 
                dim3&           bestGrid,
		const   size_t&         shared_element_bytes, 
		const   size_t&         data_size_in_x, 
		const   size_t&         data_size_in_y,
		        Tableau<DeviceAllocator>& targets, 
		        Tableau<DeviceAllocator>& input, 
                word_std_t *    block_intermediate_prefix_z,
                word_std_t *    block_intermediate_prefix_x,
		const   Commutation*    commutations,
		const   uint32&         pivot,
		const   size_t&         total_targets,
		const   size_t&         num_words_major,
		const   size_t&         num_words_minor,
		const   size_t&         num_qubits_padded,
		const   size_t&         max_blocks);

    #define CALL_INJECTCX_PASS_1_FOR_BLOCK(X, Y) \
        scan_targets_pass_1 <X, Y> \
        <<<currentgrid, currentblock, 0, stream>>> ( \
                XZ_TABLE(targets), \
                XZ_TABLE(input), \
                block_intermediate_prefix_z, \
                block_intermediate_prefix_x, \
                commutations, \
                pivot, \
                total_targets, \
                num_words_major, \ 
                num_words_minor, \
                num_qubits_padded, \
                max_blocks \
            )

}

#endif