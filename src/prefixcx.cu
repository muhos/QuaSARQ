
#include "prefix.cuh"
#include "collapse.cuh"
#include "access.cuh"
#include "vector.hpp"
#include "print.cuh"
#include "templatedim.cuh"
#include "datatypes.cuh"
#include "warp.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

namespace QuaSARQ {


    template <int BLOCKX, int BLOCKY>
    __global__ 
    void scan_targets_pass_1(
                Table *             prefix_xs, 
                Table *             prefix_zs, 
                word_std_t *        block_intermediate_prefix_z,
                word_std_t *        block_intermediate_prefix_x,
                Table*              inv_xs, 
                Table*              inv_zs,
                const_pivots_t      pivots,
        const   size_t              active_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks) {

        assert(active_targets > 0);
        assert(BLOCKX == blockDim.x);
        assert(BLOCKY == blockDim.y);

        word_std_t * __restrict__ xs = inv_xs->words();
        word_std_t * __restrict__ zs = inv_zs->words();

        typedef cub::BlockScan<word_std_t, BLOCKX> BlockScan;

        __shared__ typename BlockScan::TempStorage shared_prefix_zs[BLOCKY];
        __shared__ typename BlockScan::TempStorage shared_prefix_xs[BLOCKY];

        for_parallel_y_tiled(by, num_words_minor) {
            const grid_t w = threadIdx.y + by * BLOCKY;
    
            for_parallel_x_tiled(bx, active_targets) {
                const grid_t tid_x = threadIdx.x + bx * BLOCKX;
                bool active = (w < num_words_minor && tid_x < active_targets);

                const pivot_t pivot = __ldg(&pivots[0]);
                const size_t c_destab = TABLEAU_INDEX(w, pivot);

                pivot_t t = active ? __ldg(&pivots[tid_x + 1]) : pivot;
                assert(t != INVALID_PIVOT);
                const size_t t_destab = TABLEAU_INDEX(w, t);
                word_std_t z = active ? zs[t_destab] : 0;
                word_std_t x = active ? xs[t_destab] : 0;

                word_std_t blocksum_z, blocksum_x;
                BlockScan(shared_prefix_zs[threadIdx.y]).ExclusiveScan(z, z, 0, XOROP(), blocksum_z);
                BlockScan(shared_prefix_xs[threadIdx.y]).ExclusiveScan(x, x, 0, XOROP(), blocksum_x);

                const size_t word_idx = PREFIX_TABLEAU_INDEX(w, tid_x);
                (*prefix_zs)[word_idx] = active ? zs[c_destab] ^ z : 0;
                (*prefix_xs)[word_idx] = active ? xs[c_destab] ^ x : 0;

                if (w < num_words_minor && threadIdx.x == BLOCKX - 1) {
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, bx);
                    block_intermediate_prefix_z[bid] = blocksum_z;
                    block_intermediate_prefix_x[bid] = blocksum_x;
                    if (blocksum_z)
                        atomicXOR(zs + c_destab, blocksum_z);
                    if (blocksum_x)
                        atomicXOR(xs + c_destab, blocksum_x);
                }
            }
        }
    }

    #define CALL_INJECTCX_PASS_1_FOR_BLOCK(X, Y) \
        scan_targets_pass_1 <X, Y> \
        <<<currentgrid, currentblock, 0, stream>>> ( \
                XZ_TABLE(targets), \
                block_intermediate_prefix_z, \
                block_intermediate_prefix_x, \
                XZ_TABLE(input), \
                pivots, \
                active_targets, \
                num_words_major, \
                num_words_minor, \
                num_qubits_padded, \
                max_blocks \
            )

    template <int BLOCKX, int BLOCKY>
    __global__ 
    void scan_targets_pass_2(
                Table *             inv_xs, 
                Table *             inv_zs,
                Signs *             inv_ss,
                const_table_t       prefix_xs, 
                const_table_t       prefix_zs, 
                const_words_t       block_intermediate_prefix_z,
                const_words_t       block_intermediate_prefix_x,
                const_pivots_t      pivots,
        const   size_t              active_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks,
        const   size_t              pass_1_blocksize)
    { 
        assert(BLOCKX == blockDim.x);
        assert(BLOCKY == blockDim.y);
        word_std_t * __restrict__ xs = inv_xs->words();
        word_std_t * __restrict__ zs = inv_zs->words();
        sign_t*  __restrict__ ss = inv_ss->data();

        typedef cub::BlockReduce<sign_t, BLOCKX> BlockReduce;

        __shared__ typename BlockReduce::TempStorage shared_destab_ss[BLOCKY];
        __shared__ typename BlockReduce::TempStorage shared_stab_ss  [BLOCKY];

        for_parallel_y(w, num_words_minor) {

            sign_t local_destab_sign = 0;
            sign_t local_stab_sign = 0;

            for_parallel_x(tid_x, active_targets) {
                const pivot_t pivot = __ldg(&pivots[0]);
                assert(pivot != INVALID_PIVOT);
                const size_t t = __ldg(&pivots[tid_x + 1]);
                assert(t != pivot);
                assert(t != INVALID_PIVOT);

                const size_t c_stab = TABLEAU_INDEX(w, pivot) + TABLEAU_STAB_OFFSET;
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;

                const size_t word_idx = PREFIX_TABLEAU_INDEX(w, tid_x);
                word_std_t zc_xor_prefix = (*prefix_zs)[word_idx];
                word_std_t xc_xor_prefix = (*prefix_xs)[word_idx];

                // Compute final prefixes and hence final {x,z}'c = {x,z}'c ^ {x,z}'t expressions.
                const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, (tid_x / pass_1_blocksize));
                zc_xor_prefix ^= block_intermediate_prefix_z[bid];
                xc_xor_prefix ^= block_intermediate_prefix_x[bid];

                compute_local_sign_per_block(local_destab_sign, zs[t_stab], zc_xor_prefix, zs[c_stab], zs[t_destab]);
                compute_local_sign_per_block(local_stab_sign, xs[t_stab], xc_xor_prefix, xs[c_stab], xs[t_destab]);
            }

            sign_t block_destab_sign = BlockReduce(shared_destab_ss[threadIdx.y]).Reduce(local_destab_sign, XOROP());
            sign_t block_stab_sign   = BlockReduce(shared_stab_ss  [threadIdx.y]).Reduce(local_stab_sign,   XOROP());

            if (!threadIdx.x) {
                if (block_destab_sign)
                    atomicXOR(ss + w, block_destab_sign);
                if (block_stab_sign)
                    atomicXOR(ss + w + num_words_minor, block_stab_sign);
            }
        }
    }

    #define CALL_INJECTCX_PASS_2_FOR_BLOCK(X, Y) \
        scan_targets_pass_2 <X, Y> \
        <<<currentgrid, currentblock, 0, stream>>> ( \
                XZ_TABLE(input), \
                input.signs(), \
                XZ_TABLE(targets), \
                block_intermediate_prefix_z, \
                block_intermediate_prefix_x, \
                pivots, \
                active_targets, \
                num_words_major, \
                num_words_minor, \
                num_qubits_padded, \
                max_blocks, \
                pass_1_blocksize\
            )

	void call_injectcx_pass_1_kernel(
                Tableau& 			targets, 
                Tableau& 			input,
                word_std_t *        block_intermediate_prefix_z,
                word_std_t *        block_intermediate_prefix_x,
        const   pivot_t*            pivots,
        const   size_t&             active_targets,
        const   size_t&             num_words_major,
        const   size_t&             num_words_minor,
        const   size_t&             num_qubits_padded,
        const   size_t&             max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream) {
        
        GENERATE_SWITCH_FOR_CALL(CALL_INJECTCX_PASS_1_FOR_BLOCK)
    }

	void call_injectcx_pass_2_kernel(
                Tableau& 			targets, 
                Tableau& 			input,
                const_words_t   block_intermediate_prefix_z,
                const_words_t   block_intermediate_prefix_x,
        const   pivot_t*            pivots,
        const   size_t&             active_targets,
        const   size_t&             num_words_major,
        const   size_t&             num_words_minor,
        const   size_t&             num_qubits_padded,
        const   size_t&             max_blocks,
        const   size_t&             pass_1_blocksize,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream) {
        
        GENERATE_SWITCH_FOR_CALL(CALL_INJECTCX_PASS_2_FOR_BLOCK)
    }

    // We need to compute prefix-xor of t-th destabilizer in X,Z for t = c+1, c+2, ... c+n-1
    // so that later we can xor every prefix-xor with controlled destabilizer.
    void Prefix::scan_large(Tableau& input, const pivot_t* pivots, const size_t& active_targets, const cudaStream_t& stream) {
        assert(nextPow2(MIN_BLOCK_INTERMEDIATE_SIZE) == MIN_BLOCK_INTERMEDIATE_SIZE);
        const size_t num_qubits_padded = input.num_qubits_padded();

        // Do the first phase of prefix.
        dim3 currentblock, currentgrid;
        if (bestblockinjectprepare.x == 1)
            LOGERROR("x-block size in inject-cx is 1");
        TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblockinjectprepare, bestgridinjectprepare, num_words_minor);
        currentblock = bestblockinjectprepare, currentgrid = bestgridinjectprepare;
        FORCE_TRIM_GRID_IN_XY(active_targets, num_words_minor);
        const size_t pass_1_blocksize = currentblock.x;
        const size_t pass_1_gridsize = ROUNDUP(active_targets, pass_1_blocksize);
        if (pass_1_gridsize > max_intermediate_blocks)
            LOGERROR("too many blocks for intermediate arrays");
        LOGN2(2, " Running pass-1 kernel for %d targets with block(x:%u, y:%u) and grid(x:%u, y:%u).. ",
            active_targets, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        if (options.sync) cutimer.start(stream);
        call_injectcx_pass_1_kernel(
            targets, 
            input, 
            zblocks(), 
            xblocks(),
            pivots,
            active_targets, 
            num_words_major, 
            num_words_minor,
            num_qubits_padded,
            max_intermediate_blocks,
            currentblock,
            currentgrid,
            stream
        );
        if (options.sync) {
            LASTERR("failed to scan targets in pass 1");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        } else LOGDONE(2, 4);

        // Verify pass-1 prefix.
        if (options.check_measurement) {
            checker.check_prefix_pass_1(
                targets,
                pivots,
                zblocks(), 
                xblocks(),
                active_targets,
                max_intermediate_blocks,
                pass_1_blocksize,
                pass_1_gridsize);
        }

        // Intermeditae scan of blocks resulted in pass 1.
        scan_blocks(nextPow2(pass_1_gridsize), pass_1_blocksize, stream);

        // Verify intermediate-pass prefix.
        if (options.check_measurement) {
            checker.check_prefix_intermediate_pass(
                zblocks(), 
                xblocks(),
                max_intermediate_blocks,
                pass_1_gridsize);
        }

        // Second phase of injecting CX.
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockinjectfinal, bestgridinjectfinal, active_targets, num_words_minor);
        currentblock = bestblockinjectfinal, currentgrid = bestgridinjectfinal;
        if (currentblock.x > active_targets) {
            currentblock.x = active_targets == 1 ? 2 : MIN(currentblock.x, nextPow2(active_targets));
        }
        FORCE_TRIM_GRID_IN_XY(active_targets, num_words_minor);
        LOGN2(2, " Running pass-2 kernel for %d targets with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", \
            active_targets, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y); \
        if (options.sync) cutimer.start(stream);
        call_injectcx_pass_2_kernel(
            targets, 
            input,
            zblocks(), 
            xblocks(), 
            pivots, 
            active_targets, 
            num_words_major, 
            num_words_minor, 
            num_qubits_padded,
            max_intermediate_blocks,
            pass_1_blocksize,
            currentblock,
            currentgrid,
            stream
        );
        if (options.sync) {
            LASTERR("failed to scan targets in pass 2");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        } else LOGDONE(2, 4);

        // Verify pass-2 prefix.
        if (options.check_measurement) {
            checker.check_prefix_pass_2(
                targets, 
                input,
                active_targets, 
                max_intermediate_blocks,
                pass_1_blocksize);
        }
    }

    void Prefix::tune_inject_cx(Tableau& input, const pivot_t* pivots, const size_t& max_active_targets) {
        assert(nextPow2(MIN_BLOCK_INTERMEDIATE_SIZE) == MIN_BLOCK_INTERMEDIATE_SIZE);
        const size_t num_qubits_padded = input.num_qubits_padded();

        // Do the first phase of prefix.
        if (options.tune_injectprepare) {
            SYNCALL;
            tune_inject_pass_1(
                bestblockinjectprepare, bestgridinjectprepare,
                2 * sizeof(word_std_t), // used to skip very large blocks.
                max_active_targets,
                num_words_minor,
                targets, 
                input, 
                zblocks(), 
                xblocks(),
                pivots,
                max_active_targets, 
                num_words_major, 
                num_words_minor,
                num_qubits_padded,
                max_intermediate_blocks
            );
            SYNCALL;
        }

        const size_t pass_1_blocksize = bestblockinjectprepare.x;
        const size_t pass_1_gridsize = ROUNDUP(max_active_targets, pass_1_blocksize);
        tune_scan_blocks(nextPow2(pass_1_gridsize), pass_1_blocksize);

        if (options.tune_injectfinal) {
            SYNCALL;
            tune_inject_pass_2(
                bestblockinjectfinal, bestgridinjectfinal,
                2 * sizeof(word_std_t),
                max_active_targets,
                num_words_minor,
                targets, 
                input,
                zblocks(), 
                xblocks(), 
                pivots, 
                max_active_targets, 
                num_words_major, 
                num_words_minor, 
                num_qubits_padded,
                max_intermediate_blocks,
                pass_1_blocksize
            );
            SYNCALL;
        }
    }

}