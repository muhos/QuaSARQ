
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
                ConstTablePointer   inv_xs, 
                ConstTablePointer   inv_zs,
        const   pivot_t*            pivots,
        const   size_t              active_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks) {

        assert(active_targets > 0);

        typedef cub::BlockScan<word_std_t, BLOCKX, cub::BLOCK_SCAN_RAKING> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_z[BLOCKY];
        __shared__ typename BlockScan::TempStorage temp_storage_x[BLOCKY];

        for_parallel_y_tiled(by, num_words_minor) {
            const grid_t w = threadIdx.y + by * blockDim.y;

            for_parallel_x_tiled(bx, active_targets) {
                const grid_t tid_x = threadIdx.x + bx * blockDim.x;
                
                word_std_t z = 0;
                word_std_t x = 0;
                word_std_t init_z = 0;
                word_std_t init_x = 0;

                if (w < num_words_minor && tid_x < active_targets) {
                    const pivot_t pivot = pivots[0];
                    assert(pivot != INVALID_PIVOT);
                    const size_t t = pivots[tid_x + 1];
                    assert(t != pivot);
                    assert(t != INVALID_PIVOT);
                    const size_t t_destab = TABLEAU_INDEX(w, t);
                    z = (*inv_zs)[t_destab];
                    x = (*inv_xs)[t_destab];
                    const size_t c_destab = TABLEAU_INDEX(w, pivot);
                    init_z = (*inv_zs)[c_destab];
                    init_x = (*inv_xs)[c_destab];
                }

                word_std_t blockSum_z;
                word_std_t blockSum_x;

                BlockScan(temp_storage_z[threadIdx.y]).ExclusiveScan(z, z, 0, XOROP(), blockSum_z);
                BlockScan(temp_storage_x[threadIdx.y]).ExclusiveScan(x, x, 0, XOROP(), blockSum_x);

                if (w < num_words_minor && tid_x < active_targets) {
                    const size_t word_idx = PREFIX_TABLEAU_INDEX(w, tid_x);
                    assert(word_idx < prefix_zs->size());
                    assert(word_idx < prefix_xs->size());
                    (*prefix_zs)[word_idx] = init_z ^ z;
                    (*prefix_xs)[word_idx] = init_x ^ x;
                }

                if (w < num_words_minor && threadIdx.x == blockDim.x - 1) {
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, bx);
                    block_intermediate_prefix_z[bid] = blockSum_z;
                    block_intermediate_prefix_x[bid] = blockSum_x;
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
                ConstTablePointer   prefix_xs, 
                ConstTablePointer   prefix_zs, 
                ConstWordsPointer   block_intermediate_prefix_z,
                ConstWordsPointer   block_intermediate_prefix_x,
        const   pivot_t*            pivots,
        const   size_t              active_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks,
        const   size_t              pass_1_blocksize)
    { 
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();

        for_parallel_y(w, num_words_minor) {

            word_std_t zc_destab = 0;
            word_std_t xc_destab = 0;
            word_std_t xc_and_zt = 0;
            word_std_t not_zc_xor_xt = 0;
            word_std_t local_destab_sign = 0;
            word_std_t local_stab_sign = 0;

            for_parallel_x(tid_x, active_targets) {
                const pivot_t pivot = pivots[0];
                assert(pivot != INVALID_PIVOT);
                const size_t t = pivots[tid_x + 1];
                assert(t != pivot);
                assert(t != INVALID_PIVOT);

                const size_t c_destab = TABLEAU_INDEX(w, pivot);
                const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;

                assert(c_destab < inv_zs->size());
                assert(t_destab < inv_zs->size());

                const size_t word_idx = PREFIX_TABLEAU_INDEX(w, tid_x);
                word_std_t zc_xor_zt = (*prefix_zs)[word_idx];
                word_std_t xc_xor_xt = (*prefix_xs)[word_idx];

                // Compute final prefixes and hence final {x,z}'c = {x,z}'c ^ {x,z}'t expressions.
                const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, (tid_x / pass_1_blocksize));
                zc_xor_zt ^= block_intermediate_prefix_z[bid];
                xc_xor_xt ^= block_intermediate_prefix_x[bid];

                // Compute the CX expression for Z.
                word_std_t c_stab_word = zs[c_stab];
                word_std_t t_destab_word = zs[t_destab];
                xc_and_zt = (c_stab_word & t_destab_word);
                not_zc_xor_xt = ~(zc_xor_zt ^ zs[t_stab]);
                local_destab_sign ^= xc_and_zt & not_zc_xor_xt;
                
                // Update Z tableau.
                zs[t_stab] ^= c_stab_word;
                zc_destab ^= t_destab_word; // requires collapse.

                // Compute the CX expression for X.
                c_stab_word = xs[c_stab];
                t_destab_word = xs[t_destab];
                xc_and_zt = (c_stab_word & t_destab_word);
                not_zc_xor_xt = ~(xc_xor_xt ^ xs[t_stab]);
                local_stab_sign ^= xc_and_zt & not_zc_xor_xt;

                // Update X tableau.
                xs[t_stab] ^= c_stab_word;
                xc_destab ^= t_destab_word; // requires collapse.
            }

            typedef cub::BlockReduce<word_std_t, BLOCKX> BlockReduce;

            __shared__ typename BlockReduce::TempStorage temp_storage_zc[BLOCKY];
            __shared__ typename BlockReduce::TempStorage temp_storage_xc[BLOCKY];
            __shared__ typename BlockReduce::TempStorage temp_storage_destab_sign[BLOCKY];
            __shared__ typename BlockReduce::TempStorage temp_storage_stab_sign[BLOCKY];

            word_std_t block_zc_destab = BlockReduce(temp_storage_zc[threadIdx.y]).Reduce(zc_destab, XOROP());
            word_std_t block_xc_destab = BlockReduce(temp_storage_xc[threadIdx.y]).Reduce(xc_destab, XOROP());
            word_std_t block_local_destab_sign = BlockReduce(temp_storage_destab_sign[threadIdx.y]).Reduce(local_destab_sign, XOROP());
            word_std_t block_local_stab_sign = BlockReduce(temp_storage_stab_sign[threadIdx.y]).Reduce(local_stab_sign, XOROP());

            if (!threadIdx.x) {
                const pivot_t pivot = pivots[0];
                const size_t c_destab = TABLEAU_INDEX(w, pivot);
                if (block_zc_destab)
                    atomicXOR(zs + c_destab, block_zc_destab);
                if (block_xc_destab)
                    atomicXOR(xs + c_destab, block_xc_destab);
                if (block_local_destab_sign)
                    atomicXOR(inv_ss->data(w), block_local_destab_sign);
                if (block_local_stab_sign)
                    atomicXOR(inv_ss->data(w + num_words_minor), block_local_stab_sign);
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
                ConstWordsPointer   block_intermediate_prefix_z,
                ConstWordsPointer   block_intermediate_prefix_x,
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
        if (active_targets <= 1024) {
            LOGERROR("active targets %d are too low for large scanning", active_targets);
        }
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
                4 * sizeof(word_std_t),
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