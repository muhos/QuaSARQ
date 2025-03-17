
#include "prefix.cuh"
#include "collapse.cuh"
#include "access.cuh"

namespace QuaSARQ {

    __global__ 
    void scan_targets_pass_1(
        Table *             prefix_xs, 
        Table *             prefix_zs, 
        Table *             inv_xs, 
        Table *             inv_zs,
        word_std_t *        block_intermediate_prefix_z,
        word_std_t *        block_intermediate_prefix_x,
        const Commutation * commutations,
        const uint32        c,
        const size_t        total_targets,
        const size_t        num_words_major,
        const size_t        num_words_minor,
        const size_t        num_qubits_padded,
        const size_t        max_blocks)
    {
        grid_t padded_block_size = blockDim.x + CONFLICT_FREE_OFFSET(blockDim.x);
        grid_t slice = 2 * padded_block_size;
        word_std_t *shared = SharedMemory<word_std_t>();
        word_std_t *t_prefix_z = shared + threadIdx.y * slice;
        word_std_t *t_prefix_x = t_prefix_z + padded_block_size;
        grid_t prefix_tid = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x);
        word_t *xs = inv_xs->data();
        word_t *zs = inv_zs->data();

        for_parallel_y(w, num_words_minor) {

            const size_t c_destab = TABLEAU_INDEX(w, c);
            assert(c_destab < inv_zs->size());
            assert(c_destab < inv_xs->size());

            for_parallel_x(tid_x, total_targets) {

                const size_t t = tid_x + c + 1;
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const bool anti_commuting = commutations[t].anti_commuting;
            
                t_prefix_z[prefix_tid] = anti_commuting ? word_std_t(zs[t_destab]) : 0;
                t_prefix_x[prefix_tid] = anti_commuting ? word_std_t(xs[t_destab]) : 0;

                __syncthreads();

                word_std_t blockSum_z = scan_block_exclusive(t_prefix_z, blockDim.x);
                word_std_t blockSum_x = scan_block_exclusive(t_prefix_x, blockDim.x);

                const size_t word_idx = w * total_targets + tid_x;
                assert(word_idx < prefix_zs->size());
                assert(word_idx < prefix_xs->size());
                // Compute local zc = zc ^ zt, where zt is the zt'prefix.
                (*prefix_zs)[word_idx] = word_std_t(zs[c_destab]) ^ t_prefix_z[prefix_tid];
                // Compute local xc = xc ^ xt, where xt is the xt'prefix.
                (*prefix_xs)[word_idx] = word_std_t(xs[c_destab]) ^ t_prefix_x[prefix_tid];

                if (threadIdx.x == blockDim.x - 1) {
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    const size_t bid = w * max_blocks + (tid_x / blockDim.x);
                    block_intermediate_prefix_z[bid] = blockSum_z;
                    block_intermediate_prefix_x[bid] = blockSum_x;
                }

            }
        }
    }

    __global__ 
    void scan_targets_pass_2(
                Table *         prefix_xs, 
                Table *         prefix_zs, 
                Table *         inv_xs, 
                Table *         inv_zs,
                Signs *         inv_ss,
        const   word_std_t *    block_intermediate_prefix_z,
        const   word_std_t *    block_intermediate_prefix_x,
        const   Commutation *   commutations,
        const   uint32          c,
        const   size_t          total_targets,
        const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded,
        const   size_t          max_blocks,
        const   size_t          phase1_block_size)
    { 
        word_std_t *shared = SharedMemory<word_std_t>();
        word_std_t *shared_z = shared;
        word_std_t *shared_x = shared_z + blockDim.x;
        word_std_t *signs_destab = shared_x + blockDim.x;
        word_std_t *signs_stab = signs_destab + blockDim.x;
        grid_t      collapse_tid = threadIdx.y * 4 * blockDim.x + threadIdx.x;
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();

        for_parallel_y(w, num_words_minor) {

            const size_t c_destab = TABLEAU_INDEX(w, c);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;

            word_std_t zc_destab = 0;
            word_std_t xc_destab = 0;
            word_std_t xc_and_zt = 0;
            word_std_t not_zc_xor_xt = 0;
            word_std_t local_destab_sign = 0;
            word_std_t local_stab_sign = 0;

            for_parallel_x(tid_x, total_targets) {

                size_t t = tid_x + c + 1;

                if (commutations[t].anti_commuting) {

                    const size_t t_destab = TABLEAU_INDEX(w, t);
                    const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;

                    assert(c_destab < inv_zs->size());
                    assert(t_destab < inv_zs->size());

                    const size_t word_idx = w * total_targets + tid_x;
                    word_std_t zc_xor_zt = (*prefix_zs)[word_idx];
                    word_std_t xc_xor_xt = (*prefix_xs)[word_idx];

                    // Compute final prefixes and hence final {x,z}'c = {x,z}'c ^ {x,z}'t expressions.
                    const size_t bid = w * max_blocks + (tid_x / phase1_block_size);
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
            }

            collapse_load_shared_dual(shared_z, zc_destab, shared_x, xc_destab, collapse_tid, total_targets);
            collapse_shared_dual(shared_z, zc_destab, shared_x, xc_destab, collapse_tid);
            collapse_warp_dual(shared_z, zc_destab, shared_x, xc_destab, collapse_tid);
            collapse_load_shared_dual(signs_destab, local_destab_sign, signs_stab, local_stab_sign, collapse_tid, total_targets);
            collapse_shared_dual(signs_destab, local_destab_sign, signs_stab, local_stab_sign, collapse_tid);
            collapse_warp_dual(signs_destab, local_destab_sign, signs_stab, local_stab_sign, collapse_tid);

            if (!threadIdx.x) {
                if (zc_destab)
                    atomicXOR(zs + c_destab, zc_destab);
                if (xc_destab)
                    atomicXOR(xs + c_destab, xc_destab);
                if (local_destab_sign)
                    atomicXOR(inv_ss->data(w), local_destab_sign);
                if (local_stab_sign)
                    atomicXOR(inv_ss->data(w + num_words_minor), local_stab_sign);
            }
        }
    }

    // We need to compute prefix-xor of t-th destabilizer in X,Z for t = c+1, c+2, ... c+n-1
    // so that later we can xor every prefix-xor with controlled destabilizer.
    void Prefix::inject_CX(Tableau<DeviceAllocator>& input, const Commutation* commutations, const uint32& pivot, const qubit_t& qubit, const cudaStream_t& stream) {
        assert(num_qubits > pivot);
        assert(nextPow2(MIN_BLOCK_INTERMEDIATE_SIZE) == MIN_BLOCK_INTERMEDIATE_SIZE);
        
        const size_t num_qubits_padded = input.num_qubits_padded();

        // Calculate number of target generators.
        const size_t total_targets = num_qubits - pivot - 1;
        if (!total_targets) return;
        // Do the first phase of prefix.
        dim3 currentblock, currentgrid;
        if (options.tune_injectprepare) {
            SYNCALL;
            tune_inject_pass_1(
                scan_targets_pass_1, 
                bestblockinjectprepare, bestgridinjectprepare,
                2 * sizeof(word_std_t),
                total_targets,
                num_words_minor,
                XZ_TABLE(targets), 
                XZ_TABLE(input), 
                zblocks(), 
                xblocks(),
                commutations, 
                pivot,
                total_targets, 
                num_words_major, 
                num_words_minor,
                num_qubits_padded,
                max_intermediate_blocks
            );
            SYNCALL;
        }
        SYNCALL;
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockinjectprepare, bestgridinjectprepare, total_targets, num_words_minor);
        currentblock = bestblockinjectprepare, currentgrid = bestgridinjectprepare;
        TRIM_GRID_IN_XY(total_targets, num_words_minor);
        const size_t pass_1_blocksize = currentblock.x;
        const size_t pass_1_gridsize = ROUNDUP(total_targets, pass_1_blocksize);
        if (pass_1_gridsize > max_intermediate_blocks)
            LOGERROR("too many blocks for intermediate arrays.");
        OPTIMIZESHARED(smem_size, currentblock.y * (currentblock.x + CONFLICT_FREE_OFFSET(currentblock.x)), 2 * sizeof(word_std_t));
        scan_targets_pass_1 <<<currentgrid, currentblock, smem_size, stream>>> (
                    XZ_TABLE(targets), 
                    XZ_TABLE(input), 
                    zblocks(), 
                    xblocks(),
                    commutations, 
                    pivot,
                    total_targets, 
                    num_words_major, 
                    num_words_minor,
                    num_qubits_padded,
                    max_intermediate_blocks
                );
        if (options.sync) {
            LASTERR("failed to scan targets in pass 1");
            SYNC(stream);
        }

        // Intermeditae scan of blocks resulted in pass 1.
        scan_blocks(nextPow2(pass_1_gridsize), stream);

        // Second phase of injecting CX.
        if (options.tune_injectfinal) {
            SYNCALL;
            tune_inject_pass_2(
                scan_targets_pass_2, 
                bestblockinjectfinal, bestgridinjectfinal,
                4 * sizeof(word_std_t),
                total_targets,
                num_words_minor,
                XZ_TABLE(targets), 
                XZ_TABLE(input),
                input.signs(),
                zblocks(), 
                xblocks(), 
                commutations, 
                pivot, 
                total_targets, 
                num_words_major, 
                num_words_minor, 
                num_qubits_padded,
                max_intermediate_blocks,
                pass_1_blocksize
            );
            SYNCALL;
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockinjectfinal, bestgridinjectfinal, total_targets, num_words_minor);
        currentblock = bestblockinjectfinal, currentgrid = bestgridinjectfinal;
        TRIM_GRID_IN_XY(total_targets, num_words_minor);
        OPTIMIZESHARED(finalize_prefix_smem_size, currentblock.y * currentblock.x, 4 * sizeof(word_std_t));
        scan_targets_pass_2 <<<currentgrid, currentblock, finalize_prefix_smem_size, stream>>> (
                        XZ_TABLE(targets), 
                        XZ_TABLE(input),
                        input.signs(),
                        zblocks(), 
                        xblocks(), 
                        commutations, 
                        pivot, 
                        total_targets, 
                        num_words_major, 
                        num_words_minor, 
                        num_qubits_padded,
                        max_intermediate_blocks,
                        pass_1_blocksize
                    );
        if (options.sync) {
            LASTERR("failed to scan targets in pass 2");
            SYNC(stream);
        }
    }

}