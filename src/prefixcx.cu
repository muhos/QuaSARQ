
#include "prefix.cuh"
#include "collapse.cuh"

namespace QuaSARQ {

    __global__ 
    void scan_targets_pass_1(
        Table *prefix_xs, 
        Table *prefix_zs, 
        Table *inv_xs, 
        Table *inv_zs,
        word_std_t *block_intermediate_prefix_z,
        word_std_t *block_intermediate_prefix_x,
        const Commutation *commutations,
        const uint32 c,
        const size_t total_targets,
        const size_t num_words_major,
        const size_t num_words_minor
    )
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

            word_std_t t_delta_z = 0;
            word_std_t t_delta_x = 0;

            grid_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

            // make it grid-stride loop, looping over blocks.
            // for (size_t tile_id = blockIdx.x; tile_id < num_tiles; tile_id += gridDim.x) {
            if (global_tid < total_targets) { 
                size_t t = global_tid + c + 1;
                if (commutations[t].anti_commuting) {
                    const size_t t_destab = t * num_words_major + w;
                    assert(t_destab < inv_zs->size());
                    t_delta_z = zs[t_destab];
                    t_delta_x = xs[t_destab];
                }
            }

            t_prefix_z[prefix_tid] = t_delta_z;
            t_prefix_x[prefix_tid] = t_delta_x;

            __syncthreads();

            // In-place inclusive XOR prefix-scan
            word_std_t blockSum_z = scan_block_exclusive(t_prefix_z, blockDim.x);
            word_std_t blockSum_x = scan_block_exclusive(t_prefix_x, blockDim.x);

            // Write each thread's final prefix-scan value to output
            if (global_tid < total_targets) {
                const size_t c_destab = c * num_words_major + w;
                assert(c_destab < inv_zs->size());
                // Compute local zc = zc ^ zt, where zt is the zt'prefix.
                (*prefix_zs)[global_tid * num_words_minor + w] = word_std_t(zs[c_destab]) ^ t_prefix_z[prefix_tid];
                // Compute local xc = xc ^ xt, where xt is the xt'prefix.
                (*prefix_xs)[global_tid * num_words_minor + w] = word_std_t(xs[c_destab]) ^ t_prefix_x[prefix_tid];
            }

            if (threadIdx.x == blockDim.x - 1) {
                assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                block_intermediate_prefix_z[blockIdx.x * num_words_minor + w] = blockSum_z;
                block_intermediate_prefix_x[blockIdx.x * num_words_minor + w] = blockSum_x;
                // printf("w(%lld), t(%lld):  block intermediate prefix-xor (tz) = " B2B_STR "\n", w, global_tid, RB2B(block_intermediate_prefix_z[blockIdx.x * num_words_minor + w]));
            }
        }
    }

    __global__ 
    void scan_targets_pass_2(
        Table *prefix_xs, 
        Table *prefix_zs, 
        Table *inv_xs, 
        Table *inv_zs,
        const word_std_t *block_intermediate_prefix_z,
        const word_std_t *block_intermediate_prefix_x,
        const Commutation *commutations,
        const uint32 c,
        const qubit_t qubit,
        const size_t total_targets,
        const size_t num_words_major,
        const size_t num_words_minor,
        const size_t phase1_block_size)
    { 
        word_std_t *shared = SharedMemory<word_std_t>();
        word_std_t *shared_z = shared;
        word_std_t *shared_x = shared_z + blockDim.x;
        grid_t collapse_tid = threadIdx.y * 2 * blockDim.x + threadIdx.x;
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();
        word_std_t xc_and_zt = 0;
        word_std_t not_zc_xor_xt = 0;

        for_parallel_y(w, num_words_minor) {

            const size_t c_destab = c * num_words_major + w;
            const size_t c_stab = c_destab + num_words_minor;

            // For parallel collapsing.
            word_std_t zc_destab = 0;
            word_std_t xc_destab = 0;

            for_parallel_x(tid_x, total_targets) {

                size_t t = tid_x + c + 1;

                if (commutations[t].anti_commuting) {

                    const size_t t_destab = t * num_words_major + w;
                    const size_t t_stab = t_destab + num_words_minor;

                    assert(c_destab < inv_zs->size());
                    assert(t_destab < inv_zs->size());

                    word_std_t zc_xor_zt = (*prefix_zs)[tid_x * num_words_minor + w];
                    word_std_t xc_xor_xt = (*prefix_xs)[tid_x * num_words_minor + w];

                    // Determine from which block (phase‑1 block) this entry came.
                    grid_t bid = tid_x / phase1_block_size; // block index (0‑indexed)

                    // Compute final prefixes and hence final {x,z}'c = {x,z}'c ^ {x,z}'t expressions.
                    zc_xor_zt ^= block_intermediate_prefix_z[bid * num_words_minor + w];
                    xc_xor_xt ^= block_intermediate_prefix_x[bid * num_words_minor + w];

                    // printf("w(%lld), t(%lld): prefix-xor(" B2B_STR ") ^ block[bid: %lld](" B2B_STR ") = " B2B_STR "\n", w, t,
                    //         RB2B((word_std_t)(*prefix_zs)[tid_x * num_words_minor + w]),
                    //         bid, RB2B(block_intermediate_prefix_z[bid * num_words_minor + w]),
                    //         RB2B(zc_xor_zt));

                    // // Compute the CX expression for Z.
                    word_std_t c_stab_word = zs[c_stab];
                    word_std_t t_destab_word = zs[t_destab];
                    xc_and_zt = (c_stab_word & t_destab_word);
                    not_zc_xor_xt = ~(zc_xor_zt ^ zs[t_stab]);

                    // xc_and_zt = not_zc_xor_xt;
                    // printf("w(%lld), t(%lld): z-table: xc_and_zt:" B2B_STR " & not_zc_xor_xt:" B2B_STR " = " B2B_STR "\n", w, t, RB2B(xc_and_zt), RB2B(not_zc_xor_xt), RB2B((xc_and_zt & not_zc_xor_xt)));

                    (*prefix_zs)[tid_x * num_words_minor + w] = xc_and_zt & not_zc_xor_xt;

                    // Update Z tableau.
                    zs[t_stab] ^= c_stab_word;
                    zc_destab ^= t_destab_word; // requires collapse.

                    // // Compute the CX expression for X.
                    c_stab_word = xs[c_stab];
                    t_destab_word = xs[t_destab];
                    xc_and_zt = (c_stab_word & t_destab_word);
                    not_zc_xor_xt = ~(xc_xor_xt ^ xs[t_stab]);

                    // xc_and_zt = not_zc_xor_xt;
                    // printf("w(%lld), t(%lld): x-table: xc_and_zt:" B2B_STR " & not_zc_xor_xt:" B2B_STR " = " B2B_STR "\n", w, t, RB2B(xc_and_zt), RB2B(not_zc_xor_xt), RB2B((xc_and_zt & not_zc_xor_xt)));

                    (*prefix_xs)[tid_x * num_words_minor + w] = xc_and_zt & not_zc_xor_xt;

                    // Update X tableau.
                    xs[t_stab] ^= c_stab_word;
                    xc_destab ^= t_destab_word; // requires collapse.
                }
            }

            // Update Z, X in shared memory.
            collapse_load_shared_dual(shared_z, zc_destab, shared_x, xc_destab, collapse_tid, total_targets);
            collapse_shared_dual(shared_z, zc_destab, shared_x, xc_destab, collapse_tid);
            collapse_warp_dual(shared_z, zc_destab, shared_x, xc_destab, collapse_tid);

            if (!threadIdx.x) {
                if (zc_destab)
                    atomicXOR(zs + c_destab, zc_destab);
                if (xc_destab)
                    atomicXOR(xs + c_destab, xc_destab);
            }
        }
    }

    __global__ 
    void collapse_scanned_targets(
        Table *prefix_xs, 
        Table *prefix_zs, 
        Table *inv_xs, 
        Table *inv_zs, 
        Signs *inv_ss,
        const Commutation *commutations,
        const uint32 c,
        const qubit_t qubit,
        const size_t total_targets,
        const size_t num_words_major, 
        const size_t num_words_minor
    )
    {
        word_t *xs = inv_xs->data();
        word_t *zs = inv_zs->data();
        sign_t *ss = inv_ss->data();
        word_std_t *signs_destab = SharedMemory<word_std_t>();
        word_std_t *signs_stab = signs_destab + blockDim.x;

        grid_t collapse_tid = threadIdx.y * 2 * blockDim.x + threadIdx.x;

        for_parallel_y(w, num_words_minor) { 

            word_std_t local_destab_sign = 0;
            word_std_t local_stab_sign = 0;

            for_parallel_x(tid_x, total_targets) {
                size_t t = tid_x + c + 1;
                if (commutations[t].anti_commuting)
                {
                    local_destab_sign ^= (word_std_t)(*prefix_zs)[tid_x * num_words_minor + w];
                    local_stab_sign ^= (word_std_t)(*prefix_xs)[tid_x * num_words_minor + w];
                    // printf("w(%lld), t(%lld): xc_and_zt & not_zc_xor_xt = " B2B_STR "\n", w, t, RB2B(local_destab_sign));
                }
            }

            collapse_load_shared_dual(signs_destab, local_destab_sign, signs_stab, local_stab_sign, collapse_tid, total_targets);
            collapse_shared_dual(signs_destab, local_destab_sign, signs_stab, local_stab_sign, collapse_tid);
            collapse_warp_dual(signs_destab, local_destab_sign, signs_stab, local_stab_sign, collapse_tid);

            if (!threadIdx.x && local_destab_sign)
                atomicXOR(inv_ss->data(w), local_destab_sign);
            if (!threadIdx.x && local_stab_sign)
                atomicXOR(inv_ss->data(w + num_words_minor), local_stab_sign);
        }
    }

    // We need to compute prefix-xor of t-th destabilizer in X,Z for t = c+1, c+2, ... c+n-1
    // so that later we can xor every prefix-xor with controlled destabilizer.
    void Prefix::inject_CX(Tableau<DeviceAllocator>& input, const uint32& pivot, const qubit_t& qubit, const cudaStream_t& stream) {
        assert(num_qubits > pivot);
        // Calculate number of target generators.
        const size_t total_targets = num_qubits - pivot - 1;
        if (!total_targets) return;
        // Do the first phase of prefix.
        dim3 inject_cx_blocksize(MIN_BLOCK_INTERMEDIATE_SIZE, 2);
        nextPow2(inject_cx_blocksize.x);
        dim3 inject_cx_gridsize(1, 1);
        OPTIMIZEBLOCKS2D(inject_cx_gridsize.x, total_targets, inject_cx_blocksize.x);
        OPTIMIZEBLOCKS2D(inject_cx_gridsize.y, num_words_minor, inject_cx_blocksize.y);
        nextPow2(inject_cx_gridsize.x);
        OPTIMIZESHARED(smem_size, inject_cx_blocksize.y * (inject_cx_blocksize.x + CONFLICT_FREE_OFFSET(inject_cx_blocksize.x)), 2 * sizeof(word_std_t));
        scan_targets_pass_1 <<<inject_cx_gridsize, inject_cx_blocksize, smem_size, stream>>> (
                    XZ_TABLE(targets), 
                    XZ_TABLE(input), 
                    zblocks(), 
                    xblocks(),
                    input.commutations(), 
                    pivot,
                    total_targets, 
                    num_words_major, 
                    num_words_minor);
        LASTERR("failed to scan targets in pass 1");
        SYNC(stream);
        // Intermeditae scan of blocks resulted in pass 1.
        scan_blocks(inject_cx_gridsize.x, stream);
        // Second phase of injecting CX.
        dim3 finalize_prefix_blocksize(4, 2);
        dim3 finalize_prefix_gridsize(1, 1);
        OPTIMIZEBLOCKS2D(finalize_prefix_gridsize.x, total_targets, finalize_prefix_blocksize.x);
        OPTIMIZEBLOCKS2D(finalize_prefix_gridsize.y, num_words_minor, finalize_prefix_blocksize.y);
        OPTIMIZESHARED(finalize_prefix_smem_size, finalize_prefix_blocksize.y * finalize_prefix_blocksize.x, 2 * sizeof(word_std_t));
        scan_targets_pass_2 <<<finalize_prefix_gridsize, finalize_prefix_blocksize, finalize_prefix_smem_size, stream>>> (
                        XZ_TABLE(targets), 
                        XZ_TABLE(input),
                        zblocks(), 
                        xblocks(), 
                        input.commutations(), 
                        pivot, 
                        qubit,
                        total_targets, 
                        num_words_major, 
                        num_words_minor, 
                        inject_cx_blocksize.x);
        LASTERR("failed to scan targets in pass 2");
        SYNC(stream);
        dim3 update_signs_blocksize(4, 2);
        dim3 update_signs_gridsize(1, 1);
        OPTIMIZEBLOCKS2D(update_signs_gridsize.x, total_targets, update_signs_blocksize.x);
        OPTIMIZEBLOCKS2D(update_signs_gridsize.y, num_words_minor, update_signs_blocksize.y);
        OPTIMIZESHARED(reduce_smem_size, update_signs_blocksize.y * update_signs_blocksize.x, 2 * sizeof(word_std_t));
        // Final phase to compute the signs of the scanned targets.
        collapse_scanned_targets <<<update_signs_gridsize, update_signs_blocksize, reduce_smem_size, stream>>> (
                        XZ_TABLE(targets), 
                        XZ_TABLE(input), 
                        input.signs(), 
                        input.commutations(), 
                        pivot, 
                        qubit,
                        total_targets, 
                        num_words_major, 
                        num_words_minor);
        LASTERR("failed to collapse scanned targets");
        SYNC(stream);
    }

}