
#include "prefix.cuh"
#include "collapse.cuh"
#include "access.cuh"
#include "vector.hpp"
#include "print.cuh"

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

        for_parallel_y(w, num_words_minor) {

            const size_t c_destab = TABLEAU_INDEX(w, c);

            // Must initialize shared memory in case a partial block is scanned.
            t_prefix_z[prefix_tid] = 0;
            t_prefix_x[prefix_tid] = 0;

            __syncthreads();

            for_parallel_x(tid_x, total_targets) {

                const size_t t = tid_x + c + 1;
            
                if (commutations[t].anti_commuting) {
                    const size_t t_destab = TABLEAU_INDEX(w, t);
                    t_prefix_z[prefix_tid] = (*inv_zs)[t_destab];
                    t_prefix_x[prefix_tid] = (*inv_xs)[t_destab];
                }

                __syncthreads();

                word_std_t blockSum_z = scan_block_exclusive(t_prefix_z, blockDim.x);
                word_std_t blockSum_x = scan_block_exclusive(t_prefix_x, blockDim.x);

                const size_t word_idx = w * total_targets + tid_x;
                assert(word_idx < prefix_zs->size());
                assert(word_idx < prefix_xs->size());
                // Compute local zc = zc ^ zt, where zt is the zt'prefix.
                (*prefix_zs)[word_idx] = word_std_t((*inv_zs)[c_destab]) ^ t_prefix_z[prefix_tid];
                // Compute local xc = xc ^ xt, where xt is the xt'prefix.
                (*prefix_xs)[word_idx] = word_std_t((*inv_xs)[c_destab]) ^ t_prefix_x[prefix_tid];

                if (threadIdx.x == blockDim.x - 1 || tid_x == total_targets - 1) {
                    //printf("threadIdx.x = %d, tid_x = %lld, blockIdx.x = %d\n", threadIdx.x, tid_x, blockIdx.x);
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    const size_t bid = w * max_blocks + (tid_x / blockDim.x);
                    block_intermediate_prefix_z[bid] = blockSum_z;
                    block_intermediate_prefix_x[bid] = blockSum_x;
                    //LOGGPU("(w: %-6lld, b: %-6lld)-> block_prefix_z: " B2B_STR "\n", w, (tid_x / blockDim.x), RB2B(word_std_t(block_intermediate_prefix_z[bid])));
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
        const   size_t          pass_1_blocksize)
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
                    const size_t bid = w * max_blocks + (tid_x / pass_1_blocksize);
                    zc_xor_zt ^= block_intermediate_prefix_z[bid];
                    xc_xor_xt ^= block_intermediate_prefix_x[bid];

                    // For verifying,
                    (*prefix_zs)[word_idx] = zc_xor_zt;
                    (*prefix_xs)[word_idx] = xc_xor_xt;

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

    /// Performs an in-place, exclusive prefix-scan of `data[0..n-1]`,
    /// returning the overall XOR of all n inputs (the "block sum").
    static inline 
    word_std_t scan_block_exclusive_cpu(word_std_t *data, int n) 
    {
        // ------------------
        // 1) Up-sweep phase
        // ------------------
        int offset = 1;
        for (int d = (n >> 1); d > 0; d >>= 1) {
            for (int tid = 0; tid < d; tid++) {
                int i = offset * (2 * tid + 1) - 1;
                int j = offset * (2 * tid + 2) - 1;
                // XOR the pair
                assert(i < n);
                assert(j < n);
                data[j] ^= data[i];
            }
            offset <<= 1;  // offset *= 2
        }

        // The last element (data[n-1]) now holds the XOR of the entire block.
        word_std_t blockSum = data[n - 1];

        // In an exclusive prefix scan, set the last element to 0
        // before the down-sweep.
        data[n - 1] = 0;

        // --------------------
        // 2) Down-sweep phase
        // --------------------
        for (int d = 1; d < n; d <<= 1) {
            offset >>= 1;  // offset /= 2
            for (int tid = 0; tid < d; tid++) {
                int i = offset * (2 * tid + 1) - 1;
                int j = offset * (2 * tid + 2) - 1;
                assert(i < n);
                assert(j < n);
                word_std_t temp = data[i];
                data[i]         = data[j];
                data[j]        ^= temp;
            }
        }

        // Now `data[k]` has the exclusive prefix of the original block 
        // up to index k. Return the block-wide XOR.
        return blockSum;
    }

    /// CPU version of your scan_targets_pass_1 kernel, done in tiled fashion.
    bool PrefixChecker::check_prefix_pass_1(
        Tableau<DeviceAllocator>& other_targets,
        Tableau<DeviceAllocator>& other_input,
        const   Commutation* other_commutations,
        const   word_std_t*  other_zs,
        const   word_std_t*  other_xs,
        const   qubit_t      qubit, 
        const   uint32       pivot,
        const   size_t       total_targets,
        const   size_t       num_words_major,
        const   size_t       num_words_minor,
        const   size_t       num_qubits_padded,
        const   size_t       max_blocks,
        const   size_t       pass_1_blocksize,
        const   size_t       pass_1_gridsize
    ) 
    {
        SYNCALL;

        LOGN1(" Checking pass-1 prefix for qubit %d and pivot %d.. ", qubit, pivot);

        copy_input(other_input);
        copy_prefix(other_targets);
        copy_prefix_blocks(other_xs, other_zs, max_blocks * num_words_minor);
        copy_commutations(other_commutations, other_input.num_qubits());

        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        for(size_t i = 0; i < other_input.num_qubits(); i++)
            h_commutations[i].reset();
        for(size_t i = 0; i < total_targets; i++) {
            const size_t t = i + pivot + 1;
            const size_t word_idx = TABLEAU_INDEX(q_w, t) + TABLEAU_STAB_OFFSET;
            const word_std_t qubit_word = h_xs[word_idx];
            h_commutations[t].anti_commuting = bool(qubit_word & q_mask);
            if (h_commutations[t].anti_commuting != d_commutations[t].anti_commuting) {
                LOGERRORN("FAILED at commutations[%lld].anti_commuting", t);
                return false;
            }
        }

        h_block_intermediate_prefix_z.resize(max_blocks * num_words_minor, 0);
        h_block_intermediate_prefix_x.resize(max_blocks * num_words_minor, 0);

        Vec<word_std_t> t_prefix_z(pass_1_blocksize);
        Vec<word_std_t> t_prefix_x(pass_1_blocksize);

        // Outer loop: over each "minor word" index w
        for (size_t w = 0; w < num_words_minor; w++) {

            // The "control" row index is c_destab
            const size_t c_destab = TABLEAU_INDEX(w, pivot);

            for (size_t tid_x = 0; tid_x < pass_1_blocksize; tid_x++) {
                t_prefix_z[tid_x] = 0;
                t_prefix_x[tid_x] = 0;
            }

            // For each block in [0..gridDimX-1]:
            for (size_t blockIdx = 0; blockIdx < pass_1_gridsize; blockIdx++) {

                // 1) Load data into the block arrays (like the GPU does before the scan):
                for (size_t tid_x = 0; tid_x < pass_1_blocksize; tid_x++) {

                    // Overall "global" index among the targets
                    size_t global_tid_x = blockIdx * pass_1_blocksize + tid_x;

                    // Only load if we haven't run past total_targets
                    if (global_tid_x < total_targets) {

                        // This matches  t = tid_x + (c + 1)  from the GPU code
                        size_t t        = global_tid_x + (pivot + 1);
                        size_t t_destab = TABLEAU_INDEX(w, t);

                        // The GPU kernel used:  bool anti_commuting = commutations[t].anti_commuting;
                        bool anti_commuting = h_commutations[t].anti_commuting;

                        t_prefix_z[tid_x] = anti_commuting ? word_std_t(h_zs[t_destab]) : 0;
                        t_prefix_x[tid_x] = anti_commuting ? word_std_t(h_xs[t_destab]) : 0;
                    } 
                    else {
                        // If this thread's index is beyond total_targets, store 0
                        t_prefix_z[tid_x] = 0;
                        t_prefix_x[tid_x] = 0;
                    }
                }

                // 2) Perform the exclusive prefix scan *within* this block:
                word_std_t blockSum_z = scan_block_exclusive_cpu(t_prefix_z.data(), int(pass_1_blocksize));
                word_std_t blockSum_x = scan_block_exclusive_cpu(t_prefix_x.data(), int(pass_1_blocksize));

                // 3) Write each thread's partial prefix result out to prefix_zs/prefix_xs:
                for (size_t tid_x = 0; tid_x < pass_1_blocksize; tid_x++) {

                    size_t global_tid_x = blockIdx * pass_1_blocksize + tid_x;

                    if (global_tid_x < total_targets) {
                        // In the GPU kernel we do: prefix_zs[word_idx] = zs[c_destab] ^ t_prefix_z[tid_x];
                        size_t word_idx = w * total_targets + global_tid_x;

                        h_prefix_zs[word_idx] = word_std_t(h_zs[c_destab]) ^ t_prefix_z[tid_x];
                        h_prefix_xs[word_idx] = word_std_t(h_xs[c_destab]) ^ t_prefix_x[tid_x];

                        // LOGGPU("(w: %-6lld, t: %-6lld)-> h_prefix_z: " B2B_STR "\n", w, global_tid_x, RB2B(word_std_t(h_prefix_zs[word_idx])));
                        // LOGGPU("(w: %-6lld, t: %-6lld)-> d_prefix_z: " B2B_STR "\n", w, global_tid_x, RB2B(word_std_t(d_prefix_zs[word_idx])));

                        if (d_prefix_xs[word_idx] != h_prefix_xs[word_idx]) {
                            LOGERRORN("X-FAILED at w(%lld) and tid(%lld)", w, global_tid_x);
                            return false;
                        }
                        if (d_prefix_zs[word_idx] != h_prefix_zs[word_idx]) {
                            LOGERRORN("Z-FAILED at w(%lld) and tid(%lld)", w, global_tid_x);
                            return false;
                        }
                    }
                }

                if (pass_1_blocksize > 0 && pass_1_gridsize > 1) {
                    size_t bid = w * max_blocks + blockIdx;
                    h_block_intermediate_prefix_z[bid] = blockSum_z;
                    h_block_intermediate_prefix_x[bid] = blockSum_x;

                    //LOGGPU("(w: %-6lld, b: %-6lld)-> h_block_prefix_z: " B2B_STR "\n", w, blockIdx, RB2B(word_std_t(h_block_intermediate_prefix_z[bid])));
                    //LOGGPU("(w: %-6lld, b: %-6lld)-> d_block_prefix_z: " B2B_STR "\n", w, blockIdx, RB2B(word_std_t(d_block_intermediate_prefix_z[bid])));
                    if (h_block_intermediate_prefix_x[bid] != d_block_intermediate_prefix_x[bid]) {
                        LOGERRORN("X-FAILED at w(%lld) and block(%lld)", w, blockIdx);
                        return false;
                    }
                    if (h_block_intermediate_prefix_z[bid] != d_block_intermediate_prefix_z[bid]) {
                        LOGERRORN("Z-FAILED at w(%lld) and block(%lld)", w, blockIdx);
                        return false;
                    }
                }
                
            }
        }

        LOG0("PASSED");
        return true;
    }

    bool PrefixChecker::check_prefix_single_pass(
        const   word_std_t* other_zs,
        const   word_std_t* other_xs,
        const   qubit_t  qubit, 
        const   uint32   pivot,
        const   size_t   num_words_minor,
        const   size_t	 max_blocks,
        const 	size_t   pass_1_gridsize
    ) {
        SYNCALL;

        LOGN1(" Checking single-pass prefix for qubit %d and pivot %d.. ", qubit, pivot);

        // LOG0("");
        // LOG0("Before Scan:");
        // for (size_t w = 0; w < num_words_minor; w++) {
        //     for (size_t tid_x = 0; tid_x < pass_1_gridsize; tid_x++) {
        //         size_t bid = w * max_blocks + tid_x;
        //         LOGGPU("(w: %-6lld, b: %-6lld)-> h_block_prefix_z: " B2B_STR "\n", w, tid_x, RB2B(word_std_t(h_block_intermediate_prefix_z[bid])));
        //         LOGGPU("(w: %-6lld, b: %-6lld)-> d_block_prefix_z: " B2B_STR "\n", w, tid_x, RB2B(word_std_t(d_block_intermediate_prefix_z[bid])));
        //     }
        // }

        copy_prefix_blocks(other_xs, other_zs, max_blocks * num_words_minor);

        //LOG0("After Scan:");

        // Scan intermediate blocks.
        for (size_t w = 0; w < num_words_minor; w++) {
            word_std_t* block_z = h_block_intermediate_prefix_z + w * max_blocks;
            word_std_t* block_x = h_block_intermediate_prefix_x + w * max_blocks;
            scan_block_exclusive_cpu(block_z, int(pass_1_gridsize));
            scan_block_exclusive_cpu(block_x, int(pass_1_gridsize));

            for (size_t tid_x = 0; tid_x < pass_1_gridsize; tid_x++) {
                size_t bid = w * max_blocks + tid_x;

                //LOGGPU("(w: %-6lld, b: %-6lld)-> h_block_prefix_z: " B2B_STR "\n", w, tid_x, RB2B(word_std_t(h_block_intermediate_prefix_z[bid])));
                //LOGGPU("(w: %-6lld, b: %-6lld)-> d_block_prefix_z: " B2B_STR "\n", w, tid_x, RB2B(word_std_t(d_block_intermediate_prefix_z[bid])));

                if (h_block_intermediate_prefix_x[bid] != d_block_intermediate_prefix_x[bid]) {
                    LOGERRORN("X-FAILED at w(%lld) and i(%lld)", w, tid_x);
                    return false;
                }
                if (h_block_intermediate_prefix_z[bid] != d_block_intermediate_prefix_z[bid]) {
                    LOGERRORN("Z-FAILED at w(%lld) and i(%lld)", w, tid_x);
                    return false;
                }
            }
        }    

        LOG0("PASSED");

        return true;
    }

    bool PrefixChecker::check_prefix_pass_2(
                Tableau<DeviceAllocator>& other_targets, 
                Signs *			other_signs,
        const   qubit_t 		qubit, 
        const   uint32   		pivot,
        const   size_t          total_targets,
        const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded,
        const   size_t          max_blocks,
        const   size_t          pass_1_blocksize
    ) {
        SYNCALL;

        LOGN1(" Checking pass-2 prefix for qubit %d and pivot %d.. ", qubit, pivot);

        copy_prefix(other_targets);

        for (size_t w = 0; w < num_words_minor; w++) {

            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;

            word_std_t zc_destab = 0;
            word_std_t xc_destab = 0;
            word_std_t xc_and_zt = 0;
            word_std_t not_zc_xor_xt = 0;
            word_std_t local_destab_sign = 0;
            word_std_t local_stab_sign = 0;

            for (size_t tid_x = 0; tid_x < total_targets; tid_x++) {
            
                size_t t = tid_x + pivot + 1;

                if (h_commutations[t].anti_commuting) {

                    const size_t t_destab = TABLEAU_INDEX(w, t);
                    const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;

                    assert(c_destab < h_xs.size());
                    assert(t_destab < h_zs.size());

                    const size_t word_idx = w * total_targets + tid_x;
                    word_std_t zc_xor_zt = h_prefix_zs[word_idx];
                    word_std_t xc_xor_xt = h_prefix_xs[word_idx];

                    const size_t bid = w * max_blocks + (tid_x / pass_1_blocksize);

                    zc_xor_zt ^= h_block_intermediate_prefix_z[bid];
                    xc_xor_xt ^= h_block_intermediate_prefix_x[bid];

                    h_prefix_zs[word_idx] = zc_xor_zt;
                    h_prefix_xs[word_idx] = xc_xor_xt;

                    if (d_prefix_xs[word_idx] != h_prefix_xs[word_idx]) {
                        LOGERRORN("X-FAILED at w(%lld) and tid(%lld)", w, tid_x);
                        return false;
                    }
                    if (d_prefix_zs[word_idx] != h_prefix_zs[word_idx]) {
                        LOGERRORN("Z-FAILED at w(%lld) and tid(%lld)", w, tid_x);
                        return false;
                    }

                    // Compute the CX expression for Z.
                    word_std_t c_stab_word = h_zs[c_stab];
                    word_std_t t_destab_word = h_zs[t_destab];
                    xc_and_zt = (c_stab_word & t_destab_word);
                    not_zc_xor_xt = ~(zc_xor_zt ^ word_std_t(h_zs[t_stab]));
                    local_destab_sign ^= xc_and_zt & not_zc_xor_xt;
                    
                    // Update Z tableau.
                    //h_zs[t_stab] ^= c_stab_word;
                    //zc_destab ^= t_destab_word; // requires collapse.

                    // Compute the CX expression for X.
                    c_stab_word = h_xs[c_stab];
                    t_destab_word = h_xs[t_destab];
                    xc_and_zt = (c_stab_word & t_destab_word);
                    not_zc_xor_xt = ~(xc_xor_xt ^ word_std_t(h_xs[t_stab]));
                    local_stab_sign ^= xc_and_zt & not_zc_xor_xt;

                    // Update X tableau.
                    //h_xs[t_stab] ^= c_stab_word;
                    //xc_destab ^= t_destab_word; // requires collapse.
                }
            }
        }

        LOG0("PASSED");

        return true;
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

        // Verify pass-1 prefix.
        assert(checker.check_prefix_pass_1(
            targets,
            input,
            commutations,
            zblocks(), 
            xblocks(),
            qubit,
            pivot,
            total_targets,
            num_words_major,
            num_words_minor,
            num_qubits_padded,
            max_intermediate_blocks,
            pass_1_blocksize,
            pass_1_gridsize
        ));

        // Intermeditae scan of blocks resulted in pass 1.
        scan_blocks(nextPow2(pass_1_gridsize), stream);

        // Verify single-pass prefix.
        assert(checker.check_prefix_single_pass(
            zblocks(), 
            xblocks(),
            qubit,
            pivot,
            num_words_minor,
            max_intermediate_blocks,
            nextPow2(pass_1_gridsize)
        ));

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

        // Verify pass-1 prefix.
        assert(checker.check_prefix_pass_2(
            targets, 
            input.signs(),
            qubit,
            pivot, 
            total_targets, 
            num_words_major, 
            num_words_minor, 
            num_qubits_padded,
            max_intermediate_blocks,
            pass_1_blocksize
        ));
    }

}