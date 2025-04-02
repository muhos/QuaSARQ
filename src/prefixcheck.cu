
#include "access.cuh"
#include "prefixcheck.cuh"

namespace QuaSARQ {

    inline 
    word_std_t scan_block_exclusive_cpu(word_std_t *data, int n) {
        int offset = 1;
        for (int d = (n >> 1); d > 0; d >>= 1) {
            for (int tid = 0; tid < d; tid++) {
                int i = offset * (2 * tid + 1) - 1;
                int j = offset * (2 * tid + 2) - 1;
                assert(i < n);
                assert(j < n);
                data[j] ^= data[i];
            }
            offset <<= 1; 
        }
        word_std_t blockSum = data[n - 1];
        data[n - 1] = 0;
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
        return blockSum;
    }

     // If skip_checking_device is set to 1, checking with device
     // will be slipped but CPU computations would proceed normally.
     // Useful to determine the largest number of targets.
    void PrefixChecker::check_prefix_pass_1(
                Tableau&        other_targets,
                Tableau&        other_input,
        const   Commutation*    other_commutations,
        const   word_std_t*     other_zs,
        const   word_std_t*     other_xs,
        const   qubit_t&        qubit, 
        const   uint32&         pivot,
        const   size_t&         total_targets,
        const   size_t&         num_words_major,
        const   size_t&         num_words_minor,
        const   size_t&         num_qubits_padded,
        const   size_t&         max_blocks,
        const   size_t&         pass_1_blocksize,
        const   size_t&         pass_1_gridsize,
        const   bool&           skip_checking_device) 
    {
        SYNCALL;

        const char * title = skip_checking_device ? "Performing" : "Checking"; 
        LOGN1("  %s pass-1 prefix for qubit %d and pivot %d.. ", title, qubit, pivot);

        if (h_xs.is_colmajor()) // Do the copy only the first time.
            copy_input(other_input);
        if (!skip_checking_device) {
            copy_prefix(other_targets);
            copy_prefix_blocks(other_xs, other_zs, max_blocks * num_words_minor);
            copy_commutations(other_commutations, other_input.num_qubits());
        }

        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        for(size_t i = 0; i < other_input.num_qubits(); i++)
            h_commutations[i].reset();
        for(size_t i = 0; i < total_targets; i++) {
            const size_t t = i + pivot + 1;
            const size_t word_idx = TABLEAU_INDEX(q_w, t) + TABLEAU_STAB_OFFSET;
            const word_std_t qubit_word = h_xs[word_idx];
            h_commutations[t].anti_commuting = bool(qubit_word & q_mask);
            if (!skip_checking_device && h_commutations[t].anti_commuting != d_commutations[t].anti_commuting) {
                LOGERROR("Pass-1 FAILED at commutations[%lld].anti_commuting", t);
            }
        }

        h_block_intermediate_prefix_z.resize(max_blocks * num_words_minor, 0);
        h_block_intermediate_prefix_x.resize(max_blocks * num_words_minor, 0);

        Vec<word_std_t> t_prefix_z(pass_1_blocksize);
        Vec<word_std_t> t_prefix_x(pass_1_blocksize);

        for (size_t w = 0; w < num_words_minor; w++) {
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            for (size_t tx = 0; tx < pass_1_blocksize; tx++) {
                t_prefix_z[tx] = 0;
                t_prefix_x[tx] = 0;
            }
            for (size_t bx = 0; bx < pass_1_gridsize; bx++) {
                for (size_t tx = 0; tx < pass_1_blocksize; tx++) {
                    size_t tid_x = bx * pass_1_blocksize + tx;
                    if (tid_x < total_targets) {
                        size_t t        = tid_x + (pivot + 1);
                        size_t t_destab = TABLEAU_INDEX(w, t);
                        bool anti_commuting = h_commutations[t].anti_commuting;
                        t_prefix_z[tx] = anti_commuting ? word_std_t(h_zs[t_destab]) : 0;
                        t_prefix_x[tx] = anti_commuting ? word_std_t(h_xs[t_destab]) : 0;
                    } 
                    else {
                        t_prefix_z[tx] = 0;
                        t_prefix_x[tx] = 0;
                    }
                }

                word_std_t blockSum_z = scan_block_exclusive_cpu(t_prefix_z.data(), int(pass_1_blocksize));
                word_std_t blockSum_x = scan_block_exclusive_cpu(t_prefix_x.data(), int(pass_1_blocksize));

                for (size_t tx = 0; tx < pass_1_blocksize; tx++) {
                    size_t tid_x = bx * pass_1_blocksize + tx;
                    if (tid_x < total_targets) {
                        size_t t = tid_x + (pivot + 1);
                        if (h_commutations[t].anti_commuting) {
                            size_t word_idx = PREFIX_TABLEAU_INDEX(w, tid_x);
                            h_prefix_zs[word_idx] = word_std_t(h_zs[c_destab]) ^ t_prefix_z[tx];
                            h_prefix_xs[word_idx] = word_std_t(h_xs[c_destab]) ^ t_prefix_x[tx];
                            if (!skip_checking_device && d_prefix_xs[word_idx] != h_prefix_xs[word_idx]) {
                                LOGERROR("Pass-1 FAILED at prefix-x[w: %lld, tid: %lld]", w, tid_x);
                            }
                            if (!skip_checking_device && d_prefix_zs[word_idx] != h_prefix_zs[word_idx]) {
                                LOGERROR("Pass-1 FAILED at prefix-z[w: %lld, tid: %lld]", w, tid_x);
                            }
                        }
                    }
                }
                if (pass_1_blocksize > 0 && pass_1_gridsize > 1) {
                    size_t bid = PREFIX_INTERMEDIATE_INDEX(w, bx);
                    h_block_intermediate_prefix_z[bid] = blockSum_z;
                    h_block_intermediate_prefix_x[bid] = blockSum_x;
                    if (!skip_checking_device && h_block_intermediate_prefix_x[bid] != d_block_intermediate_prefix_x[bid]) {
                        LOGERROR("Pass-1 FAILED at block-prefix-x[w: %lld, bx: %lld]", w, bx);
                    }
                    if (!skip_checking_device && h_block_intermediate_prefix_z[bid] != d_block_intermediate_prefix_z[bid]) {
                        LOGERROR("Pass-1 FAILED at block-prefix-z[w: %lld, bx: %lld]", w, bx);
                    }
                }
                
            }
        }

        LOG0("PASSED");
    }

    void PrefixChecker::check_prefix_intermediate_pass(
        const   word_std_t*     other_zs,
        const   word_std_t*     other_xs,
        const   qubit_t&        qubit, 
        const   uint32&         pivot,
        const   size_t&         num_words_minor,
        const   size_t&	        max_blocks,
        const 	size_t&         pass_1_gridsize,
        const   bool&           skip_checking_device) {
        
        SYNCALL;
        const char * title = skip_checking_device ? "Performing" : "Checking"; 
        LOGN1("  %s pass-x prefix for qubit %d and pivot %d.. ", title, qubit, pivot);

        if (!skip_checking_device) copy_prefix_blocks(other_xs, other_zs, max_blocks * num_words_minor);

        const int nextpow2_blocksize = nextPow2(pass_1_gridsize);
        Vec<word_std_t> block_z(nextpow2_blocksize);
        Vec<word_std_t> block_x(nextpow2_blocksize);
        for (size_t w = 0; w < num_words_minor; w++) {
            for (size_t tx = pass_1_gridsize; tx < nextpow2_blocksize; tx++) {
                size_t bid = PREFIX_INTERMEDIATE_INDEX(w, tx);
                h_block_intermediate_prefix_z[bid] = 0;
                h_block_intermediate_prefix_x[bid] = 0;
            }
            for (size_t tx = 0; tx < nextpow2_blocksize; tx++) {
                size_t bid = PREFIX_INTERMEDIATE_INDEX(w, tx);
                block_z[tx] = h_block_intermediate_prefix_z[bid];
                block_x[tx] = h_block_intermediate_prefix_x[bid];
            }
            scan_block_exclusive_cpu(block_z, nextpow2_blocksize);
            scan_block_exclusive_cpu(block_x, nextpow2_blocksize);
            for (size_t tx = 0; tx < pass_1_gridsize; tx++) {
                size_t bid = PREFIX_INTERMEDIATE_INDEX(w, tx);
                h_block_intermediate_prefix_z[bid] = block_z[tx];
                h_block_intermediate_prefix_x[bid] = block_x[tx];
                if (!skip_checking_device && h_block_intermediate_prefix_x[bid] != d_block_intermediate_prefix_x[bid]) {
                    LOGERROR("Pass-x FAILED at block-prefix-x[w: %lld, tx: %lld]", w, tx);
                }
                if (!skip_checking_device && h_block_intermediate_prefix_z[bid] != d_block_intermediate_prefix_z[bid]) {
                    LOGERROR("Pass-x FAILED at block-prefix-z[w: %lld, tx: %lld]", w, tx);
                }
            }
        }    
        LOG0("PASSED");
    }

    void PrefixChecker::check_prefix_pass_2(
                Tableau& 		other_targets, 
                Tableau& 		other_input,
        const   qubit_t& 		qubit, 
        const   uint32&   		pivot,
        const   size_t&         total_targets,
        const   size_t&         num_words_major,
        const   size_t&         num_words_minor,
        const   size_t&         num_qubits_padded,
        const   size_t&         max_blocks,
        const   size_t&         pass_1_blocksize,
        const   bool&           skip_checking_device) {
        
        SYNCALL;
        const char * title = skip_checking_device ? "Performing" : "Checking"; 
        LOGN1("  %s pass-2 prefix for qubit %d and pivot %d.. ", title, qubit, pivot);

        if (!skip_checking_device) copy_input(other_input, true);

        for (size_t w = 0; w < num_words_minor; w++) {
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            word_std_t zc_destab = 0;
            word_std_t xc_destab = 0;
            word_std_t xc_and_zt = 0;
            word_std_t not_zc_xor_xt = 0;
            for (size_t tid_x = 0; tid_x < total_targets; tid_x++) {
                size_t t = tid_x + pivot + 1;
                if (h_commutations[t].anti_commuting) {
                    const size_t t_destab = TABLEAU_INDEX(w, t);
                    const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                    assert(c_destab < h_xs.size());
                    assert(t_destab < h_zs.size());

                    const size_t word_idx = PREFIX_TABLEAU_INDEX(w, tid_x);
                    word_std_t zc_xor_zt = h_prefix_zs[word_idx];
                    word_std_t xc_xor_xt = h_prefix_xs[word_idx];
                    word_std_t d_zc_xor_zt = d_prefix_zs[word_idx];
                    word_std_t d_xc_xor_xt = d_prefix_xs[word_idx];

                    const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, (tid_x / pass_1_blocksize));
                    zc_xor_zt ^= h_block_intermediate_prefix_z[bid];
                    xc_xor_xt ^= h_block_intermediate_prefix_x[bid];

                    if (!skip_checking_device) {
                        d_zc_xor_zt ^= d_block_intermediate_prefix_z[bid];
                        d_xc_xor_xt ^= d_block_intermediate_prefix_x[bid];
                        if (d_xc_xor_xt != xc_xor_xt) {
                            LOGERROR("Pass-2 FAILED at prefix-x[w: %lld, tid: %lld]", w, tid_x);
                        }
                        if (d_zc_xor_zt != zc_xor_zt) {
                            LOGERROR("Pass-2 FAILED at prefix-z[w: %lld, tid: %lld]", w, tid_x);
                        }
                    }

                    // Compute the CX expression for Z.
                    word_std_t c_stab_word = h_zs[c_stab];
                    word_std_t t_destab_word = h_zs[t_destab];
                    xc_and_zt = (c_stab_word & t_destab_word);
                    not_zc_xor_xt = ~(zc_xor_zt ^ word_std_t(h_zs[t_stab]));
                    h_ss[w] ^= (xc_and_zt & not_zc_xor_xt);
                    
                    // Update Z tableau.
                    h_zs[t_stab] ^= c_stab_word;
                    h_zs[c_destab] ^= t_destab_word;

                    // Compute the CX expression for X.
                    c_stab_word = h_xs[c_stab];
                    t_destab_word = h_xs[t_destab];
                    xc_and_zt = (c_stab_word & t_destab_word);
                    not_zc_xor_xt = ~(xc_xor_xt ^ word_std_t(h_xs[t_stab]));
                    h_ss[w + num_words_minor] ^= (xc_and_zt & not_zc_xor_xt);

                    // Update X tableau.
                    h_xs[t_stab] ^= c_stab_word;
                    h_xs[c_destab] ^= t_destab_word;

                    if (!skip_checking_device && h_xs[t_stab] != d_xs[t_stab]) {
                        LOGERROR("Pass-2 FAILED at stab-x[w: %lld, tid: %lld]", w, tid_x);
                    }
                    if (!skip_checking_device && h_zs[t_stab] != d_zs[t_stab]) {
                        LOGERROR("Pass-2 FAILED at stab-z[w: %lld, tid: %lld]", w, tid_x);
                    }
                }
            }

            if (!skip_checking_device) {
                if (h_xs[c_destab] != d_xs[c_destab]) {
                    LOGERROR("Pass-2 FAILED at destab-x[w: %lld, pivot: %lld]", w, pivot);
                }
                if (h_zs[c_destab] != d_zs[c_destab]) {
                    LOGERROR("Pass-2 FAILED at destab-z[w: %lld, pivot: %lld]", w, pivot);
                }

                if (h_ss[w] != d_ss[w]) {
                    LOGERROR("Pass-2 FAILED at destab-s[w: %lld]", w);
                }
                if (h_ss[w + num_words_minor] != d_ss[w + num_words_minor]) {
                    LOGERROR("Pass-2 FAILED at   stab-s[w: %lld]", w + num_words_minor);
                }
            }
        }

        LOG0("PASSED");
    }

}