#include "simulator.hpp"
#include "measurement.cuh"
#include "prefix.cuh"
#include "tuner.cuh"
#include "access.cuh"

namespace QuaSARQ {;

    // Xc: zs[c_stab], Zc: zs[c_destab], Xt: zs[t_stab], Zt: zs[t_destab]
    INLINE_DEVICE void do_CX_sharing_control(word_t& Xc, word_t& Zc, word_t& Xt, word_t& Zt, sign_t& s, size_t w, size_t t) {
        const word_std_t xc = Xc, zc = Zc, xt = Xt, zt = Zt;
        // Update X and Z words.
        Xt = xt ^ xc;
        Zc = zc ^ zt;
        // Update Sign words.
        word_std_t xc_and_zt = xc & zt;
        word_std_t not_zc_xor_xt = ~(zc ^ xt);
        // printf("w(%lld), t(%lld):   zs[c_stab  ]:" B2B_STR " & zs[t_destab]:" B2B_STR "  = " B2B_STR "\n", w, t, 
        //             RB2B(xc), RB2B(zt), RB2B((xc & zt)));
        // printf("w(%lld), t(%lld): ~(zs[c_destab]:" B2B_STR " ^ zs[t_stab  ]:" B2B_STR ") = " B2B_STR "\n", w, t, 
        //             RB2B(zc), RB2B(xt), RB2B(~(xt ^ zc)));
        //xc_and_zt = not_zc_xor_xt;
        // printf("w(%lld), t(%lld): not_zc_xor_xt:" B2B_STR " & not_zc_xor_xt:" B2B_STR " = " B2B_STR "\n", w, t, 
        //              RB2B(xc_and_zt), RB2B(not_zc_xor_xt), RB2B((xc_and_zt & not_zc_xor_xt)));
        // printf("w(%lld), t(%lld):             s:" B2B_STR " ^ (-----------):" B2B_STR " = " B2B_STR "\n", w, t, 
        //             RB2B(s), RB2B((xc_and_zt & not_zc_xor_xt)), RB2B(s ^ (xc_and_zt & not_zc_xor_xt)));
        //s ^= (xt ^ zc);
        s ^= (xc_and_zt & not_zc_xor_xt);
    }

    __global__ void inject_CX(Table* inv_xs, Table* inv_zs, Signs* inv_ss, 
                            Commutation* commutations, 
                            const pivot_t control,
                            const qubit_t qubit,
                            const size_t num_qubits, 
                            const size_t num_words_major, 
                            const size_t num_words_minor,
                            const size_t num_qubits_padded) {
        assert(control != INVALID_PIVOT);
        word_t *xs = inv_xs->data();
        word_t *zs = inv_zs->data();
        sign_t *ss = inv_ss->data();

        for_parallel_x(w, num_words_minor) { // Update all words in both destabs and stabs.

            const size_t c_destab = TABLEAU_INDEX(w, control);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            word_std_t zc_destab = zs[c_destab], zt_destab = 0;

            for (size_t t = control + 1; t < num_qubits; t++) { // targets: pivot + 1, ..., num_qubits - 1.
                if (commutations[t].anti_commuting) {
                    const size_t t_destab = TABLEAU_INDEX(w, t);
                    const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                    assert(c_destab < inv_zs->size());
                    assert(t_destab < inv_zs->size());
                    assert(c_stab < inv_zs->size());
                    assert(t_stab < inv_zs->size());
                    //printf("z table:\n");

                    //printf("w(%lld), t(%lld):   zc: " B2B_STR " ^ zt: " B2B_STR " ", w, t, RB2B(zc_destab), RB2B(zt_destab));

                    do_CX_sharing_control(zs[c_stab], zs[c_destab], zs[t_stab], zs[t_destab], ss[w], w, t);

                    //zc_destab ^= zt_destab, zt_destab = zs[t_destab], printf("= " B2B_STR "\n", RB2B(zc_destab));
                    //printf("w(%lld), t(%lld): prefix zt = " B2B_STR "\n", w, t, RB2B(zt_destab)), zt_destab ^= (word_std_t)zs[t_destab];

                    assert(c_destab < inv_xs->size());
                    assert(t_destab < inv_xs->size());
                    assert(c_stab < inv_xs->size());
                    assert(t_stab < inv_xs->size());
                    //printf("x table:\n");
                    do_CX_sharing_control(xs[c_stab], xs[c_destab], xs[t_stab], xs[t_destab], ss[w + num_words_minor], w, t);
                }
            }
        }
    }

    __global__ 
    void check_x_destab(
                Commutation*    commutations, 
        const   Table*          inv_xs, 
        const   pivot_t         pivot,
        const   qubit_t         qubit,
        const   size_t          num_words_major,
        const   size_t          num_qubits_padded)
    {
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        const size_t word_idx = TABLEAU_INDEX(q_w, pivot);
        const word_std_t qubit_word = (*inv_xs)[word_idx];
        commutations[pivot].commuting = bool(qubit_word & q_mask);
    }

    __global__ 
    void inject_swap_k(
                Table*          inv_xs, 
                Table*          inv_zs,
                Signs*          inv_ss, 
        const   Commutation*    commutations, 
        const   pivot_t         pivot,
        const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded) 
    {
        assert(pivot != INVALID_PIVOT);
        word_t* xs = inv_xs->data();
        word_t* zs = inv_zs->data();
        sign_t* ss = inv_ss->data();

        for_parallel_x(w, num_words_minor) { 
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            assert(c_destab < inv_zs->size());
            assert(c_stab < inv_zs->size());
            assert(c_destab < inv_xs->size());
            assert(c_stab < inv_xs->size());
            if (commutations[pivot].commuting) {
                do_YZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_YZ_Swap(xs[c_stab], xs[c_destab], ss[w + num_words_minor]);
            }
            else {
                do_XZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_XZ_Swap(xs[c_stab], xs[c_destab], ss[w + num_words_minor]);
            }
        }
    }

    bool check_inject_swap(
                Table&          h_xs, 
                Table&          h_zs,
                Signs&          h_ss, 
                Table&          d_xs, 
                Table&          d_zs,
                Signs&          d_ss, 
        const   Vec<Commutation>& d_commutations,
        const   qubit_t         qubit,
        const   pivot_t         pivot,
        const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded) 
    {
        SYNCALL;

        LOGN1("  Checking inject-swap for qubit %d and pivot %d.. ", qubit, pivot);

        assert(pivot != INVALID_PIVOT);
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        const size_t word_idx = TABLEAU_INDEX(q_w, pivot);
        const word_std_t qubit_word = h_xs[word_idx];
        const bool commuting = bool(qubit_word & q_mask);

        if (commuting != d_commutations[pivot].commuting) {
            LOGERRORN("Commuting bit not identical at pivot(%lld)", pivot);
            return false;
        }

        for (size_t w = 0; w < num_words_minor; w++) { 
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            assert(c_destab < h_zs.size());
            assert(c_stab < h_zs.size());
            assert(c_destab < h_xs.size());
            assert(c_stab < h_xs.size());
            if (commuting) {
                do_YZ_Swap(h_zs[c_stab], h_zs[c_destab], h_ss[w]);
                do_YZ_Swap(h_xs[c_stab], h_xs[c_destab], h_ss[w + num_words_minor]);
            }
            else {
                do_XZ_Swap(h_zs[c_stab], h_zs[c_destab], h_ss[w]);
                do_XZ_Swap(h_xs[c_stab], h_xs[c_destab], h_ss[w + num_words_minor]);
            }
        }

        for (size_t w = 0; w < num_words_minor; w++) { 
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            if (h_xs[c_destab] != d_xs[c_destab]) {
                LOGERRORN("X-Stabilizer failed at w(%lld), pivot(%lld)", w, pivot);
                return false;
            }
            if (h_zs[c_destab] != d_zs[c_destab]) {
                LOGERRORN("Z-Stabilizer failed at w(%lld), pivot(%lld)", w, pivot);
                return false;
            }
            if (h_ss[w] != d_ss[w]) {
                LOGERRORN("Destabilizer signs failed at w(%lld)", w);
                return false;
            }
            if (h_ss[w + num_words_minor] != d_ss[w + num_words_minor]) {
                LOGERRORN("Stabilizer signs failed at w(%lld)", w + num_words_minor);
                return false;
            }
        }

        LOG0("PASSED");
        
        return true;
    }

    void Simulator::inject_swap(const pivot_t& pivot, const qubit_t& qubit, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_qubits_padded = tableau.num_qubits_padded();

        check_x_destab<<<1, 1, 0, stream>>>
        (
            commutations,
            tableau.xtable(),
            pivot,
            qubit,
            num_words_major,
            num_qubits_padded
        );
        if (options.tune_injectswap) {
            SYNCALL;
            tune_kernel_m(inject_swap_k, "injecting swap", 
                        bestblockinjectswap, bestgridinjectswap, 
                        XZ_TABLE(tableau),
                        tableau.signs(),
                        commutations,
                        pivot,
                        num_words_major,
                        num_words_minor,
                        num_qubits_padded);
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockinjectswap, bestgridinjectswap, num_words_minor, 0);
        dim3 currentblock = bestblockinjectswap, currentgrid = bestgridinjectswap;
        TRIM_GRID_IN_1D(num_words_minor, x);
        LOGN2(2, "Running pass-1 kernel with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", \
            currentblock.x, currentblock.y, currentgrid.x, currentgrid.y); \
        inject_swap_k<<<currentgrid, currentblock, 0, stream>>>
        (
            XZ_TABLE(tableau),
            tableau.signs(),
            commutations,
            pivot,
            num_words_major,
            num_words_minor,
            num_qubits_padded
        );
        if (options.sync) {
            LASTERR("failed to inject swap");
            SYNC(stream);
        }
        LOGDONE(2, 4);
        assert(prefix.get_checker().copy_commutations(commutations, num_qubits));
        assert(prefix.get_checker().copy_input(tableau, true));
        assert(check_inject_swap(
            prefix.get_checker().h_xs,
            prefix.get_checker().h_zs,
            prefix.get_checker().h_ss,
            prefix.get_checker().d_xs,
            prefix.get_checker().d_zs,
            prefix.get_checker().d_ss,
            prefix.get_checker().d_commutations,
            qubit,
            pivot,
            num_words_major,
            num_words_minor,
            num_qubits_padded
        ));
    }

    #define DEBUG_INJECT_CX 0

    void Simulator::measure(const size_t& p, const depth_t& depth_level, const bool& reversed) {
        assert(options.streams >= 4);
        cudaStream_t copy_stream1 = copy_streams[0];
        cudaStream_t copy_stream2 = copy_streams[1];
        cudaStream_t kernel_stream1 = kernel_streams[0];
        cudaStream_t kernel_stream2 = kernel_streams[1];

        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();

        // Reset all pivots.
        reset_pivots(num_gates_per_window, kernel_stream2);

        transpose(true, kernel_stream1);

        // Sync copying gates to device.
        SYNC(copy_stream1);
        SYNC(copy_stream2);
        // Sync resetting pivots.
        SYNC(kernel_stream2);

        // Find all pivots if exist.
        find_pivots(tableau, num_gates_per_window, true, kernel_stream1);

        // Copy pivots to host.
        gpu_circuit.copypivots(kernel_stream1, num_gates_per_window);

        // Reset pivots on device side.
        reset_pivots(num_gates_per_window, kernel_stream1);

        // Sync pivots wth host.
        SYNC(kernel_stream1);

        int64 random_measures = measure_indeterminate(depth_level, kernel_stream1);

        // Transpose the tableau back into column-major format.
        transpose(false, kernel_stream1);
    }

    int64 Simulator::measure_indeterminate(const depth_t& depth_level, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();
        pivot_t* host_pivots = gpu_circuit.host_pivots();
        int64 random_measures = 0;
        bool initial_pivot = true;
        for(size_t i = 0; i < num_gates_per_window; i++) {
            const Gate& curr_gate = circuit.gate(depth_level, i);
            const pivot_t curr_pivot = host_pivots[i];
            const qubit_t qubit = curr_gate.wires[0];
            pivot_t new_pivot = INVALID_PIVOT;
            if (curr_pivot != INVALID_PIVOT) {
                if (initial_pivot) {
                    initial_pivot = false;
                    new_pivot = curr_pivot;
                    mark_commutations(qubit, stream);
                }
                else {
                    find_pivots(tableau, i, false, stream);
                    gpu_circuit.copypivotto(new_pivot, i, stream);
                    SYNC(stream);                
                }
                if (new_pivot != INVALID_PIVOT) {
                    LOG2(2, "Meauring qubit %d using pivot %d.. ", qubit, new_pivot);
                    random_measures++;

                    #if !DEBUG_INJECT_CX
                    prefix.inject_CX(tableau, commutations, new_pivot, qubit, stream);
                    #else
                    const uint32 blocksize = 8;
                    const uint32 gridsize = ROUNDUP(num_words_minor, blocksize);
                    inject_CX <<<gridsize, blocksize, 0, stream>>> (XZ_TABLE(tableau), tableau.signs(), 
                                commutations, 
                                new_pivot, qubit, 
                                num_qubits, num_words_major, num_words_minor);
                    LASTERR("failed to inject_CX");
                    SYNC(stream);
                    #endif

                    inject_swap(new_pivot, qubit, stream);
                }
            }
        }

        return random_measures;
    }
}

