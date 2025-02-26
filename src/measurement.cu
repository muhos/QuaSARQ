#include "simulator.hpp"
#include "measurement.cuh"
#include "prefix.cuh"

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
                            const uint32 control,
                            const qubit_t qubit,
                            const size_t num_qubits, 
                            const size_t num_words_major, const size_t num_words_minor) {
        assert(control != INVALID_PIVOT);
        const size_t c_row = control * num_words_major;
        word_t *xs = inv_xs->data();
        word_t *zs = inv_zs->data();
        sign_t *ss = inv_ss->data();

        for_parallel_x(w, num_words_minor) { // Update all words in both destabs and stabs.

            const size_t c_destab = c_row + w;
            word_std_t zc_destab = zs[c_destab], zt_destab = 0;

            for (size_t t = control + 1; t < num_qubits; t++) { // targets: pivot + 1, ..., num_qubits - 1.
                if (commutations[t].anti_commuting) {
                    const size_t t_row = t * num_words_major;
                    const size_t t_destab = t_row + w;
                    const size_t c_stab = c_destab + num_words_minor;
                    const size_t t_stab = t_destab + num_words_minor;
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

    __global__ void check_x_destab(Commutation* commutations, const Table* inv_xs, 
                            const uint32 pivot,
                            const qubit_t qubit,
                            const size_t num_words_major) {
        const qubit_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        const word_std_t qubit_word = (*inv_xs)[pivot * num_words_major + q_w];
        commutations[pivot].commuting = bool(qubit_word & q_mask);
        //printf("qubit(%d), destab w(%d) pivot(%d): " B2B_STR "\n", qubit, q_w, pivot, RB2B(qubit_word));
    }

    __global__ void inject_Swap(Table* inv_xs, Table* inv_zs, Signs* inv_ss, 
                            const Commutation* commutations, 
                            const uint32 c,
                            const size_t num_words_major, const size_t num_words_minor) {
        assert(c != INVALID_PIVOT);
        const size_t c_row = c * num_words_major;
        word_t* xs = inv_xs->data();
        word_t* zs = inv_zs->data();
        sign_t* ss = inv_ss->data();

        for_parallel_x(w, num_words_minor) { // Update all words in both destabs and stabs.
            const size_t c_destab = c_row + w;
            const size_t c_stab = c_destab + num_words_minor;
            assert(c_destab < inv_zs->size());
            assert(c_stab < inv_zs->size());
            assert(c_destab < inv_xs->size());
            assert(c_stab < inv_xs->size());
            if (commutations[c].commuting) {
                do_YZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_YZ_Swap(xs[c_stab], xs[c_destab], ss[w + num_words_minor]);
                //printf("H_YZ is chosen\n");
            }
            else {
                do_XZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_XZ_Swap(xs[c_stab], xs[c_destab], ss[w + num_words_minor]);
                //printf("H_XZ is chosen\n");
            }
        }
    }

    __global__ void mark_anti_commutations(Commutation* commutations, const qubit_t q, const Table* inv_xs, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        const qubit_t q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        for_parallel_x(g, num_qubits) {
            const size_t word_idx = g * num_words_major + q_w + num_words_minor;
            const word_std_t qubit_word = (*inv_xs)[word_idx];
            if (qubit_word & q_mask) {
                commutations[g].anti_commuting = true;
                //printf("qubit(%d), w(%d), t(%lld): " B2B_STR "\n", q, q_w, g, RB2B(qubit_word));
            }
            else {
                commutations[g].reset();
            }
        }
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
        bestblockallpivots.x = 32;
        bestblockallpivots.y = 16;
        find_pivots(tableau, num_gates_per_window, true, kernel_stream1);

        //print_tableau(tableau, depth_level, false);

        // Copy pivots to host.
        gpu_circuit.copypivots(kernel_stream1, num_gates_per_window);

        // Reset pivots on device side.
        reset_pivots(num_gates_per_window, kernel_stream1);

        // Sync pivots wth host.
        SYNC(kernel_stream1); 
        
        gpu_circuit.print_pivots();

        Pivot* host_pivots = gpu_circuit.host_pivots();
        int64 random_measures = 0;

        bool initial_pivot = true;
        for(size_t i = 0; i < num_gates_per_window; i++) {
            const Pivot curr_pivot = host_pivots[i];
            const Gate& curr_gate = circuit.gate(depth_level, i);
            const qubit_t qubit = curr_gate.wires[0];
            Pivot new_pivot;
            if (curr_pivot.indeterminate != INVALID_PIVOT) {
                if (initial_pivot) {
                    new_pivot = curr_pivot;
                    uint32 mark_blocksize = 4, mark_gridsize = 1;
                    OPTIMIZEBLOCKS(mark_gridsize, num_qubits, mark_blocksize);
                    mark_anti_commutations <<<mark_gridsize, mark_blocksize, 0, kernel_stream1>>> (tableau.commutations(), qubit, tableau.xtable(), num_qubits, num_words_major, num_words_minor);
                    initial_pivot = false;
                    LASTERR("failed to mark_anti_commutations");
                    SYNC(kernel_stream1);
                }
                else {
                    find_pivots(tableau, i, false, kernel_stream1);
                    gpu_circuit.copypivotto(new_pivot, i, kernel_stream1);
                    SYNC(kernel_stream1);                
                }
                if (new_pivot.indeterminate != INVALID_PIVOT) {
                    LOGN2(2, "Meauring qubit %d using pivot %d.. ", qubit, new_pivot.indeterminate);
                    cutimer.start();
                    #if !DEBUG_INJECT_CX
                    prefix.inject_CX(tableau, new_pivot.indeterminate, qubit, kernel_stream1);
                    #else
                    const uint32 blocksize = 8;
                    const uint32 gridsize = ROUNDUP(num_words_minor, blocksize);
                    inject_CX <<<gridsize, blocksize, 0, kernel_stream1>>> (XZ_TABLE(tableau), tableau.signs(), 
                                tableau.commutations(), 
                                new_pivot.indeterminate, qubit, 
                                num_qubits, num_words_major, num_words_minor);
                    LASTERR("failed to inject_CX");
                    SYNC(kernel_stream1);
                    #endif
                    cutimer.stop();
                    printf("inject CX time = %.3f\n", cutimer.time());
                    // printf("qubit(%d), pivot(%d):\n", circuit.gate(depth_level, i).wires[0], new_pivot.indeterminate), print_tableau(tableau, depth_level, false);
                    check_x_destab <<<1, 1, 0, kernel_stream1>>> (
                                tableau.commutations(), 
                                tableau.xtable(), 
                                new_pivot.indeterminate, 
                                qubit, 
                                num_words_major);

                    uint32 inject_swap_blocksize = 4;
                    uint32 inject_swap_gridsize = 1;
                    OPTIMIZEBLOCKS(inject_swap_gridsize, num_words_minor, inject_swap_blocksize);
                    inject_Swap <<<inject_swap_gridsize, inject_swap_blocksize, 0, kernel_stream1>>> (
                        XZ_TABLE(tableau), 
                        tableau.signs(), 
                        tableau.commutations(), 
                        new_pivot.indeterminate, 
                        num_words_major, 
                        num_words_minor);
                    
                    LOGDONE(2, 4);
                    
                    //printf("After inject_CX:\n"), print_tableau(prefix_tableau, depth_level, false, true);
                    SYNC(kernel_stream1); printf("After signs update for pivot %d:\n", new_pivot.indeterminate);//, print_tableau(tableau, depth_level, false, false);
                }
            }
        }

        print_tableau(tableau, depth_level, false, false);

        //
        //// Measure all determinate.
        //measure_determinate(num_gates_per_window, true, kernel_stream1);

        //// Sync copying pivots.
        //SYNC(copy_stream1);

        //Measures& measure_stats = stats.circuit.measure_stats;
        //measure_stats.random_per_window = measure_indeterminate(depth_level, kernel_stream1);
        //measure_stats.random += measure_stats.random_per_window;
        //measure_stats.definite += num_gates_per_window - measure_stats.random_per_window;

        // Transpose the tableau back into column-major format.
        transpose(false, kernel_stream1);
    }

    int64 Simulator::measure_indeterminate(const depth_t& depth_level, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_gates_per_window = circuit[depth_level].size();
        Pivot* host_pivots = gpu_circuit.host_pivots();
        Pivot new_pivot;
        int64 random_measures = 0;
        


        return random_measures;
    }
}

