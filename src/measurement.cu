#include "simulator.hpp"
#include "prefix.cuh"
#include "tuner.cuh"
#include "access.cuh"
#include "operators.cuh"

namespace QuaSARQ {;

    #define do_YZ_Swap(X, Z, S) \
    { \
        const word_std_t x = X, z = Z; \
        X = x ^ z; \
        S ^= (x & ~z); \
    }

    #define do_XZ_Swap(X, Z, S) \
    { \
        do_SWAP(X, Z); \
        S ^= word_std_t(X & Z); \
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
                Table*              inv_xs, 
                Table*              inv_zs,
                Signs*              inv_ss, 
                ConstCommsPointer   commutations, 
        const   pivot_t             pivot,
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded) 
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

    void check_inject_swap(
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
            LOGERROR("Commuting bit not identical at pivot(%lld)", pivot);
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
                LOGERROR("X-Stabilizer failed at w(%lld), pivot(%lld)", w, pivot);
            }
            if (h_zs[c_destab] != d_zs[c_destab]) {
                LOGERROR("Z-Stabilizer failed at w(%lld), pivot(%lld)", w, pivot);
            }
            if (h_ss[w] != d_ss[w]) {
                LOGERROR("Destabilizer signs failed at w(%lld)", w);
            }
            if (h_ss[w + num_words_minor] != d_ss[w + num_words_minor]) {
                LOGERROR("Stabilizer signs failed at w(%lld)", w + num_words_minor);
            }
        }

        LOG0("PASSED");
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
        LOGN2(2, "Running inject-swap kernel with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", \
            currentblock.x, currentblock.y, currentgrid.x, currentgrid.y); \
        if (options.sync) cutimer.start(stream);
        inject_swap_k<<<currentgrid, currentblock, 0, stream>>> (
            XZ_TABLE(tableau),
            tableau.signs(),
            commutations,
            pivot,
            num_words_major,
            num_words_minor,
            num_qubits_padded);
        if (options.sync) {
            LASTERR("failed to inject swap");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            prefix.get_checker().copy_commutations(commutations, num_qubits);
            prefix.get_checker().copy_input(tableau, true);
            check_inject_swap(
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
            );
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

        reset_pivots(num_gates_per_window, kernel_stream2);

        transpose(true, kernel_stream1);

        // Sync copying gates to device.
        SYNC(copy_stream1);
        SYNC(copy_stream2);
        // Sync resetting pivots.
        SYNC(kernel_stream2);

        find_pivots(tableau, num_gates_per_window, true, kernel_stream1);

        // Copy pivots to host.
        gpu_circuit.copypivots(kernel_stream1, num_gates_per_window);

        // Reset pivots on device side.
        reset_pivots(num_gates_per_window, kernel_stream1);

        // Sync pivots wth host.
        SYNC(kernel_stream1);

        int64 random_measures = measure_indeterminate(depth_level, kernel_stream1);

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
                    prefix.inject_CX(tableau, commutations, new_pivot, qubit, stream);
                    inject_swap(new_pivot, qubit, stream);
                }
            }
        }

        return random_measures;
    }
}

