#include "simulator.hpp"
#include "prefix.cuh"
#include "tuner.cuh"
#include "access.cuh"
#include "operators.cuh"
#include "swapcheck.cuh"

namespace QuaSARQ {;

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

        if (options.tune_measurement)
            tune_assuming_maximum_targets(depth_level);
        else {
            int64 random_measures = measure_indeterminate(depth_level, kernel_stream1);
            stats.circuit.measure_stats.random += random_measures;
            stats.circuit.measure_stats.random_per_window = MAX(random_measures, stats.circuit.measure_stats.random_per_window);
        }

        transpose(false, kernel_stream1);
    }

    void Simulator::tune_assuming_maximum_targets(const depth_t& depth_level) {
		const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        const size_t num_gates_per_window = circuit[depth_level].size();
		PrefixChecker& checker = prefix.get_checker();
		const size_t pass_1_blocksize = 512;
		const size_t max_intermediate_blocks = nextPow2(ROUNDUP(num_qubits, MIN_BLOCK_INTERMEDIATE_SIZE));
		Tableau dummy_input(gpu_allocator);
		Tableau dummy_targets(gpu_allocator);
		size_t max_targets = 0;
        pivot_t min_pivot = INVALID_PIVOT;
        // for(size_t i = 0; i < num_gates_per_window; i++) {
        //     const Gate& curr_gate = circuit.gate(depth_level, i);
        //     const qubit_t qubit = curr_gate.wires[0];
		// 	checker.find_new_pivot(qubit, tableau);
		// 	const pivot_t pivot = checker.pivot;
		// 	if (checker.pivot != INVALID_PIVOT) {
		// 		LOG2(2, "Meauring qubit %d using pivot %d.. ", qubit, pivot);
		// 		const size_t total_targets = num_qubits - pivot - 1;
		// 		if (!total_targets) continue;
		// 		if (max_targets < total_targets) {
        //             max_targets = total_targets;
        //             min_pivot = pivot;
        //         }
		// 		const size_t pass_1_gridsize = ROUNDUP(total_targets, pass_1_blocksize);
		// 		checker.check_prefix_pass_1(
		// 		dummy_targets,
		// 		nullptr,
		// 		nullptr, 
		// 		nullptr,
		// 		total_targets,
		// 		max_intermediate_blocks,
		// 		pass_1_blocksize,
		// 		pass_1_gridsize,
        //         true);

		// 		checker.check_prefix_intermediate_pass(
		// 			nullptr, 
		// 			nullptr,
		// 			max_intermediate_blocks,
		// 			pass_1_gridsize,
        //             true);

		// 		checker.check_prefix_pass_2(
		// 			dummy_targets, 
		// 			dummy_input,
		// 			total_targets, 
		// 			max_intermediate_blocks,
		// 			pass_1_blocksize,
        //             true);

		// 		inject_swap_cpu(
		// 			checker.h_xs,
		// 			checker.h_zs,
		// 			checker.h_ss,
		// 			qubit,
		// 			pivot,
		// 			num_words_major,
		// 			num_words_minor,
		// 			num_qubits_padded
		// 		);
		// 	}
		// }
        LOG2(2, "Maximum targets is %lld for minimum pivot %d", max_targets, min_pivot);

        min_pivot = 43, max_targets = num_qubits - min_pivot - 1;
        
        prefix.set_min_pivot(min_pivot, max_targets);
        mark_commutations(min_pivot, 0);
        prefix.tune_inject_cx(tableau, commutations);
        if (options.tune_injectswap) {
            SYNCALL;
            tune_kernel_m(inject_swap_k, "injecting swap", 
                        bestblockinjectswap, bestgridinjectswap, 
                        XZ_TABLE(tableau),
                        tableau.signs(),
                        commutations,
                        min_pivot,
                        num_words_major,
                        num_words_minor,
                        num_qubits_padded);
        }
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

