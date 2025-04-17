#include "simulator.hpp"
#include "injectswap.cuh"

namespace QuaSARQ {

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

        find_pivots(num_gates_per_window, kernel_stream1);

        // Copy source pivots to host.
        pivoting.copypivots(kernel_stream1, num_gates_per_window);

        // Reset pivots on device side.
        reset_pivots(num_gates_per_window, kernel_stream1);

        // Sync pivots wth host.
        SYNC(kernel_stream1);

        if (options.tune_measurement)
            tune_assuming_maximum_targets(depth_level);
        else {
            int64 random_measures = measure_indeterminate(depth_level, kernel_stream1);
            stats.circuit.measure_stats.random += random_measures;
            stats.circuit.measure_stats.definite += num_gates_per_window - random_measures;
            stats.circuit.measure_stats.random_per_window = MAX(random_measures, stats.circuit.measure_stats.random_per_window);
        }

        transpose(false, kernel_stream1);
    }

    void Simulator::tune_assuming_maximum_targets(const depth_t& depth_level) {
		const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        const size_t num_gates_per_window = circuit[depth_level].size();
		uint32 max_targets = ROUNDUP(num_qubits, 10);
        pivot_t min_pivot = 0;
        
        LOG2(2, "Tuning measurements for maximum targets of %u for pivot %u", max_targets, min_pivot);

        Vec<pivot_t> h_pivots(max_targets + 1);
        h_pivots[0] = min_pivot;

        for (uint32 i = 0; i < max_targets; i++) {
            h_pivots[i + 1] = rand() % num_qubits;
        }

        CHECK(cudaMemcpy(pivoting.pivots, h_pivots.data(), sizeof(pivot_t) * (max_targets + 1), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(pivoting.d_active_pivots, &max_targets, sizeof(uint32), cudaMemcpyHostToDevice));

        prefix.tune_inject_cx(tableau, pivoting.pivots, max_targets);
        if (options.tune_injectswap) {
            SYNCALL;
            tune_inject_swap(inject_swap_k,
                        bestblockinjectswap, 
                        bestgridinjectswap, 
                        XZ_TABLE(tableau),
                        tableau.signs(),
                        pivoting.pivots,
                        num_words_major,
                        num_words_minor,
                        num_qubits_padded);
        }
	}

    int64 Simulator::measure_indeterminate(const depth_t& depth_level, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        pivot_t* host_pivots = pivoting.host_pivots;
        if (options.check_measurement) {
            mchecker.copy_input(tableau);
            mchecker.check_initial_pivots(circuit, depth_level, host_pivots, num_gates_per_window);
        }
        int64 random_measures = 0;
        for(size_t i = 0; i < num_gates_per_window; i++) {
            const Gate& curr_gate = circuit.gate(depth_level, i);
            const pivot_t curr_pivot = host_pivots[i];
            const qubit_t qubit = curr_gate.wires[0];
            if (curr_pivot != INVALID_PIVOT) {
                compact_targets(qubit, stream);
                SYNC(stream);
                const uint32 active_pivots = pivoting.h_active_pivots[0];
                if (options.check_measurement)
                    mchecker.check_compact_pivots(qubit, pivoting.pivots, active_pivots);
                if (active_pivots) {
                    if (active_pivots > 1)
                        inject_cx(active_pivots - 1/*active_pivots - 1*/, stream);
                    inject_swap(qubit, stream);
                    random_measures++;
                }
            }
        }

        return random_measures;
    }
}

