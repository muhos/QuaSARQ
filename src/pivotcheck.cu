
#include "measurement.cuh"

namespace QuaSARQ {

    bool is_anti_commuting_cpu(
		const 	Table&          h_xs, 
		const   qubit_t         qubit,
		const   pivot_t         index,
		const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded) 
	{
		if (qubit == INVALID_QUBIT) {
            LOGERROR("qubit not set");
        }
        if (index >= num_qubits_padded) {
            LOGERROR("index is larger than number of qubits");
        }
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        const size_t word_idx = TABLEAU_INDEX(q_w, index) + TABLEAU_STAB_OFFSET;
        const word_std_t qubit_word = h_xs[word_idx];
        return bool(qubit_word & q_mask);
	}

    void MeasurementChecker::find_min_pivot(const qubit_t& qubit, const bool& store_pivots) {
        SYNCALL;
        if (qubit == INVALID_QUBIT) {
            LOGERROR("qubit not set");
        }
        pivot = INVALID_PIVOT;
        for(size_t i = 0; i < num_qubits; i++) {
            const 
            bool anti_commuting = is_anti_commuting_cpu(
                h_xs,
                qubit,
                i,
                num_words_major,
                num_words_minor,
                num_qubits_padded
            );
            if (anti_commuting) {
                pivot = MIN(pivot_t(i), pivot);
                if (store_pivots) {
                    h_compact_pivots.push(i);
                }
            }
        }
    }

    void MeasurementChecker::check_min_pivot(const pivot_t& other_pivot) {
        SYNCALL;
        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }
        if (qubit == INVALID_QUBIT) {
            LOGERROR("qubit not set");
        }
        find_min_pivot(qubit);
        if (pivot != other_pivot)
            LOGERROR("pivots do not match");
    }

    void MeasurementChecker::check_initial_pivots(const Circuit& circuit, const depth_t& depth_level, const pivot_t* other_pivots, const size_t& other_num_pivots) {
        SYNCALL;
        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }
        if (qubit != INVALID_QUBIT) {
            LOGERROR("qubit should not be set for initial pivots");
        }
        LOGN2(2, " Checking initial pivots for depth level %d.. ", depth_level);
        const auto num_gates = circuit[depth_level].size();
        if (num_gates != other_num_pivots)
            LOGERROR("number of initial pivots %llu does not match number of gates %llu", 
                other_num_pivots, num_gates);
        for (auto i = 0; i < num_gates; i++) {
            const Gate& m = circuit.gate(depth_level, i);
            assert(m.size == 1);
            find_min_pivot(m.wires[0]);
            if (pivot != other_pivots[i])
                LOGERROR("minimum pivot %d (calculated by CPU) and %d and do not match at gate index %lld", 
                    pivot, other_pivots[i], size_t(i));
        }
        LOG2(2, "PASSED");
    }

    void MeasurementChecker::check_compact_pivots(const qubit_t& qubit, const pivot_t* other_pivots, const size_t& other_num_pivots) {
        SYNCALL;
        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }
        LOGN2(2, " Checking compact pivots for qubit %d.. ", qubit);
        h_compact_pivots.clear();
        this->qubit = qubit;
        find_min_pivot(qubit, true);
        if (h_compact_pivots.size() != other_num_pivots)
            LOGERROR("number of compact pivots %u does not match cpu-based number %u", 
                other_num_pivots, h_compact_pivots.size());
        copy_pivots(other_pivots, other_num_pivots);
        for (size_t i = 0; i < h_compact_pivots.size(); i++) {
            if (h_compact_pivots[i] != d_compact_pivots[i])
                LOGERROR("pivots %d (calculated by CPU) and %d and do not match at index %lld", 
                    h_compact_pivots[i], d_compact_pivots[i], i);
        }
        LOG2(2, "PASSED");
    }

}