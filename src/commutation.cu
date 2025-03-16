#include "simulator.hpp"
#include "commutation.cuh"
#include "tuner.cuh"

namespace QuaSARQ {

    __global__ void mark_anti_commutations(Commutation* commutations, ConstTablePointer inv_xs, const qubit_t q, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        const qubit_t q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        for_parallel_x(g, num_qubits) {
            const size_t word_idx = g * num_words_major + q_w + num_words_minor;
            const word_std_t qubit_word = (*inv_xs)[word_idx];
            if (qubit_word & q_mask) {
                commutations[g].anti_commuting = true;
            }
            else {
                commutations[g].reset();
            }
        }
    }

    void Simulator::mark_commutations(const qubit_t& qubit, const cudaStream_t& stream) {
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_words_minor = tableau.num_words_minor();
        if (options.tune_marking) {
            SYNCALL;
            tune_kernel_m(mark_anti_commutations, "Marking anit-commutations", 
                            bestblockmarking, bestgridmarking, 
                            tableau.commutations(), tableau.xtable(), 
                            qubit, num_qubits, num_words_major, num_words_minor);
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockmarking, bestgridmarking, num_qubits, 0);
        dim3 currentblock = bestblockmarking, currentgrid = bestgridmarking;
        TRIM_GRID_IN_1D(num_qubits, x);
        mark_anti_commutations <<<currentblock, currentgrid, 0, stream>>> 
        (
            tableau.commutations(), 
            tableau.xtable(), 
            qubit, num_qubits, 
            num_words_major, 
            num_words_minor
        );
        if (options.sync) {      
            LASTERR("failed to mark_anti_commutations");
            SYNC(stream);
        }
    }

}

