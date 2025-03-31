#include "simulator.hpp"
#include "commutation.cuh"
#include "datatypes.cuh"
#include "access.cuh"
#include "tuner.cuh"

namespace QuaSARQ {

    __global__ 
    void mark_anti_commutations(
                Commutation*        commutations, 
                ConstTablePointer   inv_xs, 
        const   qubit_t             qubit, 
        const   size_t              num_qubits, 
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded) 
    {
        assert(qubit < MAX_QUBITS);
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        for_parallel_x(t, num_qubits) {
            const size_t word_idx = TABLEAU_INDEX(q_w, t) + TABLEAU_STAB_OFFSET; 
            const word_std_t qubit_word = (*inv_xs)[word_idx];
            if (qubit_word & q_mask) {
                commutations[t].anti_commuting = true;
            }
            else {
                commutations[t].reset();
            }
        }
    }

    void Simulator::mark_commutations(const qubit_t& qubit, const cudaStream_t& stream) {
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        
        if (options.tune_marking) {
            SYNCALL;
            tune_marking(
                mark_anti_commutations, 
                "Marking anit-commutations", 
                bestblockmarking, 
                bestgridmarking, 
                commutations, 
                tableau.xtable(), 
                qubit, 
                num_qubits, 
                num_words_major, 
                num_words_minor,
                num_qubits_padded);
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockmarking, bestgridmarking, num_qubits, 0);
        dim3 currentblock = bestblockmarking, currentgrid = bestgridmarking;
        TRIM_GRID_IN_1D(num_qubits, x);
        mark_anti_commutations <<<currentblock, currentgrid, 0, stream>>> 
        (
            commutations, 
            tableau.xtable(), 
            qubit, 
            num_qubits, 
            num_words_major, 
            num_words_minor,
            num_qubits_padded
        );
        if (options.sync) {      
            LASTERR("failed to mark_anti_commutations");
            SYNC(stream);
        }
    }

}

