
#ifndef __CU_COMMUTATION_H
#define __CU_COMMUTATION_H

#include "definitions.cuh"
#include "datatypes.cuh"
#include "logging.hpp"

namespace QuaSARQ {

    struct Commutation {
        bool commuting : 1;
        bool anti_commuting : 1;

        INLINE_ALL
        Commutation() : commuting(false), anti_commuting(false) {}

        INLINE_ALL
        void reset() {
            commuting = false;
            anti_commuting = false;
        }

        INLINE_ALL
        void print(const bool &nonl = false) const {
            LOGGPU("commuting: %d, anti-commuting: %d", commuting, anti_commuting);
            if (!nonl)
                LOGGPU("\n");
        }
    };

    void tune_marking(
        void (*kernel)(
                Commutation*, 
                ConstTablePointer, 
		const   qubit_t, 
        const   size_t, 
        const   size_t, 
        const   size_t, 
        const   size_t),
		const   char*               opname, 
                dim3&               bestBlock, 
                dim3&               bestGrid,
		        Commutation*        commutations, 
                ConstTablePointer   inv_xs, 
        const   qubit_t             qubit, 
		const   size_t              size, 
        const   size_t              num_words_major, 
        const   size_t              num_words_minor, 
        const   size_t              num_qubits_padded);
    
}

#endif