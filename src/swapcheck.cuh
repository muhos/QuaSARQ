
#ifndef __CU_SWAPCHECK_H
#define __CU_SWAPCHECK_H

#include "vector.hpp"
#include "pivot.cuh"
#include "table.cuh"
#include "access.cuh"
#include "operators.cuh"
#include "commutation.cuh"

namespace QuaSARQ {

	inline 
	bool is_commuting_cpu(
		const 	Table&          h_xs, 
		const   qubit_t         qubit,
		const   pivot_t         pivot,
		const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded) 
	{
		assert(pivot != INVALID_PIVOT);
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        const size_t word_idx = TABLEAU_INDEX(q_w, pivot);
        const word_std_t qubit_word = h_xs[word_idx];
        return bool(qubit_word & q_mask);
	}

	inline
	void inject_swap_cpu(
                Table&          h_xs, 
                Table&          h_zs,
                Signs&          h_ss, 
		const   qubit_t         qubit,
        const   pivot_t         pivot,
        const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded) 
    {
		const 
		bool commuting = is_commuting_cpu(
			h_xs,
			qubit,
			pivot,
			num_words_major,
			num_words_minor,
			num_qubits_padded
		);
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
    }

	inline
	void check_inject_swap(
                Table&          	h_xs, 
                Table&          	h_zs,
                Signs&          	h_ss, 
                Table&          	d_xs, 
                Table&          	d_zs,
                Signs&          	d_ss, 
        const   Vec<Commutation>& 	d_commutations,
        const   qubit_t         	qubit,
        const   pivot_t         	pivot,
        const   size_t          	num_words_major, 
        const   size_t          	num_words_minor,
        const   size_t          	num_qubits_padded) 
    {
        SYNCALL;

        LOGN1("  Checking inject-swap for qubit %d and pivot %d.. ", qubit, pivot);

        const 
		bool commuting = is_commuting_cpu(
			h_xs,
			qubit,
			pivot,
			num_words_major,
			num_words_minor,
			num_qubits_padded
		);

        if (commuting != d_commutations[pivot].commuting) {
            LOGERROR("Commuting bit not identical at pivot(%lld)", pivot);
        }

        inject_swap_cpu(
            h_xs,
            h_zs,
            h_ss,
            qubit,
            pivot,
            num_words_major,
            num_words_minor,
            num_qubits_padded
        );

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
	
}

#endif