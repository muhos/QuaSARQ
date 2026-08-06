
#pragma once

#include "definitions.hpp"
#include "tableau.cuh"
#include "operators.cuh"
#include "pivot.cuh"

namespace QuaSARQ {

	// (X, Z) are the stab/destab words being updated, (SX, SZ) the same pair
    // read in the basis-conjugated frame. They coincide in the Z-basis.
    #define do_Sdg_Swap(X, Z, SX, SZ, S) \
    { \
        const word_std_t x = X, z = Z; \
        X = x ^ z; \
        S ^= (SX & ~SZ); \
    }

    #define do_H_Swap(X, Z, SX, SZ, S) \
    { \
        do_SWAP(X, Z); \
        S ^= word_std_t(SX & SZ); \
    }

	bool is_commuting_cpu(
		const 	Table&          h_xs,
		const 	Table&          h_zs,
		const   qubit_t         qubit,
		const   pivot_t         pivot,
		const   byte_t          gate_type,
		const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded);

	__global__
    void inject_swap_k(
                Table*          inv_xs,
                Table*          inv_zs,
                Signs*          inv_ss,
                pivot_t*        pivots,
        const   byte_t          gate_type,
        const   qubit_t         qubit,
        const   sign_t          random_bit,
        const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded);

	void tune_inject_swap(
		void (*kernel)(
				Table*,
				Table*,
				Signs*,
				pivot_t*,
		const   byte_t,
		const   qubit_t,
        const   sign_t,
		const 	size_t,
		const 	size_t,
		const 	size_t),
				dim3& 			bestBlock,
				dim3& 			bestGrid,
				Table* 			xs,
				Table* 			zs,
				Signs* 			ss,
				pivot_t* 		pivots,
		const   byte_t          gate_type,
		const   qubit_t         qubit,
        const   sign_t          random_bit,
		const 	size_t& 		num_words_major,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		num_qubits_padded);
	
}