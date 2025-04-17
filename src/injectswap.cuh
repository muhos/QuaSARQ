
#pragma once

#include "definitions.hpp"
#include "tableau.cuh"
#include "pivot.cuh"

namespace QuaSARQ {

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

	bool is_commuting_cpu(
		const 	Table&          h_xs, 
		const   qubit_t         qubit,
		const   pivot_t         pivot,
		const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded);
	
	__global__ 
    void inject_swap_k(
                Table*              inv_xs, 
                Table*              inv_zs,
                Signs*              inv_ss, 
                const_pivots_t      pivots,
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded);

	void tune_inject_swap(
		void (*kernel)(
				Table*, 
				Table*,
				Signs*,
				const_pivots_t,
		const 	size_t, 
		const 	size_t, 
		const 	size_t),
				dim3& 			bestBlock,
				dim3& 			bestGrid,
				Table* 			xs,
				Table* 			zs,
				Signs* 			ss,
				const_pivots_t 	pivots,
		const 	size_t& 		num_words_major,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		num_qubits_padded);
	
}