
#pragma once

#include "definitions.hpp"
#include "tableau.cuh"
#include "operators.cuh"
#include "pivot.cuh"

namespace QuaSARQ {

	bool is_commuting_cpu(
		const 	Table&          h_zs,
		const   qubit_t         qubit,
		const   pivot_t         pivot,
		const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded);

	__global__
    void inject_swap_k(
                Table*          inv_xs,
                Table*          inv_zs,
                Signs*          inv_ss,
                pivot_t*        pivots,
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
		const   qubit_t         qubit,
        const   sign_t          random_bit,
		const 	size_t& 		num_words_major,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		num_qubits_padded);
	
}