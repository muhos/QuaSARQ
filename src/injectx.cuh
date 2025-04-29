
#pragma once

#include "definitions.hpp"
#include "tableau.cuh"
#include "operators.cuh"
#include "pivot.cuh"

namespace QuaSARQ {

	#define inject_X(Z, S) \
	{ \
		S ^= Z; \
	}
	
	__global__ 
    void inject_x_k(
                Table*              inv_xs, 
                Table*              inv_zs,
                Signs*              inv_ss, 
                const_pivots_t      pivots,
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded);

	void tune_inject_x(
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