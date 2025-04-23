#pragma once

#include "datatypes.cuh"
#include "tableau.cuh"
#include "grid.cuh"

namespace QuaSARQ {

void tune_outplace_transpose(
        void (*kernel)(
                Table*, 
                Table*, 
                const_table_t, 
                const_table_t, 
        const 	size_t, 
        const 	size_t, 
        const 	size_t),
        const 	char* 			opname, 
                dim3& 			bestBlock, 
                dim3& 			bestGrid,
        const 	size_t& 		shared_element_bytes, 
        const 	bool& 			shared_size_yextend,
        const 	size_t& 		data_size_in_x, 
        const 	size_t& 		data_size_in_y,
                Table* 			xs1, 
                Table* 			zs1,
                const_table_t 	        xs2, 
                const_table_t 	        zs2,
        const 	size_t& 		num_words_major, 
        const 	size_t& 		num_words_minor, 
        const 	size_t& 		num_qubits_padded,
        const 	bool&	 		row_major);

void tune_inplace_transpose(
    void (*transpose_tiles_kernel)(
            Table*, 
            Table*, 
    const 	size_t, 
    const 	size_t, 
    const 	bool),
    void (*swap_tiles_kernel)(
            Table*, 
            Table*, 
    const 	size_t, 
    const 	size_t),
            dim3& 	bestBlockTransposeBits, 
            dim3& 	bestGridTransposeBits,
            dim3& 	bestBlockTransposeSwap, 
            dim3& 	bestGridTransposeSwap,
            Table*	xs, 
            Table* 	zs,
    const 	size_t& num_words_major, 
    const 	size_t& num_words_minor, 
    const 	bool& 	row_major);

}