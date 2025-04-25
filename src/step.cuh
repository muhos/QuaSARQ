#pragma once

#include "datatypes.cuh"

namespace QuaSARQ {

    #define DEBUG_STEP 0

    __global__ 
    void step_2D_atomic(
                const_refs_t 	refs,
                const_buckets_t gates,
        const 	size_t 			num_gates,
        const 	size_t 			num_words_major,
                Table *			xs, 
                Table *			zs,
                Signs *			ss);

    void call_step_2D(
                const_refs_t 	refs,
                const_buckets_t gates,
                Tableau &		tableau,
        const 	size_t 			num_gates_per_window,
        const 	size_t 			num_words_major,
        const 	dim3 &			currentblock,
        const 	dim3 &			currentgrid,
        const 	size_t 			shared_size,
        const 	cudaStream_t &	stream);

    void tune_step(
                dim3 &			bestBlock,
                dim3 &			bestGrid,
        const 	size_t &		shared_element_bytes,
        const 	bool &			shared_size_yextend,
        const 	size_t &		data_size_in_x,
        const 	size_t &		data_size_in_y,
                const_refs_t 	gate_refs,
                const_buckets_t gate_buckets,
                Tableau &		tableau);

}