#ifndef __CU_STEP_H
#define __CU_STEP_H

#include "datatypes.cuh"

namespace QuaSARQ {

    __global__ 
    void step_2D_atomic(ConstRefsPointer refs, ConstBucketsPointer gates, const size_t num_gates, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss);

    void call_step_2D(
                ConstRefsPointer    refs,
                ConstBucketsPointer gates,
                Tableau&            tableau,
        const   size_t              num_gates_per_window,
        const   size_t              num_words_major,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   size_t              shared_size,
        const   cudaStream_t&       stream);

    void tune_step(
		const 	char* 				opname,
				dim3& 				bestBlock,
				dim3& 				bestGrid,
		const 	size_t& 			shared_element_bytes,
		const 	bool& 				shared_size_yextend,
		const 	size_t& 			data_size_in_x,
		const 	size_t& 			data_size_in_y,
				ConstRefsPointer 	gate_refs,
				ConstBucketsPointer gate_buckets,
				Tableau& 			tableau);

}

#endif