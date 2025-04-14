#ifndef __CU_STEP_H
#define __CU_STEP_H

#include "datatypes.cuh"
#include "collapse.cuh"
#include "circuit.cuh"
#include "vector.cuh"
#include "atomic.cuh"
#include "print.cuh"
#include "grid.cuh"
#include "timer.cuh"
#include "timer.hpp"

namespace QuaSARQ {

    // Simulate a single window per circuit.
    __global__ void step_2D_atomic(ConstRefsPointer refs, ConstBucketsPointer gates, const size_t num_gates, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss);

}

#endif