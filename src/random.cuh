#pragma once

#include <curand_kernel.h>
#include "definitions.cuh"
#include "word.cuh"

namespace QuaSARQ {

    __global__
    void setup_rand_k(
        curand_algorithm_t* states,
        uint64              seed,
        size_t              total_states);


}