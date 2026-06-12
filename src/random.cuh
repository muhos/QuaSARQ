#pragma once

#include "definitions.cuh"
#include <curand_kernel.h>
#include "word.cuh"

namespace QuaSARQ {

    // Generate one word_std_t of randomness from a curand state.
    INLINE_DEVICE
    word_std_t curand_word(curand_algorithm_t* state) {
        #if defined(WORD_SIZE_8)
            return static_cast<word_std_t>(curand(state) & 0xFFu);
        #elif defined(WORD_SIZE_32)
            return static_cast<word_std_t>(curand(state));
        #elif defined(WORD_SIZE_64)
            const word_std_t hi = static_cast<word_std_t>(curand(state));
            const word_std_t lo = static_cast<word_std_t>(curand(state));
            return (hi << 32) | lo;
        #endif
    }

    __global__
    void setup_rand_k(
        curand_algorithm_t*       states,
        const uint64              seed,
        const size_t              total_states,
        const size_t              chunk_words_minor,
        const size_t              total_words_minor,
        const size_t              chunk_word_offset);


}
