#pragma once

#include "definitions.cuh"
#include "datatypes.cuh"
#include "operators.cuh"
#include "word.cuh"
#include "grid.cuh"
#include "gate.cuh"

namespace QuaSARQ {

    __global__
    void setup_noise_k(
        curand_algorithm_t*         noise_states,
        const uint64                seed,
        const size_t                max_gates);

    __global__
   void sample_noise_k(
        curand_algorithm_t*         noise_states,
        uint32*                     noise_paulis,
        const_refs_t                refs,
        const_buckets_t             gates,
        const size_t                num_gates);

    INLINE_DEVICE
    void do_depolarize1(
        sign_t&       signs_word,
        word_t&       x_words_q1,
        word_t&       z_words_q1,
        const uint32& pauli)
    {
        if (pauli & 1u) sign_update_X_or_Z(signs_word, z_words_q1); // X error
        if (pauli & 2u) sign_update_X_or_Z(signs_word, x_words_q1); // Z error
    }

}
