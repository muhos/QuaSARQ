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

    #define do_depolarize1(signs_word, x_words_q1, z_words_q1, pauli) \
    { \
        if (pauli & 1u) sign_update_X_or_Z(signs_word, z_words_q1); /* X error */ \
        if (pauli & 2u) sign_update_X_or_Z(signs_word, x_words_q1); /* Z error */ \
    }

    #define do_depolarize2(signs_word, x_words_q1, z_words_q1, x_words_q2, z_words_q2, pauli) \
    { \
        if (pauli & 1u) sign_update_X_or_Z(signs_word, z_words_q1); /* X on q1 */ \
        if (pauli & 2u) sign_update_X_or_Z(signs_word, x_words_q1); /* Z on q1 */ \
        if (pauli & 4u) sign_update_X_or_Z(signs_word, z_words_q2); /* X on q2 */ \
        if (pauli & 8u) sign_update_X_or_Z(signs_word, x_words_q2); /* Z on q2 */ \
    }

}
