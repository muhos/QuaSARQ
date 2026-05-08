#pragma once

#include "definitions.cuh"
#include "operators.cuh"
#include "random.cuh"
#include "word.cuh"
#include "grid.cuh"

namespace QuaSARQ {

    INLINE_DEVICE
    void do_depolarize1(
        sign_t& signs_word, 
        word_t& x_words_q1,
        word_t& z_words_q1,
        curandStatePhilox4_32_10_t& state, 
        const float& noise_p, 
        const uint64& seed, 
        const grid_t& tid) 
    {
        const float r = random_uniform(state, seed, tid);
        printf("--> Depolarizing gate %llu with p = %.3f, r = %.3f\n", tid, noise_p, r);
        if (r < noise_p) {
            const uint32 p = 1 + (curand(&state) % 3);
            if (p & 1) sign_update_X_or_Z(signs_word, z_words_q1); // X
            if (p & 2) sign_update_X_or_Z(signs_word, x_words_q1); // Z
        }
    }

}