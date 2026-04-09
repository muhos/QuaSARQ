#include <curand_kernel.h>
#include "definitions.cuh"
#include "word.cuh"

namespace QuaSARQ {

    INLINE_DEVICE void randomize_word(
        word_std_t&                 word, 
        curandStatePhilox4_32_10_t& state, 
        const uint64&               seed, 
        const size_t&               tid) 
    {
        curand_init(seed, tid, 0, &state);
        #if defined(WORD_SIZE_8)
            word = static_cast<word_std_t>(curand(&state) & 0xFFu);
        #elif defined(WORD_SIZE_32)
            word = static_cast<word_std_t>(curand(&state));

        #elif defined(WORD_SIZE_64)
            word_std_t hi = static_cast<word_std_t>(curand(&state));
            word_std_t lo = static_cast<word_std_t>(curand(&state));
            word = (hi << 32) | lo;
        #endif
    }

    INLINE_DEVICE void randomize_word(
        word_t&                     word, 
        curandStatePhilox4_32_10_t& state, 
        const uint64&               seed, 
        const size_t&               tid) 
    {
        curand_init(seed, tid, 0, &state);
        #if defined(WORD_SIZE_8)
            word = static_cast<word_t>(curand(&state) & 0xFFu);
        #elif defined(WORD_SIZE_32)
            word = static_cast<word_t>(curand(&state));

        #elif defined(WORD_SIZE_64)
            word_std_t hi = static_cast<word_std_t>(curand(&state));
            word_std_t lo = static_cast<word_std_t>(curand(&state));
            word = word_t((hi << 32) | lo);
        #endif
    }

}