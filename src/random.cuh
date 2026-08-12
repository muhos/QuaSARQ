#pragma once

#include "definitions.cuh"
#include <curand_kernel.h>
#include "word.cuh"

namespace QuaSARQ {

    // Counter-based randomness. The value is a pure function of the seed and the coordinates
    // of what is being randomised. Coordinates are (shot word, row, step) and a purpose tag, 
    // which together must be unique for every draw in a run.
    #define RAND_TAG_FRAME   0u     // initial random frames
    #define RAND_TAG_RESET   1u     // frame written by a reset
    #define RAND_TAG_FLIP    2u     // measurement flip noise
    #define RAND_TAG_NOISE   3u     // gate error channels
    #define RAND_TAG_PAULI   4u     // pauli choice within a depolarising channel

    INLINE_DEVICE
    uint4 counter_bits(
        const uint64&   seed,
        const uint32&   tag_and_draw,
        const uint32&   shot_word,
        const uint32&   row,
        const uint32&   step)
    {
        return curand_Philox4x32_10(make_uint4(shot_word, row, step, tag_and_draw),
                                    make_uint2(uint32(seed), uint32(seed >> 32)));
    }

    INLINE_DEVICE
    word_std_t counter_word(
        const uint64&   seed,
        const uint32&   tag,
        const uint32&   shot_word,
        const uint32&   row,
        const uint32&   step)
    {
        const uint4 r = counter_bits(seed, tag << 16, shot_word, row, step);
        #if defined(WORD_SIZE_8)
            return static_cast<word_std_t>(r.x & 0xFFu);
        #elif defined(WORD_SIZE_32)
            return static_cast<word_std_t>(r.x);
        #else
            return (static_cast<word_std_t>(r.x) << 32) | static_cast<word_std_t>(r.y);
        #endif
    }

    // One word with each bit set independently with 'probability'. Four bits per Philox call.
    INLINE_DEVICE
    word_std_t counter_error_mask(
        const uint64&   seed,
        const uint32&   tag,
        const uint32&   shot_word,
        const uint32&   row,
        const uint32&   step,
        const float&    probability)
    {
        if (probability <= 0.0f) return 0;
        if (probability >= 1.0f) return ~word_std_t(0);
        const uint32 threshold = uint32(probability * 4294967296.0);
        word_std_t mask = 0;
        #pragma unroll
        for (uint32 draw = 0; draw < WORD_BITS / 4; draw++) {
            const uint4 r = counter_bits(seed, (tag << 16) | draw, shot_word, row, step);
            const uint32 base = draw * 4;
            if (r.x < threshold) mask |= word_std_t(1) << (base + 0);
            if (r.y < threshold) mask |= word_std_t(1) << (base + 1);
            if (r.z < threshold) mask |= word_std_t(1) << (base + 2);
            if (r.w < threshold) mask |= word_std_t(1) << (base + 3);
        }
        return mask;
    }

}
