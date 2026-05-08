#pragma once

#include "datatypes.hpp"

#if !defined(WORD_SIZE_8) && !defined(WORD_SIZE_32) && !defined(WORD_SIZE_64)
#define WORD_SIZE_64
#endif

namespace QuaSARQ {

#if defined(WORD_SIZE_8)
    #define POPC __popc
    typedef byte_t word_std_t;
    constexpr size_t WORD_POWER = 3;
    constexpr word_std_t WORDS_MAX = UINT8_MAX;
#elif defined(WORD_SIZE_32)
    #define POPC __popc
    typedef uint32 word_std_t;
    constexpr size_t WORD_POWER = 5;
    constexpr word_std_t WORDS_MAX = UINT32_MAX;
#elif defined(WORD_SIZE_64)
    #define POPC __popcll
    typedef uint64 word_std_t;
    constexpr size_t WORD_POWER = 6;
    constexpr word_std_t WORDS_MAX = UINT64_MAX;
#endif

    constexpr size_t WORD_BITS = sizeof(word_std_t) * 8ULL;
    constexpr size_t WORD_MASK = WORD_BITS - 1;

    constexpr qubit_t MAX_QUBITS = UINT32_MAX;
    constexpr qubit_t INVALID_QUBIT = UINT32_MAX;
    constexpr size_t MAX_WORDS = size_t(MAX_QUBITS - 1) * size_t(MAX_QUBITS - 1) / 2;

}