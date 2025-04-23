
#ifndef __CU_WORD_H
#define __CU_WORD_H

#include <cassert>
#include "definitions.cuh"
#include "logging.hpp"
#include "macros.cuh"

#ifdef __GNUC__ 
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
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

    typedef uint32 qubit_t;

    constexpr qubit_t MAX_QUBITS = UINT32_MAX;
    constexpr qubit_t INVALID_QUBIT = UINT32_MAX;
    constexpr size_t MAX_WORDS = size_t(MAX_QUBITS - 1) * size_t(MAX_QUBITS - 1) / 2;

    #define BITMASK_GLOBAL(VAL) (word_std_t(1) << word_std_t(VAL & WORD_MASK))
    #define BITMASK(VAL) (word_std_t(1) << word_std_t(VAL))

	class word_t {

		word_std_t word;

    public:

        INLINE_ALL constexpr word_t(): word(0) { }

        INLINE_ALL constexpr word_t(const word_std_t& value) : word(value) { }

        INLINE_ALL constexpr word_t(const word_t& other) : word(other.word) { }

        INLINE_ALL operator bool() const { return bool(word); }

        INLINE_ALL operator uint32() const { return uint32(word); }

        INLINE_ALL operator uint64() const { return uint64(word); }

        #if defined(WORD_SIZE_8)
        INLINE_ALL operator uint32() const { return uint32(word); }
        #endif

        INLINE_ALL bool operator[] (const size_t& q) const {
            return bool(word & BITMASK_GLOBAL(q));
        }

        INLINE_ALL bool operator==(const word_t& other) const {
            return word == other.word;
        }

        INLINE_ALL bool operator!=(const word_t& other) const {
            return word != other.word;
        }

        INLINE_ALL bool operator!=(const word_std_t& other) const {
            return word != other;
        }

        INLINE_ALL word_t& operator=(const word_t& other) {
            word = other.word;
            return *this;
        }

        INLINE_ALL word_t& operator&=(const word_t& other) {
            word &= other.word;
            return *this;
        }

        INLINE_ALL word_t& operator|=(const word_t& other) {
            word |= other.word;
            return *this;
        }

        INLINE_ALL word_t& operator&=(const word_std_t& other) {
            word &= other;
            return *this;
        }

        INLINE_ALL word_t& operator|=(const word_std_t& other) {
            word |= other;
            return *this;
        }

        INLINE_ALL word_t& operator^=(const word_std_t& other) {
            word ^= other;
            return *this;
        }

        INLINE_ALL bool operator!() const {
            return !word;
        }

        INLINE_ALL word_t operator~() const {
            return word_t(~word);
        }

        INLINE_ALL word_t operator^(const word_t& other) const {
            return word_t(word ^ other.word);
        }

        INLINE_ALL word_t operator&(const word_t& other) const {
            return word_t(word & other.word);
        }

        INLINE_ALL void identity(const qubit_t& word_idx) {
            word = BITMASK_GLOBAL(word_idx);
        }

        INLINE_ALL bool is_identity(const qubit_t& word_idx) {
            return word == BITMASK_GLOBAL(word_idx);
        }

	};

    constexpr size_t get_num_padded_bits(const size_t& min_qubits) {
        return (min_qubits + WORD_MASK) & ~WORD_MASK;
    }

    constexpr size_t get_num_words(size_t min_qubits) {
        const size_t num_words = get_num_padded_bits(min_qubits) / WORD_BITS;
        if (num_words > MAX_WORDS) {
            LOGERROR("Number of words %zd exceeded the supprted maximum of %zd.", num_words, size_t(MAX_QUBITS));
        }
        return num_words;
    }

#if defined(WORD_SIZE_8)

    #define B2B_STR "%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c"
    #define RB2B(WORD)  \
    ((WORD) & 0x01 ? '1' : '0'), \
    ((WORD) & 0x02 ? '1' : '0'), \
    ((WORD) & 0x04 ? '1' : '0'), \
    ((WORD) & 0x08 ? '1' : '0'), \
    ((WORD) & 0x10 ? '1' : '0'), \
    ((WORD) & 0x20 ? '1' : '0'), \
    ((WORD) & 0x40 ? '1' : '0'), \
    ((WORD) & 0x80 ? '1' : '0') 

#elif defined(WORD_SIZE_32) || defined(WORD_SIZE_64)

    #define B2B_STR "%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c%-3c"
    #define RB2B(WORD)  \
    ((WORD) & 0x00000001UL ? '1' : '0'), \
    ((WORD) & 0x00000002UL ? '1' : '0'), \
    ((WORD) & 0x00000004UL ? '1' : '0'), \
    ((WORD) & 0x00000008UL ? '1' : '0'), \
    ((WORD) & 0x00000010UL ? '1' : '0'), \
    ((WORD) & 0x00000020UL ? '1' : '0'), \
    ((WORD) & 0x00000040UL ? '1' : '0'), \
    ((WORD) & 0x00000080UL ? '1' : '0'), \
    ((WORD) & 0x00000100UL ? '1' : '0'), \
    ((WORD) & 0x00000200UL ? '1' : '0'), \
    ((WORD) & 0x00000400UL ? '1' : '0'), \
    ((WORD) & 0x00000800UL ? '1' : '0'), \
    ((WORD) & 0x00001000UL ? '1' : '0'), \
    ((WORD) & 0x00002000UL ? '1' : '0'), \
    ((WORD) & 0x00004000UL ? '1' : '0'), \
    ((WORD) & 0x00008000UL ? '1' : '0'), \
    ((WORD) & 0x00010000UL ? '1' : '0'), \
    ((WORD) & 0x00020000UL ? '1' : '0'), \
    ((WORD) & 0x00040000UL ? '1' : '0'), \
    ((WORD) & 0x00080000UL ? '1' : '0'), \
    ((WORD) & 0x00100000UL ? '1' : '0'), \
    ((WORD) & 0x00200000UL ? '1' : '0'), \
    ((WORD) & 0x00400000UL ? '1' : '0'), \
    ((WORD) & 0x00800000UL ? '1' : '0'), \
    ((WORD) & 0x01000000UL ? '1' : '0'), \
    ((WORD) & 0x02000000UL ? '1' : '0'), \
    ((WORD) & 0x04000000UL ? '1' : '0'), \
    ((WORD) & 0x08000000UL ? '1' : '0'), \
    ((WORD) & 0x10000000UL ? '1' : '0'), \
    ((WORD) & 0x20000000UL ? '1' : '0'), \
    ((WORD) & 0x40000000UL ? '1' : '0'), \
    ((WORD) & 0x80000000UL ? '1' : '0')

#endif

}

#endif 