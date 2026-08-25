#pragma once

#include "definitions.cuh"
#include "datatypes.hpp"
#include "signs.cuh"

#include <cooperative_groups.h>

namespace QuaSARQ {

    #define HOST_UNREACHABLE_ATOMIC(ADDR,VAL) \
        do { (void)(ADDR); (void)(VAL); assert(0 && "device-only atomic"); } while (0)

    #define EXTRACT_BYTE_FROM_ADDR(ADDR,VAL) \
	    uint64 addr_val = (uint64)ADDR; \
        uint32 al_offset = uint32(addr_val & 3) << 3; \
        uint32* byte_addr = reinterpret_cast<uint32*> (addr_val & (0xFFFFFFFFFFFFFFFCULL)); \
        uint32 byte = (VAL << al_offset) \

#if defined(WORD_SIZE_8)
    #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
    INLINE_DEVICE word_std_t
    #else
    INLINE_DEVICE void
    #endif
    atomicXOR(word_std_t* addr, const uint32& value) {
        #if defined(__CUDA_ARCH__)
        assert(value <= WORDS_MAX);
		EXTRACT_BYTE_FROM_ADDR(addr, value);
        #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
        return word_std_t((atomicXor(byte_addr, byte) >> al_offset) & 0xFF);
        #else
        atomicXor(byte_addr, byte);
        #endif
        #else
        HOST_UNREACHABLE_ATOMIC(addr, value);
        #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
        return 0;
        #endif
        #endif
    }
#else
    INLINE_DEVICE word_std_t atomicXOR(word_std_t* addr, const word_std_t& value) {
        #if defined(__CUDA_ARCH__)
        return atomicXor(addr, value);
        #else
        HOST_UNREACHABLE_ATOMIC(addr, value);
        return 0;
        #endif
    }
#endif

#if defined(WORD_SIZE_8)
    INLINE_DEVICE void atomicAND(word_std_t* addr, const uint32& value) {
        #if defined(__CUDA_ARCH__)
        assert(value <= WORDS_MAX);
        EXTRACT_BYTE_FROM_ADDR(addr, value);
        const uint32 byte_mask = ~(uint32(0xFF) << al_offset) | byte;
        atomicAnd(byte_addr, byte_mask);
        #else
        HOST_UNREACHABLE_ATOMIC(addr, value);
        #endif
    }
#else
    INLINE_DEVICE word_std_t atomicAND(word_std_t* addr, const word_std_t& value) {
        #if defined(__CUDA_ARCH__)
        return atomicAnd(addr, value);
        #else
        HOST_UNREACHABLE_ATOMIC(addr, value);
        return 0;
        #endif
    }
#endif

    INLINE_DEVICE void atomicByteXOR(byte_t* addr, const uint32& value) {
        #if defined(__CUDA_ARCH__)
    	EXTRACT_BYTE_FROM_ADDR(addr, value);
        atomicXor(byte_addr, byte);
        #else
        HOST_UNREACHABLE_ATOMIC(addr, value);
        #endif
    }

    INLINE_DEVICE uint32 atomicAggInc(uint32* counter) {
        #if defined(__CUDA_ARCH__)
        using namespace cooperative_groups;
        coalesced_group g = coalesced_threads();
        uint32 prev;
        if (g.thread_rank() == 0) {
            prev = atomicAdd(counter, g.num_threads());
        }
        prev = g.thread_rank() + g.shfl(prev, 0);
        return prev;
        #else
        HOST_UNREACHABLE_ATOMIC(counter, 0);
        return 0;
        #endif
    }

    #undef EXTRACT_BYTE_FROM_ADDR
    #undef HOST_UNREACHABLE_ATOMIC

}
