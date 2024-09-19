#ifndef __CU_ATOMIC_H
#define __CU_ATOMIC_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "signs.cuh"

namespace QuaSARQ {

	NOINLINE_DEVICE uint32 atomicAggInc(uint32* counter);

	NOINLINE_DEVICE uint32 atomicAggMin(uint32* min, const uint32& val);

#if defined(WORD_SIZE_8)
    #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
    NOINLINE_DEVICE sign_t
    #else
    NOINLINE_DEVICE void
    #endif
    atomicXOR(sign_t* addr, const uint32& value);
#else
    NOINLINE_DEVICE sign_t atomicXOR(sign_t* addr, const word_std_t& value);
#endif

    NOINLINE_DEVICE void atomicByteXOR(byte_t* addr, const uint32& value);

}

#endif