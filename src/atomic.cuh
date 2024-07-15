#ifndef __ATOMIC_H
#define __ATOMIC_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "word.cuh"

namespace QuaSARQ {

	NOINLINE_DEVICE uint32 atomicAggInc(uint32* counter);

	NOINLINE_DEVICE void lock(int* mutex);

	NOINLINE_DEVICE void unlock(int* mutex);
}

#endif