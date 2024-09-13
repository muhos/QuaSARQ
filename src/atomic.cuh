#ifndef __CU_ATOMIC_H
#define __CU_ATOMIC_H

#include "definitions.cuh"
#include "datatypes.hpp"

namespace QuaSARQ {

	NOINLINE_DEVICE uint32 atomicAggInc(uint32* counter);

	NOINLINE_DEVICE uint32 atomicAggMin(uint32* min, const uint32& val);
}

#endif