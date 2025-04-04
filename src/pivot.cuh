#ifndef __CU_PIVOT_H
#define __CU_PIVOT_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "logging.hpp"

namespace QuaSARQ {

    typedef uint32 pivot_t;

    constexpr pivot_t INVALID_PIVOT = UINT32_MAX;

}

#endif