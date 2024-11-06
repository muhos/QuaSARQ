#ifndef __CU_PIVOT_H
#define __CU_PIVOT_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "logging.hpp"

namespace QuaSARQ {

    constexpr uint32 INVALID_PIVOT = UINT32_MAX;

    struct Pivot {
        uint32 determinate;
        uint32 indeterminate;

        INLINE_ALL 
        Pivot() : determinate(INVALID_PIVOT), indeterminate(INVALID_PIVOT) { }

        INLINE_ALL 
        void reset() {
            determinate = INVALID_PIVOT;
            indeterminate = INVALID_PIVOT;
        }

        INLINE_ALL
        void print(const bool& nonl = false) const {
            LOGGPU("(indet: %-6d, det: %-6d)", indeterminate, determinate);
            if (!nonl) LOGGPU("\n");
        }
 
    };

}

#endif