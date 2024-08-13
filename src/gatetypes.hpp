
#ifndef __GATETYPES_H
#define __GATETYPES_H

#include "datatypes.hpp"

namespace QuaSARQ {

    enum Gatetypes {
        I,
        Z,
        X,
        Y, 
        H,
        S,
        Sdg,  // Last 1-qubit gate.
        CX,
        CY,
        CZ,
        Swap,
        iSwap // Last 2-qubit gate.
    };

    constexpr uint32 NR_GATETYPES_1 = Sdg + 1;
    constexpr uint32 NR_GATETYPES = iSwap + 1;

    // Combine 1-input gates then 2-input gates in order.
    constexpr Gatetypes gatetypes[NR_GATETYPES] =  
    {   
        I,
        Z,
        X,
        Y, 
        H,
        S,
        Sdg,
        CX,
        CY,
        CZ,
        Swap,
        iSwap
    };

    // Gate probabilities.
    extern double probabilities[NR_GATETYPES];

    // 2-input gates.
    constexpr Gatetypes gatetypes_2[NR_GATETYPES - NR_GATETYPES_1] = { CX, CY, CZ, Swap, iSwap };

    // Check if Clifford gate is 2-input gate by linear search. 
    inline bool isGate2(const Gatetypes& c) {
        assert(NR_GATETYPES > NR_GATETYPES_1);
        for (const Gatetypes* g = gatetypes_2, *e = g + (NR_GATETYPES - NR_GATETYPES_1); g != e; g++) { 
            if (c == *g) 
                return true;
        }
        return false;
    } 



}

#endif