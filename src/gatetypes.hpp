#pragma once

#include "datatypes.hpp"

namespace QuaSARQ {

    // 1-input gates first then multi-input gates.
    #define FOREACH_GATE(GATE) \
        GATE(I) \
        GATE(Z) \
        GATE(X) \
        GATE(Y) \
        GATE(H) \
        GATE(S) \
        GATE(S_DAG) \
        GATE(M) \
        GATE(CX) \
        GATE(CY) \
        GATE(CZ) \
        GATE(SWAP) \
        GATE(ISWAP) \

    #define GATE2ENUM(VAL) VAL,
    #define GATE2STR(STR) #STR,

    enum Gatetypes {
        FOREACH_GATE(GATE2ENUM)
    };

    constexpr uint32 NR_GATETYPES_1 = M + 1;
    constexpr uint32 NR_GATETYPES = ISWAP + 1;

}