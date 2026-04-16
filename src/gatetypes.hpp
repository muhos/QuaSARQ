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
        GATE(R) \
        GATE(M) \
        GATE(MR) \
        GATE(CX) \
        GATE(CY) \
        GATE(CZ) \
        GATE(SWAP) \
        GATE(ISWAP) \
        GATE(ISWAP_DAG) \

    #define GATE2ENUM(VAL) VAL,
    #define GATE2STR(STR) #STR,

    enum Gatetypes {
        FOREACH_GATE(GATE2ENUM)
    };

    constexpr uint32 NR_GATETYPES_1 = MR + 1;
    constexpr uint32 NR_GATETYPES = ISWAP_DAG + 1;

    // Returns true for 2-qubit gates.
    inline bool isGate2(const int& type) {
        return type >= int(NR_GATETYPES_1) && type < int(NR_GATETYPES);
    }

    // Returns true for measurement gates.
    inline bool isMeasurement(const int& type) {
        return type == int(R) || type == int(M) || type == int(MR);
    }

}