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
        GATE(DEPOLARIZE1) \
        GATE(X_ERROR) \
        GATE(Y_ERROR) \
        GATE(Z_ERROR) \
        GATE(PAULI_CHANNEL_1) \
        GATE(CX) \
        GATE(CY) \
        GATE(CZ) \
        GATE(SWAP) \
        GATE(ISWAP) \
        GATE(ISWAP_DAG) \
        GATE(DEPOLARIZE2) \
        GATE(PAULI_CHANNEL_2) \

    #define GATE2ENUM(VAL) VAL,
    #define GATE2STR(STR) #STR,

    enum Gatetypes {
        FOREACH_GATE(GATE2ENUM)
    };

    constexpr uint32 NR_GATETYPES_1 = PAULI_CHANNEL_1 + 1;
    constexpr uint32 NR_GATETYPES = PAULI_CHANNEL_2 + 1;

}