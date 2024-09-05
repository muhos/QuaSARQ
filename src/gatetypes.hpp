
#ifndef __GATETYPES_H
#define __GATETYPES_H

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

    // Gate probabilities.
    extern double probabilities[NR_GATETYPES];

    #define INIT_PROB(GATETYPE) (probabilities[GATETYPE] = options.GATETYPE ## _p)

    constexpr void NORMALIZE_PROBS() {
        double sum_probs = 0;
        #define SUM_PROBS(GATETYPE) \
            sum_probs += probabilities[uint32(GATETYPE)];
        FOREACH_GATE(SUM_PROBS);
        #define NORM_PROBS(GATETYPE) \
            probabilities[uint32(GATETYPE)] /= sum_probs;
        FOREACH_GATE(NORM_PROBS);
    }

    // 2-input gates.
    constexpr Gatetypes gatetypes_2[NR_GATETYPES - NR_GATETYPES_1] = { CX, CY, CZ, SWAP, ISWAP };

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