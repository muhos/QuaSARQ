
#ifndef __CU_COMMUTATION_H
#define __CU_COMMUTATION_H

#include "definitions.cuh"
#include "logging.hpp"

namespace QuaSARQ {

    struct Commutation {
        bool commuting : 1;
        bool anti_commuting : 1;

        INLINE_ALL
        Commutation() : commuting(false), anti_commuting(false) {}

        INLINE_ALL
        void reset() {
            commuting = false;
            anti_commuting = false;
        }

        INLINE_ALL
        void print(const bool &nonl = false) const {
            LOGGPU("commuting: %d, anti-commuting: %d", commuting, anti_commuting);
            if (!nonl)
                LOGGPU("\n");
        }
    };
    
}

#endif