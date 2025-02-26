#ifndef __CU_MEASUREMENT_H
#define __CU_MEASUREMENT_H

#include "operators.cuh"
#include "datatypes.cuh"
#include "circuit.cuh"
#include "locker.cuh"
#include "print.cuh"
#include "grid.cuh"
#include "timer.cuh"
#include "timer.hpp"

namespace QuaSARQ {

    INLINE_DEVICE void do_YZ_Swap(word_t& X, word_t& Z, sign_t& s) {
        const word_std_t x = X, z = Z;
        X = x ^ z;
        s ^= (x & ~z);
    }

    INLINE_DEVICE void do_XZ_Swap(word_t& X, word_t& Z, sign_t& s) {
        do_SWAP(X, Z);
        s ^= word_std_t(X & Z);
    }
}

#endif