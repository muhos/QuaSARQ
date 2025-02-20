#ifndef __CU_MEASUREMENT_H
#define __CU_MEASUREMENT_H

#include "datatypes.cuh"
#include "circuit.cuh"
#include "locker.cuh"
#include "print.cuh"
#include "grid.cuh"
#include "sum.cuh"
#include "timer.cuh"
#include "timer.hpp"

namespace QuaSARQ {

    INLINE_ALL bool _check_sign_overflow(const size_t& row, const int& a, const int& b) {
        int64 ext_a = a;
        if (ext_a <= int64(INT_MIN) || ext_a >= int64(INT_MAX)) {
            LOGGPUERROR("overflow at row %lld before addition in a: %lld\n", row, ext_a);
            return false;
        }
        int64 ext_b = b;
        if (ext_b <= int64(INT_MIN) || ext_b >= int64(INT_MAX)) {
            LOGGPUERROR("overflow at row %lld before addition in b: %lld\n", row, ext_b);
            return false;
        }
        int64 result = a + b;
        if (result <= int64(INT_MIN) || result >= int64(INT_MAX)) {
            LOGGPUERROR("overflow at row %lld during ther addition (a: %lld) + (b: %lld)\n", row, ext_a, ext_b);
            return false;
        }
        return true;
    }

    #define CHECK_SIGN_OVERFLOW(IDX,A,B) assert(_check_sign_overflow((IDX),(A),(B)));

    // POS_I = { XY=+iZ, YZ=+iX, ZX=+iY }    
    // NEG_I = { XZ=-iY, YX=-iZ, ZY=-iX }
    #define COMPUTE_POWER_I(POS_I, NEG_I, X_SRC, Z_SRC, X_DES, Z_DES) \
        const word_std_t X_SRC_ONLY =  (X_SRC) & ~(Z_SRC);  \
        const word_std_t Y_SRC      =  (X_SRC) &  (Z_SRC);  \
        const word_std_t Z_SRC_ONLY = ~(X_SRC) &  (Z_SRC);  \
        const word_std_t X_DES_ONLY =  (X_DES) & ~(Z_DES);  \
        const word_std_t Y_DES      =  (X_DES) &  (Z_DES);  \
        const word_std_t Z_DES_ONLY = ~(X_DES) &  (Z_DES);  \
        POS_I += POPC(X_SRC_ONLY & Y_DES)      + POPC(Y_SRC & Z_DES_ONLY) + POPC(Z_SRC_ONLY & X_DES_ONLY); \
        NEG_I += POPC(X_SRC_ONLY & Z_DES_ONLY) + POPC(Y_SRC & X_DES_ONLY) + POPC(Z_SRC_ONLY & Y_DES)

    // Accumulate thread-local values in shared memory.
    #define ACCUMULATE_POWER_I(GLOBAL_POWER) \
        load_shared(pos_is, pos_i, neg_is, neg_i, shared_tid, num_words_minor); \
        sum_shared(pos_is, pos_i, neg_is, neg_i, shared_tid); \
        sum_warp(pos_is, pos_i, neg_is, neg_i, shared_tid); \
        if (!tx) { \
            assert(pos_i >= 0 && pos_i < UNMEASURED); \
            assert(neg_i >= 0 && neg_i < UNMEASURED); \
            int old_value = atomicAdd(&(GLOBAL_POWER), (pos_i - neg_i) % 4); \
            CHECK_SIGN_OVERFLOW(des_idx, old_value, (pos_i - neg_i) % 4); \
        }

    #define ACCUMULATE_POWER_I_OFFSET(GLOBAL_POWER, OFFSET) \
        load_shared(pos_is, pos_i, neg_is, neg_i, shared_tid, num_words_minor); \
        sum_shared(pos_is, pos_i, neg_is, neg_i, shared_tid); \
        sum_warp(pos_is, pos_i, neg_is, neg_i, shared_tid); \
        if (!tx) { \
            assert(pos_i >= 0 && pos_i < UNMEASURED); \
            assert(neg_i >= 0 && neg_i < UNMEASURED); \
            int block_value = (pos_i - neg_i) % 4; \
            if (!blockIdx.x) block_value += (OFFSET); \
            int old_value = atomicAdd(&(GLOBAL_POWER), block_value); \
            CHECK_SIGN_OVERFLOW(des_idx, old_value, block_value); \
        }
    
}

#endif