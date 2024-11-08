#ifndef __CU_MEASUREMENT_H
#define __CU_MEASUREMENT_H

#include "timer.hpp"
#include "table.cuh"
#include "signs.cuh"
#include "circuit.cuh"
#include "print.cuh"
#include "timer.cuh"
#include "grid.cuh"
#include "sum.cuh"
#include "locker.cuh"

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
        POS_I = POPC(X_SRC_ONLY & Y_DES)      + POPC(Y_SRC & Z_DES_ONLY) + POPC(Z_SRC_ONLY & X_DES_ONLY); \
        NEG_I = POPC(X_SRC_ONLY & Z_DES_ONLY) + POPC(Y_SRC & X_DES_ONLY) + POPC(Z_SRC_ONLY & Y_DES)

    // Accumulate thread-local values in shared memory.
    #define ACCUMULATE_POWER_I(GLOBAL_POWER) \
        load_shared(pos_is, pos_i, neg_is, neg_i, shared_tid, tx, num_words_minor); \
        sum_shared(pos_is, pos_i, neg_is, neg_i, shared_tid, BX, tx); \
        sum_warp(pos_is, pos_i, neg_is, neg_i, shared_tid, BX, tx); \
        if (!tx) { \
            assert(pos_i >= 0 && pos_i <= 64); \
            assert(neg_i >= 0 && neg_i <= 64); \
            int old_value = atomicAdd(&(GLOBAL_POWER), (pos_i - neg_i) % 4); \
            CHECK_SIGN_OVERFLOW(des_idx, old_value, (pos_i - neg_i) % 4); \
        }


    // Multiply two tableau rows.
    INLINE_DEVICE void row_mul(DeviceLocker& dlocker, Gate& m, int* aux_power, Table& inv_xs, Table& inv_zs, int* inv_ss, const grid_t& des_idx, const grid_t& src_idx, const grid_t& q_w, const size_t& num_words_minor) {
        int* pos_is = aux_power;
        int* neg_is = aux_power + blockDim.x;
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        const grid_t w = blockIdx.x * BX + tx;

        // track power of i.
        if (!w) {
            //printf("ROWMUL: M(%d): row(%lld) = row(%lld) x row(%d):\n", m.wires[0], des_idx, des_idx, src_idx);
            CHECK_SIGN_OVERFLOW(des_idx, inv_ss[des_idx], inv_ss[src_idx]);
            inv_ss[des_idx] += inv_ss[src_idx];
        }
        int pos_i = 0, neg_i = 0; 
        if (w < num_words_minor) {
            const grid_t src_word_idx = src_idx * num_words_minor + w;
            const grid_t des_word_idx = des_idx * num_words_minor + w;
            const word_std_t src_x = inv_xs[src_word_idx];
            const word_std_t src_z = inv_zs[src_word_idx];
            const word_std_t des_x = inv_xs[des_word_idx];
            const word_std_t des_z = inv_zs[des_word_idx];
            // defensive condition to protect read word:
            // qubit_word = inv_xs[j * num_words_minor + q_w].
            // This will gurantee that all threads will read
            // qubit_word and enter this function if q's bit
            // is 1 in qubit_word.
            if (w != q_w) {
                inv_xs[des_word_idx] = des_x ^ src_x;
            }
            inv_zs[des_word_idx] = des_z ^ src_z;
            COMPUTE_POWER_I(pos_i, neg_i, src_x, src_z, des_x, des_z);
        }

        ACCUMULATE_POWER_I(inv_ss[des_idx]);

        // Finally, let thread ID 0 update q's word.
        if (!global_tx) {
            inv_xs[des_idx * num_words_minor + q_w] ^= inv_xs[src_idx * num_words_minor + q_w];
            // printf("ROWMUL: M(%d): dest_xs(row: %lld, w: %lld) = " B2B_STR " ( " B2B_STR " ^ " B2B_STR " )\n", m.wires[0], 
            //     des_idx, q_w, 
            //     RB2B(word_std_t(inv_xs[des_idx * num_words_minor + q_w])), 
            //     RB2B(word_std_t(des_x)), 
            //     RB2B(word_std_t(inv_xs[src_idx * num_words_minor + q_w])));
        }
    }

    // Reset pivots.
    __global__ void reset_pivots(Pivot* pivots, const size_t num_gates);

    // Find all generators commuting if exist.
    __global__ void find_pivots_initial(Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, const Table* inv_xs, 
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor);

    // Initialize measurements with generator signs.
    __global__ void initialize_determinate_measurements(Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Signs* inv_ss,
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor);

    // Measure all determinate qubits in parallel.
    __global__ void measure_determinate(DeviceLocker* dlocker, const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Table* inv_zs, 
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor);

    // For single measurements.
    __global__ void measure_indeterminate_phase1(DeviceLocker* dlocker, Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor);

    __global__ void measure_indeterminate_phase2(DeviceLocker* dlocker, Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor);

    __global__ void measure_indeterminate_single(Table* inv_xs, Table* inv_zs, Signs *inv_ss, DeviceLocker* dlocker, 
                                        bucket_t* measurements, const gate_ref_t* refs, const size_t gate_index, 
                                        const size_t num_qubits, const size_t num_words_minor);
}

#endif