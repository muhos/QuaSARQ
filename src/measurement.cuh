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
            CHECK_SIGN_OVERFLOW(src_idx, old_value, (pos_i - neg_i) % 4); \
        }

    // Let threads in x-dim find the minimum stabilizer generator commuting.
    INLINE_DEVICE void find_min_pivot(Gate& m, const Table& inv_xs, const size_t num_qubits, const size_t num_words_minor) {
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        for_parallel_x(g, num_qubits) {
            const word_std_t qubit_word = inv_xs[(g + num_qubits) * num_words_minor + q_w];
            if (qubit_word & q_mask)
                atomicMin(&m.pivot, g);
        }
    }

    

    // Copy a tableau row to another.
    INLINE_DEVICE void row_to_row(Table& inv_xs, Table& inv_zs, int* inv_ss, const grid_t& des_idx, const grid_t& src_idx, const size_t& num_words_minor) {
        for_parallel_x(w, num_words_minor) {
            const grid_t src_word_idx = src_idx * num_words_minor + w;
            const grid_t des_word_idx = des_idx * num_words_minor + w;
            inv_xs[des_word_idx] = inv_xs[src_word_idx];
            inv_zs[des_word_idx] = inv_zs[src_word_idx];
            if (!w) {
                inv_ss[des_idx] = inv_ss[src_idx];
            }
        }
    }

    // Set a stabilizer generator row to zero, then set Zq to 1.
    INLINE_DEVICE void row_set(Table& inv_xs, Table& inv_zs, int* inv_ss, const grid_t& des_idx, const size_t& q, const size_t& num_words_minor) {
        const grid_t q_w = WORD_OFFSET(q);
        const grid_t row = des_idx * num_words_minor;
        for_parallel_x(w, num_words_minor) {
            if (w != q_w) {
                const grid_t des_word_idx = row + w;
                inv_xs[des_word_idx] = 0;
                inv_zs[des_word_idx] = 0; 
            }
        }
        if (!global_tx) {
            inv_ss[des_idx] = 0;
            inv_xs[row + q_w] = 0;
            inv_zs[row + q_w] = BITMASK_GLOBAL(q);
        }
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

    // Find all stabilizer generators commuting if exist.
    __global__ void is_indeterminate_outcome(bucket_t* measurements, const gate_ref_t* refs, const Table* inv_xs, 
                                            const size_t num_qubits, const size_t num_gates, const size_t num_words_minor);

    // Measure all determinate qubits in parallel.
    __global__ void measure_determinate(Table* inv_xs, Table* inv_zs, Signs* inv_ss, DeviceLocker* dlocker, 
                                        bucket_t* measurements, const gate_ref_t* refs, const size_t num_gates, 
                                        const size_t num_qubits, const size_t num_words_minor);

    // For single measurements.
    __global__ void is_indeterminate_outcome_single(bucket_t* measurements, const gate_ref_t* refs, const Table* inv_xs, 
                                                    const size_t gate_index, const size_t num_qubits, const size_t num_words_minor);

    __global__ void measure_determinate_single(Table* inv_xs, Table* inv_zs, Signs *inv_ss, DeviceLocker* dlocker, 
                                        bucket_t* measurements, const gate_ref_t* refs, const size_t gate_index, 
                                        const size_t num_qubits, const size_t num_words_minor);

    __global__ void measure_indeterminate_single(Table* inv_xs, Table* inv_zs, Signs *inv_ss, DeviceLocker* dlocker, 
                                        bucket_t* measurements, const gate_ref_t* refs, const size_t gate_index, 
                                        const size_t num_qubits, const size_t num_words_minor);
}

#endif