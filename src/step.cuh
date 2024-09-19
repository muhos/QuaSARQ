#ifndef __CU_STEP_H
#define __CU_STEP_H

#include "timer.hpp"
#include "table.cuh"
#include "signs.cuh"
#include "vector.cuh"
#include "circuit.cuh"
#include "print.cuh"
#include "timer.cuh"
#include "grid.cuh"
#include "collapse.cuh"
#include "atomic.cuh"

namespace QuaSARQ {

    // Simulate a single window per circuit.
    __global__ void step_2D(const gate_ref_t* refs, const bucket_t* gates, const size_t num_gates, const size_t num_words_per_column, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss);

    __global__ void step_2D_warped(const gate_ref_t* refs, const bucket_t* gates, const size_t num_gates, const size_t num_words_per_column, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss);

    #define GET_GENERATOR_INDEX(WORD, WORD_IDX) ((int64(__ffsll(WORD)) - 1) + int64((WORD_IDX) << WORD_POWER))

    INLINE_DEVICE void row_to_aux(byte_t& measurement, byte_t* aux, const Table& xs, const Table& zs, const Signs& ss, const size_t& src_idx, const size_t& num_qubits, const size_t& num_words_per_column) {
        const word_std_t generator_mask = POW2(src_idx);
        const word_std_t shifts = (src_idx & WORD_MASK);
        word_std_t word = 0;
        if (!global_tx) {
            assert(measurement == UNMEASURED);
            measurement = (ss[WORD_OFFSET(src_idx)] & generator_mask) >> shifts;
            assert(measurement < 2);
        }
        for_parallel_x(q, num_qubits) {
            const size_t word_idx = q * num_words_per_column + WORD_OFFSET(src_idx);
            word = xs[word_idx];
            aux[q] = (word & generator_mask) >> shifts;
            word = zs[word_idx];
            aux[q + num_qubits] = (word & generator_mask) >> shifts;
        }
    }

    INLINE_DEVICE void row_x_aux(byte_t& measurement, uint32* shared_signs, byte_t* aux, const Table& xs, const Table& zs, const Signs& ss, const size_t& src_idx, const size_t& num_qubits, const size_t& num_words_per_column) {
        
        const word_std_t generator_mask = POW2(src_idx);
        const word_std_t shifts = (src_idx & WORD_MASK);
        const uint32 s = (ss[WORD_OFFSET(src_idx)] & generator_mask) >> shifts;
        assert(s < 2);
        // XOR s with the auxiliary sign.
        if (!global_tx) {
            measurement ^= s;
        }
        grid_t tx = threadIdx.x, bx = blockDim.x;
        grid_t global_offset = blockIdx.x * bx, collapse_tid = threadIdx.y * bx + tx;
        uint32 p = 0; // only track parity (0 for positive or 1 for i, -i, -1)
        for_parallel_x(q, num_qubits) {
            const uint32 word_idx = q * num_words_per_column + WORD_OFFSET(src_idx);
            // From here on, we deal with flags: 0 or 1.
            // We only use uint32 to make compatible with atomicXor.
            const uint32 x = ((word_std_t(xs[word_idx]) & generator_mask) >> shifts);
            const uint32 z = ((word_std_t(zs[word_idx]) & generator_mask) >> shifts);
            assert(x <= 1);
            assert(z <= 1);
            const uint32 aux_x = aux[q];
            const uint32 aux_z = aux[q + num_qubits];
            aux[q] ^= byte_t(x);
            aux[q + num_qubits] ^= byte_t(z);
            uint32 x_only = x & ~z;  // X 
            uint32 y = x & z;        // Y 
            uint32 z_only = ~x & z;  // Z 
            uint32 aux_x_only = aux_x & ~aux_z;  // aux_X 
            uint32 aux_y = aux_x & aux_z;        // aux_Y 
            uint32 aux_z_only = ~aux_x & aux_z;  // aux_Z 
            p  ^= (x_only & aux_y)        // XY=iZ
                ^ (x_only & aux_z_only)   // XZ=-iY
                ^ (y      & aux_z_only)   // YZ=iX
                ^ (y      & aux_x_only)   // YX=-iZ
                ^ (z_only & aux_x_only)   // ZX=iY
                ^ (z_only & aux_y);       // ZY=-iX
        }
        assert(p < 2);
        load_shared(shared_signs, p, collapse_tid, tx, num_qubits);
        collapse_shared(shared_signs, p, collapse_tid, bx, tx);
        collapse_warp(shared_signs, p, collapse_tid, bx, tx);
        // Do: *aux_sign ^= p, where p here holds the collapsed value of a block.
        if (!tx && global_offset < num_qubits) {
            atomicByteXOR(&measurement, p);
        }
        //assert(*aux_sign < 2);
    }

    INLINE_DEVICE uint32 measure_determinate_qubit(DeviceLocker& dlocker, Gate& m, Table& xs, Table& zs, Signs& ss, uint32* shared_signs, byte_t* aux, const size_t num_qubits, const size_t num_words_per_column) {
        const size_t col = m.wires[0] * num_words_per_column;
        word_std_t word = 0;
        for (size_t j = 0; j < num_words_per_column; j++) {
            word = xs[col + j];
            if (word) {
                const int64 generator_index = GET_GENERATOR_INDEX(word, j);
                if (generator_index < num_qubits) {
                    // TODO: It might be beneficial if we create shared memory for aux.
                    // TODO: transform the rows involved here into words. This could improve the rowmul operation.                 
                    row_to_aux(m.measurement, aux, xs, zs, ss, generator_index + num_qubits, num_qubits, num_words_per_column);
                    if (!global_tx) print_row(dlocker, m, xs, zs, ss, generator_index + num_qubits, num_qubits, num_words_per_column);
                    for (size_t k = generator_index + 1; k < num_qubits; k++) {
                        word = xs[col + WORD_OFFSET(k)];
                        if (word & POW2(k)) {
                            if (!global_tx) print_row(dlocker, m, xs, zs, ss, k + num_qubits, num_qubits, num_words_per_column);
                            row_x_aux(m.measurement, shared_signs, aux, xs, zs, ss, k + num_qubits, num_qubits, num_words_per_column);
                            if (!global_tx) print_row(dlocker, m, aux, num_qubits);
                        }
                    }
                    break;
                }
            }
        }
    }

}

#endif