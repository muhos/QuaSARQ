
#include "simulator.hpp"
#include "datatypes.cuh"
#include "print.cuh"
#include "tuner.cuh"
#include "grid.cuh"


namespace QuaSARQ {

    __global__ void transpose_to_rowmajor(Table *inv_xs, Table *inv_zs, Signs *inv_ss,
                              const Table *  xs, const Table *  zs, const Signs *  ss,
                              const size_t num_words_major, const size_t num_words_minor,
                              const size_t num_qubits) {
        if (!global_ty) {
            for_parallel_x(w, num_words_major) {
                sign_t signs_word = (*ss)[w];
                const size_t word_idx = w * WORD_BITS;
                #pragma unroll
                for (uint32 j = 0; j < WORD_BITS; j++) {
                    inv_ss->unpacked_data()[word_idx + j] = ((signs_word >> j) & 1) * 2;
                }
            }
            if (!global_tx) {
                inv_xs->flag_rowmajor();
                inv_zs->flag_rowmajor();
            }
        }

        for_parallel_y(w, 2 * num_qubits) {
            const word_std_t generator_index_per_word = (w & WORD_MASK);
            for_parallel_x(q, num_words_minor) {
                word_std_t inv_word_x = 0;
                word_std_t inv_word_z = 0;
                const size_t block_idx = q * WORD_BITS * num_words_major + WORD_OFFSET(w);
                #pragma unroll
                for (uint32 k = 0; k < WORD_BITS; k++) {
                    const size_t src_word_idx = k * num_words_major + block_idx;
                    const word_std_t generators_word_x = (*xs)[src_word_idx];
                    const word_std_t generators_word_z = (*zs)[src_word_idx];
                    const word_std_t generator_bit_x = (generators_word_x >> generator_index_per_word) & 1;
                    const word_std_t generator_bit_z = (generators_word_z >> generator_index_per_word) & 1;
                    inv_word_x |= (generator_bit_x << k);
                    inv_word_z |= (generator_bit_z << k);
                }
                const size_t dest_word_idx = q + w * num_words_minor;
                (*inv_xs)[dest_word_idx] = inv_word_x;
                (*inv_zs)[dest_word_idx] = inv_word_z;
            }
        }
    }

    __global__ void transpose_to_colmajor(Table* xs, Table* zs, Signs* ss, 
                        ConstTablePointer inv_xs, ConstTablePointer inv_zs, ConstSignsPointer  inv_ss, 
                        const size_t num_words_major, const size_t num_words_minor, 
                        const size_t num_qubits) {

        if (!global_ty) {
            sign_t *packed_signs = ss->data();
            for_parallel_x(w, num_words_major) {
                sign_t signs_word = 0;
                const size_t word_idx = w * WORD_BITS;
                #pragma unroll
                for (uint32 j = 0; j < WORD_BITS; j++) {
                    sign_t corrected_sign = ((inv_ss->unpacked_data()[word_idx + j] % 4 + 4) % 4 >> 1);
                    assert(corrected_sign >= 0 && corrected_sign <= 1);
                    signs_word |= (corrected_sign << j);
                }
                packed_signs[w] = signs_word;
            }
            if (!global_tx) {
                xs->flag_colmajor();
                zs->flag_colmajor();
            }
        }

        for_parallel_y(w, num_qubits) {
            const word_std_t qubit_index_per_word = (w & WORD_MASK);
            for_parallel_x(q, num_words_major) {
                word_std_t inv_word_x = 0;
                word_std_t inv_word_z = 0;
                const size_t block_idx = q * WORD_BITS * num_words_minor + WORD_OFFSET(w);
                #pragma unroll
                for (uint32 k = 0; k < WORD_BITS; k++) {
                    const size_t src_word_idx = k * num_words_minor + block_idx;
                    const word_std_t qubits_word_x = (*inv_xs)[src_word_idx];
                    const word_std_t qubits_word_z = (*inv_zs)[src_word_idx];
                    const word_std_t qubit_bit_x = (qubits_word_x >> qubit_index_per_word) & 1;
                    const word_std_t qubit_bit_z = (qubits_word_z >> qubit_index_per_word) & 1;
                    inv_word_x |= (qubit_bit_x << k);
                    inv_word_z |= (qubit_bit_z << k);
                }
                const size_t dest_word_idx = q + w * num_words_major;
                (*xs)[dest_word_idx] = inv_word_x;
                (*zs)[dest_word_idx] = inv_word_z;
            }
        }
    }

    void Simulator::transpose(const bool& row_major, const cudaStream_t& stream) {
        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_words_major = inv_tableau.num_words_major();
        dim3 currentblock, currentgrid;
        if (row_major) {
            if (options.tune_transpose2r) {
                SYNCALL;
                tune_transpose(transpose_to_rowmajor, "Transposing to row-major", 
                bestblocktranspose2r, bestgridtranspose2r, 
                0, false,        // shared size, extend?
                num_words_major, // x-dim
                2 * num_qubits,  // y-dim 
                XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocktranspose2r, bestgridtranspose2r, num_words_major, 2 * num_qubits);
            currentblock = bestblocktranspose2r, currentgrid = bestgridtranspose2r;
            TRIM_GRID_IN_XY(num_words_major, 2 * num_qubits);
            transpose_to_rowmajor <<< currentgrid, currentblock, 0, stream >>> (XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
            if (options.sync) {
                LASTERR("failed to launch transpose_to_rowmajor kernel");
                SYNC(stream);
            }
        }
        else {
            if (options.tune_transpose2c) {
                SYNCALL;
                tune_transpose(transpose_to_colmajor, "Transposing to column-major", 
                bestblocktranspose2c, bestgridtranspose2c, 
                0, false,        // shared size, extend?
                num_words_major, // x-dim
                num_qubits,      // y-dim 
                XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocktranspose2c, bestgridtranspose2c, num_words_major, num_qubits);
            currentblock = bestblocktranspose2c, currentgrid = bestgridtranspose2c;     
            TRIM_GRID_IN_XY(num_words_major, num_qubits);
            transpose_to_colmajor <<< currentgrid, currentblock, 0, stream >>> (XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
            if (options.sync) {
                LASTERR("failed to launch transpose_to_colmajor kernel");
                SYNC(stream);
            }
        }
    }

}

