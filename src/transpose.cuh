#ifndef __CU_TRANSPOSE_H
#define __CU_TRANSPOSE_H

#include "table.cuh"
#include "grid.cuh"

namespace QuaSARQ {

    __global__ void transpose_to_rowmajor(Table* inv_xs, Table* inv_zs, Signs* inv_ss, 
                        const Table* xs, const Table* zs, const Signs* ss, 
                        const size_t num_words_major, const size_t num_words_minor, 
                        const size_t num_qubits);

    __global__ void transpose_to_colmajor(Table* xs, Table* zs, Signs* ss, 
                        const Table* inv_xs, const Table* inv_zs, const Signs* inv_ss, 
                        const size_t num_words_major, const size_t num_words_minor, 
                        const size_t num_qubits);

}

#endif