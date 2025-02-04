
#include "simulator.hpp"
#include "datatypes.cuh"
#include "shared.cuh"
#include "print.cuh"
#include "tuner.cuh"
#include "grid.cuh"


namespace QuaSARQ {


    __device__ inline size_t compute_block_index(const size_t& i, const size_t& j, const size_t& w, const size_t& num_words_major) {
        return ((i << WORD_POWER) + j) * num_words_major + w;
    }

    __device__ void transpose_tile(word_std_t* data, word_std_t* tile, const size_t& tile_offset) {
        assert(blockDim.x == WORD_BITS);

        static const word_std_t masks[WORD_POWER] = {
#if defined(WORD_SIZE_8)
            0x55U,  // separate odds/evens
            0x33U,  // separate bit pairs
            0x0FU
#elif defined(WORD_SIZE_32)
            0x55555555ULL,  // separate odds/evens
            0x33333333ULL,  // separate bit pairs
            0x0F0F0F0FULL,
            0x00FF00FFULL,
            0x0000FFFFULL
#elif defined(WORD_SIZE_64)
            0x5555555555555555ULL,  // separate odds/evens
            0x3333333333333333ULL,  // separate bit pairs
            0x0F0F0F0F0F0F0F0FULL,
            0x00FF00FF00FF00FFULL,
            0x0000FFFF0000FFFFULL,
            0x00000000FFFFFFFFULL
#endif
        };

        static const uint32 offsets[WORD_POWER] = { 
#if defined(WORD_SIZE_8)
            1, 2, 4
#elif defined(WORD_SIZE_32)
            1, 2, 4, 8, 16
#elif defined(WORD_SIZE_64)
            1, 2, 4, 8, 16, 32
#endif
        };

        uint32 tid = threadIdx.x;
        uint32 shared_tid = threadIdx.y * blockDim.x + tid;
        tile[shared_tid] = data[tid * tile_offset];
        __syncthreads();

        #pragma unroll
        for (int pairs = 0; pairs < WORD_POWER; pairs++) {
            const word_std_t mask = masks[pairs];
            const word_std_t imask = ~mask;
            const uint32 offset = offsets[pairs];
            if (!(tid & offset)) {
                word_std_t& x = tile[shared_tid];
                word_std_t& y = tile[shared_tid + offset];
                word_std_t a = x & mask;
                word_std_t b = x & imask;
                word_std_t c = y & mask;
                word_std_t d = y & imask;
                x = a | (c << offset);
                y = (b >> offset) | d;
            }
            __syncthreads(); // ensure all threads see the updated tile before next pairs
        }

        data[tid * tile_offset] = tile[shared_tid];
    }

    __global__ void transpose_tiles_kernel(Table* xs, Table* zs, const size_t num_words_major, const size_t num_words_minor, const bool row_major) {
        word_std_t* shared = SharedMemory<word_std_t>();
        if (blockIdx.z == 0) {
            word_std_t* xdata = reinterpret_cast<word_std_t*>(xs->data());
            if (!blockIdx.x && !blockIdx.y && !threadIdx.x) {
                xs->flag_orientation(row_major);
            }
            //for (size_t a = blockIdx.y; a < num_words_minor; a += gridDim.y) {
            for_parallel_y(a, num_words_minor) {
                for (size_t b = blockIdx.x; b < num_words_major; b += gridDim.x) {
                    // Inline transpose a tile of WORD_BITS words, each word has WORD_BITS bits.
                    // Transposition is done in shared memory.
                    const size_t tile_index = compute_block_index(a, 0, b, num_words_major);
                    transpose_tile(xdata + tile_index, shared, num_words_major);
                }
            }
        }
        if (blockIdx.z == 1) {
            word_std_t* zdata = reinterpret_cast<word_std_t*>(zs->data());
            if (!blockIdx.x && !blockIdx.y && !threadIdx.x) {
                zs->flag_orientation(row_major);
            }
            //for (size_t a = blockIdx.y; a < num_words_minor; a += gridDim.y) {
            for_parallel_y(a, num_words_minor) {
                for (size_t b = blockIdx.x; b < num_words_major; b += gridDim.x) {
                    // Inline transpose a tile of WORD_BITS words, each word has WORD_BITS bits.
                    // Transposition is done in shared memory.
                    const size_t tile_index = compute_block_index(a, 0, b, num_words_major);
                    transpose_tile(zdata + tile_index, shared, num_words_major);
                }
            }
        }
        
    }

    INLINE_DEVICE void swap_tile(word_std_t* data, word_std_t* shared, const size_t& a, const size_t& b, const size_t& num_words_major, const size_t& offset) {
        word_std_t* above_diagonal = shared;
        word_std_t* below_diagonal = shared + blockDim.y * blockDim.x;
        int tid = threadIdx.x;
        int shared_tid = threadIdx.y * blockDim.x + threadIdx.x;
        const size_t a_idx = compute_block_index(a, tid, b + offset, num_words_major);
        const size_t b_idx = compute_block_index(b, tid, a + offset, num_words_major);
        above_diagonal[shared_tid] = data[a_idx];
        below_diagonal[shared_tid] = data[b_idx];
        __syncthreads();
        data[a_idx] = below_diagonal[shared_tid];
        data[b_idx] = above_diagonal[shared_tid];
    }

    __global__ void swap_tiles_kernel(Table* xs, Table* zs, const size_t num_words_major, const size_t num_words_minor) {
        word_std_t* shared = SharedMemory<word_std_t>();
        if (blockIdx.z == 0) {
            word_std_t* xdata = reinterpret_cast<word_std_t*>(xs->data());
            //for (size_t a = blockIdx.y; a < num_words_minor; a += gridDim.y) {
            for_parallel_y(a, num_words_minor) {
                for (size_t b = blockIdx.x; b < num_words_minor; b += gridDim.x) {
                    // Only swap words above diagonal
                    if (b > a) {
                        // Do the destabilizers.
                        swap_tile(xdata, shared, a, b, num_words_major, 0);
                        // Do the stabilizers.
                        swap_tile(xdata, shared, a, b, num_words_major, num_words_minor);
                    }
                }
            }
        }
        if (blockIdx.z == 1) {      
            word_std_t* zdata = reinterpret_cast<word_std_t*>(zs->data());
            //for (size_t a = blockIdx.y; a < num_words_minor; a += gridDim.y) {
            for_parallel_y(a, num_words_minor) {
                for (size_t b = blockIdx.x; b < num_words_minor; b += gridDim.x) {
                    // Only swap words above diagonal
                    if (b > a) {
                        // Do the destabilizers.
                        swap_tile(zdata, shared, a, b, num_words_major, 0);
                        // Do the stabilizers.
                        swap_tile(zdata, shared, a, b, num_words_major, num_words_minor);
                    }
                }
            }
        }
    }

    __global__ void outplace_transpose_to_rowmajor(Table *inv_xs, Table *inv_zs, Signs *inv_ss,
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

    __global__ void outplace_transpose_to_colmajor(Table* xs, Table* zs, Signs* ss, 
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

        if (options.tune_transposebits || options.tune_transposeswap) {
            SYNCALL;
            tune_inplace_transpose(transpose_tiles_kernel, swap_tiles_kernel, 
            bestblocktransposebits, bestgridtransposebits, 
            bestblocktransposeswap, bestgridtransposeswap, 
            XZ_TABLE(tableau), num_words_major, num_words_minor, row_major);
        }
        bestblocktransposebits.x = MIN(WORD_BITS, bestblocktransposebits.x);
        bestgridtransposebits.x = MIN(num_words_major, bestgridtransposebits.x); 
        bestgridtransposebits.z = 2;
        TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblocktransposebits, bestgridtransposebits, num_words_minor);
        TRIM_GRID_IN_2D(bestblocktransposebits, bestgridtransposebits, num_words_minor, y);
        currentblock = bestblocktransposebits, currentgrid = bestgridtransposebits;
        OPTIMIZESHARED(transpose_smem_size, currentblock.y * currentblock.x, sizeof(word_std_t));
        LOGN2(2, "Running transpose-tiles with block(x:%u, y:%u) and grid(x:%u, y:%u, z:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y, currentgrid.z);
        transpose_tiles_kernel << <currentgrid, currentblock, transpose_smem_size, stream >> > (XZ_TABLE(tableau), num_words_major, num_words_minor, row_major);
        if (options.sync) {
            LASTERR("failed to launch transpose-tiles kernel");
            SYNC(stream);
        }
        LOGDONE(2, 4);
        bestblocktransposeswap.x = MIN(WORD_BITS, bestblocktransposeswap.x);
        bestgridtransposeswap.x = MIN(num_words_minor, bestgridtransposeswap.x); 
        bestgridtransposeswap.z = 2;
        TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblocktransposeswap, bestgridtransposeswap, num_words_minor);
        TRIM_GRID_IN_2D(bestblocktransposeswap, bestgridtransposeswap, num_words_minor, y);
        currentblock = bestblocktransposeswap, currentgrid = bestgridtransposeswap;
        OPTIMIZESHARED(swap_smem_size, currentblock.y * currentblock.x, 2 * sizeof(word_std_t));
        LOGN2(2, "Running swap-tiles with block(x:%u, y:%u) and grid(x:%u, y:%u, z:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y, currentgrid.z);
        swap_tiles_kernel << <currentgrid, currentblock, swap_smem_size, stream >> > (XZ_TABLE(tableau), num_words_major, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch swap-tiles kernel");
            SYNC(stream);
        }
        LOGDONE(2, 4);

        // if (row_major) {
        //     if (options.tune_transpose2r) {
        //         SYNCALL;
        //         tune_outplace_transpose(outplace_transpose_to_rowmajor, "Transposing to row-major", 
        //         bestblocktranspose2r, bestgridtranspose2r, 
        //         0, false,        // shared size, extend?
        //         num_words_major, // x-dim
        //         2 * num_qubits,  // y-dim 
        //         XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
        //     }
        //     TRIM_BLOCK_IN_DEBUG_MODE(bestblocktranspose2r, bestgridtranspose2r, num_words_major, 2 * num_qubits);
        //     currentblock = bestblocktranspose2r, currentgrid = bestgridtranspose2r;
        //     TRIM_GRID_IN_XY(num_words_major, 2 * num_qubits);
        //     outplace_transpose_to_rowmajor <<< currentgrid, currentblock, 0, stream >>> (XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
        //     if (options.sync) {
        //         LASTERR("failed to launch outplace_transpose_to_rowmajor kernel");
        //         SYNC(stream);
        //     }
        // }
        // else {
        //     if (options.tune_transpose2c) {
        //         SYNCALL;
        //         tune_outplace_transpose(outplace_transpose_to_colmajor, "Transposing to column-major", 
        //         bestblocktranspose2c, bestgridtranspose2c, 
        //         0, false,        // shared size, extend?
        //         num_words_major, // x-dim
        //         num_qubits,      // y-dim 
        //         XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
        //     }
        //     TRIM_BLOCK_IN_DEBUG_MODE(bestblocktranspose2c, bestgridtranspose2c, num_words_major, num_qubits);
        //     currentblock = bestblocktranspose2c, currentgrid = bestgridtranspose2c;     
        //     TRIM_GRID_IN_XY(num_words_major, num_qubits);
        //     outplace_transpose_to_colmajor <<< currentgrid, currentblock, 0, stream >>> (XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
        //     if (options.sync) {
        //         LASTERR("failed to launch outplace_transpose_to_colmajor kernel");
        //         SYNC(stream);
        //     }
        // }
    }

}

