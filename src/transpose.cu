
#include "simulator.hpp"
#include "datatypes.cuh"
#include "print.cuh"
#include "tuner.cuh"
#include "grid.cuh"


namespace QuaSARQ {


    __device__ inline size_t compute_block_index(const size_t& i, const size_t& j, const size_t& w, const size_t& num_words_major) {
        return ((i << WORD_POWER) + j) * num_words_major + w;
    }

    __device__ void shared_inplace_transpose(word_std_t* data, size_t stride)
    {

        assert(blockDim.x == WORD_BITS);

        static const word_std_t  masks[WORD_POWER] = {
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

        static const unsigned shifts[WORD_POWER] = { 
#if defined(WORD_SIZE_8)
            1, 2, 4
#elif defined(WORD_SIZE_32)
            1, 2, 4, 8, 16
#elif defined(WORD_SIZE_64)
            1, 2, 4, 8, 16, 32
#endif
        };

        // We'll store all WORD_BITS lines in shared memory
        // tile[k] will hold the row k (one WORD_BITS-bit word)
        __shared__ word_std_t tile[WORD_BITS];

        // 1) Load from global to shared
        size_t tid = threadIdx.x;

        tile[tid] = data[tid * stride];

        __syncthreads();

#pragma unroll
        for (int pass = 0; pass < WORD_POWER; pass++) {
            word_std_t mask = masks[pass];
            word_std_t imask = ~mask;
            unsigned shift = shifts[pass];

            if ((tid & shift) == 0) {
                word_std_t& x = tile[tid];
                word_std_t& y = tile[tid + shift];

                word_std_t a = x & mask;
                word_std_t b = x & imask;
                word_std_t c = y & mask;
                word_std_t d = y & imask;

                x = a | (c << shift);
                y = (b >> shift) | d;
            }

            __syncthreads(); // ensure all threads see the updated tile before next pass
        }

        // 3) Write back from shared to global
        data[tid * stride] = tile[tid];
    }

    template <word_std_t mask, word_std_t shift>
    INLINE_DEVICE void inplace_transpose_8x8_pass(word_std_t* data, size_t stride) {
        for (size_t k = 0; k < WORD_BITS; k++) {
            if (k & shift) {
                continue;
            }
            word_std_t& x = data[stride * k];
            word_std_t& y = data[stride * (k + shift)];
            word_std_t a = x & mask;
            word_std_t b = x & ~mask;
            word_std_t c = y & mask;
            word_std_t d = y & ~mask;
            x = a | (c << shift);
            y = (b >> shift) | d;
        }
    }

    INLINE_DEVICE void inplace_transpose_8x8(word_std_t* data, size_t stride) {
        inplace_transpose_8x8_pass<0x55UL, 1>(data, stride);
        inplace_transpose_8x8_pass<0x33UL, 2>(data, stride);
        inplace_transpose_8x8_pass<0x0FUL, 4>(data, stride);
    }

    __global__ void transpose_kernel(Table* xs, Table* zs, size_t num_words_major, size_t num_words_minor)
    {
        word_std_t* data = reinterpret_cast<word_std_t*>(zs->data());
        if (!blockIdx.x && !blockIdx.y && !threadIdx.x)
            zs->flag_rowmajor();
        for (size_t a = blockIdx.y; a < num_words_minor; a += gridDim.y) {
            for (size_t b = blockIdx.x; b < num_words_minor; b += gridDim.x) {
                // Inline transpose a tile of WORD_BITS words, each word has WORD_BITS bits.
                // Transposition is done in shared memory.
                // tile_index = (a << WORD_POWER) * num_words_major + b + num_words_minor;
                const size_t tile_index = compute_block_index(a, 0, b + num_words_minor, num_words_major);
                shared_inplace_transpose(data + tile_index, num_words_major);
                //inplace_transpose_8x8(data + tile_index, num_words_major);
                //print_table(*xs, true);
            }
        }
    }

    /*
0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  1  0  0  0  0  0  0    0  0  0  0  0  0  0  0    
1    1  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    1  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    
2    0  0  1  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  1  0  0  0  0  0    0  0  0  0  0  0  0  0    
3    1  0  0  1  0  0  0  0    0  0  0  0  0  0  0  0    1  0  0  0  0  1  0  0    0  0  0  0  0  0  0  0    
4    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  1  0  0  0    0  0  0  0  0  0  0  0    
5    0  0  0  1  0  1  0  0    0  0  0  0  0  0  0  0    0  0  0  1  0  0  0  0    0  0  0  0  0  0  0  0    
6    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    1  0  0  0  0  0  0  0    
7    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  1  0  0  1    0  0  0  0  0  0  0  0    
8    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  1  0    0  0  0  0  0  0  0  0    
9    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  1  0  0  0  0  0  0    
10   0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    
11   0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    
12   0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    
13   0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    
14   0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    
15   0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0   
    */

    __global__ void swap_kernel(word_std_t* data, size_t num_words_major, size_t num_words_minor)
    {
        //assert(blockDim.x == WORD_BITS);

        // Shared memory to hold rows for sub-block (a,b) and (b,a).
        __shared__ word_std_t tileA[WORD_BITS];
        __shared__ word_std_t tileB[WORD_BITS];

        for (size_t a = blockIdx.y; a < num_words_minor; a += gridDim.y) {
            // Only swap if b > a (above-diagonal)
            for (size_t b = blockIdx.x; b < num_words_minor; b += gridDim.x) {
                if (b > a) {
                    // for (size_t k = 0; k < WORD_BITS; k++) {
                    //     // ((a << WORD_POWER) + k) * num_words_major + b;
                    //     size_t a_idx = compute_block_index(a, k, b + num_words_minor, num_words_major);
                    //     size_t b_idx = compute_block_index(b, k, a + num_words_minor, num_words_major);
                    //     word_std_t tmp = data[a_idx];
                    //     data[a_idx] = data[b_idx];
                    //     data[b_idx] = tmp;

                    // }

                    // We have WORD_BITS threads in x, each tid handles one 'maj_low' in [0..63].
                    int tid = threadIdx.x;

                    // 1) Load WORD_BITS words from global memory into tileA and tileB
                    // ((a << WORD_POWER) + tid) * num_words_major + b;
                    size_t a_idx = compute_block_index(a, tid, b + num_words_minor, num_words_major);
                    size_t b_idx = compute_block_index(b, tid, a + num_words_minor, num_words_major);

                    tileA[tid] = data[a_idx];
                    tileB[tid] = data[b_idx];

                    __syncthreads();

                    // 2) Swap: the data from (a, tid, b) should go to (b, tid, a), and vice versa.
                    //    Actually we've already loaded them, so we just need to write them back swapped.

                    // 3) Write swapped data back to global memory
                    data[a_idx] = tileB[tid];
                    data[b_idx] = tileA[tid];
                }
            }
        }
    }

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

            print_tableau(tableau, -1, false);

            // if (options.tune_transpose2r) {
            //     SYNCALL;
            //     tune_transpose(transpose_to_rowmajor, "Transposing to row-major", 
            //     bestblocktranspose2r, bestgridtranspose2r, 
            //     0, false,        // shared size, extend?
            //     num_words_major, // x-dim
            //     2 * num_qubits,  // y-dim 
            //     XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
            // }
            // TRIM_BLOCK_IN_DEBUG_MODE(bestblocktranspose2r, bestgridtranspose2r, num_words_major, 2 * num_qubits);
            // currentblock = bestblocktranspose2r, currentgrid = bestgridtranspose2r;
            // TRIM_GRID_IN_XY(num_words_major, 2 * num_qubits);
            // transpose_to_rowmajor <<< currentgrid, currentblock, 0, stream >>> (XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
            // if (options.sync) {
            //     LASTERR("failed to launch transpose_to_rowmajor kernel");
            //     SYNC(stream);
            // }

            //print_tableau(inv_tableau, -1, false);

            cudaEvent_t start, stop;
            float elapsedTime;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            CHECK(cudaEventRecord(start, 0));

            dim3 threadsPerBlock_transpose(WORD_BITS, 1);
            dim3 blocksPerGrid_transpose(num_words_minor, num_words_minor);

            transpose_kernel << <blocksPerGrid_transpose, threadsPerBlock_transpose >> > (XZ_TABLE(tableau), num_words_major, num_words_minor);
            //transpose_kernel << <1, 1 >> > (XZ_TABLE(tableau), num_words_major, num_words_minor);
            LASTERR("transpose failed");
            CHECK(cudaDeviceSynchronize());

            printf("after transpose:\n");
            print_tableau(tableau, -1, false);

            dim3 threadsPerBlock_swap(WORD_BITS, 1);
            dim3 blocksPerGrid_swap(num_words_minor, num_words_minor);

            swap_kernel << <blocksPerGrid_swap, threadsPerBlock_swap >> > (tableau.zdata(), num_words_major, num_words_minor);
            //swap_kernel << <1, 1 >> > (tableau.zdata(), num_words_major, num_words_minor);
            LASTERR("swap failed");

            CHECK(cudaEventRecord(stop, 0));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
            printf("GPU Transpose Time: %f ms\n", elapsedTime);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            printf("after swap:\n");
            print_tableau(tableau, -1, false);

            //cudaDeviceReset();
            exit(0);
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

