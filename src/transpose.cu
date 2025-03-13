
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

    __global__ void rowmajor_kernel(Table *xs, Table *zs,
                                    const size_t num_words_major,
                                    const size_t num_words_minor,
                                    const size_t num_qubits)
    {
        word_std_t* shared = SharedMemory<word_std_t>();

        word_std_t* tile_destab = shared;
        word_std_t* tile_stab = tile_destab + blockDim.y * blockDim.x;
        size_t tile_idx = threadIdx.y * blockDim.x + threadIdx.x;

        grid_t q = blockIdx.x * blockDim.x + threadIdx.x; 
        grid_t w = blockIdx.y * blockDim.y + threadIdx.y; 

        // Load from global memory (original layout: index = q * num_words_major + w)
        if (q < num_qubits && w < num_words_minor) {
            const size_t old_idx = q * num_words_major + w;
            tile_destab[tile_idx] = (*xs)[old_idx];
            tile_stab  [tile_idx] = (*xs)[old_idx + num_words_minor];
            __syncthreads();
            const size_t new_idx = w * num_qubits + q;
            (*xs)[new_idx] = tile_destab[tile_idx];
            (*xs)[new_idx + num_qubits * num_words_minor] = tile_stab[tile_idx];
        }
    }

    void Simulator::transpose(const bool& row_major, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        dim3 currentblock, currentgrid;

        print_tableau(tableau, -1, false);

        if (options.tune_transposebits || options.tune_transposeswap) {
            SYNCALL;
            tune_inplace_transpose(transpose_tiles_kernel, swap_tiles_kernel, 
            bestblocktransposebits, bestgridtransposebits, 
            bestblocktransposeswap, bestgridtransposeswap, 
            XZ_TABLE(tableau), num_words_major, num_words_minor, row_major);
        }
        bestblocktransposebits.x = WORD_BITS;
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

        bestblocktransposeswap.x = WORD_BITS;
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

        

        dim3 rowmajor_block_size(4, 2);
        dim3 rowmajor_grid_size(
            ROUNDUP(tableau.num_qubits_padded(), rowmajor_block_size.x),
            ROUNDUP(num_words_minor, rowmajor_block_size.y)
        );
        OPTIMIZESHARED(rowmajor_smem_size, rowmajor_block_size.y * rowmajor_block_size.x, 2 * sizeof(word_std_t));
        rowmajor_kernel <<<rowmajor_grid_size, rowmajor_block_size, rowmajor_smem_size, stream>>> (
            XZ_TABLE(tableau), 
            num_words_major, 
            num_words_minor, 
            tableau.num_qubits_padded()
        );
        LASTERR("failed to launch rowmajor kernel");
        SYNC(0);
        
        print_tableau(tableau, -1, false);

        exit(0);
    }

}

