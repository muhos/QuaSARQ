
#include "simulator.hpp"
#include "transpose.cuh"
#include "shared.cuh"
#include "print.cuh"
#include "access.cuh"

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
        Table& t = blockIdx.z == 0 ? *xs : *zs;
        word_std_t* data = t.words();
        if (!blockIdx.x && !blockIdx.y && !threadIdx.x) {
            t.flag_orientation(row_major);
        }
        for_parallel_y(a, num_words_minor) {
            for (size_t b = blockIdx.x; b < num_words_major; b += gridDim.x) {
                // Inline transpose a tile of WORD_BITS words, each word has WORD_BITS bits.
                // Transposition is done in shared memory.
                const size_t tile_index = compute_block_index(a, 0, b, num_words_major);
                transpose_tile(data + tile_index, shared, num_words_major);
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
        Table& t = blockIdx.z == 0 ? *xs : *zs;
        word_std_t* data = t.words();
        for_parallel_y(a, num_words_minor) {
            for (size_t b = blockIdx.x; b < num_words_minor; b += gridDim.x) {
                // Only swap words above diagonal
                if (b > a) {
                    // Do the destabilizers.
                    swap_tile(data, shared, a, b, num_words_major, 0);
                    // Do the stabilizers.
                    swap_tile(data, shared, a, b, num_words_major, num_words_minor);
                }
            }
        }
    }

    __global__ 
    void transpose_to_rowmajor(
        Table *inv_xs, 
        Table *inv_zs,
        const Table *  xs, 
        const Table *  zs,
        const size_t num_words_major, 
        const size_t num_words_minor,
        const size_t num_qubits) 
    {
        if (!global_ty && !global_tx) {
            inv_xs->flag_rowmajor();
            inv_zs->flag_rowmajor();
        }

        for_parallel_y(w, num_words_minor) {
            for_parallel_x(q, 2 * num_qubits) {
                const word_std_t generator_index_per_word = (q & WORD_MASK);
                word_std_t inv_word_x = 0;
                word_std_t inv_word_z = 0;
                const size_t block_idx = w * WORD_BITS * num_words_major + WORD_OFFSET(q);
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
                const size_t dest_word_idx = q + w * (2 * num_qubits);
                (*inv_xs)[dest_word_idx] = inv_word_x;
                (*inv_zs)[dest_word_idx] = inv_word_z;
            }
        }
    }

    __global__ 
    void transpose_to_colmajor(
                Table*          xs, 
                Table*          zs, 
                const_table_t   inv_xs, 
                const_table_t   inv_zs, 
        const   size_t          num_words_major, 
        const   size_t          num_words_minor, 
        const   size_t          num_qubits) 
    {

        if (!global_ty && !global_tx) {
            xs->flag_colmajor();
            zs->flag_colmajor();
        }

        for_parallel_y(w, num_qubits) {
            const word_std_t qubit_index_per_word = (w & WORD_MASK);
            for_parallel_x(q, num_words_major) {
                word_std_t inv_word_x = 0;
                word_std_t inv_word_z = 0;
                const size_t block_idx = q * WORD_BITS + WORD_OFFSET(w) * 2 * num_qubits;            
                #pragma unroll
                for (uint32 k = 0; k < WORD_BITS; k++) {
                    const size_t src_word_idx = k + block_idx;
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

    void check_transpose(const Table& x1, const Table& z1, const Table& x2, const Table& z2) {
        LOGN2(2, " Checking transpose correctness.. ");
        if (x1.size() != x2.size()) LOGERROR("x1 and x2 sizes do not match");
        if (z1.size() != z2.size()) LOGERROR("z1 and z2 sizes do not match");
        for (size_t i = 0; i < x1.size(); i++) {
            if (x1[i] != x2[i]) {
                LOGERROR("FAILED for x1 or x2 at word %lld.", i);
            }
        }
        for (size_t i = 0; i < z1.size(); i++) {
            if (z1[i] != z2[i]) {
                LOGERROR("FAILED for z1 or z2 at word %lld.", i);
            }
        }
        LOG2(2, "%sPASSED.%s", CGREEN, CNORMAL);
    }

    void transpose_to_colmajor_cpu(Table& xs, Table& zs, const Table& inv_xs, const Table& inv_zs) {
        xs.flag_colmajor();
        zs.flag_colmajor();
        for(size_t w = 0; w < xs.num_qubits_padded(); w++) {
            const word_std_t qubit_index_per_word = (w & WORD_MASK);
            for(size_t q = 0; q < xs.num_words_major(); q++) {
                word_std_t inv_word_x = 0;
                word_std_t inv_word_z = 0;
                const size_t block_idx = q * WORD_BITS + WORD_OFFSET(w) * 2 * xs.num_qubits_padded();
                for (uint32 k = 0; k < WORD_BITS; k++) {
                    const size_t src_word_idx = k + block_idx;
                    const word_std_t qubits_word_x = inv_xs[src_word_idx];
                    const word_std_t qubits_word_z = inv_zs[src_word_idx];
                    const word_std_t qubit_bit_x = (qubits_word_x >> qubit_index_per_word) & 1;
                    const word_std_t qubit_bit_z = (qubits_word_z >> qubit_index_per_word) & 1;
                    inv_word_x |= (qubit_bit_x << k);
                    inv_word_z |= (qubit_bit_z << k);
                }
                const size_t dest_word_idx = q + w * xs.num_words_major();
                xs[dest_word_idx] = inv_word_x;
                zs[dest_word_idx] = inv_word_z;
            }
        }
    }

    void check_outplace_transpose(Tableau& d_tab, Tableau& d_inv_tab) {
        SYNCALL;

        Table d_xs, d_zs;
        d_tab.copy_to_host(&d_xs, &d_zs);

        Table d_inv_xs, d_inv_zs;
        d_inv_xs.flag_rowmajor(), d_inv_zs.flag_rowmajor();
        d_inv_tab.copy_to_host(&d_inv_xs, &d_inv_zs);
    
        Table h_xs, h_zs;
        h_xs.alloc_host(d_tab.num_qubits_padded(), d_tab.num_words_major(), d_tab.num_words_minor());
        h_zs.alloc_host(d_tab.num_qubits_padded(), d_tab.num_words_major(), d_tab.num_words_minor());
        transpose_to_colmajor_cpu(h_xs, h_zs, d_inv_xs, d_inv_zs);

        check_transpose(h_xs, h_zs, d_xs, d_zs);
    }

    void check_inplace_transpose(Tableau& d_tab, const bool& row_major) {
        SYNCALL;

        Table h_xs, h_zs;
        h_xs.flag_orientation(row_major);
        h_zs.flag_orientation(row_major);
        d_tab.copy_to_host(&h_xs, &h_zs);

        const size_t num_words_minor = d_tab.num_words_minor();
        const size_t num_words_major = d_tab.num_words_major();
        const size_t num_qubits_padded = d_tab.num_qubits_padded();

        OPTIMIZESHARED(transpose_smem_size, bestblocktransposebits.y * bestblocktransposebits.x, sizeof(word_std_t));
        OPTIMIZESHARED(swap_smem_size, bestblocktransposeswap.y * bestblocktransposeswap.x, 2 * sizeof(word_std_t));

        transpose_tiles_kernel 
        <<< bestgridtransposebits, bestblocktransposebits, transpose_smem_size, 0 >>> (
            XZ_TABLE(d_tab), 
            num_words_major, 
            num_words_minor, 
            !row_major);
        swap_tiles_kernel 
        <<< bestgridtransposeswap, bestblocktransposeswap, swap_smem_size, 0 >>> (
            XZ_TABLE(d_tab), 
            num_words_major, 
            num_words_minor);

        transpose_tiles_kernel 
        <<< bestgridtransposebits, bestblocktransposebits, transpose_smem_size, 0 >>> (
            XZ_TABLE(d_tab), 
            num_words_major, 
            num_words_minor, 
            row_major);
        swap_tiles_kernel 
        <<< bestgridtransposeswap, bestblocktransposeswap, swap_smem_size, 0 >>> (
            XZ_TABLE(d_tab), 
            num_words_major, 
            num_words_minor);

        Table d_xs, d_zs;
        d_xs.flag_orientation(row_major);
        d_zs.flag_orientation(row_major);
        d_tab.copy_to_host(&d_xs, &d_zs);

        check_transpose(h_xs, h_zs, d_xs, d_zs);
    }

    #if ROW_MAJOR
    void Simulator::transpose(const bool& row_major, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        dim3 currentblock, currentgrid;

        if (row_major) {
            if (options.tune_transposec2r) {
                SYNCALL;
                tune_outplace_transpose(transpose_to_rowmajor, "Transposing to row-major", 
                bestblocktransposec2r, bestgridtransposec2r, 
                0, false,        // shared size, extend?
                num_words_major, // x-dim
                2 * num_qubits_padded,  // y-dim 
                XZ_TABLE(inv_tableau), XZ_TABLE(tableau), num_words_major, num_words_minor, num_qubits_padded, true);
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocktransposec2r, bestgridtransposec2r, 2 * num_qubits_padded, num_words_minor);
            currentblock = bestblocktransposec2r, currentgrid = bestgridtransposec2r;
            TRIM_GRID_IN_XY(2 * num_qubits_padded, num_words_minor);
            LOGN2(2, "Running row-major transpose with block(x:%u, y:%u) and grid(x:%u, y:%u, z:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y, currentgrid.z);
            transpose_to_rowmajor <<< currentgrid, currentblock, 0, stream >>> (
                XZ_TABLE(inv_tableau), 
                XZ_TABLE(tableau), 
                num_words_major, 
                num_words_minor, 
                num_qubits_padded
                );
            if (options.sync) {
                LASTERR("failed to launch transpose_to_rowmajor kernel");
                SYNC(stream);
            }
            LOGDONE(2, 4);
            tableau.swap_tableaus(inv_tableau);
            if (options.check_transpose) check_outplace_transpose(inv_tableau, tableau);
        }
        else {
            tableau.swap_tableaus(inv_tableau);
            if (options.tune_transposer2c) {
                SYNCALL;
                tune_outplace_transpose(transpose_to_colmajor, "Transposing to column-major", 
                bestblocktransposer2c, bestgridtransposer2c, 
                0, false,        // shared size, extend?
                num_words_major, // x-dim
                num_qubits_padded,      // y-dim 
                XZ_TABLE(tableau), XZ_TABLE(inv_tableau), num_words_major, num_words_minor, num_qubits_padded, false);
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocktransposer2c, bestgridtransposer2c, num_words_major, num_qubits_padded);
            currentblock = bestblocktransposer2c, currentgrid = bestgridtransposer2c;     
            TRIM_GRID_IN_XY(num_words_major, num_qubits_padded);
            LOGN2(2, "Running column-major transpose with block(x:%u, y:%u) and grid(x:%u, y:%u, z:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y, currentgrid.z);
            transpose_to_colmajor <<< currentgrid, currentblock, 0, stream >>> (
                XZ_TABLE(tableau), 
                XZ_TABLE(inv_tableau),
                num_words_major, 
                num_words_minor, 
                num_qubits_padded
                );
            if (options.sync) {
                LASTERR("failed to launch transpose_to_colmajor kernel");
                SYNC(stream);
            }
            LOGDONE(2, 4);
       }
    }

    #else
    void Simulator::transpose(const bool& row_major, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        dim3 currentblock, currentgrid;

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
        transpose_tiles_kernel << <currentgrid, currentblock, transpose_smem_size, stream >> > (
            XZ_TABLE(tableau), 
            num_words_major, 
            num_words_minor, 
            row_major);
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
        swap_tiles_kernel << <currentgrid, currentblock, swap_smem_size, stream >> > (
            XZ_TABLE(tableau), 
            num_words_major, 
            num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch swap-tiles kernel");
            SYNC(stream);
        }
        LOGDONE(2, 4);
        if (options.check_transpose) {
            SYNCALL;
            check_inplace_transpose(tableau, row_major);
        }
    }
    #endif

}

