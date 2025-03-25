
#include "prefix.cuh"

namespace QuaSARQ {

    __device__ word_std_t scan_block_exclusive(word_std_t* data, const int& n) {
        assert(blockDim.x >= 2);

        int tid = threadIdx.x;
        int offset = 1;

        // UP-SWEEP (reduce) phase.
        for (int d = n >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                int i = offset * (2 * tid + 1) - 1;
                int j = offset * (2 * tid + 2) - 1;
                int ai = i + CONFLICT_FREE_OFFSET(i);
                int aj = j + CONFLICT_FREE_OFFSET(j);
                data[aj] ^= data[ai];
            }
            offset *= 2;
            __syncthreads();
        }
        
        // The total block XOR is now stored at data[n-1].
        int last = n - 1;
        int last_index = last + CONFLICT_FREE_OFFSET(last);
        word_std_t blockSum = data[last_index];

        // Set the last element to (0 for XOR) to start the down‑sweep.
        if (tid == 0) {
            data[last_index] = 0;
        }

        __syncthreads();
        
        // DOWN‑SWEEP phase.
        for (int d = 1; d < n; d *= 2) {
            offset /= 2;
            if (tid < d) {
                int i = offset * (2 * tid + 1) - 1;
                int j = offset * (2 * tid + 2) - 1;
                int ai = i + CONFLICT_FREE_OFFSET(i);
                int aj = j + CONFLICT_FREE_OFFSET(j);
                word_std_t temp = data[ai];
                data[ai] = data[aj];
                data[aj] ^= temp;
            }
            __syncthreads();
        }


        // data[0] = 0,
        // data[1] = a0,
        // data[2] = a0 ^ a1
        // ...

        return blockSum;
    }

	__global__ void scan_blocks_single_pass(word_std_t* block_intermediate_prefix_z, 
                                         word_std_t* block_intermediate_prefix_x,
                                         const size_t num_chunks,
                                         const size_t num_words_minor,
                                         const size_t max_blocks) {
        assert(num_chunks <= blockDim.x);
        assert(blockDim.x >= 2);
        grid_t padded_block_size = blockDim.x + CONFLICT_FREE_OFFSET(blockDim.x);
        grid_t slice = 2 * padded_block_size;
        word_std_t* shared = SharedMemory<word_std_t>();
        word_std_t* prefix_z = shared + threadIdx.y * slice;
        word_std_t* prefix_x = prefix_z + padded_block_size;
        grid_t prefix_tid = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x);

        for_parallel_y_tiled(by, num_words_minor) {

            const grid_t w = threadIdx.y + by * blockDim.y;
            
            word_std_t z = 0;
            word_std_t x = 0;

            if (w < num_words_minor && threadIdx.x < num_chunks) {
                const size_t bid = w * max_blocks + threadIdx.x;
                z = block_intermediate_prefix_z[bid];
			    x = block_intermediate_prefix_x[bid];
            }

            prefix_z[prefix_tid] = z;
            prefix_x[prefix_tid] = x;

            __syncthreads();

            scan_block_exclusive(prefix_z, num_chunks);
            scan_block_exclusive(prefix_x, num_chunks);
            
            if (w < num_words_minor && threadIdx.x < num_chunks) {
                const size_t bid = w * max_blocks + threadIdx.x;
                block_intermediate_prefix_z[bid] = prefix_z[prefix_tid];
                block_intermediate_prefix_x[bid] = prefix_x[prefix_tid];
            }

        }
    }

    __global__ void scan_blocks_pass_1(
		word_std_t* block_intermediate_prefix_z,
		word_std_t* block_intermediate_prefix_x,
		word_std_t* subblocks_prefix_z, 
		word_std_t* subblocks_prefix_x, 
		const size_t num_blocks,
		const size_t num_words_minor,
        const size_t max_blocks,
		const size_t max_sub_blocks
	)
	{
        grid_t padded_block_size = blockDim.x + CONFLICT_FREE_OFFSET(blockDim.x);
        grid_t slice = 2 * padded_block_size;
        word_std_t* shared = SharedMemory<word_std_t>();
        word_std_t* prefix_z = shared + threadIdx.y * slice;
        word_std_t* prefix_x = prefix_z + padded_block_size;
        grid_t prefix_tid = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x);

		for_parallel_y_tiled(by, num_words_minor) {

            const grid_t w = threadIdx.y + by * blockDim.y;
			
			for_parallel_x_tiled(bx, num_blocks) {

                const grid_t tid_x = threadIdx.x + bx * blockDim.x;
                
                word_std_t z = 0;
                word_std_t x = 0;

                if (w < num_words_minor && tid_x < num_blocks) {
                    const size_t bid = w * max_blocks + tid_x;
                    z = block_intermediate_prefix_z[bid];
                    x = block_intermediate_prefix_x[bid];
                }

                prefix_z[prefix_tid] = z;
                prefix_x[prefix_tid] = x;
                
                __syncthreads();

                word_std_t blockSum_z = scan_block_exclusive(prefix_z, blockDim.x);
                word_std_t blockSum_x = scan_block_exclusive(prefix_x, blockDim.x);

                if (w < num_words_minor && tid_x < num_blocks) {
                    const size_t bid = w * max_blocks + tid_x;
                    block_intermediate_prefix_z[bid] = prefix_z[prefix_tid];
                    block_intermediate_prefix_x[bid] = prefix_x[prefix_tid];
                }

                if (w < num_words_minor && threadIdx.x == blockDim.x - 1) {
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    grid_t sub_bid = w * max_sub_blocks + bx;
                    subblocks_prefix_z[sub_bid] = blockSum_z;
                    subblocks_prefix_x[sub_bid] = blockSum_x;
                }
            }
		}
	}

    __global__
    void scan_blocks_pass_2(
        word_std_t* block_intermediate_prefix_z,
        word_std_t* block_intermediate_prefix_x,
        const word_std_t* subblocks_prefix_z,
        const word_std_t* subblocks_prefix_x,
        const size_t num_blocks,
        const size_t num_words_minor,
        const size_t max_blocks,
		const size_t max_sub_blocks,
        const size_t pass_1_blocksize
    )
    {
        for_parallel_y(w, num_words_minor) {

            for_parallel_x(tid, num_blocks) {
                
                word_std_t offsetZ = 0;
                word_std_t offsetX = 0;
                
                if ((tid / pass_1_blocksize) > 0) {
                    const size_t sub_bid = w * max_sub_blocks + (tid / pass_1_blocksize);
                    offsetZ = subblocks_prefix_z[sub_bid];
                    offsetX = subblocks_prefix_x[sub_bid];
                }

                const size_t bid = w * max_blocks + tid;
                block_intermediate_prefix_z[bid] ^= offsetZ;
                block_intermediate_prefix_x[bid] ^= offsetX;
            }
        }
    }

    void Prefix::alloc(const Tableau<DeviceAllocator>& input, const size_t& config_qubits, const size_t& max_window_bytes) {
        this->config_qubits = config_qubits;
        num_qubits = input.num_qubits();
        num_words_major = input.num_words_major();
        num_words_minor = input.num_words_minor();
        if (!num_qubits || num_qubits > MAX_QUBITS)
            LOGERROR("maximum number of qubits per window is invalid.");
        if (!num_words_minor || num_words_minor > MAX_WORDS)
            LOGERROR("number of minor words is invalid.");
        if (!num_words_major || num_words_major > MAX_WORDS)
            LOGERROR("number of major words is invalid.");
        min_blocksize_y = nextPow2(num_words_minor);
        if (!max_intermediate_blocks) {
            max_intermediate_blocks = nextPow2(ROUNDUP(num_qubits, MIN_BLOCK_INTERMEDIATE_SIZE));
            size_t max_array_size = max_intermediate_blocks * num_words_minor;
            LOGN2(2, "allocating memory for %lld intermediate blocks.. ", int64(max_intermediate_blocks));
            if (block_intermediate_prefix_z == nullptr)
                block_intermediate_prefix_z = allocator.allocate<word_std_t>(max_array_size);
            if (block_intermediate_prefix_x == nullptr)
                block_intermediate_prefix_x = allocator.allocate<word_std_t>(max_array_size);
            LOGDONE(2, 3);
            if (max_intermediate_blocks > MIN_SINGLE_PASS_THRESHOLD) {
                max_sub_blocks = max_intermediate_blocks >> 1;
                max_array_size = max_sub_blocks * num_words_minor;
                LOGN2(2, "allocating memory for %lld sub-blocks.. ", int64(max_sub_blocks));
                if (subblocks_prefix_z == nullptr)
                    subblocks_prefix_z = allocator.allocate<word_std_t>(max_array_size);
                if (subblocks_prefix_x == nullptr)
                    subblocks_prefix_x = allocator.allocate<word_std_t>(max_array_size);
                LOGDONE(2, 3);
            }
        }
        targets.alloc(num_qubits, max_window_bytes, true, false, false);
        // For verification.
        checker.alloc(num_qubits);
    }

    void Prefix::resize(const Tableau<DeviceAllocator>& input, const size_t& max_window_bytes) {
        assert(num_qubits <= input.num_qubits());
        assert(config_qubits != 0);
        num_qubits = input.num_qubits();
        num_words_major = input.num_words_major();
        num_words_minor = input.num_words_minor();
        if (!num_qubits || num_qubits > MAX_QUBITS)
            LOGERROR("maximum number of qubits per window is invalid.");
        if (!num_words_minor || num_words_minor > MAX_WORDS)
            LOGERROR("number of minor words is invalid.");
        if (!num_words_major || num_words_major > MAX_WORDS)
            LOGERROR("number of major words is invalid.");
        min_blocksize_y = nextPow2(num_words_minor);
        max_intermediate_blocks = nextPow2(ROUNDUP(num_qubits, MIN_BLOCK_INTERMEDIATE_SIZE));
        if (max_intermediate_blocks > MIN_SINGLE_PASS_THRESHOLD) {
            max_sub_blocks = max_intermediate_blocks >> 1;
        }
        targets.resize(num_qubits, max_window_bytes, true, false, false);
    }

    void Prefix::scan_blocks(const size_t& num_blocks, const size_t& inject_pass_1_blocksize, const cudaStream_t& stream) {
        assert(num_blocks <= max_intermediate_blocks);
        assert(nextPow2(num_blocks) == num_blocks);
        dim3 currentblock, currentgrid;
        bestblockprefixsingle.x = MAX(2, num_blocks);
        bestgridprefixsingle.x = 1;
        if (num_blocks <= MIN_SINGLE_PASS_THRESHOLD) {
            // Do single pass.
            if (options.tune_prefixsingle) {
                SYNCALL;
                tune_single_pass(
                    scan_blocks_single_pass, 
                    bestblockprefixsingle, bestgridprefixsingle,
                    2 * sizeof(word_std_t),
                    num_blocks,
                    num_words_minor,
                    block_intermediate_prefix_z, 
                    block_intermediate_prefix_x, 
                    num_blocks, 
                    num_words_minor,
                    max_intermediate_blocks
                );
                SYNCALL;
            }
            TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblockprefixsingle, bestgridprefixsingle, num_words_minor);
            TRIM_GRID_IN_2D(bestblockprefixsingle, bestgridprefixsingle, num_words_minor, y);
            currentblock = bestblockprefixsingle, currentgrid = bestgridprefixsingle;
            OPTIMIZESHARED(scan_blocks_smem_size, currentblock.y * (currentblock.x + CONFLICT_FREE_OFFSET(currentblock.x)), 2 * sizeof(word_std_t));
            scan_blocks_single_pass <<<currentgrid, currentblock, scan_blocks_smem_size, stream>>> (
                block_intermediate_prefix_z, 
                block_intermediate_prefix_x, 
                num_blocks, 
                num_words_minor,
                max_intermediate_blocks
            );
            if (options.sync) {
                LASTERR("failed to scan in a single pass");
                SYNC(stream);
            }
        }
        else {
            // Do triple passes.
            if (options.tune_prefixprepare) {
                SYNCALL;
                tune_prefix_pass_1(
                    scan_blocks_pass_1, 
                    bestblockprefixprepare, bestgridprefixprepare,
                    2 * sizeof(word_std_t),
                    num_blocks,
                    num_words_minor,
                    block_intermediate_prefix_z, 
                    block_intermediate_prefix_x, 
                    subblocks_prefix_z, 
                    subblocks_prefix_x, 
                    num_blocks, 
                    num_words_minor,
                    max_intermediate_blocks,
                    max_sub_blocks
                );
                SYNCALL;
            }
            assert(bestblockprefixprepare.x);
            bestgridprefixprepare.x = ROUNDUP(num_blocks, bestblockprefixprepare.x);
            while (bestgridprefixprepare.x > MIN_SINGLE_PASS_THRESHOLD &&
                   bestblockprefixprepare.x <= MIN_SINGLE_PASS_THRESHOLD) {
                bestblockprefixprepare.x <<= 1;
                bestgridprefixprepare.x = ROUNDUP(num_blocks, bestblockprefixprepare.x);
            }
            if (bestblockprefixprepare.y == 1) {
                bestblockprefixprepare.y = 2;
                OPTIMIZEBLOCKS2D(bestgridprefixprepare.y, num_words_minor, bestblockprefixprepare.y);
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblockprefixprepare, bestgridprefixprepare, num_blocks, num_words_minor);
            currentblock = bestblockprefixprepare, currentgrid = bestgridprefixprepare;
            TRIM_GRID_IN_XY(num_blocks, num_words_minor);
            const size_t pass_1_blocksize = currentblock.x;
            const size_t pass_1_gridsize = ROUNDUP(num_blocks, pass_1_blocksize);
            OPTIMIZESHARED(p1_smem_size, currentblock.y * (currentblock.x + CONFLICT_FREE_OFFSET(currentblock.x)), 2 * sizeof(word_std_t));
            LOGN2(2, "  Pass-1 scanning %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            scan_blocks_pass_1 <<<currentgrid, currentblock, p1_smem_size, stream>>> (
                block_intermediate_prefix_z, 
                block_intermediate_prefix_x, 
                subblocks_prefix_z, 
                subblocks_prefix_x, 
                num_blocks, 
                num_words_minor,
                max_intermediate_blocks,
                max_sub_blocks
            );
            LOGDONE(2, 4);

            // Single phase
            assert(bestgridprefixsingle.x == 1);
            bestblockprefixsingle.x = nextPow2(MAX(2, pass_1_gridsize));
            if (bestblockprefixsingle.x > MIN_SINGLE_PASS_THRESHOLD)
                LOGERROR("pass-1 scan exceeded (%lld) with block size (%lld).", MIN_SINGLE_PASS_THRESHOLD, bestblockprefixsingle.x);             
            if (options.tune_prefixsingle) {
                SYNCALL;
                tune_single_pass(
                    scan_blocks_single_pass, 
                    bestblockprefixsingle, bestgridprefixsingle,
                    2 * sizeof(word_std_t),
                    pass_1_gridsize,
                    num_words_minor,
                    subblocks_prefix_z, 
                    subblocks_prefix_x, 
                    pass_1_gridsize, 
                    num_words_minor,
                    max_sub_blocks
                );
                SYNCALL;
            }
            if (bestblockprefixsingle.y == 1) {
                bestblockprefixsingle.y = bestblockprefixprepare.y;
                bestgridprefixsingle.y = bestgridprefixprepare.y;
            }
            TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblockprefixsingle, bestgridprefixsingle, num_words_minor);
            TRIM_GRID_IN_2D(bestblockprefixsingle, bestgridprefixsingle, num_words_minor, y);
            currentblock = bestblockprefixsingle, currentgrid = bestgridprefixsingle;
            OPTIMIZESHARED(scan_blocks_smem_size, currentblock.y * (currentblock.x + CONFLICT_FREE_OFFSET(currentblock.x)), 2 * sizeof(word_std_t));
            LOGN2(2, "  Pass-x scanning %lld chunks with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", pass_1_gridsize, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            scan_blocks_single_pass <<<currentgrid, currentblock, scan_blocks_smem_size, stream>>> (
                subblocks_prefix_z, 
                subblocks_prefix_x, 
                pass_1_gridsize, 
                num_words_minor,
                max_sub_blocks
            );
            LOGDONE(2, 4);
            
            // Final phase.
            if (options.tune_prefixfinal) {
                SYNCALL;
                tune_prefix_pass_2(
                    scan_blocks_pass_2, 
                    bestblockprefixfinal, bestgridprefixfinal,
                    num_blocks,
                    num_words_minor,
                    block_intermediate_prefix_z, 
                    block_intermediate_prefix_x, 
                    subblocks_prefix_z, 
                    subblocks_prefix_x, 
                    num_blocks, 
                    num_words_minor,
                    max_intermediate_blocks,
                    max_sub_blocks,
                    pass_1_blocksize
                );
                SYNCALL;
            }
            if (bestblockprefixfinal.y == 1) {
                bestblockprefixfinal.y = bestblockprefixprepare.y;
                bestgridprefixfinal.y = bestgridprefixprepare.y;
            }
            if (bestblockprefixfinal.x == 1) {
                bestblockprefixfinal.x = bestblockprefixprepare.x;
                bestgridprefixfinal.x = bestgridprefixprepare.x;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblockprefixfinal, bestgridprefixfinal, num_blocks, num_words_minor);
            currentblock = bestblockprefixfinal, currentgrid = bestgridprefixfinal;
            TRIM_GRID_IN_XY(num_blocks, num_words_minor);
            LOGN2(2, " Pass-2 scanning %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            scan_blocks_pass_2 <<<currentgrid, currentblock, 0, stream>>> (
                block_intermediate_prefix_z, 
                block_intermediate_prefix_x, 
                subblocks_prefix_z, 
                subblocks_prefix_x, 
                num_blocks, 
                num_words_minor, 
                max_intermediate_blocks,
                max_sub_blocks,
                pass_1_blocksize
            );
            if (options.sync) {
                LASTERR("failed to scan in multiple passes");
                SYNC(stream);
            }
            LOGDONE(2, 4);
        }
    }

}