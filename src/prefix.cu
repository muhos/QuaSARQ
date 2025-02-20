
#include "prefix.cuh"

namespace QuaSARQ {

    __device__ word_std_t scan_block_exclusive(word_std_t* data, const int& n) {
        int tid = threadIdx.x;
        int offset = 1;

        // UP-SWEEP (reduce) phase.
        for (int d = n >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                // Compute the base indices.
                int i = offset * (2 * tid + 1) - 1;
                int j = offset * (2 * tid + 2) - 1;
                // Add conflict-free offsets.
                int ai = i + CONFLICT_FREE_OFFSET(i);
                int aj = j + CONFLICT_FREE_OFFSET(j);
                // Combine: note that XOR is our operator.
                data[aj] ^= data[ai];
            }
            offset *= 2;
            __syncthreads();
        }
        
        // The total block XOR is now stored at data[n-1].
        int last = n - 1;
        int last_index = last + CONFLICT_FREE_OFFSET(last);
        word_std_t blockSum = data[last_index];

        // Set the last element to the identity (0 for XOR) to start the down‑sweep.
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
                                         const size_t num_blocks,
                                         const size_t num_words_minor) {
        assert(num_blocks == blockDim.x);
        grid_t padded_block_size = blockDim.x + CONFLICT_FREE_OFFSET(blockDim.x);
        grid_t slice = 2 * padded_block_size;
        word_std_t* shared = SharedMemory<word_std_t>();
        word_std_t* prefix_z = shared + threadIdx.y * slice;
        word_std_t* prefix_x = prefix_z + padded_block_size;
        grid_t prefix_tid = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x);

        //for (size_t w = 0; w < num_words_minor; w++) {
        for_parallel_y(w, num_words_minor) {
            if (threadIdx.x < num_blocks) {
                // Assuming the layout is such that each block total for word w is stored contiguously.
                prefix_z[prefix_tid] = block_intermediate_prefix_z[threadIdx.x * num_words_minor + w];
                prefix_x[prefix_tid] = block_intermediate_prefix_x[threadIdx.x * num_words_minor + w];

                __syncthreads();

                //printf("w(%lld), t(%d):  loaded block intermediate prefix-xor (tz) = " B2B_STR "\n", w, threadIdx.x, RB2B(prefix_z[threadIdx.x]));

                scan_block_exclusive(prefix_z, num_blocks);
                scan_block_exclusive(prefix_x, num_blocks);

                //printf("w(%lld), t(%d):  scanned block intermediate prefix-xor (tz) = " B2B_STR "\n", w, threadIdx.x, RB2B(prefix_z[threadIdx.x]));
                
                block_intermediate_prefix_z[threadIdx.x * num_words_minor + w] = prefix_z[prefix_tid];
                block_intermediate_prefix_x[threadIdx.x * num_words_minor + w] = prefix_x[prefix_tid];
            }
        }
    }

    __global__ void scan_blocks_pass_1(
		word_std_t* block_intermediate_prefix_z,
		word_std_t* block_intermediate_prefix_x,
		word_std_t* subblocks_prefix_z, 
		word_std_t* subblocks_prefix_x, 
		const size_t num_blocks,
		const size_t num_words_minor
	)
	{
		assert(num_blocks > blockDim.x);
        grid_t padded_block_size = blockDim.x + CONFLICT_FREE_OFFSET(blockDim.x);
        grid_t slice = 2 * padded_block_size;
        word_std_t* shared = SharedMemory<word_std_t>();
        word_std_t* prefix_z = shared + threadIdx.y * slice;
        word_std_t* prefix_x = prefix_z + padded_block_size;
        grid_t prefix_tid = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x);

		for_parallel_y(w, num_words_minor) {
			
			grid_t tid = blockIdx.x * blockDim.x + threadIdx.x;

			word_std_t local_z = 0;
			word_std_t local_x = 0;

			if (tid < num_blocks) {
				local_z = block_intermediate_prefix_z[tid * num_words_minor + w];
				local_x = block_intermediate_prefix_x[tid * num_words_minor + w];
			}


			prefix_z[prefix_tid] = local_z;
			prefix_x[prefix_tid] = local_x;

			__syncthreads();

			word_std_t blockSum_z = scan_block_exclusive(prefix_z, blockDim.x);
			word_std_t blockSum_x = scan_block_exclusive(prefix_x, blockDim.x);

			if (tid < num_blocks) {
                block_intermediate_prefix_z[tid * num_words_minor + w] = prefix_z[prefix_tid];
                block_intermediate_prefix_x[tid * num_words_minor + w] = prefix_x[prefix_tid];
            }

            if (threadIdx.x == blockDim.x - 1) {
                assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                subblocks_prefix_z[blockIdx.x * num_words_minor + w] = blockSum_z;
                subblocks_prefix_x[blockIdx.x * num_words_minor + w] = blockSum_x;
            }
		}
	}

    __global__
    void scan_blocks_pass_2(
        word_std_t* block_intermediate_prefix_z,
        word_std_t* block_intermediate_prefix_x,
        const word_std_t* subblocks_prefix_z,
        const word_std_t* subblocks_prefix_x,
        size_t num_blocks,
        size_t num_words_minor,
        size_t pass_1_blocksize
    )
    {

        for_parallel_y(w, num_words_minor) {

            grid_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            grid_t bid = tid / pass_1_blocksize;

            word_std_t offsetZ = 0;
            word_std_t offsetX = 0;
            
            if (bid > 0) {
                offsetZ = subblocks_prefix_z[bid * num_words_minor + w];
                offsetX = subblocks_prefix_x[bid * num_words_minor + w];
            }

            if (tid < num_blocks) {
                block_intermediate_prefix_z[tid * num_words_minor + w] ^= offsetZ;
                block_intermediate_prefix_x[tid * num_words_minor + w] ^= offsetX;
            }
        }
    }

    void Prefix::alloc(const Tableau<DeviceAllocator>& input, const size_t& max_window_bytes) {
        num_qubits = input.num_qubits();
        num_words_major = input.num_words_major();
        num_words_minor = input.num_words_minor();
        if (!num_qubits || num_qubits > MAX_QUBITS)
            LOGERROR("maximum number of qubits per window is invalid.");
        if (!num_words_minor || num_words_minor > MAX_WORDS)
            LOGERROR("number of minor words is invalid.");
        this->num_words_minor = num_words_minor;
        if (!max_intermediate_blocks) {
            size_t min_block_size = MIN_BLOCK_INTERMEDIATE_SIZE;
            nextPow2(min_block_size);
            OPTIMIZEBLOCKS2D(max_intermediate_blocks, num_qubits, min_block_size);
            nextPow2(max_intermediate_blocks);
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
        targets.alloc(num_qubits, max_window_bytes, true, false);
    }

    void Prefix::scan_blocks(const size_t& num_blocks, const cudaStream_t& stream) {
        assert(num_blocks <= max_intermediate_blocks);
        assert(num_words_minor);
        dim3 scan_blocks_blocksize(num_blocks, MIN(num_words_minor, (1024 / num_blocks)));
        dim3 scan_blocks_gridsize(1, 1);
        OPTIMIZEBLOCKS2D(scan_blocks_gridsize.y, num_words_minor, scan_blocks_blocksize.y);
        if (num_blocks <= MIN_SINGLE_PASS_THRESHOLD) {
            // Do single pass.
            OPTIMIZESHARED(scan_blocks_smem_size, scan_blocks_blocksize.y * (scan_blocks_blocksize.x + CONFLICT_FREE_OFFSET(scan_blocks_blocksize.x)), 2 * sizeof(word_std_t));
            scan_blocks_single_pass <<<scan_blocks_gridsize, scan_blocks_blocksize, scan_blocks_smem_size, stream>>> (block_intermediate_prefix_z, block_intermediate_prefix_x, num_blocks, num_words_minor);
            LASTERR("failed to scan_blocks_single_pass");
            SYNC(stream);
        }
        else {
            // Do triple passes.
            scan_blocks_blocksize.x = MIN_SUB_BLOCK_SIZE;
            OPTIMIZEBLOCKS2D(scan_blocks_gridsize.x, num_blocks, scan_blocks_blocksize.x);
            const size_t pass_1_blocksize = scan_blocks_blocksize.x;
            const size_t pass_1_gridsize = scan_blocks_gridsize.x;
            OPTIMIZESHARED(scan_blocks_p1_smem_size, scan_blocks_blocksize.y * (scan_blocks_blocksize.x + CONFLICT_FREE_OFFSET(scan_blocks_blocksize.x)), 2 * sizeof(word_std_t));
            scan_blocks_pass_1 <<<scan_blocks_gridsize, scan_blocks_blocksize, scan_blocks_p1_smem_size, stream>>> (block_intermediate_prefix_z, block_intermediate_prefix_x, subblocks_prefix_z, subblocks_prefix_x, num_blocks, num_words_minor);
            assert(pass_1_gridsize <= MIN_SINGLE_PASS_THRESHOLD);
            scan_blocks_blocksize.x = pass_1_gridsize;
            scan_blocks_gridsize.x = 1;
            OPTIMIZESHARED(scan_blocks_smem_size, scan_blocks_blocksize.y * (scan_blocks_blocksize.x + CONFLICT_FREE_OFFSET(scan_blocks_blocksize.x)), 2 * sizeof(word_std_t));
            scan_blocks_single_pass <<<scan_blocks_gridsize, scan_blocks_blocksize, scan_blocks_smem_size, stream>>> (block_intermediate_prefix_z, block_intermediate_prefix_x, pass_1_gridsize, num_words_minor);
            scan_blocks_blocksize.x = MIN_SUB_BLOCK_SIZE; // This can be arbitrary size.
            OPTIMIZEBLOCKS2D(scan_blocks_gridsize.x, num_blocks, scan_blocks_blocksize.x); 
            scan_blocks_pass_2 <<<scan_blocks_gridsize, scan_blocks_blocksize, 0, stream>>> (block_intermediate_prefix_z, block_intermediate_prefix_x, subblocks_prefix_z, subblocks_prefix_x, num_blocks, num_words_minor, pass_1_blocksize);
            LASTERR("failed to scan_blocks_pass_2");
            SYNC(stream);
        }
    }

}