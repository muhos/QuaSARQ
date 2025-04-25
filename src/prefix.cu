
#include "timer.cuh"
#include "prefix.cuh"
#include "datatypes.cuh"
#include "templatedim.cuh"
#include <cub/block/block_scan.cuh>

namespace QuaSARQ {

    template <int BLOCKX, int BLOCKY>
    __global__ 
    void scan_blocks_single_pass(
                SINGLE_PASS_ARGS,
        const   size_t      num_chunks,
        const   size_t      num_words_minor,
        const   size_t      max_blocks) {

        typedef cub::BlockScan<word_std_t, BLOCKX, cub::BLOCK_SCAN_RAKING> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_z[BLOCKY];
        __shared__ typename BlockScan::TempStorage temp_storage_x[BLOCKY];

        for_parallel_y_tiled(by, num_words_minor) {

            const grid_t w = threadIdx.y + by * blockDim.y;

            word_std_t z = 0;
            word_std_t x = 0;

            if (w < num_words_minor && threadIdx.x < num_chunks) {
                const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, threadIdx.x);
                READ_INTERMEDIATE_PREFIX(bid, z, x);
            }

            BlockScan(temp_storage_z[threadIdx.y]).ExclusiveScan(z, z, 0, XOROP());
            BlockScan(temp_storage_x[threadIdx.y]).ExclusiveScan(x, x, 0, XOROP());

            if (w < num_words_minor && threadIdx.x < num_chunks) {
                const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, threadIdx.x);
                WRITE_INTERMEDIATE_PREFIX(bid, z, x);
            }

        }
    }

    #define CALL_SINGLE_PASS_FOR_BLOCK(X, Y) \
        scan_blocks_single_pass <X, Y> \
        <<<currentgrid, currentblock, 0, stream>>> ( \
            SINGLE_PASS_INPUT, \
            num_chunks, \
            num_words_minor, \
            max_blocks \
        )

    template <int BLOCKX, int BLOCKY>
    __global__ 
    void scan_blocks_pass_1(
		        PASS_1_ARGS_PREFIX,
		const   size_t          num_blocks,
		const   size_t          num_words_minor,
        const   size_t          max_blocks,
		const   size_t          max_sub_blocks
	)
	{
        typedef cub::BlockScan<word_std_t, BLOCKX, cub::BLOCK_SCAN_RAKING> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_z[BLOCKY];
        __shared__ typename BlockScan::TempStorage temp_storage_x[BLOCKY];

		for_parallel_y_tiled(by, num_words_minor) {

            const grid_t w = threadIdx.y + by * blockDim.y;
			
			for_parallel_x_tiled(bx, num_blocks) {

                const grid_t tid_x = threadIdx.x + bx * blockDim.x;
                
                word_std_t z = 0;
                word_std_t x = 0;

                if (w < num_words_minor && tid_x < num_blocks) {
                    const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, tid_x);
                    READ_INTERMEDIATE_PREFIX(bid, z, x);
                }

                word_std_t blocksum_z;
                word_std_t blocksum_x;

                BlockScan(temp_storage_z[threadIdx.y]).ExclusiveScan(z, z, 0, XOROP(), blocksum_z);
                BlockScan(temp_storage_x[threadIdx.y]).ExclusiveScan(x, x, 0, XOROP(), blocksum_x);

                if (w < num_words_minor && tid_x < num_blocks) {
                    const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, tid_x);
                    assert(tid_x < max_blocks);
                    WRITE_INTERMEDIATE_PREFIX(bid, z, x);
                }

                if (w < num_words_minor && threadIdx.x == blockDim.x - 1) {
                    assert((blockIdx.x * num_words_minor + w) < gridDim.x * num_words_minor);
                    grid_t sub_bid = PREFIX_SUBINTERMEDIATE_INDEX(w, bx);
                    assert(bx < max_sub_blocks);
                    WRITE_SUBBLOCK_PREFIX(sub_bid, blocksum_z, blocksum_x);
                }
            }
		}
	}

    #define CALL_PREFIX_PASS_1_FOR_BLOCK(X, Y) \
        scan_blocks_pass_1 <X, Y> \
        <<<currentgrid, currentblock, 0, stream>>> ( \
            MULTI_PASS_INPUT, \
            num_blocks, \
            num_words_minor, \
            max_blocks, \
            max_sub_blocks \
        )

    __global__
    void scan_blocks_pass_2(
                PASS_2_ARGS_PREFIX,
        const   size_t              num_blocks,
        const   size_t              num_words_minor,
        const   size_t              max_blocks,
		const   size_t              max_sub_blocks,
        const   size_t              pass_1_blocksize) {

        for_parallel_y(w, num_words_minor) {

            for_parallel_x(tid, num_blocks) {
                
                word_std_t z = 0;
                word_std_t x = 0;
                
                if ((tid / pass_1_blocksize) > 0) {
                    const size_t sub_bid = PREFIX_SUBINTERMEDIATE_INDEX(w, (tid / pass_1_blocksize));
                    assert((tid / pass_1_blocksize) < max_sub_blocks);
                    READ_SUBBLOCK_PREFIX(sub_bid, z, x);
                }

                const size_t bid = PREFIX_INTERMEDIATE_INDEX(w, tid);
                assert(tid < max_blocks);
                XOR_TO_INTERMEDIATE_PREFIX(bid, z, x);
            }
        }
    }

    void Prefix::alloc(const Tableau& input, const size_t& config_qubits, const size_t& max_window_bytes) {
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
        #if PREFIX_INTERLEAVE
        if (global_prefix == nullptr) {
            assert(num_qubits);
            assert(num_words_minor);
            const size_t max_array_size = num_qubits * num_words_minor;
            LOGN2(2, "allocating %lld memory for prefix global cells.. ", int64(max_array_size * sizeof(PrefixCell)));
            global_prefix = allocator.allocate<PrefixCell>(max_array_size);
            LOGDONE(2, 4);
        }
        #else
        targets.alloc(num_qubits, max_window_bytes, true, false, false);
        #endif
        if (!max_intermediate_blocks) {
            max_intermediate_blocks = nextPow2(ROUNDUP(num_qubits, MIN_BLOCK_INTERMEDIATE_SIZE));
            size_t max_array_size = max_intermediate_blocks * num_words_minor;
            LOGN2(2, "allocating memory for %lld intermediate blocks.. ", int64(max_intermediate_blocks));
            #if PREFIX_INTERLEAVE
            if (intermediate_prefix == nullptr)
                intermediate_prefix = allocator.allocate<PrefixCell>(max_array_size);
            #else
            if (intermediate_prefix_z == nullptr)
                intermediate_prefix_z = allocator.allocate<word_std_t>(max_array_size);
            if (intermediate_prefix_x == nullptr)
                intermediate_prefix_x = allocator.allocate<word_std_t>(max_array_size);
            #endif
            LOGDONE(2, 4);
            max_sub_blocks = MAX(1, (max_intermediate_blocks >> 1));
            max_array_size = max_sub_blocks * num_words_minor;
            LOGN2(2, "allocating memory for %lld sub-blocks.. ", int64(max_sub_blocks));
            #if PREFIX_INTERLEAVE
            if (subblock_prefix == nullptr)
                subblock_prefix = allocator.allocate<PrefixCell>(max_array_size);
            #else
            if (subblocks_prefix_z == nullptr)
                subblocks_prefix_z = allocator.allocate<word_std_t>(max_array_size);
            if (subblocks_prefix_x == nullptr)
                subblocks_prefix_x = allocator.allocate<word_std_t>(max_array_size);
            #endif
            LOGDONE(2, 4);
        }
    }

    void Prefix::resize(const Tableau& input, const size_t& max_window_bytes) {
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
        max_intermediate_blocks = nextPow2(ROUNDUP(num_qubits, MIN_BLOCK_INTERMEDIATE_SIZE));
        if (max_intermediate_blocks > MIN_SINGLE_PASS_THRESHOLD ||
            (options.tune_prefixprepare || options.tune_prefixfinal)) {
            max_sub_blocks = max_intermediate_blocks >> 1;
        }
        #if !PREFIX_INTERLEAVE
        targets.resize(num_qubits, max_window_bytes, true, false, false);
        #endif
    }

    void call_single_pass_kernel(
                SINGLE_PASS_ARGS,
        const   size_t&             num_chunks,
        const   size_t&             num_words_minor,
        const   size_t&             max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream) {

            GENERATE_SWITCH_FOR_CALL(CALL_SINGLE_PASS_FOR_BLOCK)
        }

    void call_scan_blocks_pass_1_kernel(
                PASS_1_ARGS_PREFIX,
        const   size_t&         	num_blocks,
        const   size_t&         	num_words_minor,
        const   size_t&         	max_blocks,
        const   size_t&         	max_sub_blocks,
        const   dim3&           	currentblock,
        const   dim3&           	currentgrid,
        const   cudaStream_t&   	stream) {

            GENERATE_SWITCH_FOR_CALL(CALL_PREFIX_PASS_1_FOR_BLOCK)
        }

    void Prefix::scan_blocks(const size_t& num_blocks, const size_t& inject_pass_1_blocksize, const cudaStream_t& stream) {
        assert(num_blocks <= max_intermediate_blocks);
        assert(nextPow2(num_blocks) == num_blocks);
        dim3 currentblock, currentgrid;
        bestblockprefixsingle.x = MAX(2, num_blocks);
        bestgridprefixsingle.x = 1;
        if (num_blocks <= MIN_SINGLE_PASS_THRESHOLD) {
            // Do single pass.
            if (bestblockprefixsingle.y == 1)
                bestblockprefixsingle.y = 2;
            if (bestblockprefixsingle.y * bestblockprefixsingle.x > MIN_SINGLE_PASS_THRESHOLD) 
                bestblockprefixsingle.y = 1;
            bestgridprefixsingle.y = ROUNDUP(num_words_minor, bestblockprefixsingle.y);
            TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblockprefixsingle, bestgridprefixsingle, num_words_minor);
            TRIM_GRID_IN_2D(bestblockprefixsingle, bestgridprefixsingle, num_words_minor, y);
            currentblock = bestblockprefixsingle, currentgrid = bestgridprefixsingle;
            LOGN2(2, " Running pass-x kernel scanning %lld chunks with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            call_single_pass_kernel(
                SINGLE_PASS_INPUT,
                num_blocks, 
                num_words_minor,
                max_intermediate_blocks,
                currentblock,
                currentgrid,
                stream);
            if (options.sync) {
                LASTERR("failed to scan in a single pass");
                cutimer.stop(stream);
                LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
            } else LOGDONE(2, 4);
        }
        else {
            // Do triple passes.
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
            FORCE_TRIM_GRID_IN_XY(num_blocks, num_words_minor);
            const size_t pass_1_blocksize = currentblock.x;
            const size_t pass_1_gridsize = ROUNDUP(num_blocks, pass_1_blocksize);
            LOGN2(2, "  Running pass-1 kernel scanning %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            call_scan_blocks_pass_1_kernel(
                MULTI_PASS_INPUT,
                num_blocks, 
                num_words_minor,
                max_intermediate_blocks,
                max_sub_blocks,
                currentblock,
                currentgrid,
                stream);
            if (options.sync) {
                LASTERR("failed to scan in pass-1 kernel");
                cutimer.stop(stream);
                LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
            } else LOGDONE(2, 4);

            // Single phase
            assert(bestgridprefixsingle.x == 1);
            bestblockprefixsingle.x = nextPow2(MAX(2, pass_1_gridsize));
            if (bestblockprefixsingle.x > MIN_SINGLE_PASS_THRESHOLD)
                LOGERROR("pass-1 scan exceeded (%lld) with block size (%lld).", MIN_SINGLE_PASS_THRESHOLD, bestblockprefixsingle.x);
            if (bestblockprefixsingle.y == 1)
                bestblockprefixsingle.y = 2;
            if (bestblockprefixsingle.y * bestblockprefixsingle.x > MIN_SINGLE_PASS_THRESHOLD) 
                bestblockprefixsingle.y = 1;
            bestgridprefixsingle.y = ROUNDUP(num_words_minor, bestblockprefixsingle.y);
            TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblockprefixsingle, bestgridprefixsingle, num_words_minor);
            TRIM_GRID_IN_2D(bestblockprefixsingle, bestgridprefixsingle, num_words_minor, y);
            currentblock = bestblockprefixsingle, currentgrid = bestgridprefixsingle;
            LOGN2(2, "  Running pass-x kernel scanning %lld chunks with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", pass_1_gridsize, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            call_single_pass_kernel(
                SINGLE_PASS_SUBINPUT,
                pass_1_gridsize, 
                num_words_minor,
                max_sub_blocks,
                currentblock,
                currentgrid,
                stream
            );
            if (options.sync) {
                LASTERR("failed to scan in a single pass");
                cutimer.stop(stream);
                LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
            } else LOGDONE(2, 4);
            
            // Final phase.
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
            FORCE_TRIM_GRID_IN_XY(num_blocks, num_words_minor);
            LOGN2(2, "  Running pass-2 kernel scanning %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            scan_blocks_pass_2 <<<currentgrid, currentblock, 0, stream>>> (
                MULTI_PASS_INPUT,
                num_blocks, 
                num_words_minor, 
                max_intermediate_blocks,
                max_sub_blocks,
                pass_1_blocksize
            );
            if (options.sync) {
                LASTERR("failed to scan in pass-2 kernel");
                cutimer.stop(stream);
                LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
            } else LOGDONE(2, 4);
        }
    }

    void Prefix::tune_scan_blocks(const size_t& num_blocks, const size_t& inject_pass_1_blocksize) {
        if (options.tune_prefixsingle) {
            SYNCALL;
            bestblockprefixsingle.x = MAX(2, num_blocks);
            bestgridprefixsingle.x = 1;
            tune_single_pass(
                bestblockprefixsingle, bestgridprefixsingle,
                2 * sizeof(word_std_t), // used to skip very large blocks.
                num_blocks,
                num_words_minor,
                SINGLE_PASS_INPUT,
                num_blocks, 
                num_words_minor,
                max_intermediate_blocks);
            SYNCALL;
        }
        
        // Now, assume max blocks we could get and tune for it the multiple passes.
        const size_t max_blocks = MIN(1, max_intermediate_blocks / 2);
        if (options.tune_prefixprepare) {
            SYNCALL;
            tune_prefix_pass_1(
                bestblockprefixprepare,
                bestgridprefixprepare,
                2 * sizeof(word_std_t), // used to skip very large blocks.
                max_blocks,
                num_words_minor,
                MULTI_PASS_INPUT,
                max_blocks, 
                num_words_minor,
                max_intermediate_blocks,
                max_sub_blocks);
            SYNCALL;
        }
        if (options.tune_prefixfinal) {
            SYNCALL;
            tune_prefix_pass_2(
                scan_blocks_pass_2, 
                bestblockprefixfinal, bestgridprefixfinal,
                max_blocks,
                num_words_minor,
                MULTI_PASS_INPUT,
                max_blocks, 
                num_words_minor,
                max_intermediate_blocks,
                max_sub_blocks,
                bestblockprefixprepare.x
            );
            SYNCALL;
        }
    }

}