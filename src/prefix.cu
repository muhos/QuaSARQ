
#include "prefix.cuh"
#include "prefixcub.cuh"
#include "prefixdim.cuh"
#include "timer.cuh"

namespace QuaSARQ {

    __global__
    void scan_blocks_pass_2(
                word_std_t* block_intermediate_prefix_z,
                word_std_t* block_intermediate_prefix_x,
        const   word_std_t* subblocks_prefix_z,
        const   word_std_t* subblocks_prefix_x,
        const   size_t      num_blocks,
        const   size_t      num_words_minor,
        const   size_t      max_blocks,
		const   size_t      max_sub_blocks,
        const   size_t      pass_1_blocksize) {

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

    void call_single_pass_kernel(
                word_std_t *        intermediate_prefix_z,
                word_std_t *        intermediate_prefix_x,
        const   size_t              num_chunks,
        const   size_t              num_words_minor,
        const   size_t              max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream) {

            GENERATE_SWITCH_FOR_CALL(CALL_SINGLE_PASS_FOR_BLOCK)
        }

    void call_scan_blocks_pass_1_kernel(
                word_std_t*     block_intermediate_prefix_z,
                word_std_t*     block_intermediate_prefix_x,
                word_std_t*     subblocks_prefix_z, 
                word_std_t*     subblocks_prefix_x, 
        const   size_t          num_blocks,
        const   size_t          num_words_minor,
        const   size_t          max_blocks,
        const   size_t          max_sub_blocks,
        const   dim3&           currentblock,
        const   dim3&           currentgrid,
        const   cudaStream_t&   stream) {

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
            if (options.tune_prefixsingle) {
                SYNCALL;
                tune_single_pass(
                    bestblockprefixsingle, bestgridprefixsingle,
                    2 * sizeof(word_std_t), // used to skip very large blocks.
                    num_blocks,
                    num_words_minor,
                    block_intermediate_prefix_z, 
                    block_intermediate_prefix_x, 
                    num_blocks, 
                    num_words_minor,
                    max_intermediate_blocks);
                SYNCALL;
            }
            if (bestblockprefixsingle.y == 1)
                bestblockprefixsingle.y = 2;
            if (bestblockprefixsingle.y * bestblockprefixsingle.x > MIN_SINGLE_PASS_THRESHOLD) 
                bestblockprefixsingle.y = 1;
            bestgridprefixsingle.y = ROUNDUP(num_words_minor, bestblockprefixsingle.y);
            TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblockprefixsingle, bestgridprefixsingle, num_words_minor);
            TRIM_GRID_IN_2D(bestblockprefixsingle, bestgridprefixsingle, num_words_minor, y);
            currentblock = bestblockprefixsingle, currentgrid = bestgridprefixsingle;
            OPTIMIZESHARED(scan_blocks_smem_size, currentblock.y * (currentblock.x + CONFLICT_FREE_OFFSET(currentblock.x)), 2 * sizeof(word_std_t));
            LOGN2(2, " Running pass-x kernel scanning %lld chunks with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            call_single_pass_kernel(
                block_intermediate_prefix_z, 
                block_intermediate_prefix_x, 
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
            if (options.tune_prefixprepare) {
                SYNCALL;
                tune_prefix_pass_1(
                    bestblockprefixprepare,
                    bestgridprefixprepare,
                    2 * sizeof(word_std_t), // used to skip very large blocks.
                    num_blocks,
                    num_words_minor,
                    block_intermediate_prefix_z, 
                    block_intermediate_prefix_x, 
                    subblocks_prefix_z, 
                    subblocks_prefix_x, 
                    num_blocks, 
                    num_words_minor,
                    max_intermediate_blocks,
                    max_sub_blocks);
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
            LOGN2(2, "  Running pass-1 kernel scanning %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            call_scan_blocks_pass_1_kernel(
                block_intermediate_prefix_z, 
                block_intermediate_prefix_x, 
                subblocks_prefix_z, 
                subblocks_prefix_x, 
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
            if (options.tune_prefixsingle) {
                SYNCALL;
                tune_single_pass(
                    bestblockprefixsingle, bestgridprefixsingle,
                    2 * sizeof(word_std_t), // used to skip very large blocks.
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
            if (bestblockprefixsingle.y == 1)
                bestblockprefixsingle.y = 2;
            if (bestblockprefixsingle.y * bestblockprefixsingle.x > MIN_SINGLE_PASS_THRESHOLD) 
                bestblockprefixsingle.y = 1;
            bestgridprefixsingle.y = ROUNDUP(num_words_minor, bestblockprefixsingle.y);
            TRIM_Y_BLOCK_IN_DEBUG_MODE(bestblockprefixsingle, bestgridprefixsingle, num_words_minor);
            TRIM_GRID_IN_2D(bestblockprefixsingle, bestgridprefixsingle, num_words_minor, y);
            currentblock = bestblockprefixsingle, currentgrid = bestgridprefixsingle;
            OPTIMIZESHARED(scan_blocks_smem_size, currentblock.y * (currentblock.x + CONFLICT_FREE_OFFSET(currentblock.x)), 2 * sizeof(word_std_t));
            LOGN2(2, "  Running pass-x kernel scanning %lld chunks with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", pass_1_gridsize, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            call_single_pass_kernel(
                subblocks_prefix_z, 
                subblocks_prefix_x, 
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
            LOGN2(2, "  Running pass-2 kernel scanning %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", num_blocks, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
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
                LASTERR("failed to scan in pass-2 kernel");
                cutimer.stop(stream);
                LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
            } else LOGDONE(2, 4);
        }
    }

}