#include "simulator.hpp"
#include "pivot.cuh"
#include "tuner.cuh"
#include "datatypes.cuh"
#include "shared.cuh"
#include "access.cuh"
#include <cub/device/device_select.cuh>

namespace QuaSARQ {

    #define min_pivot_load_shared(smem, val, init_val, tid, size) \
	{ \
		smem[tid] = (threadIdx.x < size) ? val : init_val; \
		__syncthreads(); \
	}

	#define min_pivot_shared(smem, val, tid) \
	{ \
    	if (blockDim.x >= 1024) { \
			if (threadIdx.x < 512) { \
                smem[tid] = val = MIN(val, smem[tid + 512]); \
            } \
			__syncthreads(); \
		} \
		if (blockDim.x >= 512) { \
			if (threadIdx.x < 256) { \
                smem[tid] = val = MIN(val, smem[tid + 256]); \
            } \
			__syncthreads(); \
		} \
		if (blockDim.x >= 256) { \
			if (threadIdx.x < 128) { \
                smem[tid] = val = MIN(val, smem[tid + 128]); \
            } \
			__syncthreads(); \
		} \
		if (blockDim.x >= 128) { \
			if (threadIdx.x < 64) { \
                smem[tid] = val = MIN(val, smem[tid + 64]); \
            } \
			__syncthreads(); \
		} \
        if (threadIdx.x < 32) { \
            if (blockDim.x >= 64) { \
                smem[tid] = val = MIN(val, smem[tid + 32]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 32) { \
                smem[tid] = val = MIN(val, smem[tid + 16]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 16) { \
                smem[tid] = val = MIN(val, smem[tid + 8]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 8) { \
                smem[tid] = val = MIN(val, smem[tid + 4]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 4) { \
                smem[tid] = val = MIN(val, smem[tid + 2]); \
                __syncthreads(); \
            } \
            if (blockDim.x >= 2) { \
                smem[tid] = val = MIN(val, smem[tid + 1]); \
                __syncthreads(); \
            } \
        } \
	}

    __global__ 
    void find_all_pivots(
                pivot_t*            pivots, 
                ConstBucketsPointer measurements, 
                ConstRefsPointer    refs, 
                ConstTablePointer   inv_xs, 
        const   size_t              num_gates, 
        const   size_t              num_qubits, 
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded) 
    {
        pivot_t* shared_mins = SharedMemory<pivot_t>();
        grid_t shared_tid = threadIdx.y * blockDim.x + threadIdx.x;

        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            const Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            const qubit_t q = m.wires[0];
            const size_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = BITMASK_GLOBAL(q);

            pivot_t local_min = INVALID_PIVOT;

            for_parallel_x(g, num_qubits) {
                const size_t word_idx = TABLEAU_INDEX(q_w, g) + TABLEAU_STAB_OFFSET;
                const word_std_t qubit_word = (*inv_xs)[word_idx];
                if (qubit_word & q_mask) {
                    local_min = MIN(pivot_t(g), local_min);
                }
            }

            min_pivot_load_shared(shared_mins, local_min, INVALID_PIVOT, shared_tid, num_qubits);
            min_pivot_shared(shared_mins, local_min, shared_tid);

            if (!threadIdx.x && local_min != INVALID_PIVOT) { 
                atomicMin(pivots + i, local_min);
            }
        }
    }

    __global__ 
    void anti_commuting_pivots (
                pivot_t*              scatter,
                ConstTablePointer     inv_xs, 
        const   qubit_t               qubit, 
        const   size_t                num_qubits, 
        const   size_t                num_words_major, 
        const   size_t                num_words_minor,
        const   size_t                num_qubits_padded) {
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        for_parallel_x(g, num_qubits) {
            const size_t word_idx = TABLEAU_INDEX(q_w, g) + TABLEAU_STAB_OFFSET;
            const word_std_t qubit_word = (*inv_xs)[word_idx];
            scatter[g] = (qubit_word & q_mask) ? pivot_t(g) : INVALID_PIVOT;
        }
    }

    __global__ 
    void compact_pivots_seq(
                pivot_t*              pivots,
                uint32*               num_compacted,
                ConstTablePointer     inv_xs, 
        const   qubit_t               qubit, 
        const   size_t                num_qubits, 
        const   size_t                num_words_major, 
        const   size_t                num_words_minor,
        const   size_t                num_qubits_padded) {
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        *num_compacted = 0;
        for (size_t g = 0; g < num_qubits; g++) {
            const size_t word_idx = TABLEAU_INDEX(q_w, g) + TABLEAU_STAB_OFFSET;
            const word_std_t qubit_word = (*inv_xs)[word_idx];
            if (qubit_word & q_mask) 
                pivots[(*num_compacted)++] = pivot_t(g);
        }
    }

    __global__ 
    void reset_all_pivots(pivot_t* pivots, const size_t num_gates) 
    {
        for_parallel_x(i, num_gates) {
            pivots[i] = INVALID_PIVOT;
        }
    }

    __global__
    void print_compacted(const pivot_t* pivots, const uint32* num_compacted) {
        LOGGPU("\nCompacted pivots:\n");
        for (uint32 i = 0; i < *num_compacted; i++) {
            LOGGPU(" pivots[%u] = %u\n", i, pivots[i]);
        }
    }

    void Pivoting::compact_pivots(const cudaStream_t& stream) {
        if (!auxiliary_bytes) {
            assert(auxiliary == nullptr);
            cub::DeviceSelect::If(nullptr, auxiliary_bytes, pivots, d_active_pivots, num_qubits, *this, stream);
            auxiliary = allocator.allocate<byte_t>(auxiliary_bytes);
        }
        if (auxiliary == nullptr) {
            LOGERROR("auxiliary buffer is not allocated");
        }
        cub::DeviceSelect::If(auxiliary,  auxiliary_bytes, pivots, d_active_pivots, num_qubits, *this, stream);
        CHECK(cudaMemcpyAsync(h_active_pivots, d_active_pivots, sizeof(uint32), cudaMemcpyDeviceToHost, stream));
    }

    void Simulator::reset_pivots(const size_t& num_pivots, const cudaStream_t& stream) {
        if (options.tune_reset) {
            SYNCALL;
            tune_kernel_m(reset_all_pivots, "Resetting pivots", bestblockreset, bestgridreset, pivoting.pivots, num_pivots);
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockreset, bestgridreset, num_pivots, 0);
        dim3 currentblock = bestblockreset, currentgrid = bestgridreset;
        TRIM_GRID_IN_1D(num_pivots, x);
        reset_all_pivots <<<currentgrid, currentblock, 0, stream>>> (pivoting.pivots, num_pivots);
        if (options.sync) {
            LASTERR("failed to launch reset_all_pivots kernel");
            SYNC(stream);
        }
    }

    void Simulator::find_pivots(const size_t& num_pivots, const cudaStream_t& stream) {
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        dim3 currentblock, currentgrid;
        if (options.tune_allpivots) {
            SYNCALL;
            tune_kernel_m(find_all_pivots, "Find all pivots", 
                            bestblockallpivots, bestgridallpivots, 
                            sizeof(pivot_t), true,   // shared size, extend?
                            num_qubits,             // x-dim
                            num_pivots,              // y-dim 
                            pivoting.pivots, 
                            gpu_circuit.gates(), 
                            gpu_circuit.references(), 
                            tableau.xtable(), 
                            num_pivots, 
                            num_qubits, 
                            num_words_major, 
                            num_words_minor,
                            num_qubits_padded);
            reset_all_pivots <<<bestgridreset, bestblockreset>>> (pivoting.pivots, num_pivots);
            SYNCALL;
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockallpivots, bestgridallpivots, num_qubits, num_pivots);
        currentblock = bestblockallpivots, currentgrid = bestgridallpivots;
        TRIM_GRID_IN_XY(num_qubits, num_pivots);
        OPTIMIZESHARED(smem_size, currentblock.y * currentblock.x, sizeof(pivot_t));
        LOGN2(2, "Finding all pivots with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        if (options.sync) cutimer.start(stream);
        find_all_pivots <<< currentgrid, currentblock, smem_size, stream >>> (
            pivoting.pivots, 
            gpu_circuit.gates(), 
            gpu_circuit.references(), 
            tableau.xtable(), 
            num_pivots, 
            num_qubits, 
            num_words_major, 
            num_words_minor,
            num_qubits_padded);
        if (options.sync) {
            LASTERR("failed to launch find_all_pivots kernel");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        } else LOGDONE(2, 4);
    }

    void Simulator::compact_targets(const qubit_t& qubit, const cudaStream_t& stream) {
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        dim3 currentblock, currentgrid;
        if (options.tune_newpivots) {
            SYNCALL;
            tune_kernel_m(anti_commuting_pivots, "New pivots", 
                bestblocknewpivots, bestgridnewpivots, 
                sizeof(pivot_t),
                pivoting.pivots, 
                tableau.xtable(), 
                qubit, 
                num_qubits, 
                num_words_major, 
                num_words_minor,
                num_qubits_padded);
            reset_all_pivots <<<bestgridreset, bestblockreset>>> (pivoting.pivots, num_qubits);
            SYNCALL;
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblocknewpivots, bestgridnewpivots, num_qubits, 0);
        currentblock = bestblocknewpivots, currentgrid = bestgridnewpivots;
        TRIM_GRID_IN_1D(num_qubits, x);
        OPTIMIZESHARED(smem_size, currentblock.x, sizeof(pivot_t));
        if (currentblock.y > 1) {
            LOGERROR("Kernel launch with 2D grid is not supported for anti_commuting_pivots kernel");
        }
        if (options.sync) cutimer.start(stream);
        #if	defined(_DEBUG) || defined(DEBUG)
        LOGN2(2, "Finding new pivots for qubit %d sequentially.. ", qubit);
        compact_pivots_seq <<<1, 1, 0, stream>>> (
            pivoting.pivots,
            pivoting.d_active_pivots,
            tableau.xtable(), 
            qubit, 
            num_qubits, 
            num_words_major, 
            num_words_minor,
            num_qubits_padded);
        CHECK(cudaMemcpyAsync(pivoting.h_active_pivots, pivoting.d_active_pivots, sizeof(uint32), cudaMemcpyDeviceToHost, stream));
        #else 
        LOGN2(2, "Finding new pivots for qubit %d using block(x:%u, y:%u) and grid(x:%u, y:%u).. ", 
            qubit, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        anti_commuting_pivots <<< currentgrid, currentblock, 0, stream >>> (
            pivoting.pivots,
            tableau.xtable(), 
            qubit, 
            num_qubits, 
            num_words_major, 
            num_words_minor,
            num_qubits_padded);
        pivoting.compact_pivots(stream);
        #endif
        if (options.sync) {
            LASTERR("failed to launch find_new_pivot_and_mark kernel");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(pivots: %d, time %.3f ms)", *(pivoting.h_active_pivots), cutimer.time());
        } else LOGDONE(2, 4);
    }
}

