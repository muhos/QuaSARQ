#include "simulator.hpp"
#include "pivot.cuh"
#include "tuner.cuh"
#include "datatypes.cuh"
#include "shared.cuh"
#include "commutation.cuh"
#include "access.cuh"

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
    void find_new_pivot_and_mark(
                Commutation*        commutations, 
                pivot_t*            pivots, 
                ConstBucketsPointer measurements, 
                ConstRefsPointer    refs, 
                ConstTablePointer   inv_xs, 
        const   size_t              gate_index, 
        const   size_t              num_qubits, 
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded) 
    {
        pivot_t* shared_mins = SharedMemory<pivot_t>();
        
        grid_t shared_tid = threadIdx.y * blockDim.x + threadIdx.x;

        const gate_ref_t r = refs[gate_index];
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
                commutations[g].anti_commuting = true;
                local_min = MIN(pivot_t(g), local_min);
            }
            else {
                commutations[g].reset();
            }
        }

        min_pivot_load_shared(shared_mins, local_min, INVALID_PIVOT, shared_tid, num_qubits);
        min_pivot_shared(shared_mins, local_min, shared_tid);

        if (!threadIdx.x && local_min != INVALID_PIVOT) { 
            atomicMin(pivots + gate_index, local_min);
        }
    }

    __global__ 
    void reset_all_pivots(pivot_t* pivots, const size_t num_gates) 
    {
        for_parallel_x(i, num_gates) {
            pivots[i] = INVALID_PIVOT;
        }
    }

    void Simulator::reset_pivots(const size_t& num_pivots, const cudaStream_t& stream) {
        if (options.tune_reset) {
            SYNCALL;
            tune_kernel_m(reset_all_pivots, "Resetting pivots", bestblockreset, bestgridreset, gpu_circuit.pivots(), num_pivots);
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockreset, bestgridreset, num_pivots, 0);
        dim3 currentblock = bestblockreset, currentgrid = bestgridreset;
        TRIM_GRID_IN_1D(num_pivots, x);
        reset_all_pivots <<<currentgrid, currentblock, 0, stream>>> (gpu_circuit.pivots(), num_pivots);
        if (options.sync) {
            LASTERR("failed to launch reset_all_pivots kernel");
            SYNC(stream);
        }
    }

    void Simulator::find_pivots(Tableau& tab, const size_t& num_pivots_or_index, const bool& bulky, const cudaStream_t& stream) {
        const size_t num_words_major = tab.num_words_major();
        const size_t num_words_minor = tab.num_words_minor();
        const size_t num_qubits_padded = tableau.num_qubits_padded();

        dim3 currentblock, currentgrid;
        if (bulky) {
            if (options.tune_allpivots) {
                SYNCALL;
                tune_kernel_m(find_all_pivots, "Find all pivots", 
                                bestblockallpivots, bestgridallpivots, 
                                sizeof(pivot_t), true,   // shared size, extend?
                                num_qubits,             // x-dim
                                num_pivots_or_index,    // y-dim 
                                gpu_circuit.pivots(), 
                                gpu_circuit.gates(), 
                                gpu_circuit.references(), 
                                tab.xtable(), 
                                num_pivots_or_index, 
                                num_qubits, 
                                num_words_major, 
                                num_words_minor,
                                num_qubits_padded);
                reset_all_pivots <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), num_pivots_or_index);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblockallpivots, bestgridallpivots, num_qubits, num_pivots_or_index);
            currentblock = bestblockallpivots, currentgrid = bestgridallpivots;
            TRIM_GRID_IN_XY(num_qubits, num_pivots_or_index);
            OPTIMIZESHARED(smem_size, currentblock.y * currentblock.x, sizeof(pivot_t));
            LOGN2(2, "Finding all pivots with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            find_all_pivots <<< currentgrid, currentblock, smem_size, stream >>> (
                gpu_circuit.pivots(), 
                gpu_circuit.gates(), 
                gpu_circuit.references(), 
                tab.xtable(), 
                num_pivots_or_index, 
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
        else {
            const size_t pivot_index = num_pivots_or_index;
            if (options.tune_newpivots) {
                SYNCALL;
                tune_kernel_m(find_new_pivot_and_mark, "New pivots", 
                    bestblocknewpivots, bestgridnewpivots, 
                    sizeof(pivot_t),
                    commutations,
                    gpu_circuit.pivots(), 
                    gpu_circuit.gates(), 
                    gpu_circuit.references(), 
                    tab.xtable(), 
                    pivot_index, 
                    num_qubits,
                    num_words_major, 
                    num_words_minor,
                    num_qubits_padded);
                reset_all_pivots <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), num_pivots_or_index);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocknewpivots, bestgridnewpivots, num_qubits, 0);
            currentblock = bestblocknewpivots, currentgrid = bestgridnewpivots;
            TRIM_GRID_IN_1D(num_qubits, x);
            OPTIMIZESHARED(smem_size, currentblock.x, sizeof(pivot_t));
            LOGN2(2, "Finding new pivot with marking using block(x:%u, y:%u) and grid(x:%u, y:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            find_new_pivot_and_mark <<< currentgrid, currentblock, smem_size, stream >>> (
                commutations, 
                gpu_circuit.pivots(), 
                gpu_circuit.gates(), 
                gpu_circuit.references(), 
                tab.xtable(), 
                pivot_index, 
                num_qubits, 
                num_words_major, 
                num_words_minor,
                num_qubits_padded);
            if (options.sync) {
                LASTERR("failed to launch find_new_pivot_and_mark kernel");
                cutimer.stop(stream);
                LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
            } else LOGDONE(2, 4);
        }
    }
}

