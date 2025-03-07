#include "simulator.hpp"
#include "pivot.cuh"
#include "tuner.cuh"
#include "datatypes.cuh"
#include "shared.cuh"
#include "commutation.cuh"

namespace QuaSARQ {

    // Find first stabilizer generator that anti-commutes with the obeservable qubit.
    INLINE_DEVICE void find_min_pivot(pivot_t& p, const qubit_t& q, const Table& inv_xs, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        uint32* shared_mins = SharedMemory<uint32>();
        
        grid_t tx = threadIdx.x;
        grid_t BX = blockDim.x;
        grid_t shared_tid = threadIdx.y * BX + tx;

        const qubit_t q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);

        uint32 local_min = INVALID_PIVOT;

        for_parallel_x(g, num_qubits) {
            const size_t word_idx = g * num_words_major + q_w + num_words_minor;
            const word_std_t qubit_word = inv_xs[word_idx];
            if (qubit_word & q_mask) {
                local_min = MIN(uint32(g), local_min);
            }
        }

        min_pivot_load_shared(shared_mins, local_min, INVALID_PIVOT, shared_tid, num_qubits);
        min_pivot_shared(shared_mins, local_min, shared_tid);

        if (!threadIdx.x && local_min != INVALID_PIVOT) { 
            atomicMin(&(p), local_min);
        }
    }

    __global__ void reset_all_pivots(pivot_t* pivots, const size_t num_gates) {
        for_parallel_x(i, num_gates) {
            pivots[i] = INVALID_PIVOT;
        }
    }

    __global__ void find_all_pivots(pivot_t* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            find_min_pivot(pivots[i], m.wires[0], *inv_xs, num_qubits, num_words_major, num_words_minor);
        }
    }

    __global__ void find_new_pivot_and_mark(Commutation* commutations, pivot_t* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
                                            const size_t gate_index, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        assert(m.size == 1);

        uint32* shared_mins = SharedMemory<uint32>();
        
        grid_t tx = threadIdx.x;
        grid_t BX = blockDim.x;
        grid_t shared_tid = threadIdx.y * BX + tx;

        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);

        uint32 local_min = INVALID_PIVOT;

        for_parallel_x(g, num_qubits) {
            const size_t word_idx = g * num_words_major + q_w + num_words_minor;
            const word_std_t qubit_word = (*inv_xs)[word_idx];
            if (qubit_word & q_mask) {
                commutations[g].anti_commuting = true;
                local_min = MIN(uint32(g), local_min);
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

    void Simulator::find_pivots(Tableau<DeviceAllocator>& tab, const size_t& num_pivots_or_index, const bool& bulky, const cudaStream_t& stream) {
        const size_t num_words_major = tab.num_words_major();
        const size_t num_words_minor = tab.num_words_minor();
        dim3 currentblock, currentgrid;
        if (bulky) {
            if (options.tune_allpivots) {
                SYNCALL;
                tune_kernel_m(find_all_pivots, "Find all pivots", 
                                bestblockallpivots, bestgridallpivots, 
                                sizeof(uint32), true,   // shared size, extend?
                                num_qubits,             // x-dim
                                num_pivots_or_index,    // y-dim 
                                gpu_circuit.pivots(), 
                                gpu_circuit.gates(), 
                                gpu_circuit.references(), 
                                tab.xtable(), 
                                num_pivots_or_index, 
                                num_qubits, 
                                num_words_major, 
                                num_words_minor);
                reset_all_pivots <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), num_pivots_or_index);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblockallpivots, bestgridallpivots, num_qubits, num_pivots_or_index);
            currentblock = bestblockallpivots, currentgrid = bestgridallpivots;
            TRIM_GRID_IN_XY(num_qubits, num_pivots_or_index);
            OPTIMIZESHARED(smem_size, currentblock.y * currentblock.x, sizeof(uint32));
            LOGN2(2, "Finding all pivots with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            find_all_pivots <<< currentgrid, currentblock, smem_size, stream >>> 
            (
                gpu_circuit.pivots(), 
                gpu_circuit.gates(), 
                gpu_circuit.references(), 
                tab.xtable(), 
                num_pivots_or_index, 
                num_qubits, 
                num_words_major, 
                num_words_minor
            );
            if (options.sync) {
                LASTERR("failed to launch find_all_pivots_indet kernel");
                SYNC(stream);
            }
            LOGDONE(2, 4);
        }
        else {
            const size_t pivot_index = num_pivots_or_index;
            if (options.tune_newpivots) {
                SYNCALL;
                tune_kernel_m(find_new_pivot_and_mark, "New pivots", 
                    bestblocknewpivots, bestgridnewpivots, 
                    sizeof(uint32),
                    tab.commutations(),
                    gpu_circuit.pivots(), 
                    gpu_circuit.gates(), 
                    gpu_circuit.references(), 
                    tab.xtable(), 
                    pivot_index, 
                    num_qubits,
                    num_words_major, 
                    num_words_minor);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocknewpivots, bestgridnewpivots, num_qubits, 0);
            currentblock = bestblocknewpivots, currentgrid = bestgridnewpivots;
            TRIM_GRID_IN_1D(num_qubits, x);
            OPTIMIZESHARED(smem_size, currentblock.x, sizeof(uint32));
            find_new_pivot_and_mark <<< currentgrid, currentblock, smem_size, stream >>> 
            (
                tab.commutations(), 
                gpu_circuit.pivots(), 
                gpu_circuit.gates(), 
                gpu_circuit.references(), 
                tab.xtable(), 
                pivot_index, 
                num_qubits, 
                num_words_major, 
                num_words_minor
            );
            if (options.sync) {
                LASTERR("failed to launch find_new_pivot_and_mark kernel");
                SYNC(stream);
            }
        }
    }
}

