#include "simulator.hpp"
#include "pivot.cuh"
#include "tuner.cuh"
#include "datatypes.cuh"

namespace QuaSARQ {

    // Find first stabilizer generator that anti-commutes with the obeservable qubit.

    // Try parallel reduction here to reduce atomicMin. 

    INLINE_DEVICE void find_min_pivot_indet(Pivot& p, const qubit_t& q, const Table& inv_xs, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        const qubit_t q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        for_parallel_x(g, num_qubits) {
            const size_t word_idx = g * num_words_major + q_w + num_words_minor;
            word_std_t qubit_word = inv_xs[word_idx];       
            if (qubit_word & q_mask) {
                printf("qubit(%d): g(%lld):" B2B_STR "\n", q, g, RB2B(qubit_word));
                atomicMin(&p.indeterminate, g);
            }
        }
    }

    #define BLOCK_SIZE 64

    INLINE_DEVICE void find_min_pivot_indet_shared(Pivot& p, const qubit_t& q, const Table& inv_xs, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {

        const qubit_t q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);

        // 1) Compute thread's global index
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x; // if we want to handle multiple passes

        // 2) We’ll hold a local min in a register
        int local_min_g = INT_MAX;

        // 3) Loop over all g values in a stride pattern
        for (int g = tid; g < num_qubits; g += stride)
        {
            // Same logic as before:
            const size_t word_idx = g * num_words_major + q_w + num_words_minor;
            word_std_t qubit_word = inv_xs[word_idx];
            if (qubit_word & q_mask)
            {
                // If set, keep track of the min index
                if (g < local_min_g)
                {
                    local_min_g = g;
                }
            }
        }

        // 4) Now we do a block-level reduction of local_min_g in shared memory
        __shared__ int shmem[BLOCK_SIZE][BLOCK_SIZE];

        shmem[threadIdx.y][threadIdx.x] = local_min_g;
        __syncthreads();

        // Basic binary reduction
        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
        {
            if (threadIdx.x < offset)
            {
                int other = shmem[threadIdx.y][threadIdx.x + offset];
                // minimum ignoring sentinel if one is INT_MAX
                if (other < shmem[threadIdx.y][threadIdx.x])
                {
                    shmem[threadIdx.y][threadIdx.x] = other;
                }
            }
            __syncthreads();
        }

        // 5) The first thread in the block writes out the result
        if (threadIdx.x == 0)
        {
            //block_mins[blockIdx.x] = shmem[0];
            atomicMin(&p.indeterminate, shmem[threadIdx.y][0]);
        }
    }


    // Let threads in x-dim find the minimum (de)stabilizer generator commuting.
    INLINE_DEVICE void find_min_pivot(Pivot& p, const qubit_t& q, const Table& inv_xs, const size_t num_qubits, const size_t num_words_minor) {
        const qubit_t q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        const grid_t stab_offset = num_qubits * num_words_minor;
        for_parallel_x(g, num_qubits) {
            const grid_t word_idx = g * num_words_minor + q_w;
            word_std_t qubit_word = inv_xs[stab_offset + word_idx];
            if (qubit_word & q_mask)
                atomicMin(&p.indeterminate, g);
            else {
                qubit_word = inv_xs[word_idx];
                if (qubit_word & q_mask)   
                    atomicMin(&p.determinate, g);
            }
        }
    }

    __global__ void reset_all_pivots(Pivot* pivots, const size_t num_gates) {
        for_parallel_x(i, num_gates) {
            pivots[i].reset();
        }
    }

    __global__ void reset_single_pivot(Pivot* pivots, const size_t gate_index) {
        pivots[gate_index].reset();
    }

    __global__ void find_all_pivots(Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor) {
        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            find_min_pivot(pivots[i], m.wires[0], *inv_xs, num_qubits, num_words_minor);
        }
    }

    __global__ void find_all_pivots_indet(Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            find_min_pivot_indet_shared(pivots[i], m.wires[0], *inv_xs, num_qubits, num_words_major, num_words_minor);
        }
    }

    __global__ void find_new_pivot(Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        assert(m.size == 1);
        find_min_pivot(pivots[gate_index], m.wires[0], *inv_xs, num_qubits, num_words_minor);
    }

    __global__ void find_new_pivot_indet(Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
                                const size_t gate_index, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor) {
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        assert(m.size == 1);
        find_min_pivot_indet(pivots[gate_index], m.wires[0], *inv_xs, num_qubits, num_words_major, num_words_minor);
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
                                0, false,               // shared size, extend?
                                num_qubits,             // x-dim
                                num_pivots_or_index,    // y-dim 
                                gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), tab.xtable(), num_pivots_or_index, num_qubits, num_words_minor);
                reset_all_pivots <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), num_pivots_or_index);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblockallpivots, bestgridallpivots, num_qubits, num_pivots_or_index);
            currentblock = bestblockallpivots, currentgrid = bestgridallpivots;
            TRIM_GRID_IN_XY(num_qubits, num_pivots_or_index);
            find_all_pivots_indet <<<currentgrid, currentblock, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), tab.xtable(), num_pivots_or_index, num_qubits, num_words_major, num_words_minor);
            if (options.sync) {
                LASTERR("failed to launch find_min_pivot_indet kernel");
                SYNC(stream);
            }
        }
        else {
            const size_t pivot_index = num_pivots_or_index;
            reset_single_pivot <<<1, 1, 0, stream>>> (gpu_circuit.pivots(), pivot_index);
            if (options.tune_newpivots) {
                SYNCALL;
                tune_kernel_m(find_new_pivot, "New pivots", bestblocknewpivots, bestgridnewpivots, gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), tab.xtable(), pivot_index, num_qubits, num_words_minor);
                reset_single_pivot <<<1, 1>>> (gpu_circuit.pivots(), pivot_index);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocknewpivots, bestgridnewpivots, num_qubits, 0);
            currentblock = bestblocknewpivots, currentgrid = bestgridnewpivots;
            TRIM_GRID_IN_1D(num_qubits, x);
            find_new_pivot_indet <<<currentgrid, currentblock, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), tab.xtable(), pivot_index, num_qubits, num_words_major, num_words_minor);
            if (options.sync) {
                LASTERR("failed to launch find_new_pivot kernel");
                SYNC(stream);
            }
        }
    }
}

