#include "simulator.hpp"
#include "pivot.cuh"
#include "tuner.cuh"
#include "datatypes.cuh"

namespace QuaSARQ {;

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

    __global__ void find_new_pivot(Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        assert(m.size == 1);
        find_min_pivot(pivots[gate_index], m.wires[0], *inv_xs, num_qubits, num_words_minor);
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

    void Simulator::find_pivots(const size_t& num_pivots_or_index, const bool& bulky, const cudaStream_t& stream) {
        const size_t num_words_minor = inv_tableau.num_words_minor();
        dim3 currentblock, currentgrid;
        if (bulky) {
            if (options.tune_allpivots) {
                SYNCALL;
                tune_kernel_m(find_all_pivots, "Find all pivots", 
                                bestblockallpivots, bestgridallpivots, 
                                0, false,               // shared size, extend?
                                num_qubits,             // x-dim
                                num_pivots_or_index,    // y-dim 
                                gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), num_pivots_or_index, num_qubits, num_words_minor);
                reset_all_pivots <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), num_pivots_or_index);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblockallpivots, bestgridallpivots, num_qubits, num_pivots_or_index);
            currentblock = bestblockallpivots, currentgrid = bestgridallpivots;
            TRIM_GRID_IN_XY(num_qubits, num_pivots_or_index);
            find_all_pivots <<<currentgrid, currentblock, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), num_pivots_or_index, num_qubits, num_words_minor);
            if (options.sync) {
                LASTERR("failed to launch find_all_pivots kernel");
                SYNC(stream);
            }
        }
        else {
            const size_t pivot_index = num_pivots_or_index;
            reset_single_pivot <<<1, 1, 0, stream>>> (gpu_circuit.pivots(), pivot_index);
            if (options.tune_newpivots) {
                SYNCALL;
                tune_kernel_m(find_new_pivot, "New pivots", bestblocknewpivots, bestgridnewpivots, gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), pivot_index, num_qubits, num_words_minor);
                reset_single_pivot <<<1, 1>>> (gpu_circuit.pivots(), pivot_index);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocknewpivots, bestgridnewpivots, num_qubits, 0);
            currentblock = bestblocknewpivots, currentgrid = bestgridnewpivots;
            TRIM_GRID_IN_1D(num_qubits, x);
            find_new_pivot <<<currentgrid, currentblock, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), pivot_index, num_qubits, num_words_minor);
            if (options.sync) {
                LASTERR("failed to launch find_new_pivot kernel");
                SYNC(stream);
            }
        }
    }
}

