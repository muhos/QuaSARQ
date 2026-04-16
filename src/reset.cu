#include "simulator.hpp"
#include "print.cuh"
#include "access.cuh"
#include "pivot.cuh"

namespace QuaSARQ {


    __global__
    void reset_signs_k(
        Signs*                      inv_ss, 
        const_refs_t                refs,
        const_buckets_t             gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor) 
    {
        sign_t* ss = inv_ss->data();
        uint64* smasks = SharedMemory<uint64>();
        smasks[threadIdx.x] = 0;
        __syncthreads();
        const Gate& base_gate = (Gate&) gates[refs[blockIdx.x * blockDim.x]];
        const size_t base_qword = WORD_OFFSET(base_gate.wires[0]);
        for_parallel_x(i, num_gates) {
            const Gate& gate = (Gate&) gates[refs[i]];
            const size_t q = gate.wires[0];
            const size_t local_qword = WORD_OFFSET(q) - base_qword;
            atomicOr(&smasks[local_qword], BITMASK_GLOBAL(q));
        }
        __syncthreads();
        // In case num_gates is not divisible by blockDim.x.
        const Gate& last_gate = (Gate&) gates[refs[MIN((blockIdx.x + 1) * blockDim.x, num_gates) - 1]];
        if (threadIdx.x < last_gate.wires[0] - base_qword) {
            word_std_t mask = static_cast<word_std_t>(smasks[threadIdx.x]);
            const size_t w = base_qword + threadIdx.x;
            mask = ~mask;
            ss[w] &= mask;
            ss[w + num_words_minor] &= mask;
        }
    }

    void Simulator::reset_signs(const size_t& num_gates, const cudaStream_t& stream) {
        dim3 currentblock, currentgrid;
        currentblock = bestblockreset, currentgrid = bestgridreset;
        TRIM_BLOCK_IN_DEBUG_MODE(currentblock, currentgrid, num_gates, 0);
        TRIM_GRID_IN_1D(num_gates, x);
        OPTIMIZESHARED(smem_size, (currentblock.x + 1), sizeof(word_std_t));
        LOGN2(2, "Resetting signs after collapsing with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        if (options.sync) cutimer.start(stream);
        reset_signs_k <<<currentgrid, currentblock, smem_size, stream>>> (
            tableau.signs(),
            gpu_circuit.references(), 
            gpu_circuit.gates(), 
            num_gates,
            tableau.num_words_minor());
        if (options.sync) {
            LASTERR("failed to reset signs");
            cutimer.stop(stream);
            double elapsed = cutimer.elapsed();
            if (options.profile) stats.profile.time.resetsigns += elapsed;
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            //mchecker.check_inject_swap(tableau, pivoting.pivots, 2);
        }
    }

}

