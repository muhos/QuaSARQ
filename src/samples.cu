#include "frame.hpp"
#include "random.cuh"
#include "locker.cuh"

namespace QuaSARQ {

    void print_record(
        const  Table&  samples, 
        const  size_t  num_qubits,
        const  size_t  num_shots) {
        string qidx = "q%-4lld";
        if (num_qubits > 1000)
            qidx = "q%-10lld";
        for (size_t q = 0; q < num_qubits; q++) {
            PRINT(qidx.c_str(), int64(q));
            int count = 0;
            for (size_t s = 0; s < num_shots; s++) {
                const size_t word_idx = q * samples.num_words_minor() + WORD_OFFSET(s);
                const word_std_t& word = samples[word_idx];
                const word_std_t bitpos = s & WORD_MASK;
                const int bit = (word >> bitpos) & 1;
                count += bit;
                PRINT("%d", bit);
            }
            PRINT(" ,   count: %d\n", count);
        }
    }

        __global__ 
    void record_sample(
                const_refs_t 	refs,
                const_buckets_t gates,
        const 	size_t 			num_gates,
        const 	size_t 			num_words_minor,
        const   uint64          seed,
                Table *			xs, 
                Table *			zs,
                Table *			samples) 
    {
        curandStatePhilox4_32_10_t st;
        for_parallel_y(i, num_gates) {
            for_parallel_x(w, num_words_minor) {
                const gate_ref_t r = refs[i];
                assert(r < NO_REF);
                const Gate& gate = (Gate&) gates[r];
                assert(gate.size <= 2);
                const size_t q = gate.wires[0];
                assert(q != INVALID_QUBIT);
                const size_t q_word_idx = q * num_words_minor + w;
                (*samples)[q_word_idx] ^= (*xs)[q_word_idx];
                randomize_word(
                    (*zs)[q_word_idx], 
                    st, 
                    seed, 
                    w * num_gates + i
                );
            }
        }
    }

    void Framing::shot(const depth_t& depth_level, const cudaStream_t& stream) {
        const size_t num_gates_per_window = circuit[depth_level].size();
        dim3 currentblock(1, 1), currentgrid(1, 1);
        currentblock = bestblockstep;
        currentgrid = bestgridstep;
        std::swap(currentblock.x, currentblock.y);
        std::swap(currentgrid.x, currentgrid.y);
        LOGN2(2, "Recording %lld measurements under %lld shots with block(x:%u, y:%u) and grid(x:%u, y:%u).. ",
            num_gates_per_window, tableau.num_words_minor() * WORD_BITS, 
            currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        double elapsed = 0;
        if (options.sync) cutimer.start(stream);
        record_sample <<< currentgrid, currentblock, 0, stream >>> (
            gpu_circuit.references(), 
            gpu_circuit.gates(), 
            num_gates_per_window, 
            tableau.num_words_minor(),
            options.seed,
            XZ_TABLE(tableau),
            samples_record.device
        );
        stats.circuit.measure_stats.random += num_gates_per_window;
        stats.circuit.measure_stats.definite = 0;
        stats.circuit.measure_stats.random_per_window = MAX(num_gates_per_window, stats.circuit.measure_stats.random_per_window);
        if (options.sync) {
            LASTERR("failed to launch randomize kernel");
            cutimer.stop(stream);
            elapsed = cutimer.elapsed();
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
    }

    void Framing::print() {
        if (!options.print_sample) return;
		if (!options.sync) SYNCALL;
		LOGHEADER(1, 4, "Sampling");
        samples_record.copy();
        print_record(
            samples_record.host, 
            num_qubits, 
            num_shots
        );
        fflush(stdout);
    }

}