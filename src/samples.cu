#include "frame.cuh"
#include "locker.cuh"

namespace QuaSARQ {

    void print_samples_qubits(
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

    void print_samples(
        const  Table&  samples,
        const  size_t  num_qubits,
        const  size_t  num_shots) {
        for (size_t s = 0; s < num_shots; s++) {
            for (size_t q = 0; q < num_qubits; q++) {
                const size_t word_idx = q * samples.num_words_minor() + WORD_OFFSET(s);
                const word_std_t& word = samples[word_idx];
                const word_std_t bitpos = s & WORD_MASK;
                PRINT("%d", int((word >> bitpos) & 1));
            }
            PRINT("\n");
        }
    }

        __global__
    void record_sample(
                const_refs_t        refs,
                const_buckets_t     gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor,
                curand_algorithm_t* rand_states,
                Table *             xs,
                Table *             zs,
                Table *             samples)
    {
        for_parallel_y(i, num_gates) {
            for_parallel_x(w, num_words_minor) {
                const gate_ref_t r = refs[i];
                assert(r < NO_REF);
                const Gate& gate = (Gate&) gates[r];
                assert(gate.size <= 2);
                const size_t q = gate.wires[0];
                assert(q != INVALID_QUBIT);
                const size_t q_word_idx = q * num_words_minor + w;

                // Pre-generate a random word consumed by all measurement types.
                // For MZ (M)/MRZ (MR): randomizes the Z frame (to guarantee anticommutation).
                // For MX/MRX: would randomize the X frame.
                // For MY/MRY: would randomize both frames.
                curand_algorithm_t local = rand_states[i * num_words_minor + w];
                #if defined(WORD_SIZE_8)
                    const word_t rnd = word_t(curand(&local) & 0xFFu);
                #elif defined(WORD_SIZE_32)
                    const word_t rnd = word_t(curand(&local));
                #elif defined(WORD_SIZE_64)
                    const word_std_t hi = word_std_t(curand(&local));
                    const word_std_t lo = word_std_t(curand(&local));
                    const word_t rnd = word_t((hi << 32) | lo);
                #endif
                rand_states[i * num_words_minor + w] = local;

                switch (gate.type) {
                case M: {
                    // X errors flip the Z-basis outcome.
                    (*samples)[q_word_idx] ^= (*xs)[q_word_idx];
                    (*zs)[q_word_idx] = rnd;
                    break;
                }
                case MR: {
                    // Record outcome, then reset qubit to 0.
                    (*samples)[q_word_idx] ^= (*xs)[q_word_idx];
                    (*xs)[q_word_idx] = 0;
                    (*zs)[q_word_idx] = rnd;
                    break;
                }
                default: break;
                }
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
        record_sample<<<currentgrid, currentblock, 0, stream>>>(
            gpu_circuit.references(),
            gpu_circuit.gates(),
            num_gates_per_window,
            tableau.num_words_minor(),
            rand_states,
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

    void Framing::print_detectors_sampled(const Vec<qubit_t, uint32>& measures_to_qubits) {
        if (!options.print_detector) return;
        const DetectorData& dets = circuit_io.detectors;
        if (dets.empty()) return;
        const Table& samples_host = samples_record.host;
        LOGHEADER(1, 4, "Detectors");
        for (size_t s = 0; s < num_shots; s++) {
            string bitstring;
            bitstring.reserve(dets.pinned.num_instructions * 2);
            uint32 fired = 0;
            for (uint32 i = 0; i < dets.pinned.num_instructions; i++) {
                bool outcome = false;
                for (uint32 j = dets.starts[i]; j < dets.starts[i] + dets.counts[i]; j++) {
                    const qubit_t q = measures_to_qubits[dets.refs[j]];
                    const word_std_t word = samples_host[q * samples_host.num_words_minor() + WORD_OFFSET(s)];
                    outcome ^= bool((word >> (s & WORD_MASK)) & 1);
                }
                if (outcome) fired++;
                bitstring += string(outcome ? CRED : CGREEN) + (outcome ? '1' : '0') + CNORMAL;
            }
            PRINT(" shot %-6zd: %s  (%u fired)\n", s, bitstring.c_str(), fired);
        }
    }

    void Framing::print_observables_sampled(const Vec<qubit_t, uint32>& measures_to_qubits) {
        if (!options.print_observable) return;
        const ObservableData& obs = circuit_io.observables;
        if (obs.empty()) return;
        const Table& samples_host = samples_record.host;
        LOGHEADER(1, 4, "Observables");
        uint32 total_errors = 0;
        for (size_t s = 0; s < num_shots; s++) {
            string bitstring;
            bitstring.reserve(obs.pinned.num_observables * 16);
            uint32 fired = 0;
            for (uint32 i = 0; i < obs.pinned.num_observables; i++) {
                bool outcome = false;
                for (uint32 j = obs.records.starts[i]; j < obs.records.starts[i] + obs.records.counts[i]; j++) {
                    const qubit_t q = measures_to_qubits[obs.records.refs[j]];
                    const word_std_t word = samples_host[q * samples_host.num_words_minor() + WORD_OFFSET(s)];
                    outcome ^= bool((word >> (s & WORD_MASK)) & 1);
                }
                if (outcome) { fired++; total_errors++; }
                bitstring += string(outcome ? CRED : CGREEN) + (outcome ? '1' : '0') + CNORMAL;
            }
            PRINT(" shot %-6zd: %s\n", s, bitstring.c_str());
        }
        LOG1(" %sLogical errors across all shots: %s%s%u / %zu%s",
            CREPORT, CNORMAL, total_errors ? CRED : CGREEN,
            total_errors, num_shots * obs.pinned.num_observables, CNORMAL);
    }

    void Framing::print() {
        if (!samples_record.needs_host()) return;
        if (!options.sync) SYNCALL;
        samples_record.copy();
        Vec<qubit_t, uint32> measures_to_qubits;
        if (options.print_detector || options.print_observable) {
            measures_to_qubits.reserve(stats.circuit.measure_stats.count);
            for (depth_t d = 0; d < depth; d++) {
                if (!circuit.is_measuring(d)) continue;
                for (uint32 g = 0; g < circuit[d].size(); g++) {
                    const Gate& gate = circuit.gate(d, g);
                    measures_to_qubits.push(gate.wires[0]);
                }
            }
        }
        if (options.print_sample) {
            LOGHEADER(1, 4, "Sampling (shot per line)");
            print_samples(samples_record.host, num_qubits, num_shots);
        }
        if (options.print_sample_qubits) {
            LOGHEADER(1, 4, "Sampling (qubit per line)");
            print_samples_qubits(samples_record.host, num_qubits, num_shots);
        }
        print_detectors_sampled(measures_to_qubits);
        print_observables_sampled(measures_to_qubits);
        fflush(stdout);
    }

}