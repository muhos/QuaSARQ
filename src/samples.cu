#include "frame.cuh"
#include "locker.cuh"
#include "random.cuh"

namespace QuaSARQ {

    void print_samples_measures(
        const  Table&   samples,
        const  size_t&  num_measurements,
        const  size_t&  num_shots) {
        string midx = "m%-4lld";
        if (num_measurements > 1000)
            midx = "m%-10lld";
        for (size_t m = 0; m < num_measurements; m++) {
            PRINT(midx.c_str(), int64(m));
            int count = 0;
            for (size_t s = 0; s < num_shots; s++) {
                const size_t word_idx = m * samples.num_words_minor() + WORD_OFFSET(s);
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
        const  Table&   samples,
        const  size_t&  num_measurements,
        const  size_t&  num_shots) {
        for (size_t s = 0; s < num_shots; s++) {
            for (size_t m = 0; m < num_measurements; m++) {
                const size_t word_idx = m * samples.num_words_minor() + WORD_OFFSET(s);
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
        const   size_t              measurement_offset,
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
                const size_t m_word_idx = (measurement_offset + i) * num_words_minor + w;

                switch (gate.type) {
                case M: {
                    (*samples)[m_word_idx] ^= (*xs)[q_word_idx];
                    (*zs)[q_word_idx] = curand_word(&rand_states[q_word_idx]);
                    break;
                }
                case MR: {
                    (*samples)[m_word_idx] ^= (*xs)[q_word_idx];
                    (*xs)[q_word_idx] = 0;
                    (*zs)[q_word_idx] = curand_word(&rand_states[q_word_idx]);
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
            measurement_offset,
            rand_states,
            XZ_TABLE(tableau),
            samples_record.device
        );
        measurement_offset += num_gates_per_window;
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

    void Framing::print_detectors_sampled() {
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
                    // dets.refs[j] is a measurement index in circuit order.
                    const size_t m_idx = dets.refs[j];
                    const word_std_t word = samples_host[m_idx * samples_host.num_words_minor() + WORD_OFFSET(s)];
                    outcome ^= bool((word >> (s & WORD_MASK)) & 1);
                }
                if (outcome) fired++;
                bitstring += string(outcome ? CRED : CGREEN) + (outcome ? '1' : '0') + CNORMAL;
            }
            PRINT(" shot %-6zd: %s  (%u fired)\n", s, bitstring.c_str(), fired);
        }
    }

    void Framing::print_observables_sampled() {
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
                    // obs.records.refs[j] is a measurement index in circuit order.
                    const size_t m_idx = obs.records.refs[j];
                    const word_std_t word = samples_host[m_idx * samples_host.num_words_minor() + WORD_OFFSET(s)];
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
        const size_t num_measurements = stats.circuit.measure_stats.count;
        if (options.print_sample) {
            LOGHEADER(1, 4, "Sampling (shot per line)");
            print_samples(samples_record.host, num_measurements, num_shots);
        }
        if (options.print_sample_qubits) {
            LOGHEADER(1, 4, "Sampling (measurement per line)");
            print_samples_measures(samples_record.host, num_measurements, num_shots);
        }
        print_detectors_sampled();
        print_observables_sampled();
        fflush(stdout);
    }

}