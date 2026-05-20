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
                case R: {
                    (*xs)[q_word_idx] = 0;
                    break;
                }
                case M: {
                    (*samples)[m_word_idx] ^= (*xs)[q_word_idx];
                    break;
                }
                case MR: {
                    (*samples)[m_word_idx] ^= (*xs)[q_word_idx];
                    (*xs)[q_word_idx] = 0;
                    break;
                }
                default: break;
                }
                (*zs)[q_word_idx] = curand_word(&rand_states[q_word_idx]);
            }
        }
    }

    __global__
    void eval_frame_refs_k(
              char*                bitstring,
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const Table*               samples,
        const size_t               num_words_minor,
        const uint32               n,
        const size_t               num_shots)
    {
        for_parallel_y(s, num_shots) {
            for_parallel_x(i, n) {
                bool result = false;
                const uint32 start = starts[i];
                const uint32 count = counts[i];
                for (uint32 j = start; j < start + count; j++) {
                    const size_t m_idx = refs[j];
                    const word_std_t word = (*samples)[m_idx * num_words_minor + WORD_OFFSET(s)];
                    result ^= bool((word >> (s & WORD_MASK)) & 1);
                }
                bitstring[s * n + i] = result ? '1' : '0';
            }
        }
    }

    inline 
    void launch_eval_frame_refs(
              char*         d_bitstring,
              char*         h_bitstring,
        const uint32*       d_refs,
        const uint32*       d_starts,
        const uint32*       d_counts,
        const Table*        d_samples,
        const size_t&       num_words_minor,
        const uint32&       n,
        const size_t&       num_shots,
        const cudaStream_t& stream,
        const char*         label)
    {
        dim3 block(32, 8), grid(1, 1);
        OPTIMIZEBLOCKS2D(grid.x, n, block.x);
        OPTIMIZEBLOCKS2D(grid.y, (uint32)num_shots, block.y);
        eval_frame_refs_k<<<grid, block, 0, stream>>>(
            d_bitstring, d_refs, d_starts, d_counts,
            d_samples, num_words_minor, n, num_shots);
        LASTERR(label);
        CHECK(cudaMemcpyAsync(h_bitstring, d_bitstring, n * num_shots * sizeof(char), cudaMemcpyDeviceToHost, stream));
    }

    inline 
    void print_frame_shot(const char* row, const uint32& n, uint32& fired) {
        if (options.color_bitstring) {
            string colored;
            colored.reserve(n * 2);
            for (uint32 i = 0; i < n; i++) {
                if (row[i] == '1') fired++;
                colored += string(row[i] == '1' ? CRED : CGREEN) + row[i];
            }
            PRINT("%s%s", colored.c_str(), CNORMAL);
        } else {
            for (uint32 i = 0; i < n; i++) {
                if (row[i] == '1') fired++;
                PRINT("%c", row[i]);
            }
        }
        PRINT("\n");
    }

    void Framing::shot(const depth_t& depth_level, const cudaStream_t& stream) {
        if (options.check_measurement) {
            mchecker.copy_input(tableau);
        }
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
        assert(!circuit.is_recording(depth_level) || "R-only window invariant violated: window mixes R with M/MR");
        const size_t prev_measurement_offset = measurement_offset;
        measurement_offset += circuit.is_recording(depth_level) ? num_gates_per_window : 0;
        stats.circuit.measure_stats.random += num_gates_per_window;
        stats.circuit.measure_stats.definite = 0;
        stats.circuit.measure_stats.random_per_window = MAX(num_gates_per_window, stats.circuit.measure_stats.random_per_window);
        if (options.sync) {
            LASTERR("failed to launch randomize kernel");
            cutimer.stop(stream);
            elapsed = cutimer.elapsed();
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            samples_record.copy();
            mchecker.check_record_samples(tableau, samples_record, circuit, depth_level, prev_measurement_offset, tableau.num_words_minor());
            mchecker.reset_state();
        }
    }

    void Framing::print_detectors_sampled() {
        if (!options.print_detector) return;
        const DetectorData& dets = circuit_io.detectors;
        if (dets.empty()) return;
        const uint32 n            = (uint32)dets.pinned.num_instructions;
        const cudaStream_t stream = kernel_streams[0];
        char* d_bitstring = gpu_allocator.allocate<char>((size_t)n * num_shots, Region::Dynamic);
        char* h_bitstring = gpu_allocator.allocate_pinned<char>((size_t)n * num_shots);
        launch_eval_frame_refs(
            d_bitstring, h_bitstring,
            dets.device.refs,
            dets.device.starts,
            dets.device.counts,
            samples_record.device,
            tableau.num_words_minor(),
            n, num_shots, stream,
            "eval_frame_refs (detectors) failed");
        SYNC(stream);
        LOGHEADER(1, 4, "Detectors");
        for (size_t s = 0; s < num_shots; s++) {
            const char* row = h_bitstring + s * n;
            uint32 fired = 0;
            print_frame_shot(row, n, fired);
            if (options.check_measurement) {
                mchecker.load_record_shot(samples_record, stats.circuit.measure_stats.count, tableau.num_words_minor(), s);
                mchecker.check_detectors(circuit_io.detectors, row, n);
            }
        }
        gpu_allocator.deallocate_pinned<char>(h_bitstring);
        gpu_allocator.deallocate<char>(d_bitstring);
    }

    void Framing::print_observables_sampled() {
        if (!options.print_observable) return;
        const ObservableData& obs = circuit_io.observables;
        if (obs.empty()) return;
        const uint32 n            = (uint32)obs.pinned.num_observables;
        const cudaStream_t stream = kernel_streams[0];
        char* d_bitstring = gpu_allocator.allocate<char>((size_t)n * num_shots, Region::Dynamic);
        char* h_bitstring = gpu_allocator.allocate_pinned<char>((size_t)n * num_shots);
        launch_eval_frame_refs(
            d_bitstring, h_bitstring,
            obs.records.device.refs,
            obs.records.device.starts,
            obs.records.device.counts,
            samples_record.device,
            tableau.num_words_minor(),
            n, num_shots, stream,
            "eval_frame_refs (observables) failed");
        SYNC(stream);
        LOGHEADER(1, 4, "Observables");
        uint32 total_errors = 0;
        for (size_t s = 0; s < num_shots; s++) {
            const char* row = h_bitstring + s * n;
            uint32 fired = 0;
            print_frame_shot(row, n, fired);
            total_errors += fired;
            if (options.check_measurement) {
                mchecker.load_record_shot(samples_record, stats.circuit.measure_stats.count, tableau.num_words_minor(), s);
                mchecker.check_observables(circuit_io.observables, row, n);
            }
        }
        LOG1(" %sLogical errors across all shots: %s%s%u / %zu%s",
            CREPORT, CNORMAL, total_errors ? CRED : CGREEN,
            total_errors, num_shots * obs.pinned.num_observables, CNORMAL);
        gpu_allocator.deallocate_pinned<char>(h_bitstring);
        gpu_allocator.deallocate<char>(d_bitstring);
    }

    void Framing::print() {
        const bool any_print = samples_record.needs_host() || options.print_detector || options.print_observable;
        if (!any_print) return;
        if (!options.sync) SYNCALL;
        if (samples_record.needs_host()) {
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
        }
        print_detectors_sampled();
        print_observables_sampled();
        fflush(stdout);
    }

}
