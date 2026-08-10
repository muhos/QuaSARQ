#include "frame.cuh"
#include "locker.cuh"
#include "random.cuh"
#include "sum.cuh"

namespace QuaSARQ {

    void print_samples_measures(
        const  Table&   samples,
        const  size_t&  num_measurements,
        const  size_t&  num_shots,
               FILE*    out) {
        string midx = "m%-4lld";
        if (num_measurements > 1000)
            midx = "m%-10lld";
        for (size_t m = 0; m < num_measurements; m++) {
            PRINTFILE(midx.c_str(), out, int64(m));
            int count = 0;
            for (size_t s = 0; s < num_shots; s++) {
                const size_t word_idx = m * samples.num_words_minor() + WORD_OFFSET(s);
                const word_std_t& word = samples[word_idx];
                const word_std_t bitpos = s & WORD_MASK;
                const int bit = (word >> bitpos) & 1;
                count += bit;
                PRINTFILE("%d", out, bit);
            }
            PRINTFILE(" ,   count: %d\n", out, count);
        }
    }

    void print_samples(
        const  Table&   samples,
        const  size_t&  num_measurements,
        const  size_t&  num_shots,
               FILE*    out) {
        for (size_t s = 0; s < num_shots; s++) {
            for (size_t m = 0; m < num_measurements; m++) {
                const size_t word_idx = m * samples.num_words_minor() + WORD_OFFSET(s);
                const word_std_t& word = samples[word_idx];
                const word_std_t bitpos = s & WORD_MASK;
                PRINTFILE("%d", out, int((word >> bitpos) & 1));
            }
            PRINTFILE("\n", out);
        }
    }

    __global__
    void record_sample(
                const_refs_t        refs,
                const_buckets_t     gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor,
        const   size_t              measurement_offset,
        const uint32* __restrict__  ordinals,
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
                const bool records = (gate.type == M || gate.type == MR);
                const size_t m_word_idx = records ?
                    size_t(ordinals[measurement_offset + i]) * num_words_minor + w : 0;

                switch (gate.type) {
                case R: {
                    (*xs)[q_word_idx] = 0;
                    (*zs)[q_word_idx] = curand_word(&rand_states[q_word_idx]);
                    break;
                }
                case RX: {
                    (*zs)[q_word_idx] = 0;
                    (*xs)[q_word_idx] = curand_word(&rand_states[q_word_idx]);
                    break;
                }
                case RY: {
                    const word_std_t r = curand_word(&rand_states[q_word_idx]);
                    (*xs)[q_word_idx] = r;
                    (*zs)[q_word_idx] = r;
                    break;
                }
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

    __global__
    void apply_reference_sample_k(
              Table*               samples,
        const bool* __restrict__   record,
        const size_t               num_measurements,
        const size_t               num_words_minor)
    {
        for_parallel_y(m, num_measurements) {
            // We need the mask since samples is measurement-major.
            // We keep it that way to avoid transposing the whole 
            // sample table just to apply the reference sample.
            const word_std_t mask = record[m] ? ~word_std_t(0) : word_std_t(0);
            if (mask) {
                for_parallel_x(w, num_words_minor) {
                    (*samples)[m * num_words_minor + w] ^= mask;
                }
            }
        }
    }

    INLINE_DEVICE
    bool eval_ref_bit(
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const Table*               samples,
        const size_t&              num_words_minor,
        const uint32&              i,
        const size_t&              s)
    {
        bool result = false;
        const uint32 start = starts[i];
        const uint32 count = counts[i];
        for (uint32 j = start; j < start + count; j++) {
            const size_t m_idx = refs[j];
            const word_std_t word = (*samples)[m_idx * num_words_minor + WORD_OFFSET(s)];
            result ^= bool((word >> (s & WORD_MASK)) & 1);
        }
        return result;
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
                const bool result = eval_ref_bit(refs, starts, counts, samples, num_words_minor, i, s);
                bitstring[s * n + i] = result ? '1' : '0';
            }
        }
    }

    __global__
    void collect_frame_refs_k(
              uint8*               out,
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const Table*               samples,
        const size_t               num_words_minor,
        const uint32               n,
        const size_t               num_shots,
        const uint32               num_units,
        const bool                 bit_packed,
        const size_t               shot_offset)
    {
        for_parallel_y(s, num_shots) {
            const size_t shot = shot_offset + s;
            for_parallel_x(u, num_units) {
                uint8 value = 0;
                if (bit_packed) {
                    const uint32 base = u * 8;
                    const uint32 lim = MIN(8u, n - base);
                    for (uint32 k = 0; k < lim; k++) {
                        if (eval_ref_bit(refs, starts, counts, samples, num_words_minor, base + k, shot))
                            value |= uint8(1) << k;
                    }
                }
                else {
                    value = uint8(eval_ref_bit(refs, starts, counts, samples, num_words_minor, u, shot));
                }
                out[s * num_units + u] = value;
            }
        }
    }

    __global__
    void eval_observable_errors_k(
              uint64*              counters,
              uint32*              shot_flags,
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const Table*               samples,
        const size_t               num_words_minor,
        const uint32               n,
        const size_t               num_shots)
    {
        extern __shared__ uint64 shared[];
        const size_t tx = threadIdx.x;
        const size_t ty = threadIdx.y;
        const size_t tile_idx = ty * blockDim.x + tx;
        for_parallel_y(s, num_shots) {
            uint64 fired = 0;
            for_parallel_x(i, n) {
                fired += eval_ref_bit(refs, starts, counts, samples, num_words_minor, i, s);
            }
            load_shared_single(shared, fired, tile_idx, n - blockIdx.x * blockDim.x);
            sum_shared_single(shared, fired, tile_idx);
            sum_warp_single(shared, fired, tile_idx);
            if (tx == 0) {
                const uint64 fired_in_tile = fired;
                if (fired_in_tile) {
                    atomicAdd(counters, fired_in_tile);
                    if (atomicOr(shot_flags + s, uint32(1)) == 0)
                        atomicAdd(counters + 1, uint64(1));
                }
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
    void launch_eval_observable_errors(
              uint64*        d_counters,
              uint32*        d_shot_flags,
              uint64*        h_counters,
        const uint32*        d_refs,
        const uint32*        d_starts,
        const uint32*        d_counts,
        const Table*         d_samples,
        const size_t&        num_words_minor,
        const uint32&        n,
        const size_t&        num_shots,
        const cudaStream_t&  stream)
    {
        dim3 block(16, 64), grid(1, 1);
        OPTIMIZEBLOCKS2D(grid.x, n, block.x);
        OPTIMIZEBLOCKS2D(grid.y, (uint32)num_shots, block.y);
        TRIM_BLOCK_IN_DEBUG_MODE(block, grid, n, num_shots);
        CHECK(cudaMemsetAsync(d_counters, 0, 2 * sizeof(uint64), stream));
        CHECK(cudaMemsetAsync(d_shot_flags, 0, num_shots * sizeof(uint32), stream));
        eval_observable_errors_k<<<grid, block, block.x * block.y * sizeof(uint64), stream>>>(
            d_counters,
            d_shot_flags,
            d_refs,
            d_starts,
            d_counts,
            d_samples,
            num_words_minor,
            n,
            num_shots);
        LASTERR("eval_observable_errors failed");
        CHECK(cudaMemcpyAsync(h_counters, d_counters, 2 * sizeof(uint64), cudaMemcpyDeviceToHost, stream));
    }

    inline
    void print_frame_shot(const char* row, const uint32& n, uint32& fired, FILE* out) {
        if (options.color_results && out == stdout) {
            string colored;
            colored.reserve(n * 2);
            for (uint32 i = 0; i < n; i++) {
                if (row[i] == '1') fired++;
                colored += string(row[i] == '1' ? CRED : CGREEN) + row[i];
            }
            PRINTFILE("%s%s", out, colored.c_str(), CNORMAL);
        } else {
            for (uint32 i = 0; i < n; i++) {
                if (row[i] == '1') fired++;
                PRINTFILE("%c", out, row[i]);
            }
        }
        PRINTFILE("\n", out);
    }

    void Framing::shot(const depth_t& depth_level, const cudaStream_t& stream) {
        const bool checking = options.check_measurement && !mchecker.record.empty();
        if (checking) {
            mchecker.copy_input(tableau, false, false);
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
            recorder.device_ordinals(),
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
        if (checking) {
            samples_record.copy(stream);
            mchecker.check_record_samples(tableau, samples_record, circuit, depth_level, prev_measurement_offset, tableau.num_words_minor());
            mchecker.reset_state();
        }
    }

    bool Framing::needs_sample_host() const {
        return options.print_sample        ||
               options.print_sample_qubits ||
               options.check_measurement   ||
               sample_host_required;
    }

    void Framing::collect_frame_refs(
              uint8*        dest,
        const size_t&       dest_stride,
        const uint32&       n,
        const RecordRefs&   refs,
        const cudaStream_t& stream,
        const char*         label)
    {
        const bool bit_packed = results->bit_packed;
        const uint32 num_units = (uint32)FrameResults::stride_of(n, bit_packed);
        if (!num_units) return;
        size_t rows_per_batch = COLLECT_STAGING_BYTES / num_units;
        if (!rows_per_batch) rows_per_batch = 1;
        if (rows_per_batch > num_shots) rows_per_batch = num_shots;
        uint8* d_out = gpu_allocator.allocate<uint8>(rows_per_batch * num_units, Region::Dynamic);
        for (size_t start = 0; start < num_shots; start += rows_per_batch) {
            const size_t batch = MIN(rows_per_batch, num_shots - start);
            dim3 block(32, 8), grid(1, 1);
            OPTIMIZEBLOCKS2D(grid.x, num_units, block.x);
            OPTIMIZEBLOCKS2D(grid.y, (uint32)batch, block.y);
            collect_frame_refs_k<<<grid, block, 0, stream>>>(
                d_out,
                refs.device.refs,
                refs.device.starts,
                refs.device.counts,
                samples_record.device,
                tableau.num_words_minor(),
                n, batch, num_units, bit_packed, start);
            LASTERR(label);
            CHECK(cudaMemcpy2DAsync(
                dest + (chunk_start + start) * dest_stride, dest_stride,
                d_out, num_units,
                num_units, batch,
                cudaMemcpyDeviceToHost, stream));
            SYNC(stream);
        }
        gpu_allocator.deallocate<uint8>(d_out);
    }

    void Framing::print_detectors_sampled(FILE* out, const cudaStream_t& stream) {
        const DetectorData& dets = circuit_io.detectors;
        if (dets.empty()) return;
        const uint32 n            = (uint32)dets.pinned.num_instructions;
        if (results != nullptr && results->detectors != nullptr) {
            if (!dets.device.is_allocated())
                LOGERROR("detector results requested but detector arrays are not on the device; construct with require_detectors.");
            collect_frame_refs(results->detectors, results->detectors_stride, n, dets, stream,
                               "collect_frame_refs (detectors) failed");
        }
        if (!options.print_detector) return;
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
        if (out == stdout) LOG2(0, "%sDetectors:%s", CHEADER, CNORMAL);
        bool all_passed = true;
        for (size_t s = 0; s < num_shots; s++) {
            const char* row = h_bitstring + s * n;
            uint32 fired = 0;
            print_frame_shot(row, n, fired, out);
            if (options.check_measurement) {
                mchecker.load_record_shot(samples_record, stats.circuit.measure_stats.count, tableau.num_words_minor(), s);
                all_passed &= mchecker.check_detectors(circuit_io.detectors, row, n, true);
            }
        }
        if (all_passed && options.check_measurement) {
            LOGN2(1, " Checking detector bitstrings ");
            LOGPASSED(1);
        }
        gpu_allocator.deallocate_pinned<char>(h_bitstring);
        gpu_allocator.deallocate<char>(d_bitstring);
    }

    void Framing::print_observables_sampled(FILE* out, const cudaStream_t& stream) {
        const ObservableData& obs = circuit_io.observables;
        if (obs.empty()) return;
        const uint32 n            = (uint32)obs.pinned.num_observables;
        if (results != nullptr && results->observables != nullptr) {
            collect_frame_refs(results->observables, results->observables_stride, n, obs.records, stream,
                               "collect_frame_refs (observables) failed");
        }
        if (!options.print_observable && !options.check_measurement) {
            uint64* d_counters = gpu_allocator.allocate<uint64>(2, Region::Dynamic);
            uint32* d_shot_flags = gpu_allocator.allocate<uint32>(num_shots, Region::Dynamic);
            uint64* h_counters = gpu_allocator.allocate_pinned<uint64>(2);
            launch_eval_observable_errors(
                d_counters, d_shot_flags, h_counters,
                obs.records.device.refs,
                obs.records.device.starts,
                obs.records.device.counts,
                samples_record.device,
                tableau.num_words_minor(),
                n, num_shots, stream);
            SYNC(stream);
            stats.logical.total_observable_errors += (size_t)h_counters[0];
            stats.logical.shots_with_error        += (size_t)h_counters[1];
            stats.logical.total_shots             += num_shots;
            stats.logical.num_observables          = obs.pinned.num_observables;
            gpu_allocator.deallocate_pinned<uint64>(h_counters);
            gpu_allocator.deallocate<uint32>(d_shot_flags);
            gpu_allocator.deallocate<uint64>(d_counters);
            return;
        }
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
        size_t total_errors = 0;
        size_t shots_with_error = 0;
        bool all_passed = true;
        if (options.print_observable && out == stdout)
            LOG2(0, "%sObservables:%s", CHEADER, CNORMAL);
        for (size_t s = 0; s < num_shots; s++) {
            const char* row = h_bitstring + s * n;
            uint32 fired = 0;
            if (options.print_observable)
                print_frame_shot(row, n, fired, out);
            else
                for (uint32 i = 0; i < n; i++) if (row[i] == '1') fired++;
            total_errors += fired;
            if (fired) shots_with_error++;
            if (options.check_measurement) {
                mchecker.load_record_shot(samples_record, stats.circuit.measure_stats.count, tableau.num_words_minor(), s);
                all_passed &= mchecker.check_observables(circuit_io.observables, row, n, true);
            }
        }
        stats.logical.shots_with_error        += shots_with_error;
        stats.logical.total_shots             += num_shots;
        stats.logical.num_observables          = obs.pinned.num_observables;
        stats.logical.total_observable_errors += total_errors;
        if (all_passed && options.check_measurement) {
            LOGN2(1, " Checking observable bitstrings ");
            LOGPASSED(1);
        }
        gpu_allocator.deallocate_pinned<char>(h_bitstring);
        gpu_allocator.deallocate<char>(d_bitstring);
    }

    void Framing::print(const cudaStream_t& stream) {
        const bool any_results = results != nullptr &&
                                 (results->detectors != nullptr || results->observables != nullptr);
        const bool any_print = samples_record.needs_host() || options.print_detector
                             || options.print_observable || !circuit_io.observables.empty()
                             || any_results;
        if (!any_print) return;
        if (!options.sync) SYNCALL;
        // XOR reference sample into all shots.
        if (recorder.step_history() > 0) {
            const size_t num_measurements = stats.circuit.measure_stats.count;
            const size_t num_words_minor  = tableau.num_words_minor();
            dim3 block(32, 8), grid(1, 1);
            OPTIMIZEBLOCKS2D(grid.x, (uint32)num_words_minor, block.x);
            OPTIMIZEBLOCKS2D(grid.y, (uint32)num_measurements, block.y);
            apply_reference_sample_k<<<grid, block, 0, stream>>>(
                samples_record.device,
                recorder.device_record(),
                num_measurements,
                num_words_minor);
            LASTERR("apply_reference_sample failed");
            SYNC(stream);
        }
        if (options.print_detector || options.print_observable) LOGHEADER(1, 4, "Results");
        if (samples_record.needs_host()) {
            samples_record.copy(stream);
            if (options.print_sample) {
                FILE* out = write_measures_to_file ? open_output_file("_samples.01", chunk_index > 0) : stdout;
                if (!write_measures_to_file) LOG2(0, "%sSampling (shot per line):%s", CHEADER, CNORMAL);
                print_samples(samples_record.host, stats.circuit.measure_stats.count, num_shots, out);
                if (write_measures_to_file) fclose(out);
            }
            if (options.print_sample_qubits) {
                FILE* out = write_measures_to_file ? open_output_file("_samples_qubits.01", chunk_index > 0) : stdout;
                if (!write_measures_to_file) LOG2(0, "%sSampling (measurement per line):%s", CHEADER, CNORMAL);
                print_samples_measures(samples_record.host, stats.circuit.measure_stats.count, num_shots, out);
                if (write_measures_to_file) fclose(out);
            }
        }
        const bool want_detectors = options.print_detector ||
                                    (results != nullptr && results->detectors != nullptr);
        if (want_detectors) {
            const bool to_file = options.print_detector && write_measures_to_file;
            FILE* out = to_file ? open_output_file("_dets.01", chunk_index > 0) : stdout;
            print_detectors_sampled(out, stream);
            if (to_file) fclose(out);
        }
        if (!circuit_io.observables.empty()) {
            FILE* out = (options.print_observable && write_measures_to_file) ? open_output_file("_obs.01", chunk_index > 0) : stdout;
            print_observables_sampled(out, stream);
            if (options.print_observable && write_measures_to_file) fclose(out);
        }
        fflush(stdout);
    }

}
