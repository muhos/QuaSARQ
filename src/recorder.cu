#include "simulator.hpp"

namespace QuaSARQ {

    __global__
    void record_signs_k(
        bool*                       record,
        const_signs_t               inv_ss,
        const_refs_t                refs,
        const_buckets_t             gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor,
        const   size_t              step_gates)
    {
        const sign_t* ss_stab = inv_ss->data(num_words_minor);
        for_parallel_x(i, num_gates) {
            const Gate& gate = (Gate&) gates[refs[i]];
            const size_t q = gate.wires[0];
            record[step_gates + i] = bool(ss_stab[WORD_OFFSET(q)] & BITMASK_GLOBAL(q));
        }
    }

    void Simulator::record_measurements(const size_t& num_gates, const depth_t& depth_level, const cudaStream_t& stream) {
        assert(circuit.is_recording(depth_level));
        dim3 currentblock, currentgrid;
        currentblock = bestblockreset, currentgrid = bestgridreset;
        TRIM_BLOCK_IN_DEBUG_MODE(currentblock, currentgrid, num_gates, 0);
        TRIM_GRID_IN_1D(num_gates, x);
        LOGN2(2, "Recording measurements with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", 
            currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        if (options.sync) cutimer.start(stream);
        record_signs_k <<<currentgrid, currentblock, 0, stream>>> (
            recorder.device_record(),
            tableau.signs(),
            gpu_circuit.references(), 
            gpu_circuit.gates(), 
            num_gates,
            tableau.num_words_minor(),
            recorder.step_history());
        recorder.reset_copied();
        recorder.advance(num_gates);
        if (options.sync) {
            LASTERR("failed to reset signs");
            cutimer.stop(stream);
            double elapsed = cutimer.elapsed();
            if (options.profile) stats.profile.time.recordsigns += elapsed;
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            recorder.copy();
            mchecker.check_record_measurements(recorder, circuit, depth_level);
        }
    }

    __global__
    void eval_record_refs_k(
              char*                bitstring,
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const bool*   __restrict__ record,
        const uint32               num_instructions)
    {
        for_parallel_x(i, num_instructions) {
            bool result = false;
            const uint32 start = starts[i];
            const uint32 count = counts[i];
            for (uint32 j = start; j < start + count; j++)
                result ^= record[refs[j]];
            bitstring[i] = result ? '1' : '0';
        }
    }

    inline void launch_eval_record_refs(
              char*        d_bitstring,
              char*        h_bitstring,
        const uint32*      d_refs,
        const uint32*      d_starts,
        const uint32*      d_counts,
        const bool*        d_record,
        const uint32       record_size,
        const uint32       n,
        const cudaStream_t stream,
        const char*        label)
    {
        dim3 block(128, 1), grid;
        OPTIMIZEBLOCKS(grid.x, n, block.x);
        eval_record_refs_k<<<grid, block, 0, stream>>>(
            d_bitstring, d_refs, d_starts, d_counts, d_record, n);
        LASTERR(label);
        CHECK(cudaMemcpyAsync(h_bitstring, d_bitstring, n * sizeof(char), cudaMemcpyDeviceToHost, stream));
    }

    inline 
    void print_bitstring(char* bs, uint32& fired, const uint32& n, const char* label, FILE* out) {
        bs[n] = '\0';
        for (uint32 i = 0; i < n; i++)
            if (bs[i] == '1') fired++;
        if (out == stdout) {
            LOG2(0, "%s%s:%s", CHEADER, label, CNORMAL);
            if (options.color_results) {
                string colored;
                colored.reserve(n * 2);
                for (uint32 i = 0; i < n; i++)
                    colored += string(bs[i] == '1' ? CRED : CGREEN) + bs[i];
                LOGN2(0, "%s%s", colored.c_str(), CNORMAL);
            } else {
                LOGN2(0, "%s", bs);
            }
            LOG2(0, " (%s%u / %u%s)", fired ? CRED : CGREEN, fired, n, CNORMAL);
        } else {
            PRINTFILE("%s\n", out, bs);
        }
    }

    inline char eval_instruction_cpu(
        const uint32*    refs,
        const uint32     start,
        const uint32     count,
        const Vec<bool>& record)
    {
        bool result = false;
        for (uint32 j = start; j < start + count; j++)
            result ^= record[refs[j]];
        return result ? '1' : '0';
    }

    inline bool check_bitstring_against_record(
        const char*      label,
        const char*      bitstring,
        const uint32*    refs,
        const uint32*    starts,
        const uint32*    counts,
        const Vec<bool>& record,
        const uint32     n)
    {
        if (!options.check_measurement) return true;
        bool all_passed = true;
        for (uint32 i = 0; i < n; i++) {
            const char expected = eval_instruction_cpu(refs, starts[i], counts[i], record);
            if (bitstring[i] != expected) {
                LOGERRORN("%s %u mismatch against copied GPU record: GPU='%c', copied-record CPU='%c'",
                    label, i, bitstring[i], expected);
                all_passed = false;
            }
        }
        return all_passed;
    }

    inline bool check_records_match(
        const Vec<bool>& copied_gpu_record,
        const Vec<bool>& checker_record,
        const size_t     record_size)
    {
        if (!options.check_measurement) return true;
        bool all_passed = true;
        for (size_t i = 0; i < record_size; i++) {
            if (copied_gpu_record[i] != checker_record[i]) {
                LOGERRORN("Measurement record mismatch before det/obs eval at history %lld: GPU-record='%d', checker-record='%d'",
                    int64(i),
                    int(copied_gpu_record[i]),
                    int(checker_record[i]));
                all_passed = false;
            }
        }
        return all_passed;
    }

    inline bool check_device_record_refs(
        const char*        label,
        const uint32*      d_refs,
        const uint32*      d_starts,
        const uint32*      d_counts,
        const uint32*      h_refs,
        const uint32*      h_starts,
        const uint32*      h_counts,
        const size_t       num_refs,
        const size_t       num_instructions,
        const size_t       num_counts)
    {
        if (!options.check_measurement) return true;
        Vec<uint32, size_t> copied_refs(num_refs);
        Vec<uint32, size_t> copied_starts(num_instructions);
        Vec<uint32, size_t> copied_counts(num_counts);
        CHECK(cudaMemcpy(copied_refs.data(), d_refs, num_refs * sizeof(uint32), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(copied_starts.data(), d_starts, num_instructions * sizeof(uint32), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(copied_counts.data(), d_counts, num_counts * sizeof(uint32), cudaMemcpyDeviceToHost));

        bool all_passed = true;
        for (size_t i = 0; i < num_refs; i++) {
            if (copied_refs[i] != h_refs[i]) {
                LOGERRORN("%s refs device mismatch at %lld: device=%u, pinned=%u",
                    label, int64(i), copied_refs[i], h_refs[i]);
                all_passed = false;
            }
        }
        for (size_t i = 0; i < num_instructions; i++) {
            if (copied_starts[i] != h_starts[i]) {
                LOGERRORN("%s starts device mismatch at %lld: device=%u, pinned=%u",
                    label, int64(i), copied_starts[i], h_starts[i]);
                all_passed = false;
            }
        }
        for (size_t i = 0; i < num_counts; i++) {
            if (copied_counts[i] != h_counts[i]) {
                LOGERRORN("%s counts device mismatch at %lld: device=%u, pinned=%u",
                    label, int64(i), copied_counts[i], h_counts[i]);
                all_passed = false;
            }
        }
        return all_passed;
    }

    void Simulator::print_observables() {
        if (!options.print_observable) return;
        const ObservableData& obs = circuit_io.observables;
        if (obs.empty()) return;
        FILE* out = write_measures_to_file ? open_output_file("_obs.01") : stdout;
        const uint32 n            = (uint32)obs.pinned.num_observables;
        const uint32 record_size  = (uint32)recorder.step_history();
        const cudaStream_t stream = kernel_streams[0];
        char* d_bitstring = gpu_allocator.allocate<char>(n, Region::Dynamic);
        char* h_bitstring = gpu_allocator.allocate_pinned<char>(n + 1);
        if (options.check_measurement) {
            check_device_record_refs(
                "Observable before eval",
                obs.records.device.refs,
                obs.records.device.starts,
                obs.records.device.counts,
                obs.records.pinned.refs,
                obs.records.pinned.starts,
                obs.records.pinned.counts,
                obs.records.pinned.num_refs,
                obs.records.pinned.num_instructions,
                obs.records.pinned.num_counts);
        }
        launch_eval_record_refs(
            d_bitstring, h_bitstring,
            obs.records.device.refs,
            obs.records.device.starts,
            obs.records.device.counts,
            recorder.device_record(),
            record_size, n, stream,
            "eval_record_refs (observables) failed");
        SYNC(stream);
        uint32 fired = 0;
        print_bitstring(h_bitstring, fired, n, "Observables", out);
        if (write_measures_to_file) fclose(out);
        if (options.check_measurement) {
            check_device_record_refs(
                "Observable after eval",
                obs.records.device.refs,
                obs.records.device.starts,
                obs.records.device.counts,
                obs.records.pinned.refs,
                obs.records.pinned.starts,
                obs.records.pinned.counts,
                obs.records.pinned.num_refs,
                obs.records.pinned.num_instructions,
                obs.records.pinned.num_counts);
            recorder.copy();
            check_records_match(recorder.host_record(), mchecker.record, record_size);
            check_bitstring_against_record(
                "Observable",
                h_bitstring,
                obs.records.pinned.refs,
                obs.records.pinned.starts,
                obs.records.pinned.counts,
                recorder.host_record(),
                n);
        }
        mchecker.check_observables(circuit_io.observables, h_bitstring, n);
        gpu_allocator.deallocate_pinned<char>(h_bitstring);
        gpu_allocator.deallocate<char>(d_bitstring);
    }

    void Simulator::print_detectors() {
        if (!options.print_detector) return;
        const DetectorData& det = circuit_io.detectors;
        if (det.empty()) return;
        FILE* out = write_measures_to_file ? open_output_file("_det.01") : stdout;
        const uint32 n            = (uint32)det.pinned.num_instructions;
        const uint32 record_size  = (uint32)recorder.step_history();
        const cudaStream_t stream = kernel_streams[0];
        char* d_bitstring = gpu_allocator.allocate<char>(n, Region::Dynamic);
        char* h_bitstring = gpu_allocator.allocate_pinned<char>(n + 1);
        if (options.check_measurement) {
            check_device_record_refs(
                "Detector before eval",
                det.device.refs,
                det.device.starts,
                det.device.counts,
                det.pinned.refs,
                det.pinned.starts,
                det.pinned.counts,
                det.pinned.num_refs,
                det.pinned.num_instructions,
                det.pinned.num_counts);
        }
        launch_eval_record_refs(
            d_bitstring, h_bitstring,
            det.device.refs,
            det.device.starts,
            det.device.counts,
            recorder.device_record(),
            record_size, n, stream,
            "eval_record_refs (detectors) failed");
        SYNC(stream);
        uint32 fired = 0;
        print_bitstring(h_bitstring, fired, n, "Detectors", out);
        if (write_measures_to_file) fclose(out);
        if (options.check_measurement) {
            check_device_record_refs(
                "Detector after eval",
                det.device.refs,
                det.device.starts,
                det.device.counts,
                det.pinned.refs,
                det.pinned.starts,
                det.pinned.counts,
                det.pinned.num_refs,
                det.pinned.num_instructions,
                det.pinned.num_counts);
            recorder.copy();
            check_records_match(recorder.host_record(), mchecker.record, record_size);
            check_bitstring_against_record(
                "Detector",
                h_bitstring,
                det.pinned.refs,
                det.pinned.starts,
                det.pinned.counts,
                recorder.host_record(),
                n);
        }
        mchecker.check_detectors(circuit_io.detectors, h_bitstring, n);
        gpu_allocator.deallocate_pinned<char>(h_bitstring);
        gpu_allocator.deallocate<char>(d_bitstring);
    }


}
