#include "simulator.hpp"
#include "pivot.cuh"
#include "atomic.cuh"
#include "sum.cuh"

namespace QuaSARQ {

    #define RECORD_SELECT_CHUNK 1024

    // The four accumulated quantities collapse into one word before reduction: only
    // (ycnt - acnt) mod 4, par mod 2 and the row sign parity reach the outcome, and
    // sum % 4 is associative, so per thread differences may be reduced field-wise.
    // bits [1:0] = (ycnt - acnt) mod 4,   bit 2 = par mod 2,  bit 3 = row signs
    #define RECORD_PACK(par, ycnt, acnt, sgn) \
        ((((ycnt) - (acnt)) & 3u) | (((par) & 1u) << 2) | ((sgn) << 3))

    #define RECORD_COMBINE(a, b) \
        (((((a) & 3u) + ((b) & 3u)) & 3u) | (((a) ^ (b)) & 0xCu))

    #define RECORD_OUTCOME(packed) \
        bool((((packed) >> 3) ^ ((packed) >> 2) ^ ((packed) >> 1)) & 1u)

    // Marks which qubits the window measures deterministically, 
    // so that the destabilizer column scan can be compacted to 
    // only those generators.
    __global__
    void record_select_setup_k(
              uint32*               counts,
              word_std_t*           masks,
        const_refs_t                refs,
        const_buckets_t             gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor)
    {
        for_parallel_x(i, num_gates) {
            const Gate& gate = (Gate&) gates[refs[i]];
            const size_t q = gate.wires[0];
            counts[q] = 0;
            const size_t basis = (gate.type == byte_t(RX)) ? 0 : (gate.type == byte_t(RY)) ? 1 : 2;
            atomicXOR(masks + basis * num_words_minor + WORD_OFFSET(q), BITMASK_GLOBAL(q));
        }
    }

    // Fill in the precomputed S(q) lists for all measurements in the window.
    // Up to RECORD_MAX_SELECTED generators are recorded per qubit.
    __global__
    void record_select_k(
                uint32*             counts,
                uint32*             lists,
        const_words_t               masks,
        const_table_t               inv_xs,
        const_table_t               inv_zs,
        const   size_t              num_qubits,
        const   size_t              num_words_major,
        const   size_t              num_words_minor)
    {
        const word_std_t* mask_x = masks;
        const word_std_t* mask_y = masks + num_words_minor;
        const word_std_t* mask_z = masks + 2 * num_words_minor;
        for_parallel_y(g, num_qubits) {
            for_parallel_x(w, num_words_minor) {
                const size_t d_idx = g * num_words_major + w;
                const word_std_t xw = (*inv_xs)[d_idx];
                const word_std_t zw = (*inv_zs)[d_idx];
                word_std_t hits = (zw & mask_x[w]) | ((xw ^ zw) & mask_y[w]) | (xw & mask_z[w]);
                while (hits) {
                    const size_t q = (w << WORD_POWER) + ctz_word(hits);
                    hits &= hits - 1;
                    const uint32 k = atomicAdd(counts + q, 1u);
                    if (k < RECORD_MAX_SELECTED)
                        lists[q * RECORD_MAX_SELECTED + k] = uint32(g);
                }
            }
        }
    }

    __global__
    void record_signs_k(
        bool*                       record,
        bool*                       flags,
                const_table_t       inv_xs,
                const_table_t       inv_zs,
        const_signs_t               inv_ss,
        const_refs_t                refs,
        const_buckets_t             gates,
        const uint32* __restrict__  counts,
        const uint32* __restrict__  lists,
        const   size_t              num_gates,
        const   size_t              num_qubits,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              step_gates)
    {
        extern __shared__ uint32 smem_record[];
        uint32* s_count = smem_record;                      // selected generators in chunk
        uint32* s_list  = smem_record + 1;                  // their indices, compacted
        uint32* s_warp  = s_list + RECORD_SELECT_CHUNK;     // one packed partial per warp

        const sign_t* ss_stab = inv_ss->data(num_words_minor);

        for (size_t i = blockIdx.x; i < num_gates; i += gridDim.x) {
            const Gate& gate = (Gate&) gates[refs[i]];
            const size_t q = gate.wires[0];
            const size_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = BITMASK_GLOBAL(q);

            uint32 par = 0, ycnt = 0, acnt = 0, sgn = 0;

            const uint32 presel = (counts != nullptr) ? counts[q] : UINT32_MAX;
            const bool preselected = presel <= RECORD_MAX_SELECTED;
            const size_t chunk = preselected ? num_qubits : RECORD_SELECT_CHUNK;

            for (size_t w0 = 0; w0 < num_words_minor; w0 += blockDim.x) {
                const size_t w = w0 + threadIdx.x;
                const bool active = w < num_words_minor;
                word_std_t acc_x = 0, acc_z = 0;
                for (size_t g0 = 0; g0 < num_qubits; g0 += chunk) {
                    if (!threadIdx.x)
                        *s_count = preselected ? presel : 0;
                    __syncthreads();
                    if (preselected) {
                        if (threadIdx.x < presel)
                            s_list[threadIdx.x] = lists[q * RECORD_MAX_SELECTED + threadIdx.x];
                    }
                    else {
                        const size_t g_end = MIN(g0 + RECORD_SELECT_CHUNK, num_qubits);
                        for (size_t g = g0 + threadIdx.x; g < g_end; g += blockDim.x) {
                            const size_t d_idx = g * num_words_major + q_w;
                            const word_std_t sel = select_anticommuting_word(
                                (*inv_xs)[d_idx], (*inv_zs)[d_idx], gate.type);
                            if (sel & q_mask)
                                s_list[atomicAggInc(s_count)] = uint32(g);
                        }
                    }
                    __syncthreads();
                    const uint32 selected = *s_count;
                    for (uint32 k = 0; k < selected; k++) {
                        const size_t g = s_list[k];
                        if (!w0 && !threadIdx.x)
                            sgn ^= uint32(bool(ss_stab[WORD_OFFSET(g)] & BITMASK_GLOBAL(g)));
                        if (active) {
                            const size_t s_idx = g * num_words_major + w + num_words_minor;
                            const word_std_t xg = (*inv_xs)[s_idx];
                            const word_std_t zg = (*inv_zs)[s_idx];
                            par  += popcount_word(acc_z & xg);
                            ycnt += popcount_word(xg & zg);
                            acc_x ^= xg;
                            acc_z ^= zg;
                        }
                    }
                    __syncthreads();
                }
                if (active)
                    acnt += popcount_word(acc_x & acc_z);
            }

            uint32 packed = RECORD_PACK(par, ycnt, acnt, sgn);
            reduce_warps(RECORD_COMBINE, 0u, packed, s_warp);
            if (!threadIdx.x) {
                const bool value = RECORD_OUTCOME(packed);
                if (record != nullptr)
                    record[step_gates + i] = value;
                if (flags != nullptr)
                    flags[i] = value;
            }
            __syncthreads();
        }
    }

    void Simulator::launch_record_signs(const size_t& num_gates, bool* record_out, bool* flags_out, const cudaStream_t& stream) {
        assert(record_out != nullptr || flags_out != nullptr);
        assert(flags_out == nullptr || num_gates <= selector.max_gates());
        const size_t num_words_minor = tableau.num_words_minor();
        uint32 blocksize = 32;
        while (blocksize < num_words_minor && blocksize < 1024) blocksize <<= 1;
        const dim3 currentblock(blocksize, 1, 1);
        const dim3 currentgrid(MIN(num_gates, size_t(65535)), 1, 1);
        const size_t smem = (1 + RECORD_SELECT_CHUNK + 32) * sizeof(uint32);
        // One batched sweep costs num_qubits * num_words_minor words regardless of the window size,
        // while the per-measurement scan costs a strided 32-byte sector per generator per gate. The
        // sweep pays off once the window holds more than roughly num_words_minor/4 measurements.
        const bool batched = selector.allocated() && (num_gates * 4 > num_words_minor);
        LOGN2(2, "Recording measurements with block(x:%u, y:%u) and grid(x:%u, y:%u)%s.. ",
            currentblock.x, currentblock.y, currentgrid.x, currentgrid.y,
            batched ? ", batched selection" : "");
        if (options.sync) cutimer.start(stream);
        if (batched) {
            CHECK(cudaMemsetAsync(selector.device_masks(), 0, selector.mask_bytes(), stream));
            dim3 setupblock(128, 1), setupgrid;
            OPTIMIZEBLOCKS(setupgrid.x, num_gates, setupblock.x);
            record_select_setup_k <<<setupgrid, setupblock, 0, stream>>> (
                selector.device_counts(),
                selector.device_masks(),
                gpu_circuit.references(),
                gpu_circuit.gates(),
                num_gates,
                num_words_minor);
            dim3 selectblock(32, 8), selectgrid;
            OPTIMIZEBLOCKS2D(selectgrid.x, num_words_minor, selectblock.x);
            OPTIMIZEBLOCKS2D(selectgrid.y, num_qubits, selectblock.y);
            record_select_k <<<selectgrid, selectblock, 0, stream>>> (
                selector.device_counts(),
                selector.device_lists(),
                selector.device_masks(),
                tableau.xtable(),
                tableau.ztable(),
                num_qubits,
                tableau.num_words_major(),
                num_words_minor);
        }
        record_signs_k <<<currentgrid, currentblock, smem, stream>>> (
            record_out,
            flags_out,
            tableau.xtable(),
            tableau.ztable(),
            tableau.signs(),
            gpu_circuit.references(),
            gpu_circuit.gates(),
            batched ? selector.device_counts() : nullptr,
            batched ? selector.device_lists()  : nullptr,
            num_gates,
            num_qubits,
            tableau.num_words_major(),
            num_words_minor,
            recorder.step_history());
        if (options.sync) {
            LASTERR("failed to record signs");
            cutimer.stop(stream);
            double elapsed = cutimer.elapsed();
            if (options.profile) stats.profile.time.recordsigns += elapsed;
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
    }

    void Simulator::record_measurements(const size_t& num_gates, const depth_t& depth_level, const cudaStream_t& stream, bool* flags) {
        assert(circuit.is_recording(depth_level));
        launch_record_signs(num_gates, recorder.device_record(), flags, stream);
        recorder.reset_copied();
        recorder.advance(num_gates);
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
