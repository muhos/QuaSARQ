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
        dim3 currentblock, currentgrid;
        currentblock = bestblockreset, currentgrid = bestgridreset;
        TRIM_BLOCK_IN_DEBUG_MODE(currentblock, currentgrid, num_gates, 0);
        TRIM_GRID_IN_1D(num_gates, x);
        LOGN2(2, "recording measurements with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", 
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
    void eval_record_refs_smem_k(
              char*                bitstring,
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const bool*   __restrict__ record,
        const uint32               record_size,
        const uint32               num_instructions)
    {
        bool* sh_record = SharedMemory<bool>();
        for (grid_t k = threadIdx.x; k < record_size; k += blockDim.x)
            sh_record[k] = record[k];
        __syncthreads();
        for_parallel_x(i, num_instructions) {
            bool result = false;
            const uint32 start = starts[i];
            const uint32 count = counts[i];
            for (uint32 j = start; j < start + count; j++)
                result ^= sh_record[refs[j]];
            bitstring[i] = result ? '1' : '0';
        }
    }

    __global__
    void eval_record_refs_tiled_k(
              char*                bitstring,
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const bool*   __restrict__ record,
        const uint32               record_size,
        const uint32               tile_size,
        const uint32               num_instructions)
    {
        bool* sh_record = SharedMemory<bool>();
        const grid_t i = global_tx;
        bool result = false;
        for (uint32 tile_start = 0; tile_start < record_size; tile_start += tile_size) {
            const uint32 this_tile = MIN(tile_size, record_size - tile_start);
            for (grid_t k = threadIdx.x; k < this_tile; k += blockDim.x)
                sh_record[k] = record[tile_start + k];
            __syncthreads();
            if (i < num_instructions) {
                const uint32 start = starts[i];
                const uint32 count = counts[i];
                for (uint32 j = start; j < start + count; j++) {
                    const uint32 ref = refs[j];
                    if (ref >= tile_start && ref < tile_start + this_tile)
                        result ^= sh_record[ref - tile_start];
                }
            }
            __syncthreads();
        }
        if (i < num_instructions)
            bitstring[i] = result ? '1' : '0';
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
        const size_t record_bytes = record_size * sizeof(bool);
        const size_t max_shared   = maxGPUSharedMem;
        if (record_bytes <= max_shared) {
            eval_record_refs_smem_k<<<grid, block, record_bytes, stream>>>(
                d_bitstring, d_refs, d_starts, d_counts, d_record, record_size, n);
        } else {
            const uint32 tile_size = (uint32)(max_shared / sizeof(bool));
            eval_record_refs_tiled_k<<<grid, block, max_shared, stream>>>(
                d_bitstring, d_refs, d_starts, d_counts, d_record, record_size, tile_size, n);
        }
        LASTERR(label);
        CHECK(cudaMemcpyAsync(h_bitstring, d_bitstring, n * sizeof(char), cudaMemcpyDeviceToHost, stream));
    }

    inline 
    void print_bitstring(char* bs, uint32& fired, const uint32& n, const char* label, FILE* out) {
        bs[n] = '\0';
        for (uint32 i = 0; i < n; i++)
            if (bs[i] == '1') fired++;
        LOGHEADER(1, 4, label);
        if (out == stdout) {
            if (options.color_bitstring) {
                string colored;
                colored.reserve(n * 2);
                for (uint32 i = 0; i < n; i++)
                    colored += string(bs[i] == '1' ? CRED : CGREEN) + bs[i];
                LOGN1(" %s%s", colored.c_str(), CNORMAL);
            } else {
                LOGN1(" %s%s%s", CLBLUE, bs, CNORMAL);
            }
            LOG1(" (%s%u / %u%s)", fired ? CRED : CGREEN, fired, n, CNORMAL);
        } else {
            PRINTFILE("%s\n", out, bs);
        }
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
        mchecker.check_detectors(circuit_io.detectors, h_bitstring, n);
        gpu_allocator.deallocate_pinned<char>(h_bitstring);
        gpu_allocator.deallocate<char>(d_bitstring);
    }


}