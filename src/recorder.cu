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
            mchecker.check_record_measurements(tableau, recorder, circuit, depth_level);
        }
    }

    __global__
    void eval_record_refs_smem_k(
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const bool*   __restrict__ record,
        const uint32               record_size,
        const uint32               num_instructions,
              bool*                outcomes)
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
            outcomes[i] = result;
        }
    }

    __global__
    void eval_record_refs_tiled_k(
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const bool*   __restrict__ record,
        const uint32               record_size,
        const uint32               tile_size,
        const uint32               num_instructions,
              bool*                outcomes)
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
            outcomes[i] = result;
    }

    inline void launch_eval_record_refs(
        const uint32*      d_refs,
        const uint32*      d_starts,
        const uint32*      d_counts,
        const bool*        d_record,
        const uint32       record_size,
        const uint32       n,
              bool*        d_outcomes,
              bool*        h_outcomes,
        const cudaStream_t stream,
        const char*        label)
    {
        dim3 block(128, 1), grid;
        OPTIMIZEBLOCKS(grid.x, n, block.x);
        const size_t record_bytes = record_size * sizeof(bool);
        const size_t max_shared = maxGPUSharedMem;
        if (record_bytes <= max_shared) {
            eval_record_refs_smem_k<<<grid, block, record_bytes, stream>>>(
                d_refs, d_starts, d_counts, d_record, record_size, n, d_outcomes);
        } 
        else {
            const uint32 tile_size = (uint32)(max_shared / sizeof(bool));
            eval_record_refs_tiled_k<<<grid, block, max_shared, stream>>>(
                d_refs, d_starts, d_counts, d_record, record_size, tile_size, n, d_outcomes);
        }
        LASTERR(label);
        CHECK(cudaMemcpyAsync(h_outcomes, d_outcomes, n * sizeof(bool), cudaMemcpyDeviceToHost, stream));
    }

    void Simulator::print_observables() {
        if (!options.print_observable) return;
        const ObservableData& obs = circuit_io.observables;
        if (obs.empty()) return;
        const uint32 n            = (uint32)obs.pinned.num_observables;
        const uint32 record_size  = (uint32)recorder.step_history();
        const cudaStream_t stream = kernel_streams[0];
        bool* d_outcomes = gpu_allocator.allocate<bool>(n, Region::Dynamic);
        bool* h_outcomes = gpu_allocator.allocate_pinned<bool>(n);
        launch_eval_record_refs(
            obs.records.device.refs,
            obs.records.device.starts,
            obs.records.device.counts,
            recorder.device_record(),
            record_size, n,
            d_outcomes, h_outcomes, stream,
            "eval_record_refs (observables) failed");
        LOGHEADER(1, 4, "Observables");
        string bitstring;
        bitstring.reserve(n * 2);
        uint32 fired = 0;
        SYNC(stream); // sync h_outcomes.
        for (uint32 i = 0; i < n; i++) {
            if (h_outcomes[i]) fired++;
            bitstring += string(h_outcomes[i] ? CRED : CGREEN) + (h_outcomes[i] ? '1' : '0');
        }
        LOG1(" %sOutcome: %s%s", CBCYAN, CNORMAL, bitstring.c_str());
        LOG1(" %sLogical errors: %s%s%u / %u%s",
            CBCYAN, CNORMAL, fired ? CRED : CGREEN, fired, n, CNORMAL);
        gpu_allocator.deallocate<bool>(d_outcomes);
        gpu_allocator.deallocate_pinned<bool>(h_outcomes);
    }

    void Simulator::print_detectors() {
        if (!options.print_detector) return;
        const DetectorData& det = circuit_io.detectors;
        if (det.empty()) return;
        const uint32 n            = (uint32)det.pinned.num_instructions;
        const uint32 record_size  = (uint32)recorder.step_history();
        const cudaStream_t stream = kernel_streams[0];
        bool* d_outcomes = gpu_allocator.allocate<bool>(n, Region::Dynamic);
        bool* h_outcomes = gpu_allocator.allocate_pinned<bool>(n);
        launch_eval_record_refs(
            det.device.refs,
            det.device.starts,
            det.device.counts,
            recorder.device_record(),
            record_size, n,
            d_outcomes, h_outcomes, stream,
            "eval_record_refs (detectors) failed");
        gpu_allocator.deallocate<bool>(d_outcomes);
        LOGHEADER(1, 4, "Detectors");
        string bitstring;
        bitstring.reserve(n * 2);
        uint32 fired = 0;
        for (uint32 i = 0; i < n; i++) {
            if (h_outcomes[i]) fired++;
            bitstring += string(h_outcomes[i] ? CRED : CGREEN) + (h_outcomes[i] ? '1' : '0');
        }
        LOG1(" %sDetection bitstring     :%s %s",
            CBCYAN, CNORMAL, bitstring.c_str());
        LOG1(" %sDetection events fired  :%s %s%u / %u%s",
            CBCYAN, CNORMAL, fired ? CRED : CGREEN, fired, n, CNORMAL);
        gpu_allocator.deallocate_pinned<bool>(h_outcomes);
    }
        

}