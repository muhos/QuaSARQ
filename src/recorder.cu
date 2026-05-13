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
    void eval_observables_k(
        const uint32* __restrict__ refs,
        const uint32* __restrict__ starts,
        const uint32* __restrict__ counts,
        const bool*   __restrict__ record,
        const uint32               num_observables,
              bool*                outcomes)
    {
        for_parallel_x(i, num_observables) {
            bool result = false;
            const uint32 start = starts[i];
            const uint32 count = counts[i];
            for (uint32 j = start; j < start + count; j++)
                result ^= record[refs[j]];
            outcomes[i] = result;
        }
    }

    void Simulator::print_observables() {
        if (!options.print_observable) return;
        const ObservableData& obs = circuit_io.observables;
        if (obs.empty()) return;
        const uint32 n = (uint32)obs.pinned.num_observables;
        const cudaStream_t stream = kernel_streams[0];
        bool* d_outcomes = gpu_allocator.allocate<bool>(n, Region::Dynamic);
        bool* h_outcomes = gpu_allocator.allocate_pinned<bool>(n);
        dim3 block(128, 1), grid;
        OPTIMIZEBLOCKS(grid.x, n, block.x);
        eval_observables_k<<<grid, block, 0, stream>>>(
            obs.records.device.refs,
            obs.records.device.starts,
            obs.records.device.counts,
            recorder.device_record(),
            n,
            d_outcomes);
        LASTERR("eval_observables_k failed");
        CHECK(cudaMemcpyAsync(h_outcomes, d_outcomes, n * sizeof(bool), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        gpu_allocator.deallocate<bool>(d_outcomes);
        LOGHEADER(1, 4, "Observables");
        string bitstring;
        bitstring.reserve(n * 2);
        uint32 fired = 0;
        for (uint32 i = 0; i < n; i++) {
            if (h_outcomes[i]) fired++;
            bitstring += string(h_outcomes[i] ? CRED : CGREEN) + (h_outcomes[i] ? '1' : '0');
        }
        LOG1(" %sOutcome: %s%s", CBCYAN, CNORMAL, bitstring.c_str());
        LOG1(" %sLogical errors: %s%s%u / %u%s",
            CBCYAN, CNORMAL, fired ? CRED : CGREEN, fired, n, CNORMAL);
        gpu_allocator.deallocate_pinned<bool>(h_outcomes);
    }

    void Simulator::print_detectors() {
        if (!options.print_detector) return;
        const DetectorData& det = circuit_io.detectors;
        if (det.empty()) return;
        if (!recorder.is_copied()) recorder.copy();
        const Vec<bool>& rec = recorder.host_record();
        const uint32 n = det.pinned.num_instructions;
        Vec<bool, uint32> outcomes(n, false);
        uint32 fired = 0;
        for (uint32 i = 0; i < n; i++) {
            for (uint32 j = det.starts[i]; j < det.starts[i] + det.counts[i]; j++)
                outcomes[i] ^= rec[det.refs[j]];
            if (outcomes[i]) fired++;
        }
        LOGHEADER(1, 4, "Detectors");
        string bitstring;
        bitstring.reserve(n * 2);
        for (uint32 i = 0; i < n; i++)
            bitstring += string(outcomes[i] ? CRED : CGREEN) + (outcomes[i] ? '1' : '0');
        LOG1(" %sDetection bitstring     :%s %s",
            CBCYAN, CNORMAL, bitstring.c_str());
        LOG1(" %sDetection events fired  :%s %s%u / %u%s",
            CBCYAN, CNORMAL, fired ? CRED : CGREEN, fired, n, CNORMAL);
    }
        

}