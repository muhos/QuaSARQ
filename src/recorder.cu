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

    void Simulator::print_observables() {
        if (!options.print_observable) return;
        const ObservableData& obs = circuit_io.observables;
        if (obs.empty()) return;
        if (!recorder.is_copied()) recorder.copy();
        const Vec<bool>& rec = recorder.host_record();
        const uint32 n = obs.num_observables();
        Vec<bool, uint32> outcomes(n, false);
        uint32 fired = 0;
        for (uint32 i = 0; i < n; i++) {
            for (uint32 j = obs.ref_starts[i]; j < obs.ref_starts[i] + obs.ref_counts[i]; j++)
                outcomes[i] ^= rec[obs.record_refs[j]];
            if (outcomes[i]) fired++;
        }
        LOGHEADER(1, 4, "Observables");
        string bitstring;
        bitstring.reserve(n * 2);
        for (uint32 i = 0; i < n; i++)
            bitstring += string(outcomes[i] ? CRED : CGREEN) + (outcomes[i] ? '1' : '0');
        LOG1(" %sOutcome: %s%s", CBCYAN, CNORMAL, bitstring.c_str());
        LOG1(" %sLogical errors: %s%s%u / %u%s",
            CBCYAN, CNORMAL, fired ? CRED : CGREEN, fired, n, CNORMAL);
    }

    void Simulator::print_detectors() {
        if (!options.print_detector) return;
        const DetectorData& det = circuit_io.detectors;
        if (det.empty()) return;
        if (!recorder.is_copied()) recorder.copy();
        const Vec<bool>& rec = recorder.host_record();
        const uint32 n = det.num_detectors();
        Vec<bool, uint32> outcomes(n, false);
        uint32 fired = 0;
        for (uint32 i = 0; i < n; i++) {
            for (uint32 j = det.ref_starts[i]; j < det.ref_starts[i] + det.ref_counts[i]; j++)
                outcomes[i] ^= rec[det.record_refs[j]];
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

    void MeasurementChecker::check_record_measurements(const Tableau& other_input, const MeasurementRecorder& other_recorder, 
        const Circuit& circuit, const depth_t& depth_level) {
        SYNCALL;

        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }

        const Vec<bool>& other_record = other_recorder.host_record();

        if (other_record.empty()) {
            LOGERROR("other record is empty");
        }

        record.resize(other_record.size());

        LOGN2(2, "  Checking measurements record at depth level %d.. ", depth_level);

        copy_input(other_input, true);

        const auto num_gates = circuit[depth_level].size();

        if (measures_count + num_gates != other_recorder.step_history()) {
            LOGERROR("measurements count mismatch: expected %lld, got %lld", measures_count + num_gates, other_recorder.step_history());
        }

        for (auto i = 0; i < num_gates; i++) {
            const Gate& m = circuit.gate(depth_level, i);
            if (!isMeasurement(m.type))
                LOGERROR("host gate %d at depth level %d is not a measurement gate", i, depth_level);
            const size_t q = m.wires[0];
            const size_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = BITMASK_GLOBAL(q);
            record[measures_count + i] = bool(h_ss[q_w + num_words_minor] & q_mask);
            if (record[measures_count + i] != other_record[measures_count + i]) {
                LOGERROR("Measurement record mismatch at history %lld", measures_count + i);
            }
        }

        LOG2(2, "%sPASSED.%s", CGREEN, CNORMAL);

        measures_count += num_gates;
    }
        

}