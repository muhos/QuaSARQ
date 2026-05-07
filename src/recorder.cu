#include "simulator.hpp"

namespace QuaSARQ {

    __global__
    void record_signs_k(
        bool*                       record,
        const_signs_t               inv_ss, 
        const_refs_t                refs,
        const_buckets_t             gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor) 
    {
        const sign_t* ss_stab = inv_ss->data(num_words_minor);
        for_parallel_x(i, num_gates) {
            const Gate& gate = (Gate&) gates[refs[i]];
            const size_t q = gate.wires[0];
            record[q] = bool(ss_stab[WORD_OFFSET(q)] & BITMASK_GLOBAL(q));
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
            tableau.num_words_minor());
        recorder.reset_copied();
        if (options.sync) {
            LASTERR("failed to reset signs");
            cutimer.stop(stream);
            double elapsed = cutimer.elapsed();
            if (options.profile) stats.profile.time.recordsigns += elapsed;
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            recorder.copy();
            mchecker.check_record_measurements(tableau, recorder.host_record(), circuit, depth_level);
        }
    }

    void MeasurementChecker::check_record_measurements(const Tableau& other_input, const Vec<bool>& other_record, const Circuit& circuit, const depth_t& depth_level) {
        SYNCALL;

        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }

        LOGN2(2, "  Checking measurements record at depth level %d.. ", depth_level);

        copy_input(other_input, true);

        record.resize(num_qubits_padded);

        const auto num_gates = circuit[depth_level].size();
        for (auto i = 0; i < num_gates; i++) {
            const Gate& m = circuit.gate(depth_level, i);
            if (!isMeasurement(m.type))
                LOGERROR("host gate %d at depth level %d is not a measurement gate", i, depth_level);
            const size_t q = m.wires[0];
            const size_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = BITMASK_GLOBAL(q);
            record[q] = bool(h_ss[q_w + num_words_minor] & q_mask);
            if (record[q] != other_record[q]) {
                LOGERROR("Measurement record mismatch at qubit %lld", q);
            }
        }

        LOG2(2, "%sPASSED.%s", CGREEN, CNORMAL);
    }
        

}