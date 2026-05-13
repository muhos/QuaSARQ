#include "measurecheck.cuh"

namespace QuaSARQ {

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