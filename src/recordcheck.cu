#include "measurecheck.cuh"

namespace QuaSARQ {

    // Mirrors record_signs_k: the record is the stabilizer sign bit of the measured qubit.
    bool MeasurementChecker::measured_value(const Gate& m) {
        const size_t q = m.wires[0];
        return bool(h_ss[num_words_minor + WORD_OFFSET(q)] & BITMASK_GLOBAL(q));
    }

    void MeasurementChecker::check_record_measurements(const MeasurementRecorder& other_recorder, const Circuit& circuit, const depth_t& depth_level) {
        SYNCALL;

        const Vec<bool>& other_record = other_recorder.host_record();

        if (other_record.empty()) {
            LOGERROR("other record is empty");
        }

        record.resize(other_record.size());

        LOGN2(2, "  Checking measurements record at depth level %d.. ", depth_level);

        const auto num_gates = circuit[depth_level].size();

        if (measures_count + num_gates != other_recorder.step_history()) {
            LOGERROR("measurements count mismatch: expected %lld, got %lld", measures_count + num_gates, other_recorder.step_history());
        }

        const Vec<uint32, size_t>& ordinals = circuit.record_ordinals();

        for (auto i = 0; i < num_gates; i++) {
            const Gate& m = circuit.gate(depth_level, i);
            if (!isMeasurement(m.type))
                LOGERROR("host gate %d at depth level %d is not a measurement gate", i, depth_level);
            if (measures_count + i >= ordinals.size())
                LOGERROR("record ordinal %lld exceeds table size %lld at depth %d",
                    measures_count + i, ordinals.size(), depth_level);
            const size_t m_idx = ordinals[measures_count + i];
            record[m_idx] = measured_value(m);
            if (record[m_idx] != other_record[m_idx]) {
                LOGERROR("Measurement record mismatch at history %lld", m_idx);
            }
        }

        LOGPASSED(2);

        measures_count += num_gates;
    }
        

}