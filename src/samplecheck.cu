#include "measurecheck.cuh"

namespace QuaSARQ {

    void MeasurementChecker::check_record_samples(
        const Tableau&  other_input,
        const Samples&  other_samples,
        const Circuit&  circuit,
        const depth_t&  depth_level,
        const size_t&   measurement_offset_before,
        const size_t&   num_words_minor)
    {
        SYNCALL;

        if (!input_copied)
            LOGERROR("device input not copied to the checker");

        if (record.empty())
            LOGERROR("samples not recorded");

        const size_t num_gates = circuit[depth_level].size();

        LOGN2(2, "  Checking samples record at depth level %d.. ", depth_level);

        if (measurement_offset_before != measures_count)
            LOGERROR("measurement offset mismatch at depth %d: expected %lld, got %lld",
                depth_level, measures_count, measurement_offset_before);

        copy_input(other_input, true);

        size_t measures_per_window = 0;
        for (size_t i = 0; i < num_gates; i++) {
            const Gate& gate = circuit.gate(depth_level, i);
            if (!isMeasurement(gate.type))
                LOGERROR("gate %lld at depth %d is not a measurement or reset gate (type: %d)", i, depth_level, int(gate.type));
            if (!isReset(gate.type))
                measures_per_window++;
        }

        for (size_t i = 0, mi = 0; i < num_gates; i++) {
            const Gate& gate = circuit.gate(depth_level, i);
            const size_t q = gate.wires[0];
            assert(q != INVALID_QUBIT);
            for (size_t w = 0; w < num_words_minor; w++) {
                const size_t q_word_idx = q * num_words_minor + w;
                switch (gate.type) {
                case R:
                    h_xs[q_word_idx] = 0;
                    break;
                case M:
                    samples[mi * num_words_minor + w] ^= word_std_t(h_xs[q_word_idx]);
                    break;
                case MR:
                    samples[mi * num_words_minor + w] ^= word_std_t(h_xs[q_word_idx]);
                    h_xs[q_word_idx] = 0;
                    break;
                default: break;
                }
                h_zs[q_word_idx] = d_zs[q_word_idx];
            }
            if (!isReset(gate.type))
                mi++;
        }

        for (size_t mi = 0; mi < measures_per_window; mi++) {
            const size_t m_idx = measurement_offset_before + mi;
            if (m_idx >= record.size())
                LOGERROR("measurement index %lld exceeds record size %lld at depth %d",
                    m_idx, record.size(), depth_level);
            for (size_t w = 0; w < num_words_minor; w++) {
                const size_t m_word_idx = m_idx * num_words_minor + w;
                const word_std_t ref_word = samples[mi * num_words_minor + w];
                const word_std_t gpu_word = word_std_t(other_samples.host[m_word_idx]);
                if (ref_word != gpu_word)
                    LOGERROR("sample mismatch at index %lld, word %lld (depth %d): ref 0x%llx, gpu 0x%llx",
                        m_idx, w, depth_level, uint64(ref_word), uint64(gpu_word));
            }
        }

        LOG2(2, "%sPASSED.%s", CGREEN, CNORMAL);
        measures_count += measures_per_window;
    }

    void MeasurementChecker::load_record_shot(
        const Samples&  samples,
        const size_t&   num_measurements,
        const size_t&   num_words_minor,
        const size_t&   shot)
    {
        for (size_t m = 0; m < num_measurements; m++) {
            const word_std_t word = word_std_t(samples.host[m * num_words_minor + WORD_OFFSET(shot)]);
            record[m] = bool((word >> (shot & WORD_MASK)) & 1);
        }
    }

}
