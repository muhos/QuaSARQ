#include "measurecheck.cuh"

namespace QuaSARQ {

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

        for (auto i = 0; i < num_gates; i++) {
            const Gate& m = circuit.gate(depth_level, i);
            if (!isMeasurement(m.type))
                LOGERROR("host gate %d at depth level %d is not a measurement gate", i, depth_level);
            const size_t q = m.wires[0];
            const size_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = BITMASK_GLOBAL(q);
            Vec<word_std_t> acc_x(num_words_minor, 0), acc_z(num_words_minor, 0);
            uint32 par = 0, ycnt = 0, acnt = 0, sgn = 0;
            for (size_t g = 0; g < num_qubits; g++) {
                const size_t d_idx = g * num_words_major + q_w;
                const word_std_t sel = select_anticommuting_word(h_xs[d_idx], h_zs[d_idx], m.type);
                if (!(sel & q_mask))
                    continue;
                for (size_t w = 0; w < num_words_minor; w++) {
                    const size_t s_idx = g * num_words_major + w + num_words_minor;
                    const word_std_t xg = h_xs[s_idx], zg = h_zs[s_idx];
                    par  += uint32(__builtin_popcountll((uint64)(acc_z[w] & xg)));
                    ycnt += uint32(__builtin_popcountll((uint64)(xg & zg)));
                    acc_x[w] ^= xg;
                    acc_z[w] ^= zg;
                }
                sgn ^= uint32(bool(h_ss[WORD_OFFSET(g) + num_words_minor] & BITMASK_GLOBAL(g)));
            }
            for (size_t w = 0; w < num_words_minor; w++)
                acnt += uint32(__builtin_popcountll((uint64)(acc_x[w] & acc_z[w])));
            acc_x.clear(true), acc_z.clear(true);
            record[measures_count + i] = bool(sgn ^ (par & 1) ^ ((((ycnt - acnt) & 3) >> 1) & 1));
            if (record[measures_count + i] != other_record[measures_count + i]) {
                LOGERROR("Measurement record mismatch at history %lld", measures_count + i);
            }
        }

        LOGPASSED(2);

        measures_count += num_gates;
    }
        

}