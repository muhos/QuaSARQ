
#include "measurecheck.cuh"

namespace QuaSARQ {

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

    void MeasurementChecker::check_observables(
        const ObservableData& obs,
        const char*           bitstring,
        const uint32&         n)
    {
        if (!options.check_measurement) return;

        if (obs.empty()) return;

        if (record.empty()) {
            LOGERROR("measurements not recorded");
        }
        LOG2(1, "");
        LOGN2(1, " Checking observable bitstring.. ");
        bool all_passed = true;
        for (uint32 i = 0; i < n; i++) {
            const char expected = eval_instruction_cpu(
                obs.records.pinned.refs,
                obs.records.pinned.starts[i],
                obs.records.pinned.counts[i],
                record);
            if (bitstring[i] != expected) {
                LOGERRORN("Observable %u mismatch: GPU='%c', CPU='%c'", i, bitstring[i], expected);
                all_passed = false;
            }
        }
        if (all_passed) LOG2(1, "%sPASSED.%s", CGREEN, CNORMAL);
    }

    void MeasurementChecker::check_detectors(
        const DetectorData& det,
        const char*         bitstring,
        const uint32&       n)
    {
        if (!options.check_measurement) return;

        if (det.empty()) return;

        if (record.empty()) {
            LOGERROR("measurements not recorded");
        }
        LOG2(1, "");
        LOGN2(1, " Checking detector bitstring.. ");
        bool all_passed = true;
        for (uint32 i = 0; i < n; i++) {
            const char expected = eval_instruction_cpu(
                det.pinned.refs,
                det.pinned.starts[i],
                det.pinned.counts[i],
                record);
            if (bitstring[i] != expected) {
                LOGERRORN("Detector %u mismatch: GPU='%c', CPU='%c'", i, bitstring[i], expected);
                all_passed = false;
            }
        }
        if (all_passed) LOG2(1, "%sPASSED.%s", CGREEN, CNORMAL);
    }


}
