
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

    bool MeasurementChecker::check_observables(
        const ObservableData& obs,
        const char*           bitstring,
        const uint32&         n,
        const bool&           skip_reporting_passed)
    {
        if (!options.check_measurement) return false;

        if (obs.empty()) return false;

        if (record.empty()) {
            LOGERROR("measurements not recorded");
        }
        if (!skip_reporting_passed) {
            LOG2(1, "");
            LOGN2(1, " Checking observable bitstring.. ");
        }
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
        if (all_passed && !skip_reporting_passed) LOGPASSED(1);
        return all_passed;
    }

    bool MeasurementChecker::check_detectors(
        const DetectorData& det,
        const char*         bitstring,
        const uint32&       n,
        const bool&         skip_reporting_passed)
    {
        if (!options.check_measurement) return false;

        if (det.empty()) return false;

        if (record.empty()) {
            LOGERROR("measurements not recorded");
        }
        if (!skip_reporting_passed) {
            LOG2(1, "");
            LOGN2(1, " Checking detector bitstring.. ");
        }
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
        if (all_passed && !skip_reporting_passed) LOGPASSED(1);
        return all_passed;
    }


}
