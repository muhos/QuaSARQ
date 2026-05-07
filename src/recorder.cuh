#pragma once

#include "options.hpp"
#include "definitions.cuh"
#include "memory.cuh"
#include "vector.cuh"
#include "vector.hpp"

namespace QuaSARQ {

    class MeasurementRecorder {

        DeviceAllocator& allocator;

        bool*       device;
        Vec<bool>   host;
        size_t      step_gates;
        bool        copied;

    public:

        MeasurementRecorder(DeviceAllocator& allocator) :
            allocator(allocator),
            device(nullptr),
            step_gates(0),
            copied(false)
        {}

        ~MeasurementRecorder() {
            device = nullptr;
            host.clear(true);
        }

        inline void reset_copied() { copied = false; }
        inline bool is_copied()  const { return copied; }
        inline size_t step_history() const { return step_gates; }
        inline size_t total_history()  const { return host.size(); }

        inline void alloc(const size_t& measures_count) {
            device = allocator.allocate<bool>(measures_count, Region::Stable);
            assert(device != nullptr);
            host.resize(measures_count);
            step_gates = 0;
        }

        inline void advance(const size_t& num_gates) { step_gates += num_gates; }

        inline void copy() {
            if (device != nullptr && step_gates > 0) {
                CHECK(cudaMemcpy(host.data(), device, step_gates, cudaMemcpyDeviceToHost));
                copied = true;
            }
        }

        inline void print(const size_t& num_gates) {
            if (!options.print_record) return;
            if (!options.sync) SYNCALL;
            LOGHEADER(1, 4, "Recorded measurements");
            copy();
            const size_t from = step_gates - num_gates;
            for (size_t i = from; i < step_gates; i++) {
                PRINT("%-2d", host[i]);
            }
            PRINT("\n");
            fflush(stdout);
        }

        inline bool* device_record() {
            if (device == nullptr) LOGERROR("recorder not allocated");
            return device;
        }

        inline Vec<bool>& host_record() {
            if (host.empty()) LOGERROR("recorder not allocated");
            if (!copied) LOGERROR("record not copied to host");
            return host;
        }

        inline
        const Vec<bool>& host_record() const {
            if (host.empty()) LOGERROR("recorder not allocated");
            if (!copied) LOGERROR("record not copied to host");
            return host;
        }
    };

}