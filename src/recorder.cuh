#pragma once

#include "options.hpp"
#include "definitions.cuh"
#include "memory.cuh"
#include "vector.cuh"
#include "vector.hpp"
#include "word.cuh"

namespace QuaSARQ {


    class MeasurementRecorder {

        DeviceAllocator& allocator;

        bool*       device;
        uint32*     device_map;
        Vec<bool>   host;
        size_t      step_gates;
        size_t      last_gates;
        bool        copied;

    public:

        MeasurementRecorder(DeviceAllocator& allocator) :
            allocator(allocator),
            device(nullptr),
            device_map(nullptr),
            step_gates(0),
            last_gates(0),
            copied(false)
        {}

        ~MeasurementRecorder() {
            destroy();
        }

        inline void destroy() noexcept {
            if (allocator.gpu_capacity() > 0) {
                allocator.deallocate<bool>(device);
                allocator.deallocate<uint32>(device_map);
            }
            device = nullptr;
            device_map = nullptr;
            host.clear(true);
            step_gates = 0;
            last_gates = 0;
            copied = false;
        }

        inline void reset_copied() { copied = false; }
        inline bool is_copied()  const { return copied; }
        inline size_t step_history() const { return step_gates; }
        inline size_t total_history()  const { return host.size(); }

        inline void alloc(const size_t& measures_count, const cudaStream_t& stream = 0) {
            LOGN2(1, "Allocating %lld MB for %lld measurements recording.. ",
                (int64)ratio(int64(measures_count * sizeof(bool)), MB), int64(measures_count));
            device = allocator.allocate<bool>(measures_count, Region::Stable);
            assert(device != nullptr);
            // A reused pool hands back the previous call's bytes. The reference run only
            // writes the measurements it executes, so any old non-zero byte here would
            // invert that measurement's reference bit for every shot.
            CHECK(cudaMemsetAsync(device, 0, measures_count * sizeof(bool), stream));
            device_map = allocator.allocate<uint32>(measures_count, Region::Stable);
            assert(device_map != nullptr);
            host.resize(measures_count);
            host.reset();
            step_gates = 0;
            last_gates = 0;
            LOGDONE(1, 4);
        }

        inline void copy_ordinals(const Vec<uint32, size_t>& ordinals, const cudaStream_t& stream = 0) {
            if (device_map == nullptr || ordinals.empty()) return;
            CHECK(cudaMemcpyAsync(device_map, ordinals.data(),
                ordinals.size() * sizeof(uint32), cudaMemcpyHostToDevice, stream));
        }

        inline void advance(const size_t& num_gates) {
            last_gates = num_gates;
            step_gates += num_gates;
        }

        inline void copy() {
            if (device != nullptr && step_gates > 0) {
                CHECK(cudaMemcpy(host.data(), device, step_gates, cudaMemcpyDeviceToHost));
                copied = true;
            }
        }

        inline void print() {
            if (!options.print_record) return;
            if (!last_gates) return;
            if (!options.sync) SYNCALL;
            LOGHEADER(1, 4, "Recorded measurements");
            copy();
            const size_t from = step_gates - last_gates;
            for (size_t i = from; i < step_gates; i++) {
                PRINT("%-2d", host[i]);
            }
            PRINT("\n");
            fflush(stdout);
            last_gates = 0;
        }

        inline bool* device_record() {
            if (device == nullptr) LOGERROR("recorder not allocated");
            return device;
        }

        inline const uint32* device_ordinals() const { return device_map; }

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
