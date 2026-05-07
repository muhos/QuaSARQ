#pragma once

#include "options.hpp"
#include "definitions.cuh"
#include "memory.cuh"
#include "vector.cuh"
#include "vector.hpp"

namespace QuaSARQ {

    class MeasurementRecorder {

        DeviceAllocator& allocator;

        bool* device;
        Vec<bool> host;

        bool copied;

    public:

        MeasurementRecorder(DeviceAllocator& allocator) :
            allocator(allocator),
            device(nullptr),
            copied(false)
        {}

        ~MeasurementRecorder() {
            device = nullptr;
            host.clear(true);
        }

        inline void reset_copied() {
            copied = false;
        }

        inline bool is_copied() const {
            return copied;
        }

        inline void alloc(const size_t& num_qubits) {
            const size_t num_qubits_padded = get_num_padded_bits(num_qubits);
            device = allocator.allocate<bool>(num_qubits_padded, Region::Stable);
            host.resize(num_qubits_padded);
        }

        inline void copy() {
            if (device != nullptr) {
                CHECK(cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost));
                copied = true;
            }
        }

        inline void print(const size_t& num_qubits) {
            if (!options.print_record) return;
            if (!options.sync) SYNCALL;
            LOGHEADER(1, 4, "Recorded measurements");
            copy();
            for (size_t q = 0; q < num_qubits; q++) {
                PRINT("%-2d", host[q]);
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