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

    public:

        MeasurementRecorder(DeviceAllocator& allocator) :
            allocator(allocator),
            device(nullptr)
        {}

        ~MeasurementRecorder() {
            device = nullptr;
            host.clear(true);
        }

        inline void alloc(const size_t& num_qubits) {
            const size_t num_qubits_padded = get_num_padded_bits(num_qubits);
            device = allocator.allocate<bool>(num_qubits_padded, Region::Stable);
            host.resize(num_qubits_padded);
        }

        inline void copy() {
            if (device != nullptr) {
                CHECK(cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost));
            }
        }

        inline void print(const size_t& num_qubits = 0) {
            if (!options.print_record) return;
            if (!options.sync) SYNCALL;
            LOGHEADER(1, 4, "Recorded measurements");
            copy();
            const size_t size = !num_qubits ? host.size() : num_qubits;
            for (size_t q = 0; q < size; q++) {
                PRINT("%-2d\n", host[q]);
            }
            fflush(stdout);
        }

        inline bool* device_record() {
            if (device == nullptr) LOGERROR("recorder not allocated");
            return device;
        }
    };

}