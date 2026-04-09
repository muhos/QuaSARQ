#pragma once

#include "options.hpp"
#include "tableau.cuh"
#include "definitions.cuh"

namespace QuaSARQ {

    struct Recorder {
        Table *device;
        word_t *device_data;
        Table  host;

        Recorder() : device(nullptr), device_data(nullptr) {}
        ~Recorder() {
            device = nullptr;
            device_data = nullptr;
            if (options.print_record) {
                host.destroy();
            }
        }

        void alloc(const Tableau& tableau, DeviceAllocator& gpu_allocator) {
            device = gpu_allocator.allocate<Table>(1);
            device_data = gpu_allocator.allocate<word_t>(tableau.num_words_per_table());
            Table tmp;
            tmp.alloc(device_data, tableau.num_qubits_padded(), tableau.num_words_major(), tableau.num_words_minor());
            CHECK(cudaMemcpyAsync(device, &tmp, sizeof(Table), cudaMemcpyHostToDevice));
            if (options.print_record)
                host.alloc_host(tableau.num_qubits_padded(), tableau.num_words_major(), tableau.num_words_minor());
        }

        void copy() {
            if (device_data != nullptr) {
                CHECK(cudaMemcpy(host.data(), device_data, host.size(), cudaMemcpyDeviceToHost));
            }
        }
    };

}