#pragma once

#include "options.hpp"
#include "tableau.cuh"
#include "definitions.cuh"

namespace QuaSARQ {

    struct Samples {
        Table *device;
        word_t *device_data;
        Table  host;

        Samples() : device(nullptr), device_data(nullptr) {}
        
        inline
        bool needs_host() {
            return options.print_sample        ||
                   options.print_sample_qubits;
        }

        ~Samples() {
            device = nullptr;
            device_data = nullptr;
            if (needs_host())
                host.destroy();
        }

        void alloc(const size_t& num_measurements, const size_t& num_words_minor, DeviceAllocator& gpu_allocator) {
            const size_t num_words_major = get_num_words(num_measurements);
            const size_t num_measures_padded = num_words_major * WORD_BITS;
            const size_t num_words = num_words_major * (num_words_minor * WORD_BITS);
            device = gpu_allocator.allocate<Table>(1, Region::Stable);
            device_data = gpu_allocator.allocate<word_t>(num_words, Region::Stable);
            Table tmp;
            tmp.alloc(device_data, num_measures_padded, num_words_major, num_words_minor);
            CHECK(cudaMemcpy(device, &tmp, sizeof(Table), cudaMemcpyHostToDevice));
            if (needs_host())
                host.alloc_host(num_measures_padded, num_words_major, num_words_minor);
        }

        void copy() {
            if (device_data != nullptr) {
                CHECK(cudaMemcpy(host.data(), device_data, host.size() * sizeof(word_t), cudaMemcpyDeviceToHost));
            }
        }
    };

}