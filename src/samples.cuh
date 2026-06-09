#pragma once

#include "options.hpp"
#include "tableau.cuh"
#include "definitions.cuh"

namespace QuaSARQ {

    struct Samples {
        Table *device;
        word_t *device_data;
        Table  host;
        size_t num_words;

        Samples() : device(nullptr), device_data(nullptr), num_words(0) {}
        
        inline
        bool needs_host() const {
            return options.print_sample        ||
                   options.print_sample_qubits ||
                   options.check_measurement;
        }

        ~Samples() {
            device = nullptr;
            device_data = nullptr;
            num_words = 0;
            if (needs_host())
                host.destroy();
        }

        void alloc(const size_t& num_measurements, const size_t& num_words_minor, DeviceAllocator& gpu_allocator) {
            const size_t num_words_major = get_num_words(num_measurements);
            const size_t num_measures_padded = num_words_major * WORD_BITS;
            num_words = num_words_major * (num_words_minor * WORD_BITS);
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

        size_t device_bytes() const {
            return device_data != nullptr ? num_words * sizeof(word_t) : 0;
        }

        size_t host_bytes() const {
            return needs_host() ? host.size() * sizeof(word_t) : 0;
        }
    };

}
