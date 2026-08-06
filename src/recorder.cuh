#pragma once

#include "options.hpp"
#include "definitions.cuh"
#include "memory.cuh"
#include "vector.cuh"
#include "vector.hpp"
#include "word.cuh"

namespace QuaSARQ {

    // Per-measurement capacity of the precomputed S(q) lists. |S(q)| is 4 at most and 1.81 on
    // average on surface codes; a measurement that exceeds this falls back to scanning its own
    // column in record_signs_k, so the bound is a heuristic and never a correctness one.
    #define RECORD_MAX_SELECTED 8

    class RecordSelector {

        DeviceAllocator& allocator;

        uint32*     counts;
        uint32*     lists;
        word_std_t* masks;

        size_t      num_qubits;
        size_t      num_words_minor;

    public:

        RecordSelector(DeviceAllocator& allocator) :
            allocator(allocator),
            counts(nullptr),
            lists(nullptr),
            masks(nullptr),
            num_qubits(0),
            num_words_minor(0)
        {}

        ~RecordSelector() {
            destroy();
        }

        inline void destroy() noexcept {
            if (allocator.gpu_capacity() > 0) {
                allocator.deallocate<uint32>(counts);
                allocator.deallocate<uint32>(lists);
                allocator.deallocate<word_std_t>(masks);
            }
            counts = nullptr;
            lists = nullptr;
            masks = nullptr;
            num_qubits = 0;
            num_words_minor = 0;
        }

        inline void alloc(const size_t& num_qubits, const size_t& num_words_minor) {
            destroy();
            this->num_qubits = num_qubits;
            this->num_words_minor = num_words_minor;
            counts = allocator.allocate<uint32>(num_qubits, Region::Stable);
            lists  = allocator.allocate<uint32>(num_qubits * RECORD_MAX_SELECTED, Region::Stable);
            masks  = allocator.allocate<word_std_t>(3 * num_words_minor, Region::Stable);
        }

        inline bool allocated() const { return counts != nullptr && lists != nullptr && masks != nullptr; }
        inline uint32* device_counts() { return counts; }
        inline uint32* device_lists() { return lists; }
        inline word_std_t* device_masks() { return masks; }
        inline size_t mask_bytes() const { return 3 * num_words_minor * sizeof(word_std_t); }
    };

    class MeasurementRecorder {

        DeviceAllocator& allocator;

        bool*       device;
        Vec<bool>   host;
        size_t      step_gates;
        size_t      last_gates;
        bool        copied;

    public:

        MeasurementRecorder(DeviceAllocator& allocator) :
            allocator(allocator),
            device(nullptr),
            step_gates(0),
            last_gates(0),
            copied(false)
        {}

        ~MeasurementRecorder() {
            destroy();
        }

        inline void destroy() noexcept {
            if (allocator.gpu_capacity() > 0)
                allocator.deallocate<bool>(device);
            device = nullptr;
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
            host.resize(measures_count);
            host.reset();
            step_gates = 0;
            last_gates = 0;
            LOGDONE(1, 4);
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
