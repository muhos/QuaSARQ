#pragma once

#include "vector.hpp"
#include "memory.cuh"

namespace QuaSARQ {

    struct RecordRefs {

        Vec<uint32, uint32>  refs; // measurement-history indices per instruction.
        Vec<uint32, uint32>  starts;  // the start index in record_refs for each instruction.
        Vec<uint32, uint32>  counts;  // the number of record_refs for each instruction.

        struct RawArrays {
            uint32* refs;
            uint32* starts;
            uint32* counts;

            size_t num_instructions;
            size_t num_counts;
            size_t num_refs;

            RawArrays() : 
                refs(nullptr), starts(nullptr), counts(nullptr),
                num_instructions(0), num_counts(0), num_refs(0) { }

            bool is_allocated() const {
                return !(refs == nullptr || starts == nullptr || counts == nullptr);
            }
        };

        RawArrays pinned, device;

        bool moved_to_pinned;

        RecordRefs() : pinned(), device(), moved_to_pinned(false) {}

        void init() {
            moved_to_pinned = false;
            refs.reserve(64);
            starts.reserve(16);
            counts.reserve(16);
        }

        void destroy() {
            refs.clear(true);
            starts.clear(true);
            counts.clear(true);
        }

        size_t bytes() const {
            return refs.size() * sizeof(uint32) + 
                   starts.size() * sizeof(uint32) + 
                   counts.size() * sizeof(uint32);
        }

        void alloc_pinned(DeviceAllocator& allocator);
        void alloc_device(DeviceAllocator& allocator);

        void move_to_pinned();
        void copy_to_device(const cudaStream_t& stream);

        bool empty() const { return !pinned.num_instructions; }
    };

    typedef RecordRefs DetectorData;

    struct ObservableData {

        RecordRefs records;
        Vec<uint32, uint32>  ids; // the observable id (the k in OBSERVABLE_INCLUDE(k))

        struct RawIds {
            uint32* ids;
            size_t num_observables;

            RawIds() : ids(nullptr), num_observables(0) {}
        };

        RawIds pinned, device;

        bool moved_to_pinned;

        ObservableData() : records(), pinned(), device(), moved_to_pinned(false) {}

        void init() {
            records.init();
            ids.reserve(4);
        }

        void destroy() {
            records.destroy();
            ids.clear(true);
        }

        size_t bytes() const {
            return records.bytes() +
                   ids.size() * sizeof(uint32);
        }

        void alloc_pinned(DeviceAllocator& allocator);
        void alloc_device(DeviceAllocator& allocator);
        void move_to_pinned();
        void copy_to_device(const cudaStream_t& stream);

        // Collapse the per-OBSERVABLE_INCLUDE entries into one entry per observable id.
        // The observable is the XOR of every entry that names its id.
        void merge_by_id() {
            if (!ids.size()) return;
            uint32 max_id = 0;
            for (uint32 i = 0; i < ids.size(); i++) {
                if (ids[i] > max_id) max_id = ids[i];
            }
            const uint32 num_ids = max_id + 1;
            bool grouped = (num_ids == ids.size());
            for (uint32 i = 0; grouped && i < ids.size(); i++)
                grouped = (ids[i] == i);
            if (grouped) return;

            Vec<uint32, uint32> merged_refs, merged_starts, merged_counts;
            for (uint32 k = 0; k < num_ids; k++) {
                merged_starts.push(merged_refs.size());
                uint32 count = 0;
                for (uint32 i = 0; i < ids.size(); i++) {
                    if (ids[i] != k) continue;
                    const uint32 start = records.starts[i];
                    const uint32 n = records.counts[i];
                    for (uint32 j = start; j < start + n; j++)
                        merged_refs.push(records.refs[j]);
                    count += n;
                }
                merged_counts.push(count);
            }

            records.refs.resize(merged_refs.size());
            for (uint32 i = 0; i < merged_refs.size(); i++)
                records.refs[i] = merged_refs[i];
            records.starts.resize(num_ids);
            records.counts.resize(num_ids);
            ids.resize(num_ids);
            for (uint32 k = 0; k < num_ids; k++) {
                records.starts[k] = merged_starts[k];
                records.counts[k] = merged_counts[k];
                ids[k] = k;
            }
            pinned.num_observables = ids.size();
            records.pinned.num_instructions = records.starts.size();
            records.pinned.num_counts = records.counts.size();
            records.pinned.num_refs = records.refs.size();
        }

        bool empty() const { return !pinned.num_observables; }
    };

}