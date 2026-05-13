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

        bool empty() const { return !pinned.num_observables; }
    };

}