#include "simulator.hpp"

namespace QuaSARQ {

    void DetectorData::alloc_pinned(DeviceAllocator& allocator) {
        if (pinned.is_allocated()) return;
        pinned.refs = allocator.allocate_pinned<uint32>(refs.size());
        pinned.starts = allocator.allocate_pinned<uint32>(starts.size());
        pinned.counts = allocator.allocate_pinned<uint32>(counts.size());
        moved_to_pinned = false;
    }

    void DetectorData::alloc_device(DeviceAllocator& allocator) {
        if (device.is_allocated()) return;
        device.refs = allocator.allocate<uint32>(refs.size());
        device.starts = allocator.allocate<uint32>(starts.size());
        device.counts = allocator.allocate<uint32>(counts.size());
    }

    void DetectorData::move_to_pinned() {
        if (!pinned.is_allocated()) {
            LOGERROR("pinned memory not allocated for record refs");
        }
        std::memcpy(pinned.refs, refs.data(), refs.size() * sizeof(uint32));
        std::memcpy(pinned.starts, starts.data(), starts.size() * sizeof(uint32));
        std::memcpy(pinned.counts, counts.data(), counts.size() * sizeof(uint32));
        pinned.num_instructions = starts.size();
        pinned.num_counts = counts.size();
        pinned.num_refs = refs.size();
        destroy();
        moved_to_pinned = true;
    }

    void DetectorData::copy_to_device(const cudaStream_t& stream) {
        if (!device.is_allocated()) {
            LOGERROR("device memory not allocated for record refs");
        }
        if (!pinned.is_allocated()) {
            LOGERROR("pinned memory not allocated for record refs");
        }
        if (!moved_to_pinned) {
            LOGERROR("record refs not moved to pinned memory");
        }
        CHECK(cudaMemcpyAsync(device.refs, pinned.refs, pinned.num_refs * sizeof(uint32), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(device.counts, pinned.counts, pinned.num_counts * sizeof(uint32), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(device.starts, pinned.starts, pinned.num_instructions * sizeof(uint32), cudaMemcpyHostToDevice, stream));
    }

    void ObservableData::alloc_pinned(DeviceAllocator& allocator) {
        if (records.pinned.is_allocated()) return;
        records.alloc_pinned(allocator);
        pinned.ids = allocator.allocate_pinned<uint32>(ids.size());
        moved_to_pinned = false;
    }

    void ObservableData::alloc_device(DeviceAllocator& allocator) {
        if (records.device.is_allocated()) return;
        records.alloc_device(allocator);
        device.ids = allocator.allocate<uint32>(ids.size());
    }

    void ObservableData::move_to_pinned() {
        if (!records.pinned.is_allocated()) {
            LOGERROR("pinned memory not allocated for observable records");
        }
        if (!pinned.ids) {
            LOGERROR("pinned memory not allocated for observable ids");
        }
        records.move_to_pinned();
        std::memcpy(pinned.ids, ids.data(), ids.size() * sizeof(uint32));
        pinned.num_observables = ids.size();
        destroy();
        moved_to_pinned = true;
    }

    void ObservableData::copy_to_device(const cudaStream_t& stream) {
        if (!device.ids) {
            LOGERROR("device memory not allocated for observable ids");
        }
        if (!records.device.is_allocated()) {
            LOGERROR("device memory not allocated for observable records");
        }
        if (!moved_to_pinned) {
            LOGERROR("observable data not moved to pinned memory");
        }
        records.copy_to_device(stream);
        CHECK(cudaMemcpyAsync(device.ids, pinned.ids, pinned.num_observables * sizeof(uint32), cudaMemcpyHostToDevice, stream));
    }

    void Simulator::alloc_detectors() {
        if (!options.print_detector) return;
        circuit_io.detectors.alloc_pinned(gpu_allocator);
        circuit_io.detectors.alloc_device(gpu_allocator);
        circuit_io.detectors.move_to_pinned();
    }

    void Simulator::alloc_observables() {
        if (!options.print_observable) return;
        circuit_io.observables.alloc_pinned(gpu_allocator);
        circuit_io.observables.alloc_device(gpu_allocator);
        circuit_io.observables.move_to_pinned();
    }

    void Simulator::copy_detectors(const cudaStream_t& stream) {
        if (!options.print_detector) return;
        circuit_io.detectors.copy_to_device(stream);
    }

    void Simulator::copy_observables(const cudaStream_t& stream) {
        if (!options.print_observable) return;
        circuit_io.observables.copy_to_device(stream);
    }
        

}