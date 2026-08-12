#pragma once

// nanobind over pybind11: nb::ndarray is DLPack-backed, so a device (GPU) array can be
// handed to numpy/cupy/torch directly with no copy to the host. pybind11's py::array_t is
// numpy- and host-only and cannot express that.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include <sys/stat.h>

#include "sampler.hpp"

namespace nb = nanobind;

namespace Module{

    inline std::mutex sampling_mutex;

    // Output buffers are handed to numpy, which owns them until Python drops the array. A
    // fresh allocation costs a page fault per page on first write, ~10 ms for the 32 MB an
    // unpacked d=15 batch needs. Recycling a released block instead costs one memset.
    // This class tracks down all live allocations and recycles them when possible.
    class Cacher {

        static constexpr size_t MAX_SLOTS  = 4;
        static constexpr size_t MAX_POOLED = size_t(512) << 20;

        std::mutex                                  lock;
        std::unordered_map<void*, size_t>           live;
        std::vector<std::pair<uint8_t*, size_t>>    idle;
        size_t                                      idle_bytes = 0;

    public:
        
        static Cacher& getCacher() {
            static Cacher* pool = new Cacher();
            return *pool;
        }

        uint8_t* alloc(const size_t& bytes) {
            const size_t needed = bytes ? bytes : 1;
            std::lock_guard<std::mutex> guard(lock);
            for (size_t i = 0; i < idle.size(); i++) {
                if (idle[i].second != needed) continue;
                uint8_t* reused = idle[i].first;
                idle_bytes -= needed;
                idle.erase(idle.begin() + i);
                std::memset(reused, 0, needed);
                live.emplace(reused, needed);
                return reused;
            }
            uint8_t* fresh = new uint8_t[needed]();
            live.emplace(fresh, needed);
            return fresh;
        }

        void release(void* p) noexcept {
            if (p == nullptr) return;
            uint8_t* block = static_cast<uint8_t*>(p);
            size_t bytes = 0;
            {
                std::lock_guard<std::mutex> guard(lock);
                const auto found = live.find(p);
                if (found == live.end()) return;
                bytes = found->second;
                live.erase(found);
                if (idle.size() < MAX_SLOTS && idle_bytes + bytes <= MAX_POOLED) {
                    idle.emplace_back(block, bytes);
                    idle_bytes += bytes;
                    return;
                }
            }
            delete[] block;
        }
    };

    struct HostBuffer {

        uint8_t* data;
        size_t   rows;
        size_t   cols;

        HostBuffer(const size_t& rows, const size_t& cols) : data(nullptr), rows(rows), cols(cols) {
            data = Cacher::getCacher().alloc(rows * cols);
        }
    };

    inline void discard(HostBuffer& buffer) {
        Cacher::getCacher().release(buffer.data);
        buffer.data = nullptr;
    }

    inline void raise_from_core(const char* fallback) {
        const std::string message = QuaSARQ::binding_last_error();
        throw std::runtime_error(message.empty() ? std::string(fallback) : message);
    }

    inline std::string circuit_to_text(nb::handle circuit) {
        if (nb::isinstance<nb::str>(circuit))
            return nb::cast<std::string>(circuit);
        return nb::cast<std::string>(nb::str(circuit));
    }

    inline nb::object to_numpy(HostBuffer& buffer, const bool& bit_packed) {
        uint8_t* data = buffer.data;
        buffer.data = nullptr;
        // Capsule now holds the pointer and a function to run 
        // when the pointer is not needed anymore.
        // This acts like a smart pointer to track the pointer 
        // and free it when all references to capsule are deleted.
        nb::capsule owner(data, 
            [](void* p) noexcept { Cacher::getCacher().release(p); }
        );
        if (bit_packed) {
            return nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(
                data, { buffer.rows, buffer.cols }, owner));
        }
        return nb::cast(nb::ndarray<nb::numpy, bool, nb::ndim<2>>(
            reinterpret_cast<bool*>(data), { buffer.rows, buffer.cols }, owner));
    }

    class Engine {

    protected:

        std::unique_ptr<QuaSARQ::Sampling> engine;

        void execute(const QuaSARQ::SampleRequest& request, std::initializer_list<HostBuffer*> outputs) {
            QuaSARQ::binding_clear_error();
            std::lock_guard<std::mutex> lock(sampling_mutex);
            bool failed = false;
            {
                nb::gil_scoped_release release;
                try {
                    engine->run(request);
                }
                catch (...) {
                    failed = true;
                }
            }
            if (failed) {
                for (HostBuffer* buffer : outputs)
                    discard(*buffer);
                raise_from_core("sampling failed");
            }
        }

    public:

        Engine(nb::handle circuit, const uint64_t& seed) {
            const std::string text = circuit_to_text(circuit);
            QuaSARQ::binding_clear_error();
            try {
                engine.reset(new QuaSARQ::Sampling(text, seed));
            }
            catch (const std::exception&) {
                raise_from_core("failed to parse the circuit");
            }
            catch (...) {
                raise_from_core("failed to parse the circuit");
            }
        }

        bool holds_device_memory() const { return engine->holds_device_memory(); }
        void release() { engine->release(); }
    };

    class DetectorSampler : public Engine {

    public:

        using Engine::Engine;

        size_t num_detectors() const { return engine->detectors(); }
        size_t num_observables() const { return engine->observables(); }

        nb::object sample(const size_t& shots,
                          const bool& separate_observables,
                          const bool& bit_packed,
                          const bool& append_observables,
                          const bool& prepend_observables);
    };

    class MeasurementSampler : public Engine {

    public:

        using Engine::Engine;

        size_t num_measurements() const { return engine->measurements(); }

        nb::object sample(const size_t& shots, const bool& bit_packed);
    };

    std::string locate_kernel_config();
}
