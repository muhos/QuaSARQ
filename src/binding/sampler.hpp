#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <random>
#include <chrono>
#include <cstring>
#include <memory>

#include "frame.cuh"
#include "logging.hpp"
#include "version.hpp"
#include "options.hpp"
#include "definitions.hpp"

#define CONFIG2CLEARPRINT(CONFIG, MSG) \
    options.print_ ## CONFIG = false;

#define CONFIG2CLEARCHECK(CONFIG) \
    options.check_ ## CONFIG = false;

namespace QuaSARQ {

    inline bool         options_ready = false;
    inline std::string  last_error;

    inline uint64_t splitmix64(uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    inline void strip_escapes(std::string& target, const char* text) {
        for (const char* p = text; *p != '\0'; p++) {
            if (*p == '\x1b') {
                while (*p != '\0' && *p != 'm') p++;
                if (*p == '\0') return;
                continue;
            }
            target += *p;
        }
    }

    inline void library_sink(int stream, const char* text, void*) {
        if (stream == LOG_STREAM_ERR) {
            strip_escapes(last_error, text);
            return;
        }
        if (options.verbose > 0)
            std::fputs(text, stdout);
    }

    inline void apply_library_defaults() {
        options.report_en = false;
        options.force_report_en = false;
        options.progress_en = false;
        options.sync = false;
        options.profile = false;
        options.equivalence_en = false;
        options.tuner_en = false;
        options.check_all = false;
        options.write_rc = 0;
        options.timeout = 0;
        options.quiet_en = options.verbose <= 0;
        FOREACH_PRINT(CONFIG2CLEARPRINT);
        FOREACH_CHECK(CONFIG2CLEARCHECK);
    }

    inline void binding_set_verbosity(const int& level) {
        options.verbose = level;
        options.quiet_en = level <= 0;
        options.progress_en = false;
        SET_LOGGER_VERBOSITY(level);
    }

    inline void binding_clear_error() { last_error.clear(); }

    inline int binding_get_verbosity() { return options.verbose; }

    inline void binding_set_chunk_shots(const int& shots) {
        if (shots < 0)
            throw std::invalid_argument("chunk shots cannot be negative");
        options.chunk_shots = shots;
    }

    inline int binding_get_chunk_shots() { return options.chunk_shots; }

    inline std::string binding_version() { return std::string(version()); }

    inline size_t binding_stride_of(const size_t& units, const bool& bit_packed) { return FrameResults::stride_of(units, bit_packed); }

    inline std::string binding_device_name() {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
            return std::string();
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
            return std::string();
        return std::string(prop.name);
    }

    inline uint64_t binding_random_seed() {
        std::random_device device;
        const uint64_t entropy = (uint64_t(device()) << 32) ^ uint64_t(device());
        const uint64_t clock = uint64_t(
            std::chrono::steady_clock::now().time_since_epoch().count());
        return splitmix64(entropy ^ clock);
    }

    inline std::string binding_last_error() {
        while (!last_error.empty() && (last_error.back() == '\n' || last_error.back() == ' '))
            last_error.pop_back();
        return last_error;
    }

    void binding_initialize(const std::string& config_path);

    struct SampleRequest {
        size_t      num_shots;
        bool        bit_packed;
        uint8_t*    detectors;
        uint8_t*    observables;
        uint8_t*    measurements;
        size_t      detectors_stride;
        size_t      observables_stride;
        size_t      measurements_stride;

        SampleRequest() :
            num_shots(0)
            , bit_packed(false)
            , detectors(nullptr)
            , observables(nullptr)
            , measurements(nullptr)
            , detectors_stride(0)
            , observables_stride(0)
            , measurements_stride(0) {}
    };

    class Sampling {

        std::string              circuit_text;
        uint64_t                 base_seed;
        uint64_t                 call_index;
        size_t                   num_detectors;
        size_t                   num_observables;
        size_t                   num_measurements;
        std::unique_ptr<Framing> framing;

    public:

        Sampling(const std::string& circuit, const uint64_t& seed);
        ~Sampling();

        Sampling            (const Sampling&) = delete;
        Sampling& operator= (const Sampling&) = delete;

        size_t detectors() const { return num_detectors; }
        size_t observables() const { return num_observables; }
        size_t measurements() const { return num_measurements; }
        bool holds_device_memory() const { return framing != nullptr; }

        void run(const SampleRequest& request);
        void release();
    };

}

#undef CONFIG2CLEARPRINT
#undef CONFIG2CLEARCHECK
