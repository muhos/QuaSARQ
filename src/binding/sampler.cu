
#include "sampler.hpp"

namespace QuaSARQ {

    void binding_initialize(const std::string& config_path) {
        if (!options_ready) {
            options.initialize();
            options_ready = true;
        }
        options.verbose = 0;
        apply_library_defaults();
        if (!config_path.empty()) {
            const size_t limit = OPTION_PATH_LEN - 1;
            const size_t length = config_path.size() < limit ? config_path.size() : limit;
            std::memset(options.configpath, 0, OPTION_PATH_LEN);
            std::memcpy(options.configpath, config_path.c_str(), length);
        }
        SET_LOGGER_VERBOSITY(0);
        SET_LOGGER_SINK(library_sink, nullptr);
    }

    // reserve() sizes the GPU pool to almost all free device memory, so at most one
    // Sampling may hold a live Framing. A second one would find nothing left.
    static Sampling* pool_owner = nullptr;

    Sampling::Sampling(const std::string& circuit, const uint64_t& seed) :
        circuit_text(circuit)
        , base_seed(seed)
        , call_index(0)
        , num_detectors(0)
        , num_observables(0)
    {
        if (circuit_text.empty())
            LOGERROR("circuit is empty.");
        OptionsGuard guard;
        CircuitIO io;
        char* str = io.read(circuit_text.data(), circuit_text.size());
        while (str < io.eof) {
            eatWS(str);
            if (*str == '\0') break;
            if (*str == '#') {
                eatLine(str);
                continue;
            }
            io.read_gate(str);
        }
        io.observables.merge_by_id();
        num_detectors = io.detectors.pinned.num_instructions;
        num_observables = io.observables.pinned.num_observables;
    }

    Sampling::~Sampling() {
        release();
    }

    void Sampling::release() {
        framing.reset();
        if (pool_owner == this)
            pool_owner = nullptr;
    }

    void Sampling::run(const SampleRequest& request) {
        if (!request.num_shots) return;
        OptionsGuard guard;
        apply_library_defaults();
        options.num_shots = int(request.num_shots);
        options.seed = size_t(splitmix64(base_seed ^ splitmix64(call_index)));
        call_index++;
        FrameResults results;
        results.bit_packed = request.bit_packed;
        results.detectors = num_detectors ? request.detectors : nullptr;
        results.observables = num_observables ? request.observables : nullptr;
        results.detectors_stride = request.detectors_stride;
        results.observables_stride = request.observables_stride;
        if (pool_owner != nullptr && pool_owner != this)
            pool_owner->release();
        // Shot count does not size the pool -- reserve() takes nearly all free device memory
        // and choose_chunk_shots() re-derives chunking per call -- so one Framing serves any
        // count. Rebuilding when the request grows would pay the whole pool cost every call.
        if (framing == nullptr) {
            framing.reset(new Framing(circuit_text.data(), circuit_text.size(),
                                      request.num_shots, num_detectors > 0));
        }
        pool_owner = this;
        framing->set_shots(request.num_shots);
        framing->set_results(&results);
        try {
            framing->sample();
        }
        catch (...) {
            framing->set_results(nullptr);
            release();
            throw;
        }
        framing->set_results(nullptr);
    }
}
