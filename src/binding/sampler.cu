
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

    CircuitScan scan_circuit(std::string& circuit_text) {
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
        CircuitScan scan;
        scan.qubits = io.max_qubits;
        scan.measurements = io.measures_count;
        scan.detectors = io.detectors.pinned.num_instructions;
        scan.observables = io.observables.pinned.num_observables;
        return scan;
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
        , num_measurements(0)
        , num_qubits(0)
        , built_cap(0)
    {
        const CircuitScan scan = scan_circuit(circuit_text);
        num_detectors = scan.detectors;
        num_observables = scan.observables;
        num_measurements = scan.measurements;
        num_qubits = scan.qubits;
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
        results.measurements = num_measurements ? request.measurements : nullptr;
        results.detectors_stride = request.detectors_stride;
        results.observables_stride = request.observables_stride;
        results.measurements_stride = request.measurements_stride;

        if (auto_device_memory) {
            const int cap = binding_auto_device_memory(num_qubits, num_measurements, request.num_shots);
            if (framing != nullptr && cap > built_cap)
                release();
            options.max_gpu_memory = cap;
            built_cap = cap;
        }
        if (pool_owner != nullptr && pool_owner != this)
            pool_owner->release();

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
