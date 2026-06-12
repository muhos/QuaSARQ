#include "../src/frame.cuh"
#include "helper.hpp"

#include <cmath>
#include <fstream>

using namespace QuaSARQ;

constexpr size_t SAMPLE_SHOTS = 1024;
constexpr double LOGICAL_RATE_TOLERANCE = 0.12;

struct LogicalRateBaseline {
    const char* circuit;
    double rate;
};

constexpr LogicalRateBaseline LOGICAL_RATE_BASELINES[] = {
    {"surface_code_d10_r10.stim", 0.40823700},
    {"surface_code_d10_r3.stim",  0.23896900},
    {"surface_code_d20_r10.stim", 0.48462500},
    {"surface_code_d20_r3.stim",  0.36828200},
    {"surface_code_d30_r10.stim", 0.49802300},
    {"surface_code_d30_r3.stim",  0.43301200},
    {"surface_code_d40_r10.stim", 0.49985500},
    {"surface_code_d40_r3.stim",  0.46659600},
    {"surface_code_d50_r10.stim", 0.50056400},
    {"surface_code_d50_r3.stim",  0.48294200},
    {"surface_code_d60_r10.stim", 0.49990200},
    {"surface_code_d60_r3.stim",  0.49188400},
};

const LogicalRateBaseline* find_logical_rate_baseline(const std::string& filename) {
    for (const LogicalRateBaseline& baseline : LOGICAL_RATE_BASELINES) {
        if (filename == baseline.circuit)
            return &baseline;
    }
    return nullptr;
}

class SamplingHarness : public Framing {

public:

    using Framing::Framing;

    const Statistics& statistics() const {
        return stats;
    }

    const bool& has_measurements() const {
        return measuring;
    }

    const bool& is_reference_run() const {
        return reference_mode;
    }

};

void reset_sampling_options(const char* circuit_path) {
    options.initialize();
    options.verbose = 0;
    options.quiet_en = true;
    options.report_en = false;
    options.progress_en = false;
    options.force_report_en = false;
    options.check_scheduler = false;
    options.check_tableau = false;
    options.check_measurement = false;
    options.print_observable = false;
    options.print_detector = false;
    options.print_sample = false;
    options.print_sample_qubits = false;
    options.print_finalstate = false;
    options.print_finaltableau = false;
    options.num_shots = SAMPLE_SHOTS;
    options.seed = 123;
    options.min_shots_write = 0;
    options.min_measures_write = 0;
    options.streams = 6;
    options.tuner_en = false;
    options.ignore_ticks = false;
    SET_LOGGER_VERBOSITY(options.verbose);
    copy_test_path(options.configpath, kernel_config_path());
    copy_test_path(options.statepath, test_build_path() / "test_sampling.qstate");
    options.check(circuit_path);
}

void check_sampled_circuit(SamplingHarness& framing) {
    framing.sample();
    const Statistics& stats = framing.statistics();
    TCHECK(framing.has_measurements());
    TCHECK(stats.circuit.measure_stats.count > 0);
    TCHECK(stats.tableau.count == 1);
    TCHECK(stats.tableau.istates == 1);
    TCHECK(stats.tableau.gigabytes > 0.0);
    TCHECK(stats.time.simulation >= 0.0);
    TCHECK(!framing.is_reference_run());
}

template<typename Configure>
void run_for_all_circuits(const char* suffix, Configure configure) {
    const auto paths = circuit_paths_up_to_distance(50);
    TCHECK(!paths.empty());
    for (const std::string& path : paths) {
        const std::string name = path + " " + suffix;
        run_test(name.c_str(), [&] {
            reset_sampling_options(path.c_str());
            configure();
            SamplingHarness framing(path, SAMPLE_SHOTS);
            check_sampled_circuit(framing);
        });
    }
}

void test_sampling_paths() {
    section("Sampling surface-code circuits");

    run_for_all_circuits("check samples only", [] {
        options.check_measurement = true;
        options.print_sample = false;
        options.print_sample_qubits = false;
        options.print_detector = false;
        options.print_observable = false;
    });

    run_for_all_circuits("prints shot-per-line samples", [] {
        options.check_measurement = true;
        options.print_sample = true;
    });

    run_for_all_circuits("prints measurement-per-line samples", [] {
        options.check_measurement = true;
        options.print_sample = false;
        options.print_sample_qubits = true;
    });

    run_for_all_circuits("prints sampled detectors", [] {
        options.check_measurement = true;
        options.print_sample = false;
        options.print_detector = true;
    });

    run_for_all_circuits("prints sampled observables", [] {
        options.check_measurement = true;
        options.print_sample = false;
        options.print_observable = true;
    });

    run_for_all_circuits("prints all sampled outputs", [] {
        options.check_measurement = true;
        options.print_sample = true;
        options.print_sample_qubits = true;
        options.print_detector = true;
        options.print_observable = true;
    });
}

std::string read_observable_file(const std::string& circuit_path) {
    std::filesystem::path path(circuit_path);
    path.replace_extension();
    path += "_obs.01";
    std::ifstream in(path, std::ios::binary);
    TCHECK(in.good());
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct SamplingOutcome {
    size_t total_observable_errors;
    size_t shots_with_error;
    std::string observables;
};

SamplingOutcome run_chunked_sampling(const std::string& path, const int& chunk_shots) {
    reset_sampling_options(path.c_str());
    options.print_observable = true;
    options.chunk_shots = chunk_shots;
    SamplingHarness framing(path, SAMPLE_SHOTS);
    check_sampled_circuit(framing);
    const Statistics& stats = framing.statistics();
    TCHECK(stats.logical.total_shots == SAMPLE_SHOTS);
    return SamplingOutcome {
        stats.logical.total_observable_errors,
        stats.logical.shots_with_error,
        read_observable_file(path)
    };
}

void test_chunked_sampling() {
    section("Chunked sampling consistency");

    const auto paths = circuit_paths_up_to_distance(100);
    TCHECK(!paths.empty());
    for (const std::string& path : paths) {
        run_test((path + " chunked sampling matches single chunk").c_str(), [&] {
            const SamplingOutcome baseline = run_chunked_sampling(path, int(SAMPLE_SHOTS));
            TCHECK(!baseline.observables.empty());
            for (const int chunk_shots : {int(SAMPLE_SHOTS) / 4, int(SAMPLE_SHOTS) / 8}) {
                const SamplingOutcome chunked = run_chunked_sampling(path, chunk_shots);
                TCHECK(chunked.total_observable_errors == baseline.total_observable_errors);
                TCHECK(chunked.shots_with_error == baseline.shots_with_error);
                TCHECK(chunked.observables == baseline.observables);
            }
        });
    }

    const std::string& rounding_path = paths.front();
    run_test((rounding_path + " rounds chunk shots down to word boundary").c_str(), [&] {
        reset_sampling_options(rounding_path.c_str());
        options.chunk_shots = 100;
        SamplingHarness framing(rounding_path, SAMPLE_SHOTS);
        check_sampled_circuit(framing);
        const Statistics& stats = framing.statistics();
        TCHECK(stats.sampling.requested_shots == SAMPLE_SHOTS);
        TCHECK(stats.sampling.chunk_shots == 64);
        TCHECK(stats.sampling.chunks == SAMPLE_SHOTS / 64);
        TCHECK(stats.logical.total_shots == SAMPLE_SHOTS);
    });
}

void test_logical_error_rates() {
    section("Sampling logical error rates");

    const auto paths = non_patch_circuit_paths();
    TCHECK(!paths.empty());
    for (const std::string& circuit_path : paths) {
        const std::filesystem::path path(circuit_path);
        const LogicalRateBaseline* baseline = find_logical_rate_baseline(path.filename().string());
        const std::string name = path.string() + " logical error rate near Stim baseline";
        run_test(name.c_str(), [&] {
            TCHECK(baseline != nullptr);
            reset_sampling_options(path.string().c_str());
            SamplingHarness framing(path.string(), SAMPLE_SHOTS);
            check_sampled_circuit(framing);

            const Statistics& stats = framing.statistics();
            TCHECK(stats.logical.total_shots == SAMPLE_SHOTS);
            TCHECK(stats.logical.num_observables > 0);
            TCHECK(std::fabs(stats.logical.rate() - baseline->rate) <= LOGICAL_RATE_TOLERANCE);
        });
    }
}

int main() {
    test_sampling_paths();
    test_chunked_sampling();
    test_logical_error_rates();

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    cleanup_generated_measure_files();
    return (passed == total) ? 0 : 1;
}
