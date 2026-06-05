#include "../src/frame.cuh"
#include "helper.hpp"

using namespace QuaSARQ;

constexpr size_t SAMPLE_SHOTS = 1024;

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

int main() {
    test_sampling_paths();

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    cleanup_generated_measure_files();
    return (passed == total) ? 0 : 1;
}
