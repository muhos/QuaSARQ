#include "../src/simulator.hpp"
#include "helper.hpp"

#include <cstring>

using namespace QuaSARQ;

#define RESET_GATE_PROB(GATE) options.GATE ## _p = 0.0;

class SimulatorHarness : public Simulator {

public:

    using Simulator::Simulator;

    const size_t& qubits() const {
        return num_qubits;
    }

    const depth_t& scheduled_depth() const {
        return depth;
    }

    const Statistics& statistics() const {
        return stats;
    }

    const WindowInfo& window_info() const {
        return winfo;
    }

    const bool& has_measurements() const {
        return measuring;
    }

    const bool& is_reference_run() const {
        return reference_mode;
    }

    cudaStream_t* streams() const {
        return custreams;
    }

    cudaStream_t copy_stream(const int& index) const {
        return copy_streams[index];
    }

    cudaStream_t kernel_stream(const int& index) const {
        return kernel_streams[index];
    }
};

void reset_options(const char* circuit_path) {
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
    options.print_finalstate = false;
    options.print_finaltableau = false;
    options.num_shots = 0;
    options.min_measures_write = 0;
    options.streams = 6;
    options.tuner_en = false;
    options.ignore_ticks = false;
    SET_LOGGER_VERBOSITY(options.verbose);
    copy_test_path(options.configpath, kernel_config_path());
    copy_test_path(options.statepath, test_build_path() / "test_simulator.qstate");
    options.check(circuit_path);
}

void reset_random_options(const size_t& qubits, const size_t& depth) {
    reset_options(nullptr);
    options.num_qubits = qubits;
    options.depth = depth;
    FOREACH_GATE(RESET_GATE_PROB);
    options.H_p = 1.0;
    options.M_p = 0.4;
    options.MR_p = 0.6;
}

void reset_measurement_and_noise_gates() {
    options.R_p           = 0.0;
    options.RX_p          = 0.0;
    options.RY_p          = 0.0;
    options.M_p           = 0.0;
    options.MR_p          = 0.0;
    options.MX_p          = 0.0;
    options.MY_p          = 0.0;
    options.MRX_p         = 0.0;
    options.MRY_p         = 0.0;
    options.X_ERROR_p     = 0.0;
    options.Y_ERROR_p     = 0.0;
    options.Z_ERROR_p     = 0.0;
    options.DEPOLARIZE1_p = 0.0;
    options.DEPOLARIZE2_p = 0.0;
    options.PAULI_CHANNEL_1_p = 0.0;
    options.PAULI_CHANNEL_2_p = 0.0;
}

void check_loaded_surface_code(SimulatorHarness& sim) {
    const Circuit& circuit = sim.get_circuit();
    const Statistics& parse_stats = sim.statistics();

    TCHECK(sim.qubits() > 0);
    TCHECK(sim.scheduled_depth() > 0);
    TCHECK(circuit.depth() == sim.scheduled_depth());
    TCHECK(circuit.num_gates() > 0);
    TCHECK(parse_stats.circuit.num_gates >= circuit.num_gates());
    TCHECK(parse_stats.circuit.max_parallel_gates > 0);
    TCHECK(parse_stats.circuit.num_parallel_gates > 0);
    TCHECK(sim.window_info().max_parallel_gates > 0);
    TCHECK(sim.window_info().max_window_bytes > 0);
    TCHECK(sim.has_measurements());
    TCHECK(parse_stats.circuit.measure_stats.count > 0);
    TCHECK(parse_stats.circuit.measure_stats.depth > 0);
    TCHECK(parse_stats.circuit.measure_stats.depth < circuit.depth());

    depth_t measuring_windows = 0;
    depth_t recording_windows = 0;
    for (depth_t d = 0; d < circuit.depth(); d++) {
        if (circuit.is_measuring(d))
            measuring_windows++;
        if (circuit.is_recording(d))
            recording_windows++;
    }
    TCHECK(measuring_windows == parse_stats.circuit.measure_stats.depth);
    TCHECK(recording_windows > 0);
    TCHECK(sim.streams() != nullptr);
    TCHECK(sim.copy_stream(0) == sim.streams()[0]);
    TCHECK(sim.copy_stream(1) == sim.streams()[1]);
    TCHECK(sim.copy_stream(2) == sim.streams()[2]);
    TCHECK(sim.copy_stream(3) == sim.streams()[3]);
    TCHECK(sim.kernel_stream(0) == sim.streams()[4]);
    TCHECK(sim.kernel_stream(1) == sim.streams()[5]);
}

void check_simulated_surface_code(SimulatorHarness& sim) {
    sim.simulate();
    const Statistics& sim_stats = sim.statistics();
    TCHECK(sim.has_measurements());
    TCHECK(sim_stats.circuit.measure_stats.count > 0);
    TCHECK(sim_stats.tableau.count == 1);
    TCHECK(sim_stats.tableau.istates == 1);
    TCHECK(sim_stats.tableau.gigabytes > 0.0);
    TCHECK(sim_stats.time.simulation >= 0.0);
    TCHECK(!sim.is_reference_run());
    TCHECK(options.check_measurement);
}

void test_surface_code_lifecycle() {
    section("Simulator surface code lifecycle");

    const auto paths = circuit_paths();
    TCHECK(!paths.empty());
    for (const std::string& path : paths) {
        const std::string name = path + " loads and schedules";
        run_test(name.c_str(), [&] {
            reset_options(path.c_str());
            SimulatorHarness sim(path);
            check_loaded_surface_code(sim);
        });
    }
}

void test_surface_code_simulation() {
    section("Simulator surface code simulation with active built-in checker");

    const auto paths = circuit_paths_up_to_distance(50);
    TCHECK(!paths.empty());
    for (const std::string& path : paths) {
        const std::string name = path + " simulates with dets/obs checks";
        run_test(name.c_str(), [&] {
            reset_options(path.c_str());
            options.check_measurement = true;
            options.print_observable = true;
            options.print_detector = true;
            SimulatorHarness sim(path);
            check_simulated_surface_code(sim);
        });
    }
}

void test_random_simulation(const bool& with_checking = false) {
    section(("Simulator random circuit simulation" + std::string(with_checking ? " with built-in checker" : "")).c_str());
    for (size_t qubits = 1000; qubits <= 5000; qubits += 1000) {
        const size_t depth = 100;
        run_test(("simulates q" + std::to_string(qubits) + "-d" + std::to_string(depth) + " circuit").c_str(), [&] {
            reset_random_options(qubits, depth);
            if (with_checking) {
                options.check_measurement = true;
            }
            SimulatorHarness sim;
            TCHECK(sim.has_measurements());
            TCHECK(sim.statistics().circuit.measure_stats.count > 0);
            sim.simulate();
            TCHECK(sim.statistics().circuit.measure_stats.count > 0);
            TCHECK(sim.statistics().tableau.count == 1);
        });
    }

    section(("Simulator random circuit simulation " + 
        std::string(with_checking ? "with built-in checker" : "") + " - no measurements").c_str());
    for (size_t qubits = 10000; qubits <= 50000; qubits += 10000) {
        const size_t depth = 500;
        run_test(("simulates q" + std::to_string(qubits) + "-d" + std::to_string(depth) + " circuit").c_str(), [&] {
            reset_random_options(qubits, depth);
            reset_measurement_and_noise_gates();
            if (with_checking) {
                options.check_all = true;
            }
            SimulatorHarness sim;
            TCHECK(!sim.has_measurements());
            TCHECK(sim.statistics().circuit.measure_stats.count == 0);
            TCHECK(sim.statistics().circuit.measure_stats.depth == 0);
            TCHECK(sim.qubits() == qubits);
            TCHECK(sim.scheduled_depth() > 0);
            TCHECK(sim.statistics().circuit.num_gates > 0);
            const Circuit& circuit = sim.get_circuit();
            for (depth_t d = 0; d < circuit.depth(); d++)
                TCHECK(!circuit.is_measuring(d));
            sim.simulate();
            TCHECK(!sim.has_measurements());
            TCHECK(sim.statistics().circuit.measure_stats.count == 0);
            TCHECK(sim.statistics().circuit.measure_stats.depth == 0);
            TCHECK(sim.statistics().tableau.count == 1);
            TCHECK(sim.statistics().tableau.gigabytes > 0.0);
            TCHECK(sim.statistics().time.simulation >= 0.0);
            TCHECK(!sim.is_reference_run());
        });
    }
}

int main() {
    test_surface_code_lifecycle();
    test_surface_code_simulation();
    test_random_simulation();
    test_random_simulation(true);

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    cleanup_generated_measure_files();
    return (passed == total) ? 0 : 1;
}

#undef RESET_GATE_PROB
