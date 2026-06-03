#include "../src/simulator.hpp"
#include "helper.hpp"

#include <cstring>

using namespace QuaSARQ;

constexpr const char* SURFACE_CODE_D10_R3 = "circuits/surface_code_d10_r3.stim";
constexpr const char* SURFACE_CODE_D50_R10 = "circuits/surface_code_d50_r10.stim";

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
    std::strcpy(options.configpath, "../src/kernel.config");
    std::strcpy(options.statepath, "../build/test_simulator.qstate");
    options.check(circuit_path);
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

    run_test("loads and schedules d10 r3 surface code", [] {
        reset_options(SURFACE_CODE_D10_R3);
        options.check_measurement = true;
        SimulatorHarness sim(SURFACE_CODE_D10_R3);
        check_loaded_surface_code(sim);
    });

    run_test("loads and schedules d50 r10 surface code", [] {
        reset_options(SURFACE_CODE_D50_R10);
        options.check_measurement = true;
        SimulatorHarness sim(SURFACE_CODE_D50_R10);
        check_loaded_surface_code(sim);
    });
}

void test_surface_code_simulation() {
    section("Simulator surface code simulation");

    run_test("simulates d10 r3 surface code with dets/obs checks", [] {
        reset_options(SURFACE_CODE_D10_R3);
        options.check_measurement = true;
        options.print_observable = true;
        options.print_detector = true;
        SimulatorHarness sim(SURFACE_CODE_D10_R3);
        check_simulated_surface_code(sim);
    });

    run_test("simulates d50 r10 surface code with dets/obs checks", [] {
        reset_options(SURFACE_CODE_D50_R10);
        options.check_measurement = true;
        options.print_observable = true;
        options.print_detector = true;
        SimulatorHarness sim(SURFACE_CODE_D50_R10);
        check_simulated_surface_code(sim);
    });
}

int main() {
    test_surface_code_lifecycle();
    test_surface_code_simulation();

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    return (passed == total) ? 0 : 1;
}
