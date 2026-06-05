#include "../src/equivalence.hpp"
#include "helper.hpp"

using namespace QuaSARQ;

class EquivalenceHarness : public Equivalence {

protected:

    void print_result(const bool&, const char&) const override {}

public:

    using Equivalence::Equivalence;

    const Statistics& statistics() const {
        return stats;
    }

    const Circuit& scheduled_circuit() const {
        return circuit;
    }

    const size_t& qubits() const {
        return num_qubits;
    }

};

std::string write_temp_circuit(const char* name, const char* body) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to create temporary circuit");
    out << body;
    return path.string();
}

void reset_equivalence_options(const char* left, const char* right);

bool is_equivalence_ignored_gate(const byte_t& type) {
    return isNoise(type) ||
           type == R || type == RX || type == RY ||
           type == M || type == MR || type == MX ||
           type == MY || type == MRX || type == MRY;
}

byte_t daggered_gate_type(const byte_t& type) {
    if (type == S_DAG) return S;
    if (type == S) return S_DAG;
    if (type == ISWAP) return ISWAP_DAG;
    if (type == ISWAP_DAG) return ISWAP;
    if (type == SQRT_X) return SQRT_X_DAG;
    if (type == SQRT_X_DAG) return SQRT_X;
    if (type == SQRT_Y) return SQRT_Y_DAG;
    if (type == SQRT_Y_DAG) return SQRT_Y;
    return type;
}

void write_gate(std::ofstream& out, const Gate& gate, const byte_t& type) {
    out << G2S_STIM[type] << " " << gate.wires[0];
    if (gate.size > 1)
        out << " " << gate.wires[1];
    out << "\n";
}

void write_full_identity_barrier(std::ofstream& out, const size_t& num_qubits) {
    for (int layer = 0; layer < 2; layer++) {
        out << "H";
        for (qubit_t q = 0; q < num_qubits; q++)
            out << " " << q;
        out << "\n";
    }
}

void write_tick_barrier(std::ofstream& out) {
    out << "TICK\n";
}

std::string write_filtered_identity_temp_circuit(const size_t& num_qubits) {
    std::filesystem::path identity_path = std::filesystem::temp_directory_path() /
        ("quasarq_equiv_identity_q" + std::to_string(num_qubits) + ".stim");
    std::ofstream out(identity_path);
    if (!out)
        throw std::runtime_error("failed to create identity temporary circuit");
    out << "#q" << num_qubits << "\n";
    write_full_identity_barrier(out, num_qubits);
    return identity_path.string();
}

std::string write_filtered_composed_inverse_temp_circuit(const std::string& source_path, size_t& num_qubits) {
    reset_equivalence_options(source_path.c_str(), source_path.c_str());
    EquivalenceHarness equivalence(source_path, source_path);
    const Circuit& circuit = equivalence.scheduled_circuit();
    num_qubits = equivalence.qubits();
    std::filesystem::path composed_path = std::filesystem::temp_directory_path() /
        (std::filesystem::path(source_path).stem().string() + "_filtered_composed_inverse.stim");
    std::ofstream out(composed_path);
    if (!out)
        throw std::runtime_error("failed to create composed inverse temporary circuit");
    out << "#q" << num_qubits << "\n";
    for (depth_t d = 0; d < circuit.depth(); d++) {
        const Window& window = circuit[d];
        for (size_t i = 0; i < window.size(); i++) {
            const Gate& gate = circuit.gate(window[i]);
            if (is_equivalence_ignored_gate(gate.type) || gate.type == I)
                continue;
            write_gate(out, gate, gate.type);
        }
        write_tick_barrier(out);
    }
    for (depth_t d = circuit.depth(); d > 0; --d) {
        const Window& window = circuit[d - 1];
        for (size_t i = window.size(); i > 0; --i) {
            const Gate& gate = circuit.gate(window[i - 1]);
            if (is_equivalence_ignored_gate(gate.type) || gate.type == I)
                continue;
            write_gate(out, gate, daggered_gate_type(gate.type));
        }
        write_tick_barrier(out);
    }
    return composed_path.string();
}

void reset_equivalence_options(const char* left, const char* right) {
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
    options.num_shots = 0;
    options.min_shots_write = 0;
    options.min_measures_write = 0;
    options.streams = 6;
    options.tuner_en = false;
    options.disable_concurrency = false;
    options.ignore_ticks = false;
    SET_LOGGER_VERBOSITY(options.verbose);
    copy_test_path(options.configpath, kernel_config_path());
    copy_test_path(options.statepath, test_build_path() / "test_equivalence.qstate");
    options.check(left, right);
}

void check_equivalence_pair(const std::string& left, const std::string& right, const bool expected) {
    EquivalenceHarness equivalence(left, right);
    equivalence.check();
    TCHECK(equivalence.is_equivalent() == expected);
    if (expected) {
        const Statistics& stats = equivalence.statistics();
        TCHECK(stats.tableau.count == 2);
        TCHECK(stats.tableau.istates == 2);
        TCHECK(stats.time.simulation >= 0.0);
    }
}

void test_identical_checked_in_circuits() {
    section("Equivalence checked-in circuits");

    const auto paths = circuit_paths();
    TCHECK(!paths.empty());
    for (const std::string& path : paths) {
        const std::string name = path + " equivalent to itself";
        run_test(name.c_str(), [&] {
            reset_equivalence_options(path.c_str(), path.c_str());
            check_equivalence_pair(path, path, true);
        });
    }
}

void test_checked_in_inverse_circuits() {
    section("Equivalence checked-in inverse circuits");

    const auto paths = circuit_paths();
    TCHECK(!paths.empty());
    for (const std::string& path : paths) {
        const std::string name = path + " filtered circuit composed with inverse is identity";
        run_test(name.c_str(), [&] {
            size_t num_qubits = 0;
            const std::string composed = write_filtered_composed_inverse_temp_circuit(path, num_qubits);
            const std::string identity = write_filtered_identity_temp_circuit(num_qubits);
            reset_equivalence_options(composed.c_str(), identity.c_str());
            check_equivalence_pair(composed, identity, true);
        });
    }
}

void test_explicit_equivalence_paths() {
    section("Equivalence explicit paths");

    const std::string h0_a = write_temp_circuit("quasarq_equiv_h0_a.stim", "H 0\n");
    const std::string h0_b = write_temp_circuit("quasarq_equiv_h0_b.stim", "H 0\n");
    const std::string s0   = write_temp_circuit("quasarq_equiv_s0.stim", "S 0\n");
    const std::string cx01 = write_temp_circuit("quasarq_equiv_cx01.stim", "CX 0 1\n");

    run_test("identical tiny circuits are equivalent", [&] {
        reset_equivalence_options(h0_a.c_str(), h0_b.c_str());
        check_equivalence_pair(h0_a, h0_b, true);
    });

    run_test("different tiny circuits are not equivalent", [&] {
        reset_equivalence_options(h0_a.c_str(), s0.c_str());
        check_equivalence_pair(h0_a, s0, false);
    });

    run_test("qubit-count mismatch is not equivalent", [&] {
        reset_equivalence_options(h0_a.c_str(), cx01.c_str());
        check_equivalence_pair(h0_a, cx01, false);
    });

    run_test("disable-concurrency path keeps equivalent result", [&] {
        reset_equivalence_options(h0_a.c_str(), h0_b.c_str());
        options.disable_concurrency = true;
        const bool sync_before = options.sync;
        check_equivalence_pair(h0_a, h0_b, true);
        TCHECK(options.sync == sync_before);
    });
}

int main() {
    test_identical_checked_in_circuits();
    test_checked_in_inverse_circuits();
    test_explicit_equivalence_paths();

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    cleanup_generated_measure_files();
    return (passed == total) ? 0 : 1;
}
