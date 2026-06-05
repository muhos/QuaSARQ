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

};

std::string write_temp_circuit(const char* name, const char* body) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to create temporary circuit");
    out << body;
    return path.string();
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
    SET_LOGGER_VERBOSITY(options.verbose);
    std::strcpy(options.configpath, "../src/kernel.config");
    std::strcpy(options.statepath, "../build/test_equivalence.qstate");
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
        check_equivalence_pair(h0_a, h0_b, true);
        TCHECK(options.sync);
    });
}

int main() {
    test_identical_checked_in_circuits();
    test_explicit_equivalence_paths();

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    cleanup_generated_measure_files();
    return (passed == total) ? 0 : 1;
}
