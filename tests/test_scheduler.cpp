#include "../src/parser.hpp"
#include "../src/scheduler.hpp"
#include "../src/options.hpp"
#include "helper.hpp"

using namespace QuaSARQ;
struct SchedulerHarness {
    CircuitIO circuit_io;
    Circuit circuit;
    WindowInfo winfo;

    SchedulerHarness() {
        circuit_io.init();
    }

    void feed(const char* text) {
        const size_t len = strlen(text);
        char* buf = new char[len + 1];
        memcpy(buf, text, len + 1);
        char* str = buf;
        circuit_io.eof = buf + len;
        while (str < circuit_io.eof) {
            eatWS(str);
            if (str >= circuit_io.eof || *str == '\0') break;
            if (*str == '#') { eatLine(str); continue; }
            circuit_io.read_gate_into(str, circuit_io.circuit_queue, circuit_io.gate_stats);
        }
        delete[] buf;
    }

    void feed_file(const char* path) {
        char* str = circuit_io.read(path);
        while (str < circuit_io.eof) {
            eatWS(str);
            if (str >= circuit_io.eof || *str == '\0') break;
            if (*str == '#') { eatLine(str); continue; }
            circuit_io.read_gate_into(str, circuit_io.circuit_queue, circuit_io.gate_stats);
        }
    }

    void schedule(const size_t& num_qubits) {
        schedule_circuit(circuit_io, circuit, winfo, num_qubits);
    }

    depth_t depth() const {
        return circuit.depth();
    }

    size_t gates_in_window(const depth_t& d) const {
        TCHECK(d < circuit.depth());
        return circuit[d].size();
    }

    std::set<qubit_t> qubits_in_window(const depth_t& d) const {
        TCHECK(d < circuit.depth());
        std::set<qubit_t> qubits;
        for (uint32 g = 0; g < circuit[d].size(); g++) {
            gate_ref_t ref = circuit[d][g];
            const Gate& gate = circuit.gate(ref);
            for (input_size_t i = 0; i < gate.size; i++) {
                qubits.insert(gate.wires[i]);
            }
        }
        return qubits;
    }

    bool no_qubit_overlap_in_window(const depth_t& d) const {
        TCHECK(d < circuit.depth());
        std::set<qubit_t> seen;
        for (uint32 g = 0; g < circuit[d].size(); g++) {
            gate_ref_t ref = circuit[d][g];
            const Gate& gate = circuit.gate(ref);
            for (input_size_t i = 0; i < gate.size; i++) {
                if (seen.count(gate.wires[i]))
                    return false;
                seen.insert(gate.wires[i]);
            }
        }
        return true;
    }

    Gatetypes gate_type_at(const depth_t& d, const uint32& g) const {
        TCHECK(d < circuit.depth());
        TCHECK(g < circuit[d].size());
        gate_ref_t ref = circuit[d][g];
        return Gatetypes(circuit.gate(ref).type);
    }

    qubit_t control_qubit_at(const depth_t& d, const uint32& g) const {
        TCHECK(d < circuit.depth());
        TCHECK(g < circuit[d].size());
        gate_ref_t ref = circuit[d][g];
        return circuit.gate(ref).wires[0];
    }

    bool all_gates_have_type(const depth_t& d, const Gatetypes& type) const {
        TCHECK(d < circuit.depth());
        for (uint32 g = 0; g < circuit[d].size(); g++) {
            if (gate_type_at(d, g) != type)
                return false;
        }
        return true;
    }

    depth_t find_first_window_with_type(const Gatetypes& type) const {
        for (depth_t d = 0; d < circuit.depth(); d++) {
            if (all_gates_have_type(d, type))
                return d;
        }
        return circuit.depth();
    }

    size_t count_type_in_window(const depth_t& d, const Gatetypes& type) const {
        TCHECK(d < circuit.depth());
        size_t count = 0;
        for (uint32 g = 0; g < circuit[d].size(); g++) {
            if (gate_type_at(d, g) == type)
                count++;
        }
        return count;
    }

    bool same_qubit_in_different_windows(const qubit_t& qubit) const {
        depth_t first_d = circuit.depth();
        for (depth_t d = 0; d < circuit.depth(); d++) {
            std::set<qubit_t> qubits = qubits_in_window(d);
            if (qubits.count(qubit)) {
                if (first_d == circuit.depth())
                    first_d = d;
                else
                    return true;
            }
        }
        return false;
    }
};

void test_window_separation() {
    section("Window separation — M/MR in own window");

    run_test("single H and single M: two windows", [] {
        SchedulerHarness h;
        h.feed("H 0\nM 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 2);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, M));
    });

    run_test("H M H: H-M-H windows separate", [] {
        SchedulerHarness h;
        h.feed("H 0\nM 0\nH 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 3);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, M));
        TCHECK(h.all_gates_have_type(2, H));
    });

    run_test("TICK separates otherwise independent gates", [] {
        SchedulerHarness h;
        h.feed("H 0\nTICK\nH 1\n");
        h.schedule(2);
        TCHECK(h.depth() == 2);
        TCHECK(h.gates_in_window(0) == 1);
        TCHECK(h.gates_in_window(1) == 1);
    });

    run_test("ignore-ticks allows independent gates to compact", [] {
        SchedulerHarness h;
        options.ignore_ticks = true;
        h.feed("H 0\nTICK\nH 1\n");
        options.ignore_ticks = false;
        h.schedule(2);
        TCHECK(h.depth() == 1);
        TCHECK(h.gates_in_window(0) == 2);
    });

    run_test("MX expansion: phase-batched H-M-H windows", [] {
        SchedulerHarness h;
        h.feed("MX 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 3);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, M));
        TCHECK(h.all_gates_have_type(2, H));
    });

    run_test("multiple measurements same type: single M window", [] {
        SchedulerHarness h;
        h.feed("H 0 1\nM 0 1\n");
        h.schedule(2);
        depth_t m_window = h.find_first_window_with_type(M);
        TCHECK(m_window < h.depth());
        TCHECK(h.gates_in_window(m_window) == 2);
        TCHECK(h.count_type_in_window(m_window, M) == 2);
    });

    run_test("M then MR: separate windows", [] {
        SchedulerHarness h;
        h.feed("H 0 1\nM 0\nMR 1\n");
        h.schedule(2);
        depth_t m_window = h.find_first_window_with_type(M);
        depth_t mr_window = h.find_first_window_with_type(MR);
        TCHECK(m_window != mr_window);
        TCHECK(m_window < mr_window);
    });

    run_test("no mixing of M and non-M in same window", [] {
        SchedulerHarness h;
        h.feed("H 0\nH 1\nM 0\nM 1\n");
        h.schedule(2);
        for (depth_t d = 0; d < h.depth(); d++) {
            bool has_m = h.count_type_in_window(d, M) > 0;
            bool has_h = h.count_type_in_window(d, H) > 0;
            TCHECK(!(has_m && has_h));
        }
    });
}

void test_qubit_locking() {
    section("Qubit locking — same qubit in different windows");

    run_test("two gates same qubit: different windows", [] {
        SchedulerHarness h;
        h.feed("H 0\nS 0\n");
        h.schedule(1);
        TCHECK(h.depth() >= 2);
        TCHECK(h.same_qubit_in_different_windows(0));
    });

    run_test("three gates same qubit: three windows", [] {
        SchedulerHarness h;
        h.feed("H 0\nS 0\nH 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 3);
        TCHECK(h.gates_in_window(0) == 1);
        TCHECK(h.gates_in_window(1) == 1);
        TCHECK(h.gates_in_window(2) == 1);
    });

    run_test("parallel gates on different qubits: same window", [] {
        SchedulerHarness h;
        h.feed("H 0\nH 1\n");
        h.schedule(2);
        TCHECK(h.depth() == 1);
        TCHECK(h.gates_in_window(0) == 2);
    });

    run_test("qubit 0: two-gate sequence keeps gates separated", [] {
        SchedulerHarness h;
        h.feed("H 0\nS 0\n");
        h.schedule(1);
        std::set<qubit_t> w0 = h.qubits_in_window(0);
        std::set<qubit_t> w1 = h.qubits_in_window(1);
        TCHECK(w0.count(0) && w1.count(0));
    });

    run_test("no qubit overlap within a window", [] {
        SchedulerHarness h;
        h.feed("H 0\nH 1\nH 2\nH 3\n");
        h.schedule(4);
        for (depth_t d = 0; d < h.depth(); d++) {
            TCHECK(h.no_qubit_overlap_in_window(d));
        }
    });
}

void test_gate_ordering() {
    section("Gate ordering — dependent gates in right depth order");

    run_test("two sequential gates same qubit: order preserved", [] {
        SchedulerHarness h;
        h.feed("H 0\nS 0\n");
        h.schedule(1);
        TCHECK(h.gate_type_at(0, 0) == H);
        TCHECK(h.gate_type_at(1, 0) == S);
    });

    run_test("H-S-H sequence depth order: H(0) S(1) H(2)", [] {
        SchedulerHarness h;
        h.feed("H 0\nS 0\nH 0\n");
        h.schedule(1);
        TCHECK(h.gate_type_at(0, 0) == H);
        TCHECK(h.gate_type_at(1, 0) == S);
        TCHECK(h.gate_type_at(2, 0) == H);
    });

    run_test("CX-CX same control: second in later window", [] {
        SchedulerHarness h;
        h.feed("CX 0 1\nCX 0 2\n");
        h.schedule(3);
        TCHECK(h.depth() >= 2);
        depth_t first_d = h.depth();
        for (depth_t d = 0; d < h.depth(); d++) {
            if (h.count_type_in_window(d, CX) > 0) {
                first_d = d;
                break;
            }
        }
        bool found_second = false;
        for (depth_t d = first_d + 1; d < h.depth(); d++) {
            if (h.count_type_in_window(d, CX) > 0) {
                found_second = true;
                break;
            }
        }
        TCHECK(found_second);
    });

    run_test("input gates stay before operation gates", [] {
        SchedulerHarness h;
        h.feed("MX 0\nH 0\n");
        h.schedule(1);
        depth_t m_window = h.find_first_window_with_type(M);
        depth_t h_after_m = h.circuit.depth();
        for (depth_t d = m_window + 1; d < h.depth(); d++) {
            if (h.all_gates_have_type(d, H)) {
                h_after_m = d;
                break;
            }
        }
        TCHECK(h_after_m > m_window);
    });
}

void test_expansion_batching() {
    section("Expansion batching — MX/MY phases to windows");

    run_test("MX 0: H M H windows", [] {
        SchedulerHarness h;
        h.feed("MX 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 3);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, M));
        TCHECK(h.all_gates_have_type(2, H));
    });

    run_test("MY 0: S_DAG H M H S windows", [] {
        SchedulerHarness h;
        h.feed("MY 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 5);
        TCHECK(h.gate_type_at(0, 0) == S_DAG);
        TCHECK(h.gate_type_at(1, 0) == H);
        TCHECK(h.gate_type_at(2, 0) == M);
        TCHECK(h.gate_type_at(3, 0) == H);
        TCHECK(h.gate_type_at(4, 0) == S);
    });

    run_test("MX 0 1 2: phase-batched H H H M M M H H H", [] {
        SchedulerHarness h;
        h.feed("MX 0 1 2\n");
        h.schedule(3);
        TCHECK(h.depth() == 3);
        TCHECK(h.gates_in_window(0) == 3);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.gates_in_window(1) == 3);
        TCHECK(h.all_gates_have_type(1, M));
        TCHECK(h.gates_in_window(2) == 3);
        TCHECK(h.all_gates_have_type(2, H));
    });

    run_test("MRX expansion: H MR H", [] {
        SchedulerHarness h;
        h.feed("MRX 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 3);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, MR));
        TCHECK(h.all_gates_have_type(2, H));
    });

    run_test("H_YZ expansion: H S H S S", [] {
        SchedulerHarness h;
        h.feed("H_YZ 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 5);
        TCHECK(h.gate_type_at(0, 0) == H);
        TCHECK(h.gate_type_at(1, 0) == S);
        TCHECK(h.gate_type_at(2, 0) == H);
        TCHECK(h.gate_type_at(3, 0) == S);
        TCHECK(h.gate_type_at(4, 0) == S);
    });
}

void test_repeat_unroll() {
    section("REPEAT unroll integration");

    run_test("REPEAT 2 { H 0 }: 2 gates", [] {
        SchedulerHarness h;
        h.feed("REPEAT 2 {\n  H 0\n}\n");
        h.schedule(1);
        TCHECK(h.depth() == 2);
        TCHECK(h.gates_in_window(0) == 1);
        TCHECK(h.gates_in_window(1) == 1);
    });

    run_test("REPEAT 3 { CX 0 1 }: 3 CX gates", [] {
        SchedulerHarness h;
        h.feed("REPEAT 3 {\n  CX 0 1\n}\n");
        h.schedule(2);
        TCHECK(h.depth() == 3);
        for (depth_t d = 0; d < h.depth(); d++) {
            TCHECK(h.gates_in_window(d) == 1);
            TCHECK(h.all_gates_have_type(d, CX));
        }
    });

    run_test("REPEAT 2 { H 0 CX 0 1 }: 4 gates alternating", [] {
        SchedulerHarness h;
        h.feed("REPEAT 2 {\n  H 0\n  CX 0 1\n}\n");
        h.schedule(2);
        TCHECK(h.depth() == 4);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, CX));
        TCHECK(h.all_gates_have_type(2, H));
        TCHECK(h.all_gates_have_type(3, CX));
    });

    run_test("REPEAT measurements separate by intervening gates", [] {
        SchedulerHarness h;
        h.feed("REPEAT 2 {\n  H 0\n  M 0\n}\n");
        h.schedule(1);
        depth_t m_window = h.find_first_window_with_type(M);
        TCHECK(m_window != h.depth());
        TCHECK(h.count_type_in_window(m_window, M) == 1);
        depth_t second_m = h.find_first_window_with_type(M);
        for (depth_t d = m_window + 1; d < h.depth(); d++) {
            if (h.count_type_in_window(d, M) > 0) {
                second_m = d;
                break;
            }
        }
        TCHECK(second_m > m_window);
    });
}

void test_edge_cases() {
    section("Edge cases — empty, single gate, all-parallel");

    run_test("single H gate: one window", [] {
        SchedulerHarness h;
        h.feed("H 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 1);
        TCHECK(h.gates_in_window(0) == 1);
    });

    run_test("single CX gate: one window", [] {
        SchedulerHarness h;
        h.feed("CX 0 1\n");
        h.schedule(2);
        TCHECK(h.depth() == 1);
        TCHECK(h.gates_in_window(0) == 1);
    });

    run_test("four independent single-qubit gates: one window", [] {
        SchedulerHarness h;
        h.feed("H 0\nH 1\nH 2\nH 3\n");
        h.schedule(4);
        TCHECK(h.depth() == 1);
        TCHECK(h.gates_in_window(0) == 4);
    });

    run_test("four independent two-qubit gates: one window", [] {
        SchedulerHarness h;
        h.feed("CX 0 1\nCX 2 3\nCX 4 5\nCX 6 7\n");
        h.schedule(8);
        TCHECK(h.depth() == 1);
        TCHECK(h.gates_in_window(0) == 4);
    });

    run_test("single M gate: one window", [] {
        SchedulerHarness h;
        h.feed("M 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 1);
        TCHECK(h.all_gates_have_type(0, M));
    });

    run_test("five independent measurements: one window", [] {
        SchedulerHarness h;
        h.feed("M 0 1 2 3 4\n");
        h.schedule(5);
        TCHECK(h.depth() == 1);
        TCHECK(h.gates_in_window(0) == 5);
    });

    run_test("identity gates skipped in window count", [] {
        SchedulerHarness h;
        h.feed("I 0\nH 0\n");
        h.schedule(1);
        TCHECK(h.depth() == 2);
    });
}

void test_integration() {
    section("Integration — complex circuits");

    run_test("bell state circuit: H-CX-M", [] {
        SchedulerHarness h;
        h.feed("H 0\nCX 0 1\nM 0 1\n");
        h.schedule(2);
        TCHECK(h.depth() >= 3);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, CX));
        depth_t m_window = h.find_first_window_with_type(M);
        TCHECK(m_window > 1);
    });

    run_test("GHZ state: H-CX-CX-M", [] {
        SchedulerHarness h;
        h.feed("H 0\nCX 0 1\nCX 0 2\nM 0 1 2\n");
        h.schedule(3);
        TCHECK(h.depth() >= 4);
        for (depth_t d = 0; d < h.depth(); d++) {
            TCHECK(h.no_qubit_overlap_in_window(d));
        }
    });

    run_test("surface code layer: H CX H M", [] {
        SchedulerHarness h;
        h.feed(
            "H 0 1 2 3\n"
            "CX 0 1\nCX 0 4\nCX 2 1\nCX 2 3\n"
            "H 0 1 2 3\n"
            "M 0 1 2 3\n"
        );
        h.schedule(5);
        TCHECK(h.depth() >= 5);
        for (depth_t d = 0; d < h.depth(); d++) {
            TCHECK(h.no_qubit_overlap_in_window(d));
        }
    });

    run_test("MX with following gate: H-M-H-next", [] {
        SchedulerHarness h;
        h.feed("MX 0\nH 0\n");
        h.schedule(1);
        TCHECK(h.depth() >= 4);
        TCHECK(h.all_gates_have_type(0, H));
        TCHECK(h.all_gates_have_type(1, M));
        TCHECK(h.all_gates_have_type(2, H));
        TCHECK(h.all_gates_have_type(3, H));
    });

    run_test("REPEAT creates separate M windows per iteration", [] {
        SchedulerHarness h;
        h.feed("REPEAT 3 {\n  H 0\n  M 0\n}\n");
        h.schedule(1);
        size_t m_count = 0;
        for (depth_t d = 0; d < h.depth(); d++) {
            m_count += h.count_type_in_window(d, M);
        }
        TCHECK(m_count == 3);
        size_t windows_with_m = 0;
        for (depth_t d = 0; d < h.depth(); d++) {
            if (h.count_type_in_window(d, M) > 0) {
                windows_with_m++;
            }
        }
        TCHECK(windows_with_m == 3);
    });

    const auto paths = circuit_paths();
    TCHECK(!paths.empty());
    for (const std::string& path : paths) {
        const std::string no_conflicts_name = path + " no qubit conflicts in any window";
        run_test(no_conflicts_name.c_str(), [&] {
            SchedulerHarness h;
            h.feed_file(path.c_str());
            h.schedule(h.circuit_io.max_qubits);
            for (depth_t d = 0; d < h.depth(); d++) {
                TCHECK(h.no_qubit_overlap_in_window(d));
            }
        });

        const std::string measurements_name = path + " measurements isolated from gates";
        run_test(measurements_name.c_str(), [&] {
            SchedulerHarness h;
            h.feed_file(path.c_str());
            h.schedule(h.circuit_io.max_qubits);
            for (depth_t d = 0; d < h.depth(); d++) {
                bool has_m = h.count_type_in_window(d, M) > 0 || h.count_type_in_window(d, MR) > 0;
                bool has_gates = false;
                for (uint32 g = 0; g < h.gates_in_window(d); g++) {
                    Gatetypes type = h.gate_type_at(d, g);
                    if (type != M && type != MR) {
                        has_gates = true;
                        break;
                    }
                }
                TCHECK(!(has_m && has_gates));
            }
        });
    }
}

int main() {
    test_window_separation();
    test_qubit_locking();
    test_gate_ordering();
    test_expansion_batching();
    test_repeat_unroll();
    test_edge_cases();
    test_integration();

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    cleanup_generated_measure_files();
    return (passed == total) ? 0 : 1;
}
