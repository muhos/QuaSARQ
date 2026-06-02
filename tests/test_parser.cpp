#include "../src/parser.hpp"
#include "helper.hpp"

using namespace QuaSARQ;
struct ParserHarness {
    CircuitIO io;

    ParserHarness() { io.init(); }

    void reset() {
        io.circuit_queue.clear(true);
        io.gate_stats.destroy();
        io.gate_stats.alloc();
        io.measures_count = 0;
        io.measuring = false;
        io.max_qubits = 0;
        io.detectors.destroy(); io.detectors.init();
        io.observables.destroy(); io.observables.init();
    }

    void feed(const char* text) {
        const size_t len = strlen(text);
        char* buf = new char[len + 1];
        memcpy(buf, text, len + 1);
        char* str = buf;
        io.eof = buf + len;
        while (str < io.eof) {
            eatWS(str);
            if (str >= io.eof || *str == '\0') break;
            if (*str == '#') { eatLine(str); continue; }
            io.read_gate_into(str, io.circuit_queue, io.gate_stats);
        }
        delete[] buf;
    }

    size_t queue_size() const { return io.circuit_queue.size(); }

    const ParsedGate& gate(size_t i) { return io.circuit_queue[i]; }

    size_t count(Gatetypes t) const { return io.gate_stats.types[t]; }
};

static int translate(const char* name) {
    CircuitIO io; io.init();
    char buf[64];
    strncpy(buf, name, 63); buf[63] = '\0';
    return io.translate_gate(buf, (int)strlen(name));
}

void test_translate_gate() {
    section("translate_gate — aliases and canonical names");

    run_test("CNOT -> CX",         [] { TCHECK(translate("CNOT")       == int(CX));    });
    run_test("ZCX -> CX",          [] { TCHECK(translate("ZCX")        == int(CX));    });
    run_test("ZCY -> CY",          [] { TCHECK(translate("ZCY")        == int(CY));    });
    run_test("ZCZ -> CZ",          [] { TCHECK(translate("ZCZ")        == int(CZ));    });
    run_test("H_XZ -> H",          [] { TCHECK(translate("H_XZ")       == int(H));     });
    run_test("SQRT_Z -> S",        [] { TCHECK(translate("SQRT_Z")     == int(S));     });
    run_test("SQRT_Z_DAG -> S_DAG",[] { TCHECK(translate("SQRT_Z_DAG") == int(S_DAG)); });
    run_test("MZ -> M",            [] { TCHECK(translate("MZ")         == int(M));     });
    run_test("MRZ -> MR",          [] { TCHECK(translate("MRZ")        == int(MR));    });
    run_test("RZ -> R",            [] { TCHECK(translate("RZ")         == int(R));     });

    run_test("H canonical",        [] { TCHECK(translate("H")    == int(H));    });
    run_test("S canonical",        [] { TCHECK(translate("S")    == int(S));    });
    run_test("CX canonical",       [] { TCHECK(translate("CX")   == int(CX));   });
    run_test("ISWAP canonical",    [] { TCHECK(translate("ISWAP") == int(ISWAP)); });
    run_test("unknown -> -1",      [] { TCHECK(translate("BOGUS") == -1);       });
    run_test("empty -> -1",        [] { TCHECK(translate("") == -1);            });
    run_test("prefix only -> -1",  [] { TCHECK(translate("CN") == -1);          });
}

void test_m_variants() {
    section("M-variant expansion (MX / MY / MRX / MRY)");

    run_test("MX 0: queue H M H", [] {
        ParserHarness h;
        h.feed("MX 0\n");
        TCHECK(h.queue_size() == 3);
        TCHECK(h.gate(0).type == byte_t(H)  && h.gate(0).c == 0);
        TCHECK(h.gate(1).type == byte_t(M)  && h.gate(1).c == 0);
        TCHECK(h.gate(2).type == byte_t(H)  && h.gate(2).c == 0);
    });

    run_test("MY 0: queue S_DAG H M H S", [] {
        ParserHarness h;
        h.feed("MY 0\n");
        TCHECK(h.queue_size() == 5);
        TCHECK(h.gate(0).type == byte_t(S_DAG));
        TCHECK(h.gate(1).type == byte_t(H));
        TCHECK(h.gate(2).type == byte_t(M));
        TCHECK(h.gate(3).type == byte_t(H));
        TCHECK(h.gate(4).type == byte_t(S));
    });

    run_test("MRX 0: queue H MR H", [] {
        ParserHarness h;
        h.feed("MRX 0\n");
        TCHECK(h.queue_size() == 3);
        TCHECK(h.gate(0).type == byte_t(H));
        TCHECK(h.gate(1).type == byte_t(MR));
        TCHECK(h.gate(2).type == byte_t(H));
    });

    run_test("MRY 0: queue S_DAG H MR H S", [] {
        ParserHarness h;
        h.feed("MRY 0\n");
        TCHECK(h.queue_size() == 5);
        TCHECK(h.gate(0).type == byte_t(S_DAG));
        TCHECK(h.gate(2).type == byte_t(MR));
        TCHECK(h.gate(4).type == byte_t(S));
    });

    run_test("expanded_from set on all MX gates", [] {
        ParserHarness h;
        h.feed("MX 0\n");
        TCHECK(h.gate(0).expanded_from == byte_t(MX));
        TCHECK(h.gate(1).expanded_from == byte_t(MX));
        TCHECK(h.gate(2).expanded_from == byte_t(MX));
    });

    run_test("gstats counts MX not H/M", [] {
        ParserHarness h;
        h.feed("MX 0\n");
        TCHECK(h.count(MX) == 1);
        TCHECK(h.count(H)  == 0);
        TCHECK(h.count(M)  == 0);
    });

    run_test("measures_count incremented once for MX 0", [] {
        ParserHarness h;
        h.feed("MX 0\n");
        TCHECK(h.io.measures_count == 1);
    });

    run_test("MX 0 1 2: phase-batched H H H M M M H H H", [] {
        ParserHarness h;
        h.feed("MX 0 1 2\n");
        TCHECK(h.queue_size() == 9);
        TCHECK(h.gate(0).type == byte_t(H) && h.gate(0).c == 0);
        TCHECK(h.gate(1).type == byte_t(H) && h.gate(1).c == 1);
        TCHECK(h.gate(2).type == byte_t(H) && h.gate(2).c == 2);
        TCHECK(h.gate(3).type == byte_t(M) && h.gate(3).c == 0);
        TCHECK(h.gate(4).type == byte_t(M) && h.gate(4).c == 1);
        TCHECK(h.gate(5).type == byte_t(M) && h.gate(5).c == 2);
        TCHECK(h.gate(6).type == byte_t(H) && h.gate(6).c == 0);
    });

    run_test("MX 0 1 2: measures_count == 3", [] {
        ParserHarness h;
        h.feed("MX 0 1 2\n");
        TCHECK(h.io.measures_count == 3);
    });

    run_test("MX 0 1 2: gstats counts MX=3", [] {
        ParserHarness h;
        h.feed("MX 0 1 2\n");
        TCHECK(h.count(MX) == 3);
    });

    run_test("MY 0 1: phase-batched S_DAG S_DAG H H M M H H S S", [] {
        ParserHarness h;
        h.feed("MY 0 1\n");
        TCHECK(h.queue_size() == 10);
        TCHECK(h.gate(0).type == byte_t(S_DAG) && h.gate(0).c == 0);
        TCHECK(h.gate(1).type == byte_t(S_DAG) && h.gate(1).c == 1);
        TCHECK(h.gate(2).type == byte_t(H)     && h.gate(2).c == 0);
        TCHECK(h.gate(3).type == byte_t(H)     && h.gate(3).c == 1);
        TCHECK(h.gate(4).type == byte_t(M)     && h.gate(4).c == 0);
        TCHECK(h.gate(5).type == byte_t(M)     && h.gate(5).c == 1);
    });
}

void test_clifford_expansion() {
    section("Clifford expansion");

    run_test("H_YZ 0: H S H S S", [] {
        ParserHarness h;
        h.feed("H_YZ 0\n");
        TCHECK(h.queue_size() == 5);
        TCHECK(h.gate(0).type == byte_t(H));
        TCHECK(h.gate(1).type == byte_t(S));
        TCHECK(h.gate(2).type == byte_t(H));
        TCHECK(h.gate(3).type == byte_t(S));
        TCHECK(h.gate(4).type == byte_t(S));
    });

    run_test("H_XY 0: H S S H S", [] {
        ParserHarness h;
        h.feed("H_XY 0\n");
        TCHECK(h.queue_size() == 5);
        TCHECK(h.gate(0).type == byte_t(H));
        TCHECK(h.gate(1).type == byte_t(S));
        TCHECK(h.gate(2).type == byte_t(S));
        TCHECK(h.gate(3).type == byte_t(H));
        TCHECK(h.gate(4).type == byte_t(S));
    });

    run_test("XCX 0 1: H(0) CX(0,1) H(0)", [] {
        ParserHarness h;
        h.feed("XCX 0 1\n");
        TCHECK(h.queue_size() == 3);
        TCHECK(h.gate(0).type == byte_t(H)  && h.gate(0).c == 0);
        TCHECK(h.gate(1).type == byte_t(CX) && h.gate(1).c == 0 && h.gate(1).t == 1);
        TCHECK(h.gate(2).type == byte_t(H)  && h.gate(2).c == 0);
    });

    run_test("XCZ 0 1: CX(1,0) reversed", [] {
        ParserHarness h;
        h.feed("XCZ 0 1\n");
        TCHECK(h.queue_size() == 1);
        TCHECK(h.gate(0).type == byte_t(CX) && h.gate(0).c == 1 && h.gate(0).t == 0);
    });

    run_test("YCZ 0 1: S_DAG(0) CX(1,0) S(0)", [] {
        ParserHarness h;
        h.feed("YCZ 0 1\n");
        TCHECK(h.queue_size() == 3);
        TCHECK(h.gate(0).type == byte_t(S_DAG) && h.gate(0).c == 0);
        TCHECK(h.gate(1).type == byte_t(CX)    && h.gate(1).c == 1 && h.gate(1).t == 0);
        TCHECK(h.gate(2).type == byte_t(S)     && h.gate(2).c == 0);
    });

    run_test("SQRT_ZZ 0 1: H(1) CX(0,1) H(1) S(0) S(1)", [] {
        ParserHarness h;
        h.feed("SQRT_ZZ 0 1\n");
        TCHECK(h.queue_size() == 5);
        TCHECK(h.gate(0).type == byte_t(H)  && h.gate(0).c == 1);
        TCHECK(h.gate(1).type == byte_t(CX) && h.gate(1).c == 0 && h.gate(1).t == 1);
        TCHECK(h.gate(2).type == byte_t(H)  && h.gate(2).c == 1);
        TCHECK(h.gate(3).type == byte_t(S)  && h.gate(3).c == 0);
        TCHECK(h.gate(4).type == byte_t(S)  && h.gate(4).c == 1);
    });

    run_test("SQRT_ZZ_DAG 0 1: S_DAG instead of S", [] {
        ParserHarness h;
        h.feed("SQRT_ZZ_DAG 0 1\n");
        TCHECK(h.queue_size() == 5);
        TCHECK(h.gate(3).type == byte_t(S_DAG) && h.gate(3).c == 0);
        TCHECK(h.gate(4).type == byte_t(S_DAG) && h.gate(4).c == 1);
    });

    run_test("CXSWAP 0 1: CX(1,0) CX(0,1)", [] {
        ParserHarness h;
        h.feed("CXSWAP 0 1\n");
        TCHECK(h.queue_size() == 2);
        TCHECK(h.gate(0).type == byte_t(CX) && h.gate(0).c == 1 && h.gate(0).t == 0);
        TCHECK(h.gate(1).type == byte_t(CX) && h.gate(1).c == 0 && h.gate(1).t == 1);
    });

    run_test("CZSWAP == SWAPCZ", [] {
        ParserHarness a, b;
        a.feed("CZSWAP 0 1\n");
        b.feed("SWAPCZ 0 1\n");
        TCHECK(a.queue_size() == b.queue_size());
        for (size_t i = 0; i < a.queue_size(); i++) {
            TCHECK(a.gate(i).type == b.gate(i).type);
            TCHECK(a.gate(i).c    == b.gate(i).c);
            TCHECK(a.gate(i).t    == b.gate(i).t);
        }
    });

    run_test("XCX 0 1 2 3: phase-batched H H CX CX H H", [] {
        ParserHarness h;
        h.feed("XCX 0 1 2 3\n");
        TCHECK(h.queue_size() == 6);
        TCHECK(h.gate(0).type == byte_t(H)  && h.gate(0).c == 0);
        TCHECK(h.gate(1).type == byte_t(H)  && h.gate(1).c == 2);
        TCHECK(h.gate(2).type == byte_t(CX) && h.gate(2).c == 0 && h.gate(2).t == 1);
        TCHECK(h.gate(3).type == byte_t(CX) && h.gate(3).c == 2 && h.gate(3).t == 3);
        TCHECK(h.gate(4).type == byte_t(H)  && h.gate(4).c == 0);
        TCHECK(h.gate(5).type == byte_t(H)  && h.gate(5).c == 2);
    });

    run_test("unknown clifford name -> not consumed (fallthrough)", [] {
        ParserHarness h;
        h.feed("H 0\n");
        TCHECK(h.queue_size() == 1);
        TCHECK(h.gate(0).type == byte_t(H));
    });
}

void test_standard_gate() {
    section("Standard gate path");

    run_test("H 0: single qubit gate", [] {
        ParserHarness h;
        h.feed("H 0\n");
        TCHECK(h.queue_size() == 1);
        TCHECK(h.gate(0).type == byte_t(H) && h.gate(0).c == 0 && h.gate(0).t == 0);
        TCHECK(h.count(H) == 1);
    });

    run_test("CX 0 1: two qubit gate", [] {
        ParserHarness h;
        h.feed("CX 0 1\n");
        TCHECK(h.queue_size() == 1);
        TCHECK(h.gate(0).type == byte_t(CX) && h.gate(0).c == 0 && h.gate(0).t == 1);
    });

    run_test("DEPOLARIZE1(0.1) 0: prob stored", [] {
        ParserHarness h;
        h.feed("DEPOLARIZE1(0.1) 0\n");
        TCHECK(h.queue_size() == 1);
        TCHECK(h.gate(0).type == byte_t(DEPOLARIZE1));
        TCHECK(h.gate(0).probs[0] > 0.09f && h.gate(0).probs[0] < 0.11f);
    });

    run_test("multiple qubits on one line: H 0 1 2", [] {
        ParserHarness h;
        h.feed("H 0 1 2\n");
        TCHECK(h.queue_size() == 3);
        TCHECK(h.gate(0).c == 0);
        TCHECK(h.gate(1).c == 1);
        TCHECK(h.gate(2).c == 2);
        TCHECK(h.count(H) == 3);
    });

    run_test("M: measuring flag set, measures_count incremented", [] {
        ParserHarness h;
        h.feed("M 0 1\n");
        TCHECK(h.io.measuring == true);
        TCHECK(h.io.measures_count == 2);
    });

    run_test("TICK/QUBIT_COORDS/SHIFT_COORDS silently dropped", [] {
        ParserHarness h;
        h.feed("TICK\nQUBIT_COORDS(1,1) 0\nSHIFT_COORDS(0,1,0)\nH 0\n");
        TCHECK(h.queue_size() == 1);
        TCHECK(h.gate(0).type == byte_t(H));
    });

    run_test("unknown gate silently skipped", [] {
        ParserHarness h;
        h.feed("UNKNOWNGATE 0\nH 1\n");
        TCHECK(h.queue_size() == 1);
        TCHECK(h.gate(0).type == byte_t(H));
    });

    run_test("max_qubits tracks highest index", [] {
        ParserHarness h;
        h.feed("H 5\nCX 2 7\n");
        TCHECK(h.io.max_qubits == 8);
    });

    run_test("CNOT alias: same queue as CX", [] {
        ParserHarness a, b;
        a.feed("CNOT 0 1\n");
        b.feed("CX 0 1\n");
        TCHECK(a.queue_size() == 1 && b.queue_size() == 1);
        TCHECK(a.gate(0).type == b.gate(0).type);
    });
}

void test_repeat() {
    section("REPEAT block unroll");

    run_test("REPEAT 3 { H 0 }: 3 gates in queue", [] {
        ParserHarness h;
        h.feed("REPEAT 3 {\n  H 0\n}\n");
        TCHECK(h.queue_size() == 3);
        TCHECK(h.count(H) == 3);
        for (size_t i = 0; i < 3; i++)
            TCHECK(h.gate(i).type == byte_t(H) && h.gate(i).c == 0);
    });

    run_test("REPEAT 2 { M 0 }: measures_count == 2", [] {
        ParserHarness h;
        h.feed("REPEAT 2 {\n  M 0\n}\n");
        TCHECK(h.io.measures_count == 2);
    });

    run_test("REPEAT 4 { CX 0 1 }: 4 CX gates", [] {
        ParserHarness h;
        h.feed("REPEAT 4 {\n  CX 0 1\n}\n");
        TCHECK(h.queue_size() == 4);
        TCHECK(h.count(CX) == 4);
    });

    run_test("REPEAT 2 { H 0 CX 0 1 }: 4 gates total", [] {
        ParserHarness h;
        h.feed("REPEAT 2 {\n  H 0\n  CX 0 1\n}\n");
        TCHECK(h.queue_size() == 4);
        TCHECK(h.count(H) == 2);
        TCHECK(h.count(CX) == 2);
    });

    run_test("REPEAT 1 { M 0 } DETECTOR rec[-1]: detector stored", [] {
        ParserHarness h;
        h.feed("REPEAT 1 {\n  M 0\n  DETECTOR rec[-1]\n}\n");
        TCHECK(h.io.detectors.starts.size() == 1);
        TCHECK(h.io.detectors.counts[0] == 1);
    });

    run_test("REPEAT 3 { M 0 } DETECTOR refs rebased per iteration", [] {
        ParserHarness h;
        h.feed("REPEAT 3 {\n  M 0\n  DETECTOR rec[-1]\n}\n");
        TCHECK(h.io.detectors.starts.size() == 3);
        TCHECK(h.io.detectors.refs[0] == 0);
        TCHECK(h.io.detectors.refs[1] == 1);
        TCHECK(h.io.detectors.refs[2] == 2);
    });

    run_test("MX inside REPEAT: gstats counts MX correctly", [] {
        ParserHarness h;
        h.feed("REPEAT 3 {\n  MX 0\n}\n");
        TCHECK(h.count(MX) == 3);
        TCHECK(h.io.measures_count == 3);
    });
}

void test_detector_observable() {
    section("DETECTOR / OBSERVABLE_INCLUDE");

    run_test("DETECTOR rec[-1] after M 0", [] {
        ParserHarness h;
        h.feed("M 0\nDETECTOR rec[-1]\n");
        TCHECK(h.io.detectors.starts.size() == 1);
        TCHECK(h.io.detectors.counts[0] == 1);
        TCHECK(h.io.detectors.refs[0] == 0);
    });

    run_test("DETECTOR rec[-1] rec[-2] after M 0 1", [] {
        ParserHarness h;
        h.feed("M 0 1\nDETECTOR rec[-1] rec[-2]\n");
        TCHECK(h.io.detectors.counts[0] == 2);
        TCHECK(h.io.detectors.refs[0] == 1);
        TCHECK(h.io.detectors.refs[1] == 0);
    });

    run_test("DETECTOR with coords: coords skipped", [] {
        ParserHarness h;
        h.feed("M 0\nDETECTOR(1, 2, 0) rec[-1]\n");
        TCHECK(h.io.detectors.starts.size() == 1);
        TCHECK(h.io.detectors.counts[0] == 1);
    });

    run_test("OBSERVABLE_INCLUDE(0) rec[-1]", [] {
        ParserHarness h;
        h.feed("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
        TCHECK(h.io.observables.ids.size() == 1);
        TCHECK(h.io.observables.ids[0] == 0);
        TCHECK(h.io.observables.records.counts[0] == 1);
    });

    run_test("OBSERVABLE_INCLUDE(2) correct id", [] {
        ParserHarness h;
        h.feed("M 0\nOBSERVABLE_INCLUDE(2) rec[-1]\n");
        TCHECK(h.io.observables.ids[0] == 2);
    });

    run_test("two detectors: separate starts and counts", [] {
        ParserHarness h;
        h.feed("M 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n");
        TCHECK(h.io.detectors.starts.size() == 2);
        TCHECK(h.io.detectors.counts[0] == 1);
        TCHECK(h.io.detectors.counts[1] == 1);
        TCHECK(h.io.detectors.refs[0] == 1);
        TCHECK(h.io.detectors.refs[1] == 0);
    });
}

void test_integration() {
    section("Integration");

    run_test("simple 3-qubit circuit: gate count and types", [] {
        ParserHarness h;
        h.feed(
            "R 0 1 2\n"
            "H 0\n"
            "CX 0 1\n"
            "CX 0 2\n"
            "M 0 1 2\n"
        );
        TCHECK(h.count(R)  == 3);
        TCHECK(h.count(H)  == 1);
        TCHECK(h.count(CX) == 2);
        TCHECK(h.count(M)  == 3);
        TCHECK(h.io.measures_count == 3);
        TCHECK(h.io.max_qubits == 3);
    });

    run_test("MX + REPEAT + DETECTOR end-to-end", [] {
        ParserHarness h;
        h.feed(
            "MX 0 1\n"
            "REPEAT 2 {\n"
            "  H 0\n"
            "  CX 0 1\n"
            "  MX 0\n"
            "  DETECTOR rec[-1]\n"
            "}\n"
        );
        TCHECK(h.count(MX) == 4);
        TCHECK(h.io.measures_count == 4);
        TCHECK(h.io.detectors.starts.size() == 2);
    });

    run_test("surface-code circuit parses without error", [] {
        CircuitIO io; io.init();
        char* stream = io.read("circuits/surface_code_d10_r3.stim");
        char* str = stream;
        while (str < io.eof) {
            eatWS(str);
            if (str >= io.eof || *str == '\0') break;
            if (*str == '#') { eatLine(str); continue; }
            io.read_gate_into(str, io.circuit_queue, io.gate_stats);
        }
        TCHECK(io.circuit_queue.size() > 0);
        TCHECK(io.max_qubits > 0);
        TCHECK(io.measures_count > 0);
    });
}

int main() {
    test_translate_gate();
    test_m_variants();
    test_clifford_expansion();
    test_standard_gate();
    test_repeat();
    test_detector_observable();
    test_integration();

    std::cout << std::format("\n{}{}/{} tests passed{}\n\n",
        passed == total ? CPASS : CFAIL, passed, total, CNORMAL);

    return (passed == total) ? 0 : 1;
}
