#include "checker.hpp"

using namespace QuaSARQ;

Checker::Checker() : 
    Simulator() { }

Checker::Checker(const string& path) :
    Simulator(path) { }

void Checker::check_parallel_gates() {
    locked.resize(num_qubits, 0);
    if (circuit.empty()) {
        LOG0("");
        LOGERROR(" found circuit is empty");
    }
    for (depth_t d = 0; d < depth; d++) {
        const Window& w = circuit[d];
        for (uint32 g = 0; g < w.size(); g++) {
            gate_ref_t r = w[g];
            if (r == NO_REF) {
                LOG0("");
                LOGERROR(" found gate reference is invalid");
            }
            const Gate& gate = circuit.gate(r);
            for (input_size_t i = 0; i < gate.size; i++) {
                if (locked[gate.wires[i]]) {
                    LOG0("");
                    LOGERROR(" found input %d of gate %d at depth %d is a duplicate", gate.wires[i], g, d);
                }
                locked[gate.wires[i]] = 1;
            }
        }
        locked.reset();
    }
}

void Checker::check_random_circuit() {
    if (!options.check_parallel_gates) return;
    LOG2(1, "");
    LOGN2(1, "Checking parallelism in %s circuit.. ", 
        circuit_mode == RANDOM_CIRCUIT ? "generated random" : "parsed");
    parse();
    check_parallel_gates();
    LOG2(1, "%sVERIFIED%s", CGREEN, CNORMAL);
}

inline bool verify_inline_transpose(const Table& x1, const Table& z1, const Table& x2, const Table& z2) {
    if (x1.size() != x2.size()) return false;
    if (z1.size() != z2.size()) return false;
    for (auto i = 0; i < x1.size(); i++) {
        if (x1[i] != x2[i]) {
            LOGERRORN("incorrect transpose of x1 or x2.");
            return false;
        }
    }
    for (auto i = 0; i < z1.size(); i++) {
        if (z1[i] != z2[i]) {
            LOGERRORN("incorrect transpose of z1 or z2.");
            return false;
        }
    }
    return true;
}

void Checker::check_transpose() {
    Table in_xs, in_zs;
    tableau.copy_to_host(&in_xs, &in_zs);

    // printf("Before:\n");
    // print_tableau(tableau, -1, false);

    transpose(true, 0);
    transpose(true, 0);

    // printf("After:\n");
    // print_tableau(tableau, -1, false);

    Table out_xs, out_zs;
    tableau.copy_to_host(&out_xs, &out_zs);
    verify_inline_transpose(in_xs, out_xs, in_zs, out_zs);
}

void Checker::check_integrity() {
    if (!options.check_integrity) return;
    LOG2(1, "");
    LOG2(1, "Checking tableau integrity.. ");
    LOG2(1, "");

    // Create a tableau in GPU memory.
    timer.start();
    num_partitions = tableau.alloc(num_qubits, winfo.max_window_bytes, false, measuring);
    const size_t num_qubits_per_partition = num_partitions > 1 ? tableau.num_words_major() * WORD_BITS : num_qubits;
    gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
    timer.stop();
    stats.time.initial += timer.time();

    // For all partitions.
    timer.start();
    for (size_t p = 0; p < num_partitions; p++) {  
        // Create identity.
        const size_t prev_num_qubits = num_qubits_per_partition * p;
        assert(prev_num_qubits < num_qubits);
        identity(tableau, prev_num_qubits, (p == num_partitions - 1) ? (num_qubits - prev_num_qubits) : num_qubits_per_partition, custreams, options.initialstate);

        // Simulate forward.
        simulate(p, false);

        // Simulate backward.
        simulate(p, true);

        // Check diagonal.
        LOG2(1, "");
        LOGN2(1, "Partition %zd: tableau integrity", p);
        if (check_identity(prev_num_qubits, (p == num_partitions - 1) ? (num_qubits - prev_num_qubits) : num_qubits_per_partition))
            LOG2(1, "%s VERIFIED.%s", CGREEN, CNORMAL);
        else
            LOG2(1, "%s NOT VERIFIED.%s", CRED, CNORMAL);
    }
    timer.stop();
    stats.time.simulation = timer.time();
}

void Checker::run() {
    LOGHEADER(1, 3, "Checker");
    check_random_circuit();
    check_integrity();
    report();
}