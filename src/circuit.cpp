
#include "simulator.hpp"
#include <queue>

namespace QuaSARQ {
    double probabilities[NR_GATETYPES] = {0};
}

using namespace QuaSARQ;

// Inside-out variant of Fisher-Yates algorithm.
void Simulator::shuffle_qubits() {
    shuffled.resize(num_qubits);
    for (qubit_t i = 0; i < num_qubits; i++) {
        qubit_t j = random.irand() % (i + 1);
        if (j != i)
            shuffled[i] = shuffled[j]; 
        shuffled[j] = i;
    }
}

void Simulator::get_rand_qubit(const qubit_t& control, qubit_t& random_qubit) {
    assert(num_qubits);
    assert(shuffled.size() == num_qubits);
    qubit_t* q = shuffled, * e = shuffled.end();
    while (q != e && (control == (random_qubit = *q) || locked[random_qubit])) q++;
}

Gatetypes Simulator::get_rand_gate(const bool& multi_input, const bool& force_multi_input) {
    double p = random.drand();
    double sum_probs = 0;
    Gatetypes type = I;
    uint32 offset = 0;
    uint32 limit = NR_GATETYPES_1;
    if (force_multi_input) {
        offset = NR_GATETYPES_1;
        limit = NR_GATETYPES;
    }
    else if (multi_input) 
        limit = NR_GATETYPES;
    for (uint32 i = offset; i < limit; i++) {
        sum_probs += probabilities[i];
        if (p <= sum_probs) {
            type = Gatetypes(i);
            break;
        }
    }
    return type;
}

// Add measurements to circuit.
void add_measurements(Circuit& circuit, Vec<qubit_t, size_t>& measurements, WindowInfo& winfo, const depth_t& depth_level) {
    size_t num_gate_buckets_per_window = circuit.num_buckets();
    for (size_t i = 0; i < measurements.size(); i++)
        circuit.addGate(depth_level, M, measurements[i]);
    measurements.clear();
    circuit.markMeasure(depth_level);
    winfo.max(circuit[depth_level].size(), (circuit.num_buckets() - num_gate_buckets_per_window));
}

void Simulator::generate() {
    assert(circuit_mode == RANDOM_CIRCUIT);
    if (!num_qubits) {
        LOGERROR("Number of qubits for random circuit cannot be zero.");
    }
    if (!depth) {
        LOGERROR("Depth for random circuit cannot be zero.");
    }
    LOGN2(1, "Generating random circuit for %s%zd qubits%s and %s%d-level%s depth.. ", 
            CREPORTVAL, num_qubits, CNORMAL, CREPORTVAL, depth, CNORMAL);
    LOG2(2, "");
    timer.start();
    circuit.init_depth(depth);
    locked.resize(num_qubits, 0);
    // Initialize gate probabilities.
    if (options.write_rc == 2) { // CHP format
        FOREACH_GATE(RESET_PROB);
    }
    else {
        FOREACH_GATE(UNIFORM_PROB);
        INIT_PROB(I);
    }
    INIT_PROB(H);
    INIT_PROB(S);
    INIT_PROB(CX);
    INIT_PROB(M);
    NORMALIZE_PROBS();
    size_t* types = stats.circuit.gate_stats.types;
    size_t parallel_gates_per_window = 0;
    for (depth_t d = 0; d < depth; d++) {
        size_t num_gate_buckets_per_window = circuit.num_buckets();
        // Add measurements to circuit if exist.
        if (measurements.size()) {
            add_measurements(circuit, measurements, winfo, d);
            continue;
        }
        // Shuffle qubits used for multi-input gates.
        shuffle_qubits();
        // Loop over all qubits with these assumptions:
        //  1-input gate acts on q or
        //  control qubit in 2-input gate = q.
        for (qubit_t q = 0; q < num_qubits; q++) {
            if (!locked[q]) {
                Gatetypes type = get_rand_gate();
                assert(type < NR_GATETYPES);
                assert(type < 256);    
                // 0: q (control), 1: target.
                Gate_inputs gate_inputs;
                gate_inputs.push(q);
                if (isGate2(type)) {
                    // Get (independent) random target qubit and lock it.
                    gate_inputs.push(q);
                    get_rand_qubit(q, gate_inputs[1]);
                    // If we could not find a free qubit, then we're done.
                    if (locked[gate_inputs[1]]) 
                        break;
                    else if (gate_inputs[0] == gate_inputs[1]) {
                        // Last (unlocked) qubit must be a 1-input gate.
                        type = get_rand_gate(false);
                        gate_inputs.pop();
                    }
                    else {                        
                        // Lock target to avoid being control later.
                        locked[gate_inputs[1]] = 1;
                    }                 
                }
                if (type == M) {
                    if (d < depth - 1) {
                        measurements.push(q);
                        measuring = true;
                    }
                    else
                        continue;
                }
                else  
                    circuit.addGate(d, type, gate_inputs);
                // This lock is necessary to avoid generating
                // same control qubit as target.
                locked[q] = 1;
                // Collect statistics.
                types[type]++;
                if (type != I) {
                    stats.circuit.num_parallel_gates++;
                    parallel_gates_per_window++;
                }
            }
        }
        stats.circuit.max_parallel_gates = MAX(stats.circuit.max_parallel_gates, parallel_gates_per_window);
        locked.reset();
        parallel_gates_per_window = 0;
        assert(circuit.num_buckets() >= num_gate_buckets_per_window);
        assert(circuit.depth() > d);
        winfo.max(circuit[d].size(), (circuit.num_buckets() - num_gate_buckets_per_window));
    }
    assert(circuit.depth() <= depth + 1);
    stats.circuit.num_gates = MAX(stats.circuit.num_gates, circuit.num_gates());
    stats.circuit.bytes = stats.circuit.num_gates * sizeof(gate_ref_t) + circuit.num_buckets() * BUCKETSIZE;
    shuffled.clear(true);
    locked.clear(true);
    measurements.clear(true);
    timer.stop();
    stats.time.schedule = timer.time();
    LOGDONE(1, 2);
    LOG2(1, "Generated a total of %s%zd gates%s with a maximum of %s%zd parallel gates%s.", 
    CREPORTVAL, circuit.num_gates(), CNORMAL, 
    CREPORTVAL, stats.circuit.max_parallel_gates, CNORMAL);
    if (options.write_rc)
        circuit_io.write(circuit, num_qubits, options.write_rc, stats);
    if (options.verbose > 1)
        circuit.print();
}

size_t Simulator::parse(Statistics& stats, const char* path) {
    timer.start();
    assert(!circuit_io.size);
    circuit_io.init();
    char* str = circuit_io.read(path);
    size_t max_qubits = 0;
    while (str < circuit_io.eof) {
        eatWS(str);
        if (*str == '\0') break;
        if (*str == '#') {
            if (*++str == 'q' && !max_qubits) {
                uint32 sign = 0;
                max_qubits = toInteger(++str, sign);
                if (sign) LOGERROR("number of qubits in header is negative.");
                LOG2(1, "Found header %s%zd%s.", CREPORTVAL, max_qubits, CNORMAL);   
            }        
        }
        circuit_io.read_gate(str);  
    }
    if (!max_qubits)
        max_qubits = circuit_io.max_qubits;
    assert(circuit_io.circuit_queue.size() == circuit_io.gate_stats.all());
    stats.circuit.num_gates = circuit_io.circuit_queue.size();
    stats.circuit.gate_stats = circuit_io.gate_stats;
    measuring = circuit_io.measuring;
    timer.stop();
    stats.time.initial += timer.time();
    return max_qubits;
}

size_t Simulator::schedule(Statistics& stats, Circuit& circuit) {
    LOGN2(1, "Scheduling %s%zd%s gates for parallel simulation.. ", CREPORTVAL, stats.circuit.num_gates, CNORMAL);
    LOG2(2, "");
    timer.start();
    // For locking qubits
    assert(num_qubits);
    locked.resize(num_qubits, 0);
    Vec<qubit_t, qubit_t> locked_qubits;
    locked_qubits.reserve(num_qubits);
    circuit.init_depth(stats.circuit.num_gates / 2);
    size_t max_depth = 0;
    size_t parallel_gates_per_window = 0;
    // To prevent stagnation if empty wires exist per depth level.
    // In that scenario, locked qubits will never reach a fixpoint.
    qubit_t last_num_locked_qubits = 0;
    qubit_t max_locked_qubits = 0;
    for (; max_depth < MAX_DEPTH && !circuit_io.circuit_queue.empty(); max_depth++) {

        size_t num_gate_buckets_per_window = circuit.num_buckets();

        // Add measurements to circuit if exist.
        if (measurements.size()) {
            for (size_t i = 0; i < measurements.size(); i++)
                circuit.addGate(max_depth, M, measurements[i]);
            measurements.clear();
            circuit.markMeasure(max_depth);
            winfo.max(circuit[max_depth].size(), (circuit.num_buckets() - num_gate_buckets_per_window));
            continue;
        }

        // Forall gates in order find an independent gate.
        while (!circuit_io.circuit_queue.empty()) {
                const ParsedGate gate = circuit_io.circuit_queue.front();
                const qubit_t c = gate.c;
                const qubit_t t = gate.t;
                const bool is_c_unlocked = !locked[c];
                if (c == t) {
                    if (is_c_unlocked) {
                        circuit_io.circuit_queue.pop_front();
                        if (gate.type == M) {
                            measurements.push(c);
                            measuring = true;
                        }
                        else     
                            circuit.addGate(max_depth, gate.type, c);
                        locked_qubits.push(c);
                        locked[c] = 1;
                        if (gate.type != I) {
                            stats.circuit.num_parallel_gates++;
                            parallel_gates_per_window++;
                        }
                    }
                }
                else {
                    const bool is_t_unlocked = !locked[t];
                    if (is_c_unlocked && is_t_unlocked) {
                        circuit_io.circuit_queue.pop_front();
                        assert(gate.type != M);
                        circuit.addGate(max_depth, gate.type, c, t);
                        locked_qubits.push(c);
                        locked_qubits.push(t);
                        locked[c] = 1;
                        locked[t] = 1;
                        if (gate.type != I) {
                            stats.circuit.num_parallel_gates++;
                            parallel_gates_per_window++;
                        }
                    }
                    // Either one of them may not be locked.
                    // Thus, make sure the other is locked.
                    // This prevent bypassing a locked 2-gate
                    // and not preserving the order.
                    else if (is_c_unlocked) {
                        locked_qubits.push(c);
                        locked[c] = 1;                         
                    }
                    else if (is_t_unlocked) {
                        locked_qubits.push(t);
                        locked[t] = 1;
                    }
                }
                // If this is true, we know that no more gates
                // can be scheduled at themax_depth++; same depth. Thus, 
                // we have to start a new depth level.
                // In other words, all gates in between 0 and n - 1
                // have been already scheduled if they are not locked.
                max_locked_qubits = locked_qubits.size();
                if (last_num_locked_qubits == max_locked_qubits || max_locked_qubits == num_qubits)
                    break;
                last_num_locked_qubits = max_locked_qubits;
        }
        max_depth++;

        stats.circuit.max_parallel_gates = MAX(stats.circuit.max_parallel_gates, parallel_gates_per_window);

        parallel_gates_per_window = 0;
        last_num_locked_qubits = 0;
        if (max_locked_qubits) {
            forall_vector(locked_qubits, lq) {
                locked[*lq] = 0;
            }
            locked_qubits.clear();
        }
    
        assert(circuit.num_buckets() >= num_gate_buckets_per_window);
        winfo.max(circuit[max_depth].size(), (circuit.num_buckets() - num_gate_buckets_per_window));
    }
    timer.stop();
    stats.time.schedule = timer.time();
    assert(max_depth <= circuit.depth());
    assert(circuit.num_gates() == stats.circuit.num_gates);
    stats.circuit.num_gates = MAX(stats.circuit.num_gates, circuit.num_gates());
    stats.circuit.bytes = stats.circuit.num_gates * sizeof(gate_ref_t) + circuit.num_buckets() * BUCKETSIZE;
    locked.clear(true);
    locked_qubits.clear(true);
    circuit_io.destroy();
    LOGDONE(1, 2);
    LOG2(1, "Scheduled %s%zd%s gates with a maximum of %s%zd%s parallel gates in %s%.3f%s ms.",
        CREPORTVAL, stats.circuit.num_gates, CNORMAL,
        CREPORTVAL, stats.circuit.max_parallel_gates, CNORMAL, 
        CREPORTVAL, stats.time.schedule, CNORMAL);
    return max_depth;
}

void Simulator::parse() {
    if (!circuit.empty()) return;
    if (circuit_mode == RANDOM_CIRCUIT) {
        generate();
    }
    else {
        assert(circuit_mode == PARSED_CIRCUIT);
        num_qubits = parse(stats, circuit_path.c_str());
        depth = schedule(stats, circuit);
    }
}