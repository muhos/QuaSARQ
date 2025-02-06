#include "equivalence.hpp"
#include "power.cuh"

using namespace QuaSARQ;

#define OTHER_GATE2STATISTIC(GATE) \
	LOG1(" %s %-5s  %24s %s%-12zd (%%%-3.0f)%s", \
			CREPORT, #GATE, ":", CREPORTVAL, other_stats.circuit.gate_stats.types[GATE], \
			percent((double)other_stats.circuit.gate_stats.types[GATE], other_stats.circuit.num_gates), CNORMAL);	\

Equivalence::Equivalence() : 
    other_num_qubits(options.num_qubits)
    , other_num_partitions(1)
    , other_depth(options.depth)
    , other_circuit(MB)
    , other_tableau(gpu_allocator)
    , other_gpu_circuit(gpu_allocator)
    , other_custreams(nullptr)
    , Simulator() 
    { 
        assert(!circuit.empty());
        create_streams(other_custreams);
        inject();
        gpu_allocator.resize_cpu_pool(winfo.max_window_bytes + other_wininfo.max_window_bytes + KB * 2);
    }

Equivalence::Equivalence(const string& path_to_circuit, const string& path_to_other) :
    other_num_qubits(options.num_qubits)
    , other_num_partitions(1)
    , other_depth(options.depth)
    , other_circuit(MB)
    , other_tableau(gpu_allocator)
    , other_gpu_circuit(gpu_allocator)
    , other_custreams(nullptr)
    , Simulator(path_to_circuit) 
    {
        create_streams(other_custreams);
        if (path_to_other.empty()) {
            inject();
        }
        else {
            assert(!circuit_io.size);
            other_num_qubits = parse(other_stats, path_to_other.c_str());
            other_depth = schedule(other_stats, other_circuit);
            stats.time.initial += other_stats.time.initial;
            stats.time.schedule += other_stats.time.schedule;
        }
        gpu_allocator.resize_cpu_pool(winfo.max_window_bytes + other_wininfo.max_window_bytes + KB * 2);
    }

void Equivalence::inject() {
    circuit.copyTo(other_circuit);
    other_wininfo = winfo;
    other_num_qubits = num_qubits;
    other_depth = depth;
    other_stats = stats;
    assert(other_depth == other_circuit.depth());
    random.seed(other_num_qubits);
    depth_t depth_level = random.irand() % other_depth;
    gate_ref_t gate_index = random.irand() % other_circuit[depth_level].size();
    Gate& random_gate = other_circuit.gate(depth_level, gate_index);
    Gatetypes type = I;
    if (random_gate.size == 1) {
        type = get_rand_gate(false); 
    }
    else {
        type = get_rand_gate(true, true); 
    }
    assert(type != I);
    assert(other_stats.circuit.gate_stats.types[random_gate.type]);
    other_stats.circuit.gate_stats.types[random_gate.type]--;
    other_stats.circuit.gate_stats.types[type]++;
    ogate = G2S[random_gate.type], rgate = G2S[type];
    random_gate.type = type;
}

bool Equivalence::check(const InitialState& istate, const size_t& num_qubits_per_partition, const size_t& other_num_qubits_per_partition) {
    stats.tableau.istates++;
    char state = '0';
    if (istate == Plus)
        state = '+';
    else if (istate == Imag)
        state = 'i';
    size_t all_equivalent = 0;
    for (size_t p = 0; p < num_partitions; p++) {
        if (p < num_partitions) {
            const size_t prev_num_qubits = num_qubits_per_partition * p;
            assert(prev_num_qubits < num_qubits);
            LOGN2(1, "Partition %zd: ", p);
            identity(tableau, prev_num_qubits, (p == num_partitions - 1) ? (num_qubits - prev_num_qubits) : num_qubits_per_partition, custreams, istate);
            gpu_circuit.reset_circuit_offset(0);
        }
        if (p < other_num_partitions) {
            const size_t other_prev_num_qubits = other_num_qubits_per_partition * p;
            assert(other_prev_num_qubits < other_num_qubits);
            LOGN2(1, "Partition %zd: ", p);
            identity(other_tableau, other_prev_num_qubits, (p == other_num_partitions - 1) ? (other_num_qubits - other_prev_num_qubits) : other_num_qubits_per_partition, other_custreams, istate);
            other_gpu_circuit.reset_circuit_offset(0);
        }
        if (check(p, custreams, other_custreams)) {
            all_equivalent++;
        }
        else {
            failed_state = state;
            break;
        }
    }
    LOGN2(1, "Tableau");
    if (all_equivalent == num_partitions) {
        LOG2(1, "%s EQUIVALENT%s for \'%c\' initial state.", CGREEN, CNORMAL, state);
        return true;
    }
    else {
        LOG2(1, "%s NOT EQUIVALENT%s for \'%c\' initial state.", CRED, CNORMAL, state);
        return false;
    }
}

void Equivalence::check() {
    LOGHEADER(1, 3, "Equivalence checking");
    if (num_qubits != other_num_qubits) {
        LOG2(1, "%s NOT EQUIVALENT%s due to misaligned qubits.", CRED, CNORMAL);
    }
    // Create two tableaus in GPU memory.
    Power power;
    timer.start();
    size_t estimated_num_partitions = get_num_partitions(2, num_qubits, winfo.max_window_bytes + other_wininfo.max_window_bytes, gpu_allocator.gpu_capacity());
    num_partitions = tableau.alloc(num_qubits, winfo.max_window_bytes, false, false, estimated_num_partitions);
    other_num_partitions = other_tableau.alloc(other_num_qubits, other_wininfo.max_window_bytes, false, false, estimated_num_partitions);
    assert(num_partitions == other_num_partitions);
    const size_t num_qubits_per_partition = num_partitions > 1 ? tableau.num_words_major() * WORD_BITS : num_qubits;
    const size_t other_num_qubits_per_partition = other_num_partitions > 1 ? other_tableau.num_words_major() * WORD_BITS : other_num_qubits;
    gpu_circuit.initiate(winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
    other_gpu_circuit.initiate(other_wininfo.max_parallel_gates, other_wininfo.max_parallel_gates_buckets);
    timer.stop();
    stats.time.initial += timer.time();
    // Start step-wise equivalence.
    timer.start();
    const bool equivalent = check(Zero, num_qubits_per_partition, other_num_qubits_per_partition) && check(Plus, num_qubits_per_partition, other_num_qubits_per_partition);
	SYNCALL;
	timer.stop();
	stats.time.simulation = timer.time();
    stats.power.wattage = power.measure();
    stats.power.joules = stats.power.wattage * (stats.time.simulation / 1000.0);
    stats.tableau.count = 2;
    stats.tableau.gigabytes = ratio((double)tableau.size() * sizeof(word_std_t), double(GB));
    stats.tableau.seconds = (stats.time.simulation / 1000.0) / (num_partitions * depth);
    stats.tableau.calc_speed();
    report(equivalent);
}

void Equivalence::report(const bool& equivalent) {
    Simulator::report();
    LOG1(" %sOther circuit depth            : %s%-12u%s", CREPORT, CREPORTVAL, other_depth, CNORMAL);
    LOG1(" %sOther circuit qubits           : %s%-12zd%s", CREPORT, CREPORTVAL, other_num_qubits, CNORMAL);
    double circuit_mb = ratio((double)other_stats.circuit.bytes, double(MB));
    LOG1(" %sOther circuit memory           : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, circuit_mb, CNORMAL);
    LOG1(" %sOther maximum parallel gates   : %s%-12zd%s", CREPORT, CREPORTVAL, other_stats.circuit.max_parallel_gates, CNORMAL);
    LOG1(" %sOther Clifford gates           : %s%-12zd%s", CREPORT, CREPORTVAL, other_stats.circuit.num_gates, CNORMAL);
    FOREACH_GATE(OTHER_GATE2STATISTIC)
    if (!ogate.empty())
        LOG1(" %sInjected gates                 : %s%s -> %s%s", CREPORT, CREPORTVAL, ogate.c_str(), rgate.c_str(), CNORMAL);
    if (equivalent)
        LOG1(" %sCircuits check                 : %sEQUIVALENT%s", CREPORT, CGREEN, CNORMAL);
    else {
        LOG1(" %sCircuits check                 : %sNOT EQUIVALENT%s", CREPORT, CRED, CNORMAL);
        LOG1(" %sFailed state                   : %s%c%s", CREPORT, CREPORTVAL, failed_state, CNORMAL);
    }
}