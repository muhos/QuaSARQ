
#include "simulator.hpp"
#include "power.cuh"

using namespace QuaSARQ;

Simulator::~Simulator() { 
    if (!gpu_allocator.destroy_cpu_pool()) {
		LOGERROR("Failed.");
	}
	if (!gpu_allocator.destroy_gpu_pool()) {
		LOGERRORN("Failed.");
	}
    if (custreams != nullptr) {
        for (int i = 0; i < options.streams; i++) 
            cudaStreamDestroy(custreams[i]);
        delete[] custreams;
        custreams = nullptr;
    }
}

Simulator::Simulator() :
	num_qubits(options.num_qubits)
    , num_partitions(1)
	, depth((depth_t)options.depth)
	, random(1)
	, circuit(MB)
	, circuit_mode(RANDOM_CIRCUIT)
	, tableau(gpu_allocator)
	, gpu_circuit(gpu_allocator)
	, configfile(nullptr)
	, custreams(nullptr)
{
    initialize();
}

Simulator::Simulator(const string& path) :
    num_qubits(options.num_qubits)
    , num_partitions(1)
    , depth((depth_t)options.depth)
    , random(1)
    , circuit(MB)
    , circuit_path(path)
    , circuit_mode(PARSED_CIRCUIT)
    , tableau(gpu_allocator)
    , gpu_circuit(gpu_allocator)
    , configfile(nullptr)
    , custreams(nullptr)
{
    initialize();
}

void Simulator::initialize() {
    LOGHEADER(1, 3, "Build");
    getCPUInfo();
    getGPUInfo();
    LOGHEADER(1, 3, "Initial");
    gpu_allocator.create_gpu_pool();
    parse();
    // Creating CPU pool (pinned memory)
    // is done after parsing as it causes
    // degradation to CPU performance.
    // The KB extra space is used for tableau allocations.
    gpu_allocator.create_cpu_pool(stats.circuit.max_window_bytes + KB);
    if (!options.tuner_en) register_config();
    create_streams(custreams);
    SYNC(0); // Sync gpu memory pool allocation.
    FAULT_DETECTOR;
}

void Simulator::create_streams(cudaStream_t*& streams) {
    if (streams == nullptr) {
        LOGN2(1, "Allocating %d GPU streams..", options.streams);
        streams = new cudaStream_t[options.streams];
        for (int i = 0; i < options.streams; i++) 
            cudaStreamCreate(streams + i);
        LOGDONE(1, 3);
    }
}

// Simulate circuit by moving window
// over parallel gates of each depth level.
void Simulator::simulate(const size_t& p, const bool& reversed) {
    if (!options.check_integrity) {
        if (reversed) { LOGHEADER(1, 3, "Reversed Simulation"); }
        else { LOGHEADER(1, 3, "Simulation"); }
    }
    if (!tableau.size()) {
        LOGERRORN("cannot run simulation without allocating the tableau.");
        throw GPU_memory_exception();
    }
    if (reversed) {
        gpu_circuit.reset_circuit_offsets(circuit[depth - 1].size(), circuit.reference(depth - 1, 0), options.overlap);
        for (depth_t d = 0; d < depth; d++) {
            const depth_t b = depth - d - 1;
            step(p, b, custreams, true);
        }
    }
    else {
        gpu_circuit.reset_circuit_offsets(0, 0, options.overlap);
        for (depth_t d = 0; d < depth; d++) {
            step(p, d, custreams);
        }
    }
    print_tableau_final(tableau, reversed);
}

void Simulator::simulate() {
    // Create a tableau in GPU memory.
    Power power;
    timer.start();
    num_partitions = tableau.alloc(num_qubits, options.overlap ? stats.circuit.bytes : stats.circuit.max_window_bytes);
    const size_t num_qubits_per_partition = num_partitions > 1 ? tableau.num_words_major() * WORD_BITS : num_qubits;
    if (options.overlap)
        gpu_circuit.initiate(circuit, stats.circuit.max_parallel_gates, stats.circuit.max_parallel_gates_buckets);
    else
        gpu_circuit.initiate(stats.circuit.max_parallel_gates, stats.circuit.max_parallel_gates_buckets);
    timer.stop();
    stats.time.initial += timer.time();
    // Start step-wise simulation.
    timer.start();
    for (size_t p = 0; p < num_partitions; p++) {
        // Create identity.
        const size_t prev_num_qubits = num_qubits_per_partition * p;
        assert(prev_num_qubits < num_qubits);
        LOGN2(1, "Partition %zd: ", p);
        identity(tableau, prev_num_qubits, (p == num_partitions - 1) ? (num_qubits - prev_num_qubits) : num_qubits_per_partition, custreams, options.initialstate);
        // Stepwise simulation.
        simulate(p, false);
    }
	SYNCALL;
	timer.stop();
	stats.time.simulation = timer.time();
    stats.power.wattage = power.measure();
    stats.power.joules = stats.power.wattage * (stats.time.simulation / 1000.0);
    stats.tableau.count = stats.tableau.istates = 1;
    stats.tableau.gigabytes = ratio((double)tableau.size() * sizeof(word_std_t), double(GB));
    stats.tableau.seconds = (stats.time.simulation / 1000.0) / (num_partitions * depth);
    stats.tableau.calc_speed();
    report();
}