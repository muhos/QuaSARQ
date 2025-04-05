
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
	, gpu_circuit(gpu_allocator)
    , locker(gpu_allocator)
    , tableau(gpu_allocator)
    , inv_tableau(gpu_allocator)
    , prefix(gpu_allocator)
	, config_file(nullptr)
    , config_qubits(0)
	, custreams(nullptr)
    , copy_streams{ 0, 0 }
    , kernel_streams{ 0, 0 }
    , measuring(false)
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
    , gpu_circuit(gpu_allocator)
    , locker(gpu_allocator)
    , tableau(gpu_allocator)
    , inv_tableau(gpu_allocator)
    , prefix(gpu_allocator)
    , config_file(nullptr)
    , config_qubits(0)
    , custreams(nullptr)
    , copy_streams{ 0, 0 }
    , kernel_streams{ 0, 0 }
    , measuring(false)
{
    initialize();
}

void Simulator::initialize() {
    LOGHEADER(1, 4, "Build");
    getCPUInfo();
    getGPUInfo();
    LOGHEADER(1, 4, "Initial");
    gpu_allocator.create_gpu_pool();
    parse();
    // Creating CPU pool (pinned memory)
    // is done after parsing as it causes
    // degradation to CPU performance.
    // The KB extra space is used for tableau allocations.
    gpu_allocator.create_cpu_pool(winfo.max_window_bytes 
                                + winfo.max_parallel_gates * sizeof(pivot_t)
                                + KB);
    if (!options.tuner_en) register_config();
    create_streams(custreams);
    locker.alloc();
    SYNC(0); // Sync gpu memory pool allocation.
    FAULT_DETECTOR;
}

void Simulator::create_streams(cudaStream_t*& streams) {
    if (streams == nullptr) {
        assert(options.streams >= 4);
        LOGN2(1, "Allocating %d GPU streams..", options.streams);
        streams = new cudaStream_t[options.streams];
        for (int i = 0; i < options.streams; i++) 
            cudaStreamCreate(streams + i);
        // We need at least two copy streams and two kernel streams.
        copy_streams[0] = streams[0];
        copy_streams[1] = streams[1];
        kernel_streams[0] = streams[2];
        kernel_streams[1] = streams[3];
        LOGDONE(1, 4);
    }
}

// Simulate circuit by moving window
// over parallel gates of each depth level.
void Simulator::simulate(const size_t& p, const bool& reversed) {
    if (!tableau.size()) {
        LOGERRORN("cannot run simulation without allocating the tableau.");
        throw GPU_memory_exception();
    }
    if (!options.check_tableau) {
        if (reversed) { LOGHEADER(1, 4, "Reversed Simulation"); }
        else { LOGHEADER(1, 4, "Simulation"); }
    }
    if (options.progress_en) print_progress_header();
    if (reversed) {
        gpu_circuit.reset_circuit_offset(circuit.reference(depth - 1, 0));
        for (depth_t d = 0; d < depth; d++)
            step(p, depth - d - 1, true);
    }
    else {
        gpu_circuit.reset_circuit_offset(0);
        for (depth_t d = 0; d < depth; d++)
            step(p, d);
    }
    if (options.print_finaltableau) print_tableau(tableau, depth, reversed);
    if (options.print_finalstate) print_paulis(tableau, depth, reversed);
}

void Simulator::simulate() {
    // Create tableau(s) in GPU memory.
    Power power;
    timer.start();
    num_partitions = tableau.alloc(num_qubits, winfo.max_window_bytes, false, measuring, true);
    if (measuring) {
        inv_tableau.alloc(num_qubits, winfo.max_window_bytes, false, measuring, false);
        prefix.alloc(tableau, config_qubits, winfo.max_window_bytes);
        commutations = gpu_allocator.allocate<Commutation>(num_qubits);
    }
    const size_t num_qubits_per_partition = num_partitions > 1 ? tableau.num_words_major() * WORD_BITS : num_qubits;
    gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
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