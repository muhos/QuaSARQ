
#include "simulator.hpp"
#include "control.hpp"
#include "power.cuh"
#include "identitycheck.cuh"

using namespace QuaSARQ;

bool Simulator::timeout = false;

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
	, crand(1)
    , mrand(1)
	, circuit(MB)
	, circuit_mode(RANDOM_CIRCUIT)
	, gpu_circuit(gpu_allocator)
    , locker(gpu_allocator)
    , tableau(gpu_allocator)
    , inv_tableau(gpu_allocator)
    , ref_tableau(gpu_allocator)
    , pivoting(gpu_allocator)
    , recorder(gpu_allocator)
    , prefix(gpu_allocator, mchecker)
	, config_file(nullptr)
    , state_file(nullptr)
    , config_qubits(0)
	, custreams(nullptr)
    , copy_streams{ 0, 0, 0, 0 }
    , kernel_streams{ 0, 0 }
    , measuring(false)
    , write_measures_to_file(false)
    , reference_mode(false)
{
    initialize();
}

Simulator::Simulator(const string& path) :
    num_qubits(options.num_qubits)
    , num_partitions(1)
    , depth((depth_t)options.depth)
    , crand(1)
    , mrand(1)
    , circuit(MB)
    , circuit_path(path)
    , circuit_mode(PARSED_CIRCUIT)
    , gpu_circuit(gpu_allocator)
    , locker(gpu_allocator)
    , tableau(gpu_allocator)
    , inv_tableau(gpu_allocator)
    , ref_tableau(gpu_allocator)
    , pivoting(gpu_allocator)
    , recorder(gpu_allocator)
    , prefix(gpu_allocator, mchecker)
    , config_file(nullptr)
    , state_file(nullptr)
    , config_qubits(0)
    , custreams(nullptr)
    , copy_streams{ 0, 0, 0, 0 }
    , kernel_streams{ 0, 0 }
    , measuring(false)
    , write_measures_to_file(false)
    , reference_mode(false)
{
    initialize();
}

void Simulator::reserve() {
    const size_t circuit_obs_dets_bytes = circuit_mode == RANDOM_CIRCUIT ? 0 :
                                          circuit_io.observables.bytes() + circuit_io.detectors.bytes();
    // Creating GPU pool (device memory)
    size_t gfree = 0, gtot = 0;
    cudaMemGetInfo(&gfree, &gtot);
    static constexpr size_t CUARENA_GPU_PENALTY = 256 * MB;
    static constexpr size_t MIN_DYNAMIC = 128 * MB;
    const size_t pool_est = (gfree > CUARENA_GPU_PENALTY) ? gfree - CUARENA_GPU_PENALTY : 0;
    const size_t stable_bytes = circuit_obs_dets_bytes + 
                                (pool_est > MIN_DYNAMIC) ? pool_est - MIN_DYNAMIC : 0;
    gpu_allocator.create_gpu_pool(0, cuArena::GPUMemoryType::Device, 0, stable_bytes);
    // Creating CPU pool (pinned memory)
    const size_t sample_host_bytes = options.print_sample || options.print_sample_qubits || options.check_measurement ?
        get_num_words(stats.circuit.measure_stats.count) * WORD_BITS * get_num_words(options.num_shots) * sizeof(word_t) : 0;
    const size_t detector_bitstring_bytes = options.print_detector ?
        circuit_io.detectors.starts.size() * options.num_shots * sizeof(char) : 0;
    const bool eval_observables = options.print_observable || !circuit_io.observables.empty();
    const size_t observable_bitstring_bytes = eval_observables ?
        circuit_io.observables.ids.size() * options.num_shots * sizeof(char) : 0;
    const size_t bitstring_bytes = circuit_mode == RANDOM_CIRCUIT ? 0 :
        sample_host_bytes + detector_bitstring_bytes + observable_bitstring_bytes;
    const size_t pinned_bytes =
        sizeof(Table) * 3 +
        winfo.max_window_bytes +
        (num_qubits + 2) * sizeof(pivot_t) +
        KB * gpu_allocator.alignment() +
        circuit_obs_dets_bytes +
        circuit_io.observables.pinned.num_observables +
        bitstring_bytes;
    gpu_allocator.create_cpu_pool(pinned_bytes);
    if (circuit_mode == PARSED_CIRCUIT) {
        alloc_observables();
        alloc_detectors();
    }
}

void Simulator::initialize() {
    LOGHEADER(1, 4, "Build");
    getCPUInfo(1);
    getGPUInfo(1);
    LOGHEADER(1, 4, "Initial");
    parse();
    reserve();
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
        for (int i = 0; i < NUM_COPY_STREAMS; i++) 
            copy_streams[i] = streams[i];
        for (int i = 0; i < NUM_COMPUTE_STREAMS; i++) 
            kernel_streams[i] = streams[NUM_COPY_STREAMS + i];
        LOGDONE(1, 4);
    }
}

void Simulator::rsample() {
    if (!measuring || !stats.circuit.measure_stats.count) return;
    reference_mode = true;
    // Disable checking during reference run — mchecker is not allocated here.
    const bool saved_check = options.check_measurement;
    options.check_measurement = false;
    tableau.swap_tableaus(ref_tableau);
    num_partitions = tableau.alloc(num_qubits, 0, winfo.max_window_bytes, false, measuring, true);
    #if ROW_MAJOR
    inv_tableau.alloc(num_qubits, 0, 0, false, measuring, false);
    #endif
    prefix.alloc(tableau, config_qubits);
    pivoting.alloc(num_qubits);
    recorder.alloc(stats.circuit.measure_stats.count);
    const size_t num_qubits_per_partition = num_partitions > 1 ? tableau.num_words_major() * WORD_BITS : num_qubits;
    gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
    for (size_t p = 0; p < num_partitions && !timeout; p++) {
        const size_t prev_num_qubits = num_qubits_per_partition * p;
        assert(prev_num_qubits < num_qubits);
        identity(tableau, prev_num_qubits,
                 (p == num_partitions - 1) ? (num_qubits - prev_num_qubits) : num_qubits_per_partition,
                 custreams, options.initialstate);
        simulate(p, false);
    }
    SYNCALL;
    tableau.swap_tableaus(ref_tableau);
    options.check_measurement = saved_check;
    reference_mode = false;
}

// Simulate circuit by moving window
// over parallel gates of each depth level.
void Simulator::simulate(const size_t& p, const bool& reversed) {
    if (!tableau.size()) {
        LOGERRORN("cannot run simulation without allocating the tableau.");
        throw tableau_memory_error();
    }
    if (reference_mode) LOGHEADER(1, 4, "Simulation (reference mode)");
    else LOGHEADER(1, 4, "Simulation");
    if (options.progress_en) print_progress_header();
    if (reversed) {
        gpu_circuit.reset_circuit_offset(circuit.reference(depth - 1, 0));
        for (depth_t d = 0; d < depth && !timeout; d++)
            step(p, depth - d - 1, true);
    }
    else {
        gpu_circuit.reset_circuit_offset(0);
        for (depth_t d = 0; d < depth && !timeout; d++)
            step(p, d);
    }
    if (options.print_finaltableau) print_tableau(tableau, depth, reversed);
    if (options.print_finalstate) print_paulis(tableau, depth, reversed);
}

void check_simulate(Simulator& sim, const size_t& p, const size_t& prev_num_qubits, const size_t& num_qubits, const bool& timeout) {
    Tableau& tableau = sim.get_tableau();
    Circuit& circuit = sim.get_circuit();
    DeviceCircuit& gpu_circuit = sim.get_gpu_circuit();
    if (!tableau.size()) {
        LOGERRORN("cannot run simulation without allocating the tableau.");
        throw tableau_memory_error();
    }
    LOGHEADER(1, 4, "Checking Simulation");
    if (options.progress_en) sim.print_progress_header();
    const depth_t max_depth = circuit.depth();
    depth_t start_depth = 0, end_depth = 1;
    while (start_depth < end_depth && end_depth < max_depth && !timeout) {
        if (circuit.is_measuring(start_depth)) {
            start_depth++;
            end_depth++;
            continue;
        }
        gpu_circuit.reset_circuit_offset(circuit.reference(start_depth, 0));
        for (depth_t d = start_depth; d < end_depth && !timeout; d++)
            if (!circuit.is_measuring(d)) 
                sim.step(p, d);
        gpu_circuit.reset_circuit_offset(circuit.reference(end_depth - 1, 0));
        for (depth_t d = start_depth; d < end_depth && !timeout; d++)
            if (!circuit.is_measuring(d)) 
                sim.step(p, end_depth - d - 1, true);
        
        LOGN2(2, "  checking tableau integrity from %d up to %d depth levels.. ", start_depth, end_depth);
        const bool passed = check_identity(tableau, prev_num_qubits, num_qubits, sim.is_measuring());
        if (passed) LOGPASSED(2);
        else LOG2(2, "%sFAILED.%s", CRED, CNORMAL);
        if (options.progress_en) sim.print_progress(p, end_depth - 1, passed);
        end_depth++;
    }
}

void Simulator::simulate() {
    Power power;
    timer.start();
    num_partitions = tableau.alloc(num_qubits, 0, winfo.max_window_bytes, false, measuring, true);
    if (measuring) {
        #if ROW_MAJOR
        inv_tableau.alloc(num_qubits, 0, 0, false, measuring, false);
        #endif
        prefix.alloc(tableau, config_qubits);
        pivoting.alloc(num_qubits);
        if (!stats.circuit.measure_stats.count)
            LOGERRORN("cannot run simulation with measurement gates but no measurements.");
        recorder.alloc(stats.circuit.measure_stats.count);
        if (options.check_measurement)
            mchecker.alloc(num_qubits);
    }
    const size_t num_qubits_per_partition = num_partitions > 1 ? tableau.num_words_major() * WORD_BITS : num_qubits;
    gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
    if (!reference_mode)
        gpu_circuit.init_noise_states(options.seed, winfo.max_parallel_gates, kernel_streams[0]);
    timer.stop();
    stats.time.initial += timer.elapsed();
    timer.start();
    copy_detectors(copy_streams[2]);
    copy_observables(copy_streams[3]);
    for (size_t p = 0; p < num_partitions && !timeout; p++) {
        const size_t prev_num_qubits = num_qubits_per_partition * p;
        assert(prev_num_qubits < num_qubits);
        LOGN2(1, "Partition %zd: ", p);
        identity(tableau, prev_num_qubits, 
            (p == num_partitions - 1) ? (num_qubits - prev_num_qubits) : num_qubits_per_partition, 
            custreams, options.initialstate);
        if (options.check_tableau)
            check_simulate(*this, p, prev_num_qubits, num_qubits_per_partition, timeout);
        else
            simulate(p, false);
    }
	SYNCALL;
	timer.stop();
	stats.time.simulation = timer.elapsed();
    stats.power.wattage = power.measure();
    stats.power.joules = stats.power.wattage * (stats.time.simulation / 1000.0);
    stats.tableau.count = stats.tableau.istates = 1;
    stats.tableau.gigabytes = ratio((double)tableau.size() * sizeof(word_std_t), double(GB));
    stats.tableau.seconds = (stats.time.simulation / 1000.0) / (num_partitions * depth);
    stats.tableau.calc_speed();
    if (!reference_mode) {
        if ((options.print_observable && !circuit_io.observables.empty()) ||
            (options.print_detector  && !circuit_io.detectors.empty()))
            LOGHEADER(1, 4, "Results");
        print_observables();
        print_detectors();
        report();
    }
}
