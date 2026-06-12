
#include "frame.cuh"
#include "print.cuh"
#include "power.cuh"
#include "identitycheck.cuh"

using namespace QuaSARQ;

Framing::Framing(const string& path, const size_t& num_shots) :
    Simulator(path)
    , num_shots(num_shots)
    , total_shots(num_shots)
    , chunk_start(0)
    , chunk_index(0)
    , max_chunk_shots(num_shots)
    , measurement_offset(0)
    , rand_states(nullptr)
    , rand_states_size(0)
{ 
    write_measures_to_file |= (num_shots > options.min_shots_write);
}

size_t Framing::sample_device_bytes() const {
    return samples_record.device_bytes();
}

size_t Framing::sample_host_bytes() const {
    return samples_record.host_bytes();
}

size_t Framing::choose_chunk_shots() const {
    if (!total_shots) return 0;
    if (options.chunk_shots > 0) {
        size_t chunk = MIN(size_t(options.chunk_shots), total_shots);
        if (chunk < total_shots)
            chunk = MAX(size_t(WORD_BITS), (chunk / WORD_BITS) * size_t(WORD_BITS));
        return chunk;
    }
    const size_t measure_words = get_num_words(stats.circuit.measure_stats.count);
    const size_t qubit_words = get_num_words(num_qubits);
    const size_t usable = gpu_stable_avail(gpu_allocator);
    const size_t safety = MIN(usable / 5, size_t(512) * MB);
    const size_t reserved = winfo.max_window_bytes + sizeof(Table) * 2;
    const size_t available = usable > safety + reserved ? usable - safety - reserved : 0;
    const size_t rand_state_rows = MAX(get_num_padded_bits(num_qubits), winfo.max_parallel_gates);
    const size_t bytes_per_shot_word =
        (measure_words + 2 * qubit_words) * sizeof(word_std_t) * WORD_BITS
        + rand_state_rows * sizeof(curand_algorithm_t);
    if (!bytes_per_shot_word) return total_shots;
    const size_t chunk_words = available / bytes_per_shot_word;
    size_t chunk = chunk_words ? chunk_words * WORD_BITS : 1;
    chunk = MIN(chunk, total_shots);
    return chunk;
}

void Framing::sample() {
    Power power;
    timer.start();
    rsample();
    timer.stop();
    const double reference_sim_time = timer.elapsed();
    LOGRULER(1, '-', RULERLEN);
    max_chunk_shots = choose_chunk_shots();
    stats.sampling.requested_shots = total_shots;
    stats.sampling.chunk_shots = max_chunk_shots;
    stats.sampling.chunks = ROUNDUP(total_shots, max_chunk_shots);
    stats.logical.shots_with_error = 0;
    stats.logical.total_shots = 0;
    stats.logical.num_observables = circuit_io.observables.pinned.num_observables;
    stats.logical.total_observable_errors = 0;
    const Measures scheduled_measures = stats.circuit.measure_stats;
    timer.start();
    gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
    copy_detectors(copy_streams[2]);
    copy_observables(copy_streams[3]);
    const size_t total_words_minor = get_num_words(total_shots);
    for (chunk_start = 0, chunk_index = 0; chunk_start < total_shots && !timeout; chunk_start += num_shots, chunk_index++) {
        num_shots = MIN(max_chunk_shots, total_shots - chunk_start);
        const size_t chunk_word_offset = WORD_OFFSET(chunk_start);
        const uint64 sample_seed = options.seed ^ 0x9e3779b97f4a7c15ULL;
        num_partitions = tableau.alloc(num_qubits, num_shots, winfo.max_window_bytes, false, false, false, 0, "frame ");
        if (options.check_measurement) {
            mchecker.alloc(num_qubits, tableau.num_words_minor());
            mchecker.record.resize(stats.circuit.measure_stats.count);
            mchecker.samples.resize(stats.circuit.measure_stats.count * tableau.num_words_minor(), word_std_t(0));
        }
        tableau.reset_xtable();
        init_rand_states(sample_seed, tableau.num_words_per_table(), total_words_minor, chunk_word_offset, kernel_streams[1]);
        randomize(tableau.zdata(), tableau.num_words_per_table(), kernel_streams[1]);
        samples_record.alloc(stats.circuit.measure_stats.count, tableau.num_words_minor(), gpu_allocator, kernel_streams[0]);
        stats.sampling.sample_device_bytes = MAX(stats.sampling.sample_device_bytes, samples_record.device_bytes());
        stats.sampling.sample_host_bytes = MAX(stats.sampling.sample_host_bytes, samples_record.host_bytes());
        gpu_circuit.reset_circuit_offset(0);
        measurement_offset = 0;
        SYNCALL;
        if (!chunk_index) {
            timer.stop();
            stats.time.initial += timer.elapsed();
            timer.start();
            LOGHEADER(1, 4, "Sampling");
            if (options.progress_en && ref_tableau.empty()) print_progress_header();
        }
        for (depth_t d = 0; d < depth && !timeout; d++)
            step(d);
        SYNCALL;
        if (options.print_finaltableau) print_tableau(tableau, depth);
        print(kernel_streams[0]);
        samples_record.destroy(gpu_allocator);
        tableau.destroy();
    }
    recorder.destroy();
    gpu_allocator.deallocate<curand_algorithm_t>(rand_states);
    rand_states = nullptr;
    rand_states_size = 0;
    num_shots = total_shots;
    stats.circuit.measure_stats = scheduled_measures;
	timer.stop();
	stats.time.simulation = reference_sim_time + timer.elapsed();
    stats.power.wattage = power.measure();
    stats.power.joules = stats.power.wattage * (stats.time.simulation / 1000.0);
    stats.tableau.count = stats.tableau.istates = 1;
    stats.tableau.gigabytes = ratio(double(get_num_words(num_qubits)) * double(get_num_padded_bits(max_chunk_shots)) * 2.0 * sizeof(word_std_t), double(GB));
    stats.tableau.seconds = (stats.time.simulation / 1000.0) / (num_partitions * depth);
    stats.tableau.calc_speed();
    report();
}
