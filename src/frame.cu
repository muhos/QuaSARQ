
#include "frame.cuh"
#include "print.cuh"
#include "power.cuh"
#include "identitycheck.cuh"

using namespace QuaSARQ;

Framing::Framing(const string& path, const size_t& num_shots) :
    Simulator(path)
    , num_shots(num_shots)
    , rand_states(nullptr)
    , rand_states_size(0)
{ 
    write_measures_to_file |= (num_shots > options.min_shots_write);
}

void Framing::sample() {
    Power power;
    timer.start();
    num_partitions = tableau.alloc(num_qubits, num_shots, winfo.max_window_bytes, false, false, false);
    const size_t frame_num_partitions = num_partitions;
    rsample();
    num_partitions = frame_num_partitions;
    if (options.check_measurement) {
        mchecker.record.resize(stats.circuit.measure_stats.count);
        mchecker.samples.resize(stats.circuit.measure_stats.count * tableau.num_words_minor(), word_std_t(0));
    }
    tableau.reset_xtable();
    gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
    gpu_circuit.init_noise_states(options.seed, winfo.max_parallel_gates, kernel_streams[0]);
    init_rand_states(options.seed ^ 0x9e3779b97f4a7c15ULL, tableau.num_words_per_table(), kernel_streams[1]);
    randomize(tableau.zdata(), tableau.num_words_per_table(), kernel_streams[1]);
    SYNCALL;
    timer.stop();
    stats.time.initial += timer.elapsed();
    timer.start();
    LOGHEADER(1, 4, "Sampling");
    if (options.progress_en && ref_tableau.empty()) print_progress_header();
    samples_record.alloc(stats.circuit.measure_stats.count, tableau.num_words_minor(), gpu_allocator);
    gpu_circuit.reset_circuit_offset(0);
    measurement_offset = 0;
    copy_detectors(copy_streams[2]);
    copy_observables(copy_streams[3]);
    for (depth_t d = 0; d < depth && !timeout; d++)
        step(d);
	SYNCALL;
	timer.stop();
    if (options.print_finaltableau) print_tableau(tableau, depth);
	stats.time.simulation = timer.elapsed();
    stats.power.wattage = power.measure();
    stats.power.joules = stats.power.wattage * (stats.time.simulation / 1000.0);
    stats.tableau.count = stats.tableau.istates = 1;
    stats.tableau.gigabytes = ratio((double)tableau.size() * sizeof(word_std_t), double(GB));
    stats.tableau.seconds = (stats.time.simulation / 1000.0) / (num_partitions * depth);
    stats.tableau.calc_speed();
    print();
    report();
}

void Framing::report() {
    if (options.quiet_en && options.force_report_en) {
        LOGHEADER(0, 4, "Statistics");
        PRINT("%-30s : %-12zd\n", "Shots", num_shots);
    }
    Simulator::report();
}