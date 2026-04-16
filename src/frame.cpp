
#include "frame.hpp"
#include "print.cuh"
#include "power.cuh"
#include "identitycheck.cuh"

using namespace QuaSARQ;

Framing::Framing(const string& path, const size_t& num_shots) :
    Simulator(path)
    , num_shots(num_shots)
{ }

void Framing::sample() {
    Power power;
    timer.start();
    // Create and randomize tableau in GPU memory.
    num_partitions = tableau.alloc(num_qubits, num_shots, winfo.max_window_bytes, false, false, false);
    tableau.reset_xtable();
    randomize(tableau.zdata(), tableau.num_words_per_table(), 0);
    gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
    timer.stop();
    stats.time.initial += timer.elapsed();
    // Start step-wise simulation.
    timer.start();
    LOGHEADER(1, 4, "Simulation");
    if (options.progress_en) print_progress_header();
    samples_record.alloc(tableau, gpu_allocator);
    gpu_circuit.reset_circuit_offset(0);
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
    if (options.quiet_en) {
        PRINT("%-30s : %-12zd\n", "Shots", num_shots);
    }
    Simulator::report();
}