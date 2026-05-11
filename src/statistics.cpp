#include "simulator.hpp" 

using namespace QuaSARQ;

#define GATE2STATISTIC(GATE) \
	LOG1(" %s %-16s  %13s %s%-12zd (%%%-3.0f)%s", \
			CREPORT, #GATE, ":", CREPORTVAL, stats.circuit.gate_stats.types[GATE], \
			percent((double)stats.circuit.gate_stats.types[GATE], stats.circuit.num_gates), CNORMAL);	\

void Simulator::report()
{
	if (options.profile) {
		const double total = stats.profile.total();
		stats.profile.percentage.gaterules = percent(stats.profile.time.gaterules, total);
		stats.profile.percentage.transpose = percent(stats.profile.time.transpose, total);
		stats.profile.percentage.maxrandom = percent(stats.profile.time.maxrandom, total);
		stats.profile.percentage.compactpivots = percent(stats.profile.time.compactpivots, total);
		stats.profile.percentage.injectswap = percent(stats.profile.time.injectswap, total);
		stats.profile.percentage.injectx = percent(stats.profile.time.injectx, total);
		stats.profile.percentage.injectcx = percent(stats.profile.time.injectcx, total);
    }
	if (options.report_en) {
		LOGHEADER(1, 4, "Statistics");
		LOG1(" %sInitial time                   : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.initial / 1000.0, CNORMAL);
		LOG1(" %sSchedule time                  : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.schedule / 1000.0, CNORMAL);
		if (options.sync)
			LOG1(" %sTransfer time                  : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.transfer / 1000.0, CNORMAL);
		LOG1(" %sSimulation time                : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.simulation / 1000.0, CNORMAL);
		if (options.profile) {
			LOG1(" %s %-30s: %s%-12.3f  msec (%%%-3.0f)%s", CREPORT, "Gate Rules", CREPORTVAL, stats.profile.time.gaterules, stats.profile.percentage.gaterules, CNORMAL);
			LOG1(" %s %-30s: %s%-12.3f  msec (%%%-3.0f)%s", CREPORT, "Transpose", CREPORTVAL, stats.profile.time.transpose, stats.profile.percentage.transpose, CNORMAL);
			LOG1(" %s %-30s: %s%-12.3f  msec (%%%-3.0f)%s", CREPORT, "All Random Measures", CREPORTVAL, stats.profile.time.maxrandom, stats.profile.percentage.maxrandom, CNORMAL);
			LOG1(" %s %-30s: %s%-12.3f  msec (%%%-3.0f)%s", CREPORT, "Compact Pivots", CREPORTVAL, stats.profile.time.compactpivots, stats.profile.percentage.compactpivots, CNORMAL);
			LOG1(" %s %-30s: %s%-12.3f  msec (%%%-3.0f)%s", CREPORT, "Inject CX", CREPORTVAL, stats.profile.time.injectcx, stats.profile.percentage.injectcx, CNORMAL);
			LOG1(" %s %-30s: %s%-12.3f  msec (%%%-3.0f)%s", CREPORT, "Inject Swap", CREPORTVAL, stats.profile.time.injectswap, stats.profile.percentage.injectswap, CNORMAL);
			LOG1(" %s %-30s: %s%-12.3f  msec (%%%-3.0f)%s", CREPORT, "Inject X", CREPORTVAL, stats.profile.time.injectx, stats.profile.percentage.injectx, CNORMAL);
		}
		LOG1(" %sPower consumption              : %s%-12.3f  watt%s", CREPORT, CREPORTVAL, stats.power.wattage, CNORMAL);
		LOG1(" %sEnergy consumption             : %s%-12.3f  joules%s", CREPORT, CREPORTVAL, stats.power.joules, CNORMAL);
		string WORD_SIZE_STR = "64 bits";
		#if defined(WORD_SIZE_8)
		WORD_SIZE_STR = "8 bits";
		#elif defined(WORD_SIZE_32)
		WORD_SIZE_STR = "32 bits";
		#endif
		LOG1(" %sTableau word size              : %s%-12s%s", CREPORT, CREPORTVAL, WORD_SIZE_STR.c_str(), CNORMAL);
		LOG1(" %sTableau partitions             : %s%-12zd%s", CREPORT, CREPORTVAL, num_partitions, CNORMAL);
		LOG1(" %sTableau memory                 : %s%-12.3f  GB%s", CREPORT, CREPORTVAL, stats.tableau.count * stats.tableau.gigabytes, CNORMAL);
		LOG1(" %sTableau step speed             : %s%-12.3f  GB/sec%s", CREPORT, CREPORTVAL, stats.tableau.speed, CNORMAL);
		LOG1(" %sTableau initial states         : %s%-12zd%s", CREPORT, CREPORTVAL, stats.tableau.istates, CNORMAL);
		const double tableau_gb   = stats.tableau.count * stats.tableau.gigabytes;
		const size_t noise_bytes  = gpu_circuit.max_noise_gates() * (sizeof(curand_algorithm_t) + sizeof(uint32));
		const size_t win_gpu_bytes = winfo.max_window_bytes;
		const size_t win_cpu_bytes = winfo.max_window_bytes;
		const size_t rec_bytes    = stats.circuit.measure_stats.count * sizeof(bool);
		const size_t pool_used    = gpu_allocator.gpu_used();
		const size_t pool_cap     = gpu_allocator.gpu_capacity();
		const size_t cpu_used_b   = gpu_allocator.cpu_used();
		const size_t cpu_cap_b    = gpu_allocator.cpu_capacity();
		const double pct_gpu      = pool_cap ? percent((double)pool_used, (double)pool_cap) : 0.0;
		LOG1(" %sGPU pool capacity              : %s%-12.3f  GB%s", CREPORT, CREPORTVAL, ratio((double)pool_cap,   double(GB)), CNORMAL);
		LOG1(" %sGPU pool used                  : %s%-12.3f  GB  (%.0f%%)%s",
			CREPORT, CREPORTVAL, ratio((double)pool_used, double(GB)), pct_gpu, CNORMAL);
		LOG1(" %s  Tableau                      : %s%-12.3f  GB%s", CREPORT, CREPORTVAL, tableau_gb, CNORMAL);
		LOG1(" %s  Noise (states + Paulis)      : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, ratio((double)noise_bytes,   double(MB)), CNORMAL);
		LOG1(" %s  Circuit window (GPU)         : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, ratio((double)win_gpu_bytes, double(MB)), CNORMAL);
		if (rec_bytes)
			LOG1(" %s  Recorder                     : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, ratio((double)rec_bytes, double(MB)), CNORMAL);
		LOG1(" %sCPU pinned capacity            : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, ratio((double)cpu_cap_b,    double(MB)), CNORMAL);
		LOG1(" %sCPU pinned used                : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, ratio((double)cpu_used_b,   double(MB)), CNORMAL);
		LOG1(" %s  Circuit window (CPU)         : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, ratio((double)win_cpu_bytes, double(MB)), CNORMAL);
		LOG1(" %sCircuit depth                  : %s%-12u%s", CREPORT, CREPORTVAL, depth, CNORMAL);
		LOG1(" %sCircuit qubits                 : %s%-12zd%s", CREPORT, CREPORTVAL, num_qubits, CNORMAL);
		double circuit_mb = ratio((double)stats.circuit.bytes, double(MB));
		LOG1(" %sCircuit memory                 : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, circuit_mb, CNORMAL);
		LOG1(" %sMaximum parallel gates         : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.max_parallel_gates, CNORMAL);
		LOG1(" %sMeasurement depth              : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.measure_stats.depth, CNORMAL);
		LOG1(" %sTotal measurements             : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.measure_stats.count, CNORMAL);
		LOG1(" %s  Random                       : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.measure_stats.random, CNORMAL);
		LOG1(" %s  Definite                     : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.measure_stats.definite, CNORMAL);
		if (!circuit_io.observables.empty())
			LOG1(" %sObservables                    : %s%-12u%s", CREPORT, CREPORTVAL, circuit_io.observables.num_observables(), CNORMAL);
		if (!circuit_io.detectors.empty())
			LOG1(" %sDetectors                      : %s%-12u%s", CREPORT, CREPORTVAL, circuit_io.detectors.num_detectors(), CNORMAL);
		LOG1(" %sClifford gates                 : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.num_gates, CNORMAL);
		FOREACH_GATE(GATE2STATISTIC);
	}
	if (options.quiet_en) {
		PRINT("%-30s : %-12zd\n", "Qubits",   num_qubits);
		PRINT("%-30s : %-12u\n",  "Depth",    depth);
		PRINT("%-30s : %-12zd\n", "Gates",    stats.circuit.num_gates);
		PRINT("%-30s : %-12zd\n", "Measures", stats.circuit.measure_stats.random + stats.circuit.measure_stats.definite);
		PRINT("%-30s : %-12.3f  sec\n", "Time", stats.time.total() / 1000.0);
		if (options.profile) {
			PRINT(" %-30s: %-12.3f  msec (%%%-3.0f)\n", "Gate Rules", stats.profile.time.gaterules, stats.profile.percentage.gaterules);
			PRINT(" %-30s: %-12.3f  msec (%%%-3.0f)\n", "Transpose", stats.profile.time.transpose, stats.profile.percentage.transpose);
			PRINT(" %-30s: %-12.3f  msec (%%%-3.0f)\n", "All Random Measures", stats.profile.time.maxrandom, stats.profile.percentage.maxrandom);
			PRINT(" %-30s: %-12.3f  msec (%%%-3.0f)\n", "Compact Pivots", stats.profile.time.compactpivots, stats.profile.percentage.compactpivots);
			PRINT(" %-30s: %-12.3f  msec (%%%-3.0f)\n", "Inject CX", stats.profile.time.injectcx, stats.profile.percentage.injectcx);
			PRINT(" %-30s: %-12.3f  msec (%%%-3.0f)\n", "Inject Swap", stats.profile.time.injectswap, stats.profile.percentage.injectswap);
			PRINT(" %-30s: %-12.3f  msec (%%%-3.0f)\n", "Inject X", stats.profile.time.injectx, stats.profile.percentage.injectx);
		}
		PRINT("%-30s : %-12.3f  GB\n", "Memory", ratio((double)gpu_allocator.gpu_used(), double(GB)));
		PRINT("%-30s : %-12.3f  joules\n", "Energy", stats.power.joules);
	}
}