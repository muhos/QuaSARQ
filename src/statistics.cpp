#include "simulator.hpp" 

using namespace QuaSARQ;

#define GATE2STATISTIC(GATE) \
	LOG1(" %s %-5s  %24s %s%-12zd (%%%-3.0f)%s", \
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
		LOG1(" %sCircuit depth                  : %s%-12u%s", CREPORT, CREPORTVAL, depth, CNORMAL);
		LOG1(" %sCircuit qubits                 : %s%-12zd%s", CREPORT, CREPORTVAL, num_qubits, CNORMAL);
		double circuit_mb = ratio((double)stats.circuit.bytes, double(MB));
		LOG1(" %sCircuit memory                 : %s%-12.3f  MB%s", CREPORT, CREPORTVAL, circuit_mb, CNORMAL);
		LOG1(" %sMaximum parallel gates         : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.max_parallel_gates, CNORMAL);
		LOG1(" %sRandom measurements            : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.measure_stats.random, CNORMAL);
		LOG1(" %sDefinite measurements          : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.measure_stats.definite, CNORMAL);
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
		PRINT("%-30s : %-12.3f  GB\n", "Memory", ratio((double)stats.circuit.bytes, double(GB)) + stats.tableau.count * stats.tableau.gigabytes);
		PRINT("%-30s : %-12.3f  joules\n", "Energy", stats.power.joules);
	}
}