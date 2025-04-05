#include "simulator.hpp" 

using namespace QuaSARQ;

#define GATE2STATISTIC(GATE) \
	LOG1(" %s %-5s  %24s %s%-12zd (%%%-3.0f)%s", \
			CREPORT, #GATE, ":", CREPORTVAL, stats.circuit.gate_stats.types[GATE], \
			percent((double)stats.circuit.gate_stats.types[GATE], stats.circuit.num_gates), CNORMAL);	\

void Simulator::report()
{
	if (options.report_en) {
		LOGHEADER(0, 4, "Statistics");
		LOG1(" %sInitial time                   : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.initial / 1000.0, CNORMAL);
		LOG1(" %sSchedule time                  : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.schedule / 1000.0, CNORMAL);
		if (options.sync)
			LOG1(" %sTransfer time                  : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.transfer / 1000.0, CNORMAL);
		LOG1(" %sSimulation time                : %s%-12.3f  sec%s", CREPORT, CREPORTVAL, stats.time.simulation / 1000.0, CNORMAL);
		LOG1(" %sPower consumption              : %s%-12.3f  watt%s", CREPORT, CREPORTVAL, stats.power.wattage, CNORMAL);
		LOG1(" %sEnergy consumption             : %s%-12.3f  joules%s", CREPORT, CREPORTVAL, stats.power.joules, CNORMAL);
		string INTERLEAVING_STR = "disabled";
		string WINTERLEAVING_STR = "";
		#ifdef INTERLEAVE_WORDS
		WINTERLEAVING_STR = "(mixed words)";
		#endif
		#ifdef INTERLEAVE_XZ
		INTERLEAVING_STR = to_string(INTERLEAVE_COLS) + "-way";
		#endif
		LOG1(" %sTableau interleaving           : %s%-12s  %s%s", CREPORT, CREPORTVAL, INTERLEAVING_STR.c_str(), WINTERLEAVING_STR.c_str(), CNORMAL);
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
}