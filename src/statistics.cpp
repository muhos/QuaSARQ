#include "simulator.hpp" 

using namespace QuaSARQ;

void Simulator::report()
{
	if (options.report_en) {
		LOGHEADER(1, 5, "Statistics");
		LOG1(" %sInitial time                   : %s%-12.3f  msec%s", CREPORT, CREPORTVAL, stats.time.initial, CNORMAL);
		LOG1(" %sSchedule time                  : %s%-12.3f  msec%s", CREPORT, CREPORTVAL, stats.time.schedule, CNORMAL);
		if (options.sync)
			LOG1(" %sTransfer time                  : %s%-12.3f  msec%s", CREPORT, CREPORTVAL, stats.time.transfer, CNORMAL);
		LOG1(" %sSimulation time                : %s%-12.3f  msec%s", CREPORT, CREPORTVAL, stats.time.simulation, CNORMAL);
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
		LOG1(" %sAverage parallel gates         : %s%-12.3f%s", CREPORT, CREPORTVAL, stats.circuit.average_parallel_gates, CNORMAL);
		LOG1(" %sClifford gates                 : %s%-12zd%s", CREPORT, CREPORTVAL, stats.circuit.max_gates, CNORMAL);
		LOG1(" %s X %s%12zd  (%%%-3.0f)%s CX %s%12zd  (%%%-3.0f)%s", 
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[X], 
			percent((double)stats.circuit.gate_stats.types[X], stats.circuit.max_gates),
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[CX], 
			percent((double)stats.circuit.gate_stats.types[CX], stats.circuit.max_gates), CNORMAL);
		LOG1(" %s Y %s%12zd  (%%%-3.0f)%s CZ %s%12zd  (%%%-3.0f)%s",
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[Y],
			percent((double)stats.circuit.gate_stats.types[Y], stats.circuit.max_gates),
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[CZ], 
			percent((double)stats.circuit.gate_stats.types[CZ], stats.circuit.max_gates), CNORMAL);
		LOG1(" %s Z %s%12zd  (%%%-3.0f)%s CY %s%12zd  (%%%-3.0f)%s",
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[Z],
			percent((double)stats.circuit.gate_stats.types[Z], stats.circuit.max_gates),
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[CY], 
			percent((double)stats.circuit.gate_stats.types[CY], stats.circuit.max_gates), CNORMAL);
		LOG1(" %s H %s%12zd  (%%%-3.0f)%s Swap%s%11zd  (%%%-3.0f)%s",
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[H],
			percent((double)stats.circuit.gate_stats.types[H], stats.circuit.max_gates),
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[Swap], 
			percent((double)stats.circuit.gate_stats.types[Swap], stats.circuit.max_gates), CNORMAL);
		LOG1(" %s S %s%12zd  (%%%-3.0f)%s iSwap%s%10zd  (%%%-3.0f)%s",
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[S],
			percent((double)stats.circuit.gate_stats.types[S], stats.circuit.max_gates),
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[iSwap],
			percent((double)stats.circuit.gate_stats.types[iSwap], stats.circuit.max_gates), CNORMAL);
		LOG1(" %s Sdg%s%11zd  (%%%-3.0f)%s ", 
			CREPORT, CREPORTVAL, stats.circuit.gate_stats.types[Sdg], 
			percent((double)stats.circuit.gate_stats.types[Sdg], stats.circuit.max_gates), CNORMAL);		
	}
}