
#ifndef __CHECKER_H
#define __CHECKER_H

#include "simulator.hpp"
#include "definitions.cuh"

namespace QuaSARQ {

	class Checker : public Simulator {

	public:

		Checker();
		Checker(const string& path);
		~Checker() { }

		// Check if:
        // type 0: Z table is identity.
        // type 1: X table is identity.
        // type other: both tables are identity.
        bool check_identity(const size_t& offset_per_partition, const size_t& num_qubits_per_partition);
		void check_parallel_gates();
		void check_random_circuit();
		void check_integrity();
		void run();

	};

}

#endif 
