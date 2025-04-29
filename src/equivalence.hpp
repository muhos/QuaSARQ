#pragma once

#include "simulator.hpp"
#include "definitions.cuh"

namespace QuaSARQ {

	class Equivalence : public Simulator {

		// Used for equivalence of other circuit.
		size_t                          other_num_qubits;
        size_t                          other_num_partitions;
        depth_t                         other_depth;
		Circuit 						other_circuit;
		Tableau        					other_tableau;
        DeviceCircuit 					other_gpu_circuit;
		Statistics                      other_stats;
		cudaStream_t*                   other_custreams;
		WindowInfo                      other_wininfo;
		string 							ogate, rgate;
		char							failed_state;

	public:

		Equivalence();
		Equivalence(const string& path_to_circuit, const string& path_to_other = "");
		~Equivalence() { }

		void report(const bool& equivalent);

		void inject_faulty();

		// Check if two circuits are equivalent.
		void check();
		bool check(const size_t& p, const cudaStream_t* streams, const cudaStream_t* other_streams);
		bool check(const InitialState& initstate, const size_t& num_qubits_per_partition, const size_t& other_num_qubits_per_partition);

	};

}