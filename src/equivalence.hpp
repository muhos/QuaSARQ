#pragma once

#include "simulator.hpp"
#include "definitions.cuh"

namespace QuaSARQ {

	class Equivalence : public Simulator {

		// Used for equivalence of other circuit. Initialised here so every constructor gets the
		// same starting state without restating it.
		size_t                          other_num_qubits = options.num_qubits;
        size_t                          other_num_partitions = 1;
        depth_t                         other_depth = depth_t(options.depth);
		Circuit 						other_circuit{MB};
		Tableau        					other_tableau{gpu_allocator};
        DeviceCircuit 					other_gpu_circuit{gpu_allocator};
		Statistics                      other_stats;
		cudaStream_t*                   other_custreams = nullptr;
		WindowInfo                      other_winfo;
		string 							ogate, rgate;
		char							failed_state = '?';
		bool                            last_equivalent = false;

	protected:

		virtual void print_result(const bool& equivalent, const char& failed_state) const;

		void open_other();
		void adopt_other(const size_t& qubits);
		void finish_other();

	public:

		Equivalence();
		Equivalence(const string& path_to_circuit, const string& path_to_other = "");
		Equivalence(char* circuit, const size_t& length, char* other, const size_t& other_length);
		~Equivalence() { }

		void report(const bool& equivalent);
		bool is_equivalent() const { return last_equivalent; }

		void inject_faulty();

		// Check if two circuits are equivalent.
		void check();
		bool check(const size_t& p, const cudaStream_t* streams, const cudaStream_t* other_streams, const char& state);
		bool check(const InitialState& initstate, const size_t& num_qubits_per_partition, const size_t& other_num_qubits_per_partition);

	};

}
