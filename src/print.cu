#include "simulator.hpp"
#include "print.cuh"

namespace QuaSARQ {

	__global__ void print_tableau_k(const Table* ps, const Signs* ss, const depth_t level, const bool measuring) {
		if (!blockIdx.x && !threadIdx.x) {
			print_tables(*ps, *ss, level == MAX_DEPTH ? -1 : int64(level), measuring);
		}
	}

	__global__ void print_tableau_k(const Table* xs, const Table* zs, const Signs* ss, const depth_t level, const bool measuring) {
		if (!blockIdx.x && !threadIdx.x) {
			print_tables(*xs, *zs, *ss, level == MAX_DEPTH ? -1 : int64(level), measuring);
		}
	}

	__global__ void print_paulis_k(const Table* xs, const Table* zs, const Signs* ss, const size_t num_words_per_column, const size_t num_qubits, const bool extended) {
		if (!blockIdx.x && !threadIdx.x) {
			print_state(*xs, *zs, *ss, 0, num_qubits, num_qubits, num_words_per_column);
			if (extended) {
				REPCH_GPU("-", num_qubits + 1);
				LOGGPU("\n");
				print_state(*xs, *zs, *ss, num_qubits, 2*num_qubits, num_qubits, num_words_per_column);
			}
		}
	}
	

	__global__ void print_paulis_k(const Table* ps, const Signs* ss, const size_t num_words_per_column, const size_t num_qubits, const depth_t level) {
		if (!blockIdx.x && !threadIdx.x) {
			const word_t *words = ps->data();
			for (size_t w = 0; w < num_qubits; w++) {
				const word_t pow2 = BITMASK_GLOBAL(w);
				for (size_t q = 0; q < num_qubits; q++) {
					if (q == 0 && (*ss)[WORD_OFFSET(q)] & sign_t(pow2)) {
						LOGGPU("-");
					}
					else if (q == 0) {
						LOGGPU("+");
					}
					const size_t x_word_idx = X_OFFSET(q) * num_words_per_column + X_WORD_OFFSET(WORD_OFFSET(w));
					const size_t z_word_idx = Z_OFFSET(q) * num_words_per_column + Z_WORD_OFFSET(WORD_OFFSET(w));
					if ((!(words[x_word_idx] & pow2)) && (!(words[z_word_idx] & pow2)))
						LOGGPU("I");
					if ((words[x_word_idx] & pow2) && (!(words[z_word_idx] & pow2)))
						LOGGPU("X");
					if ((words[x_word_idx] & pow2) && (words[z_word_idx] & pow2))
						LOGGPU("Y");
					if ((!(words[x_word_idx] & pow2)) && (words[z_word_idx] & pow2))
						LOGGPU("Z");
				}
				LOGGPU("\n");
			}
		}
	}

	__global__ void print_gates_k(const gate_ref_t* refs, const bucket_t* gates, const gate_ref_t num_gates) {
		if (!blockIdx.x && !threadIdx.x) {
			for (gate_ref_t i = 0; i < num_gates; i++) {
				const gate_ref_t r = refs[i];
				LOGGPU(" Gate(%3d , r:%3d):", i, r);
				const Gate &gate = (Gate &)gates[r];
				gate.print();
			}
		}
	}

	__global__ void print_measurements_k(const gate_ref_t* refs, const bucket_t* measurements, const gate_ref_t num_gates) {
		if (!blockIdx.x && !threadIdx.x) {
			for (gate_ref_t i = 0; i < num_gates; i++) {
				const gate_ref_t r = refs[i];
				const Gate &m = (Gate &)measurements[r];
				LOGGPU(" %8d     %10s    %2c\n", m.wires[0], 
					m.pivot == MAX_QUBITS ? "definite" : "random",  
					m.measurement != UNMEASURED ? char(((m.measurement % 4 + 4) % 4 >> 1) + 48) : 'U');
			}
		}
	}

	void Simulator::print_paulis(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level, const bool& reversed) {
		if (!options.sync) SYNCALL;
		if (depth_level == -1) 
			LOGHEADER(0, 3, "Initial state");
		else if (options.print_step_state)
			LOG2(0, "State after %d-step", depth_level);
		else if (options.print_final_state)
			LOGHEADER(0, 3, "Final state");
		if (num_qubits > 100) {
            LOGWARNING("State is too large to print.");
			fflush(stdout);
		}
        print_paulis_k << <1, 1 >> > (XZ_TABLE(tab), tab.signs(), tab.num_words_per_column(), num_qubits, measuring);
        LASTERR("failed to launch print-paulis kernel");
        SYNCALL;
        fflush(stdout);
	}

	void Simulator::print_tableau(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level, const bool& reversed) {
		if (!options.sync) SYNCALL;
		LOG2(0, "");
		if (depth_level == -1)
			LOG2(0, "Initial tableau before simulation");
		else if (depth_level == depth)
			LOG2(0, "Final tableau after %d %ssimulation steps", depth, reversed ? "reversed " : "");
		else
			LOG2(0, "Tableau after %d-step", depth_level);
        print_tableau_k << <1, 1 >> > (XZ_TABLE(tab), tab.signs(), depth_level, measuring);
        LASTERR("failed to launch print-tableau kernel");
        SYNCALL;
        fflush(stdout);
	}

	void Simulator::print_gates(const DeviceCircuit<DeviceAllocator>& gates, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_gates) return;
		if (!options.sync) SYNCALL;
		LOG2(0, " Gates on GPU for %d-time step:", depth_level);
		print_gates_k << <1, 1 >> > (gates.references(), gates.gates(), num_gates);
		LASTERR("failed to launch print-gates kernel");
		SYNCALL;
		fflush(stdout);
	}

	void Simulator::print_measurements(const DeviceCircuit<DeviceAllocator>& gates, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_measurements) return;
		if (!circuit.is_measuring(depth_level)) return;
		if (!options.sync) SYNCALL;
		LOG2(0, " Measurements on GPU for %d-time step:", depth_level);
		LOG2(0, "%10s   %10s     %5s", "Qubit", "Type", "Outcome");
		print_measurements_k << <1, 1 >> > (gates.references(), gates.gates(), num_gates);
		LASTERR("failed to launch print-gates kernel");
		SYNCALL;
		fflush(stdout);
	}

}

