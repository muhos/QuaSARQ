#include "simulator.hpp"
#include "print.cuh"

namespace QuaSARQ {

	__global__ void print_tableau(const Table* ps, const Signs* ss, const depth_t level) {
		if (!blockIdx.x && !threadIdx.x) {
			print_tables(*ps, *ss, level == MAX_DEPTH ? -1 : int64(level));
		}
	}

	__global__ void print_tableau(const Table* xs, const Table* zs, const Signs* ss, const depth_t level) {
		if (!blockIdx.x && !threadIdx.x) {
			print_tables(*xs, *zs, *ss, level == MAX_DEPTH ? -1 : int64(level));
		}
	}

	__global__ void print_gates(const gate_ref_t* refs, const bucket_t* gates, const gate_ref_t num_gates) {
		if (!blockIdx.x && !threadIdx.x) {
			print_gates_per_window(refs, gates, num_gates);
		}
	}

	__global__ void print_signs(const Signs* ss) {
		if (!blockIdx.x && !threadIdx.x) {
			print_signs(*ss);
		}
	}

	void Simulator::print_tableau_step(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level) {
		if (!options.print_tableau_step) return;
		LOG2(1, "");
		LOG2(1, "Tableau after %d-step", depth_level);
		PRINT_TABLEAU(tab, depth_level);
	}

	void Simulator::print_tableau_final(const Tableau<DeviceAllocator>& tab, const bool& reversed) {
		if (!options.print_tableau_final) return;
		LOG2(1, "");
		LOG2(1, "Final tableau after %d %ssimulation steps", depth, reversed ? "reversed " : "");
		PRINT_TABLEAU(tab, depth);
	}

	void Simulator::print_tableau_initial(const Tableau<DeviceAllocator>& tab) {
		if (!options.print_tableau_initial) return;
		LOG2(1, "");
		LOG2(1, "Initial tableau before simulation");
		PRINT_TABLEAU(tab, -1);
	}

	void Simulator::print_gates_step(const DeviceCircuit<DeviceAllocator>& gates, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_gates) return;
		LOG2(1, "");
		LOG2(1, " Gates on GPU for %d-time step:", depth_level);
		if (!options.sync) SYNCALL;
		print_gates << <1, 1 >> > (gates.references(), gates.gates(), num_gates);
		LASTERR("failed to launch print-gates kernel");
		SYNCALL;
		fflush(stdout);
	}

	void Simulator::print_signs_step(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level) {
		LOG2(1, "");
		LOG2(1, " Signs on GPU for %d-time step:", depth_level);
		if (!options.sync) SYNCALL;
		print_signs << <1, 1 >> > (tab.signs());
		LASTERR("failed to launch print-signs kernel");
		SYNCALL;
		fflush(stdout);
	}

}

