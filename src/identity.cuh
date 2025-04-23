#pragma once

#include "table.cuh"
#include "grid.cuh"

namespace QuaSARQ {

	// Set the tableau into identity.
	__global__ void identity_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs);
	__global__ void identity_Z_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs);
	__global__ void identity_X_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs);
	__global__ void identity_extended_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs);
    __global__ void identity_Z_extended_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs);
    __global__ void identity_X_extended_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs);

	void tune_identity(
		void (*kernel)(
		const 	size_t, 
		const 	size_t, 
				Table*, 
				Table*),
				dim3& 	bestBlock,
				dim3& 	bestGrid,
		const 	size_t& offset,
		const 	size_t& size,
				Table* 	xs,
				Table* 	zs);

}