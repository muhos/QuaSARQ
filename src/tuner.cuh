
#ifndef __CU_TUNE_H
#define __CU_TUNE_H

#include "timer.cuh"
#include "grid.cuh"
#include "vector.cuh"
#include "tableau.cuh"
#include "datatypes.cuh"

namespace QuaSARQ {


	void tune_kernel(void (*kernel)(const size_t, const size_t, Table*),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& offset, const size_t& size, Table* ps);

	void tune_kernel(void (*kernel)(const size_t, 
		#ifdef INTERLEAVE_XZ
		Table*,
		#else
		Table*, Table*, 
		#endif
		Signs *),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& size, 
		#ifdef INTERLEAVE_XZ
		Table* ps, 
		#else
		Table* xs, Table* zs, 
		#endif
		Signs *ss);

	void tune_kernel(void (*kernel)(ConstRefsPointer, ConstBucketsPointer, const size_t, const size_t, 
		#ifdef INTERLEAVE_XZ
		Table*,
		#else
		Table*, Table*, 
		#endif
		Signs *),
		const char* opname,
		dim3& bestBlock, dim3& bestGrid, 
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		ConstRefsPointer gate_refs, ConstBucketsPointer gate_buckets,
		#ifdef INTERLEAVE_XZ
		Table* ps, 
		#else
		Table* xs, Table* zs, 
		#endif
		Signs *ss);
	
	void tune_kernel_m(void (*kernel)(const size_t, const size_t, Table*, Table*),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& offset, const size_t& size, Table* xs, Table* zs);

	void tune_kernel_m(void (*kernel)(pivot_t*, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		pivot_t* pivots, const size_t size);

	void tune_kernel_m(void (*kernel)(Table*, Table*, Signs*, const Commutation* commutations, 
		const pivot_t, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		Table* inv_xs, Table* inv_zs, Signs* ss, const Commutation* commutations, const pivot_t new_pivot, 
		const size_t num_words_major, const size_t num_words_minor, const size_t num_qubits_padded);

	void tune_kernel_m(void (*kernel)(pivot_t*, bucket_t*, ConstRefsPointer, ConstTablePointer, 
		const size_t, const size_t, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid, const size_t& shared_element_bytes, const bool& shared_size_yextend,
		const size_t& data_size_in_x, const size_t& data_size_in_y,
		pivot_t* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t num_gates, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor, const size_t num_qubits_padded);

	void tune_kernel_m(void (*kernel)(Commutation* commutations, pivot_t*, bucket_t*, ConstRefsPointer, ConstTablePointer, 
        const size_t, const size_t, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid, const size_t& shared_element_bytes, 
		Commutation* commutations, pivot_t* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t& gate_index, const size_t& num_qubits, const size_t num_words_major, const size_t num_words_minor, const size_t num_qubits_padded);

	void tune_outplace_transpose(void (*kernel)(Table*, Table*, ConstTablePointer, ConstTablePointer, const size_t, const size_t, const size_t),
		const char* opname, 
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table* xs1, Table* zs1, 
        ConstTablePointer xs2, ConstTablePointer zs2,
        const size_t& num_words_major, const size_t& num_words_minor, const size_t& num_qubits_padded);

	void tune_inplace_transpose(
		void (*transpose_tiles_kernel)(Table*, Table*, const size_t, const size_t, const bool),
		void (*swap_tiles_kernel)(Table*, Table*, const size_t, const size_t),
		dim3& bestBlockTransposeBits, dim3& bestGridTransposeBits,
		dim3& bestBlockTransposeSwap, dim3& bestGridTransposeSwap,
		Table* xs, Table* zs,
        const size_t& num_words_major, const size_t& num_words_minor, const bool& row_major);

	class Tuner : public Simulator {

		void reset();

	public:

		Tuner() : Simulator() {}

		Tuner(const string& path) : Simulator(path) {}
		
		void write();
		void run();
	};

}

#endif 