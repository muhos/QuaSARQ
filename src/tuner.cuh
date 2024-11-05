
#ifndef __CU_TUNE_H
#define __CU_TUNE_H

#include "timer.cuh"
#include "grid.cuh"
#include "gate.cuh"
#include "vector.cuh"
#include "tableau.cuh"

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

	void tune_kernel(void (*kernel)(const gate_ref_t*, const bucket_t*, const size_t, const size_t, 
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
		const gate_ref_t* gate_refs, const bucket_t* gate_buckets, const size_t& num_gates, const size_t& num_words_major, 
		#ifdef INTERLEAVE_XZ
		Table* ps, 
		#else
		Table* xs, Table* zs, 
		#endif
		Signs *ss);

	class Tuner : public Simulator {

		void reset();

	public:

		Tuner();
		
		void write();
		void run();
	};

}

#endif 