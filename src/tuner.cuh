
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

	// With measurements.
	struct TableauState
	{
		word_t* saving_xs, *saving_zs;
		word_t* org_xs, *org_zs;
		size_t num_words;
		bool recover;

		TableauState() :
			saving_xs(nullptr), saving_zs(nullptr), org_xs(nullptr), org_zs(nullptr),
			num_words(0), recover(false) {}

		void set_original_pointers(word_t* org_xdata, word_t* org_zdata, const size_t& num_words) {
			org_xs = org_xdata, org_zs, org_zdata;
			this->num_words = num_words;
		}

		void set_saving_pointers(word_t* saving_xdata, word_t* saving_zdata) {
			saving_xs = saving_xdata, saving_zs, saving_zdata;
		}

		void save_state() {
			if (!recover) return;
			SYNCALL;
			assert(num_words);
			if (saving_xs != nullptr) CHECK(cudaMemcpy(saving_xs, org_xs, num_words * sizeof(word_t), cudaMemcpyDeviceToDevice));
			if (saving_zs != nullptr) CHECK(cudaMemcpy(saving_zs, org_zs, num_words * sizeof(word_t), cudaMemcpyDeviceToDevice));
		}

		void recover_state() {
			if (!recover) return;
			SYNCALL;
			assert(num_words);
			if (org_xs != nullptr) CHECK(cudaMemcpy(org_xs, saving_xs, num_words * sizeof(word_t), cudaMemcpyDeviceToDevice));
			if (org_zs != nullptr) CHECK(cudaMemcpy(org_zs, saving_zs, num_words * sizeof(word_t), cudaMemcpyDeviceToDevice));
		}
	};
	extern TableauState ts; 
	
	void tune_kernel_m(void (*kernel)(const size_t, const size_t, Table*, Table*),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& offset, const size_t& size, Table* xs, Table* zs);

	void tune_kernel_m(void (*kernel)(Pivot*, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		Pivot* pivots, const size_t size);

	void tune_kernel_m(void (*kernel)(Pivot*, bucket_t*, ConstRefsPointer, ConstTablePointer, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid, const size_t& shared_element_bytes, const bool& shared_size_yextend,
		const size_t& data_size_in_x, const size_t& data_size_in_y,
		Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor);

	void tune_kernel_m(void (*kernel)(Pivot*, bucket_t*, ConstRefsPointer, ConstTablePointer, 
        const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid, 
		Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t& gate_index, const size_t& num_qubits, const size_t& num_words_minor);

	void tune_transpose(void (*kernel)(Table*, Table*, Signs*, ConstTablePointer, ConstTablePointer, ConstSignsPointer, const size_t, const size_t, const size_t),
		const char* opname, 
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table* xs1, Table* zs1, Signs* ss1, 
        ConstTablePointer xs2, ConstTablePointer zs2, ConstSignsPointer ss2, 
        const size_t& num_words_major, const size_t& num_words_minor, const size_t& num_qubits);

	void tune_determinate(void (*kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, ConstTablePointer, ConstTablePointer, ConstSignsPointer, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid, const size_t& shared_element_bytes, const bool& shared_size_yextend,
		const size_t& data_size_in_x, const size_t& data_size_in_y,
		ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs,
        ConstTablePointer inv_xs, ConstTablePointer inv_zs, ConstSignsPointer inv_ss, 
        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor);

	void tune_single_determinate(void (*kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, ConstTablePointer, ConstTablePointer, ConstSignsPointer, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid, const size_t& shared_element_bytes, 
		ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs,
        ConstTablePointer inv_xs, ConstTablePointer inv_zs, ConstSignsPointer inv_ss, 
        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor);

	void tune_indeterminate(
		void (*copy_kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, Table*, Table*, Signs*, const size_t, const size_t, const size_t),
		void (*phase1_kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, Table*, Table*, Signs*, const size_t, const size_t, const size_t),
		void (*phase2_kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, Table*, Table*, Signs*, const size_t, const size_t, const size_t),
		dim3& bestBlockCopy, dim3& bestGridCopy,
		dim3& bestBlockPhase1, dim3& bestGridPhase1,
		dim3& bestBlockPhase2, dim3& bestGridPhase2,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs, 
        Table* inv_xs, Table* inv_zs, Signs *inv_ss,
        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor);

	class Tuner : public Simulator {

		void reset();

	public:

		Tuner() : Simulator() { ts.recover = false; }
		
		void write();
		void run();
	};

}

#endif 