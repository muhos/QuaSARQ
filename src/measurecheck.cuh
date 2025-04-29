
#pragma once

#include "definitions.hpp"
#include "tableau.cuh"
#include "pivot.cuh"
#include "prefixintra.cuh"
#include "vector.hpp"
#include "circuit.hpp"

namespace QuaSARQ {

	// Xc: xs/zs[c_stab], Zc: xs/zs[c_destab], Xt: xs/zs[t_stab], Zt: xs/zs[t_destab]
    #define do_CX_sharing_control(Xc, Zc, Xt, Zt, s) \
    { \
        const word_std_t xc = Xc, zc = Zc, xt = Xt, zt = Zt; \
        Xt = xt ^ xc; \
        Zc = zc ^ zt; \
        const word_std_t xc_and_zt = xc & zt; \
        const word_std_t not_zc_xor_xt = ~(zc ^ xt); \
        s ^= (xc_and_zt & not_zc_xor_xt); \
    }

	bool is_anti_commuting_cpu(
		const 	Table&          h_xs, 
		const   qubit_t         qubit,
		const   pivot_t         pivot,
		const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded);
	
	struct MeasurementChecker {
		Table h_xs, h_zs;
		Table d_xs, d_zs;
		Signs h_ss;
		Signs d_ss;

		Table h_prefix_xs, h_prefix_zs;
		Vec<word_std_t> h_intermediate_prefix_z;
        Vec<word_std_t> h_intermediate_prefix_x;
		#if PREFIX_INTERLEAVE
		Vec<PrefixCell> d_prefix;
		Vec<PrefixCell> d_intermediate_prefix;
		#else
		Table d_prefix_xs, d_prefix_zs;
		Vec<word_std_t> d_intermediate_prefix_z;
        Vec<word_std_t> d_intermediate_prefix_x;
		#endif

		Vec<pivot_t> h_compact_pivots;
		Vec<pivot_t> d_compact_pivots;
		Vec<bool> anticommuting;

		size_t num_qubits;
		size_t num_qubits_padded;
		size_t num_words_major;
        size_t num_words_minor;

		pivot_t pivot;
		qubit_t qubit;

		bool input_copied;

		MeasurementChecker() : 
			num_qubits(0) 
			, num_qubits_padded(0)
			, num_words_major(0)
			, num_words_minor(0)
			, pivot(INVALID_PIVOT)
			, qubit(INVALID_QUBIT)
			, input_copied(false)
			{}

		~MeasurementChecker() { destroy();}

		void destroy() {
			num_qubits = 0; 
			num_qubits_padded = 0;
			num_words_major = 0;
			num_words_minor = 0;
			pivot = INVALID_PIVOT;
			qubit = INVALID_QUBIT;
		}

		void alloc(const size_t& num_qubits) {
			this->num_qubits = num_qubits;
			num_qubits_padded = get_num_padded_bits(num_qubits);
			num_words_minor = get_num_words(num_qubits);
			num_words_major = num_words_minor * 2;
			h_prefix_xs.alloc_host(num_qubits_padded, num_words_minor, num_words_minor);
			h_prefix_zs.alloc_host(num_qubits_padded, num_words_minor, num_words_minor);
			#if !PREFIX_INTERLEAVE
			d_prefix_xs.alloc_host(num_qubits_padded, num_words_minor, num_words_minor);
			d_prefix_zs.alloc_host(num_qubits_padded, num_words_minor, num_words_minor);
			#endif
			h_xs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			h_zs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			h_ss.alloc_host(num_qubits_padded, num_words_major);
			d_ss.alloc_host(num_qubits_padded, num_words_major);
		}

		void copy_pivots(const pivot_t* other_pivots, const size_t& num_pivots) {
			SYNCALL;
            assert(num_pivots <= num_qubits);
			d_compact_pivots.resize(num_pivots);
			CHECK(cudaMemcpy(d_compact_pivots.data(), other_pivots, sizeof(pivot_t) * num_pivots, cudaMemcpyDeviceToHost));
		}

		void copy_input(const Tableau& other, const bool& to_device = false) {
			SYNCALL;
			if (input_copied && !to_device)
				return;
			Table& dest_xs = to_device ? d_xs : h_xs;
			Table& dest_zs = to_device ? d_zs : h_zs;
			Signs& dest_ss = to_device ? d_ss : h_ss;
			dest_xs.flag_rowmajor(), dest_zs.flag_rowmajor();
			other.copy_to_host(&dest_xs, &dest_zs, &dest_ss);
			if (!to_device) 
				input_copied = true;
		}

		#if PREFIX_INTERLEAVE
		void copy_prefix(const PrefixCell* other) {
			SYNCALL;
			const size_t size = num_qubits_padded * num_words_minor;
			d_prefix.resize(size);
			CHECK(cudaMemcpy(d_prefix.data(), other, sizeof(PrefixCell) * size, cudaMemcpyDeviceToHost));
		}

		void copy_prefix_blocks(const PrefixCell* other, const size_t& size) {
			SYNCALL;
			d_intermediate_prefix.resize(size);
			CHECK(cudaMemcpy(d_intermediate_prefix.data(), other, sizeof(PrefixCell) * size, cudaMemcpyDeviceToHost));
		}
		#else
		void copy_prefix(const Tableau& other) {
			SYNCALL;
			d_prefix_xs.flag_rowmajor(), d_prefix_zs.flag_rowmajor();
			other.copy_to_host(&d_prefix_xs, &d_prefix_zs);
		}

		void copy_prefix_blocks(const word_std_t* other_xs, const word_std_t* other_zs, const size_t& size) {
			SYNCALL;
			d_intermediate_prefix_x.resize(size);
			d_intermediate_prefix_z.resize(size);
			CHECK(cudaMemcpy(d_intermediate_prefix_x.data(), other_xs, sizeof(word_std_t) * size, cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(d_intermediate_prefix_z.data(), other_zs, sizeof(word_std_t) * size, cudaMemcpyDeviceToHost));
		}
		#endif

		void find_min_pivot(const qubit_t& qubit, const bool& store_pivots = false);

		void check_min_pivot(const pivot_t& other_pivot);

		void check_compact_pivots(const qubit_t& qubit, const pivot_t* other_pivots, const size_t& other_num_pivots);

		void check_initial_pivots(const Circuit& circuit, const depth_t& depth_level, const pivot_t* other_pivots, const size_t& other_num_pivots);

		void check_prefix_intermediate_pass(
			CHECK_PREFIX_SINGLE_PASS_ARGS,
			const   size_t&	        max_blocks,
			const 	size_t&         pass_1_gridsize,
			const   bool&           skip_checking_device = false);

		void check_prefix_pass_1(
			CHECK_PREFIX_PASS_1_ARGS,
			const   pivot_t*    	other_pivots,
			const   size_t&         active_targets,
			const   size_t&         max_blocks,
			const   size_t&         pass_1_blocksize,
			const   size_t&         pass_1_gridsize,
			const   bool&           skip_checking_device = false);

		void check_prefix_pass_2(
			const 	Tableau& 		other_input,
			const   size_t&         active_targets,
			const   size_t&         max_blocks,
			const   size_t&         pass_1_blocksize,
			const   bool&           skip_checking_device = false);

		void check_inject_cx(const Tableau& other_input);

		void check_inject_swap(const Tableau& other_input, const pivot_t* other_pivots, const size_t& num_pivots);

		void check_inject_x(const Tableau& other_input, const pivot_t* other_pivots, const size_t& num_pivots, const sign_t& random_bit);

		void inject_swap_cpu();

		void inject_x_cpu();

	};
	
}