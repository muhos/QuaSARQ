
#ifndef __CU_PREFIXCHECK_H
#define __CU_PREFIXCHECK_H

#include "definitions.hpp"
#include "commutation.cuh"
#include "tableau.cuh"
#include "vector.hpp"

namespace QuaSARQ {

	struct PrefixChecker {
		Table h_xs, h_zs;
		Table d_xs, d_zs;

		Table h_prefix_xs, h_prefix_zs;
		Table d_prefix_xs, d_prefix_zs;

		Signs h_ss;
		Signs d_ss;

		Vec<Commutation> h_commutations;
		Vec<Commutation> d_commutations;

		Vec<word_std_t> h_block_intermediate_prefix_z;
        Vec<word_std_t> h_block_intermediate_prefix_x;
		Vec<word_std_t> d_block_intermediate_prefix_z;
        Vec<word_std_t> d_block_intermediate_prefix_x;

		bool prefix;
		bool signs;

		PrefixChecker() :
			prefix(false)
		, 	signs(false)
		{}

		~PrefixChecker() {}

		void alloc(const size_t& num_qubits) {
			h_commutations.resize(num_qubits);
			d_commutations.resize(num_qubits);
			const size_t num_qubits_padded = get_num_padded_bits(num_qubits);
			size_t num_words_major = get_num_words(num_qubits);
			size_t num_words_minor = num_words_major;
			h_prefix_xs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			h_prefix_zs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			d_prefix_xs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			d_prefix_zs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			num_words_major <<= 1;
			h_xs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			h_zs.alloc_host(num_qubits_padded, num_words_major, num_words_minor);
			h_ss.alloc_host(num_qubits_padded, num_words_major);
			d_ss.alloc_host(num_qubits_padded, num_words_major);
		}

		bool copy_input(Tableau<DeviceAllocator>& other, const bool& to_device = false) {
			SYNCALL;
			Table& dest_xs = to_device ? d_xs : h_xs;
			Table& dest_zs = to_device ? d_zs : h_zs;
			Signs& dest_ss = to_device ? d_ss : h_ss;
			dest_xs.flag_rowmajor(), dest_zs.flag_rowmajor();
			other.copy_to_host(&dest_xs, &dest_zs, &dest_ss);
			return true;
		}

		bool copy_prefix(Tableau<DeviceAllocator>& other) {
			SYNCALL;
			d_prefix_xs.flag_rowmajor(), d_prefix_zs.flag_rowmajor();
			other.copy_to_host(&d_prefix_xs, &d_prefix_zs);
			return true;
		}

		bool copy_prefix_blocks(const word_std_t* other_xs, const word_std_t* other_zs, const size_t& size) {
			SYNCALL;
			d_block_intermediate_prefix_x.resize(size);
			d_block_intermediate_prefix_z.resize(size);
			CHECK(cudaMemcpy(d_block_intermediate_prefix_x.data(), other_xs, sizeof(word_std_t) * size, cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(d_block_intermediate_prefix_z.data(), other_zs, sizeof(word_std_t) * size, cudaMemcpyDeviceToHost));
			return true;
		}

		bool copy_commutations(const Commutation* other, const size_t& size) {
			SYNCALL;
			assert(size <= d_commutations.size());
			CHECK(cudaMemcpy(d_commutations.data(), other, sizeof(Commutation) * size, cudaMemcpyDeviceToHost));
			return true;
		}

		bool check_prefix_intermediate_pass(
			const   word_std_t* other_zs,
			const   word_std_t* other_xs,
			const   qubit_t  qubit, 
			const   uint32   pivot,
			const   size_t   num_words_minor,
			const   size_t	 max_blocks,
			const 	size_t   pass_1_gridsize);
		
		bool check_prefix_pass_1(
			Tableau<DeviceAllocator>& other_targets,
			Tableau<DeviceAllocator>& other_input,
			const   Commutation* other_commutations,
			const   word_std_t*  other_zs,
			const   word_std_t*  other_xs,
			const   qubit_t      qubit, 
			const   uint32       pivot,
			const   size_t       total_targets,
			const   size_t       num_words_major,
			const   size_t       num_words_minor,
			const   size_t       num_qubits_padded,
			const   size_t       max_blocks,
			const   size_t       pass_1_blocksize,
			const   size_t       pass_1_gridsize
		);

		bool check_prefix_pass_2(
			Tableau<DeviceAllocator>& other_targets, 
			Tableau<DeviceAllocator>& other_input,
			const   qubit_t 		qubit, 
			const   uint32   		pivot,
			const   size_t          total_targets,
			const   size_t          num_words_major,
			const   size_t          num_words_minor,
			const   size_t          num_qubits_padded,
			const   size_t          max_blocks,
			const   size_t          pass_1_blocksize
		);

	};
	
}

#endif