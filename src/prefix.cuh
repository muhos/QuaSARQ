
#ifndef __CU_PREFIX_H
#define __CU_PREFIX_H

#include "definitions.hpp"
#include "definitions.cuh"
#include "commutation.cuh"
#include "shared.cuh"
#include "word.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "tableau.cuh"
#include "vector.hpp"

namespace QuaSARQ {

	#define NUM_BANKS 16
    #define LOG_NUM_BANKS 4
    #ifdef ZERO_BANK_CONFLICTS
    #define CONFLICT_FREE_OFFSET(n) (((n) >> NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))
    #else
    #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
    #endif

	#define MIN_BLOCK_INTERMEDIATE_SIZE 32
	#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
	constexpr int64 MIN_SINGLE_PASS_THRESHOLD = 512;
	#else
	constexpr int64 MIN_SINGLE_PASS_THRESHOLD = 1024;
	#endif


    __device__ word_std_t scan_block_exclusive(word_std_t* data, const int& n);

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

	class Prefix {

		DeviceAllocator& allocator;

		Tableau<DeviceAllocator> targets;

		PrefixChecker checker;

		word_std_t* block_intermediate_prefix_z;
		word_std_t* block_intermediate_prefix_x;
		word_std_t* subblocks_prefix_z;
		word_std_t* subblocks_prefix_x;

		size_t max_intermediate_blocks;
		size_t max_sub_blocks;
		size_t min_blocksize_y;

		size_t num_qubits;
		size_t config_qubits;
		size_t num_words_major;
		size_t num_words_minor;

		uint32 pivot;

	public:

		Prefix(DeviceAllocator& allocator) : 
			allocator(allocator)
		,	targets(allocator)
		,	block_intermediate_prefix_z(nullptr)
		,	block_intermediate_prefix_x(nullptr)
		,	subblocks_prefix_z(nullptr)
		,	subblocks_prefix_x(nullptr)
		,	max_intermediate_blocks(0)
		,	max_sub_blocks(0)
		,	min_blocksize_y(0)
		,   num_qubits(0)
		,	config_qubits(0)
		,   num_words_major(0)
		,   num_words_minor(0)
		,   pivot(0)
		{}

		PrefixChecker& get_checker		() { return checker; }

		word_std_t* zblocks			() { assert(block_intermediate_prefix_z != nullptr); return block_intermediate_prefix_z; }
		word_std_t* xblocks			() { assert(block_intermediate_prefix_x != nullptr); return block_intermediate_prefix_x; }

		void 		alloc			(const Tableau<DeviceAllocator>& input, const size_t& config_qubits, const size_t& max_window_bytes);
		void 		resize			(const Tableau<DeviceAllocator>& input, const size_t& max_window_bytes);
		void 		scan_blocks		(const size_t& num_blocks, const size_t& inject_pass_1_blocksize, const cudaStream_t& stream);
		void 		inject_CX		(Tableau<DeviceAllocator>& input,  const Commutation* commutations, const uint32& pivot, const qubit_t& qubit, const cudaStream_t& stream);

	};

	void tune_prefix_pass_1(
		void (*kernel)(word_std_t*, word_std_t*, word_std_t*, word_std_t*, 
						const size_t, const size_t, const size_t, const size_t),
		dim3& bestBlockPass1, dim3& bestGridPass1,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		word_std_t* block_intermediate_prefix_z,
		word_std_t* block_intermediate_prefix_x,
		word_std_t* subblocks_prefix_z, 
		word_std_t* subblocks_prefix_x,
		const size_t& num_blocks,
		const size_t& num_words_minor,
		const size_t& max_blocks,
		const size_t& max_sub_blocks);

	void tune_prefix_pass_2(
		void (*kernel)(word_std_t*, word_std_t*, const word_std_t*, const word_std_t*, 
						const size_t, const size_t, const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		word_std_t* block_intermediate_prefix_z,
		word_std_t* block_intermediate_prefix_x,
		const word_std_t* subblocks_prefix_z, 
		const word_std_t* subblocks_prefix_x,
		const size_t& num_blocks,
		const size_t& num_words_minor,
		const size_t& max_blocks,
		const size_t& max_sub_blocks,
		const size_t& pass_1_blocksize);

	void tune_single_pass(
		void (*kernel)(word_std_t*, word_std_t*, const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		word_std_t* block_intermediate_prefix_z, 
		word_std_t* block_intermediate_prefix_x,
		const size_t num_chunks,
		const size_t num_words_minor,
		const size_t max_blocks
	);

	void tune_inject_pass_1(
		void (*kernel)(Table*, Table*, Table*, Table*, word_std_t *, word_std_t *, 
						const Commutation*, const uint32, const size_t, const size_t, const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table *prefix_xs, 
        Table *prefix_zs, 
        Table *inv_xs, 
        Table *inv_zs,
        word_std_t *block_intermediate_prefix_z,
        word_std_t *block_intermediate_prefix_x,
		const Commutation* commutations,
		const uint32& pivot,
		const size_t& total_targets,
		const size_t& num_words_major,
		const size_t& num_words_minor,
		const size_t& num_qubits_padded,
		const size_t& max_blocks);

	void tune_inject_pass_2(
		void (*kernel)(Table*, Table*, Table*, Table*, Signs*, const word_std_t *, const word_std_t *, 
						const Commutation*, const uint32, 
						const size_t, const size_t, const size_t, const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table *prefix_xs, 
		Table *prefix_zs, 
		Table *inv_xs, 
		Table *inv_zs,
		Signs *inv_ss,
		const word_std_t *block_intermediate_prefix_z,
		const word_std_t *block_intermediate_prefix_x,
		const Commutation* commutations,
		const uint32& pivot,
		const size_t& total_targets,
		const size_t& num_words_major,
		const size_t& num_words_minor,
		const size_t& num_qubits_padded,
		const size_t& max_blocks,
		const size_t& pass_1_blocksize);
	
}

#endif