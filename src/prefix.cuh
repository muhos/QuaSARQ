
#ifndef __CU_PREFIX_H
#define __CU_PREFIX_H

#include "definitions.hpp"
#include "definitions.cuh"
#include "commutation.cuh"
#include "prefixcheck.cuh"
#include "shared.cuh"
#include "word.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "tableau.cuh"
#include "vector.hpp"

namespace QuaSARQ {

	#define MIN_BLOCK_INTERMEDIATE_SIZE 32
	#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
	constexpr int64 MIN_SINGLE_PASS_THRESHOLD = 512;
	#else
	constexpr int64 MIN_SINGLE_PASS_THRESHOLD = 1024;
	#endif

	struct XOROP {
        __device__ __forceinline__ word_std_t operator()(const word_std_t &a, const word_std_t &b) const {
            return a ^ b;
        }
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

	void call_single_pass_kernel(
                word_std_t *        intermediate_prefix_z,
                word_std_t *        intermediate_prefix_x,
        const   size_t              num_chunks,
        const   size_t              num_words_minor,
        const   size_t              max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);

    void call_scan_blocks_pass_1_kernel(
                word_std_t*     block_intermediate_prefix_z,
                word_std_t*     block_intermediate_prefix_x,
                word_std_t*     subblocks_prefix_z, 
                word_std_t*     subblocks_prefix_x, 
        const   size_t          num_blocks,
        const   size_t          num_words_minor,
        const   size_t          max_blocks,
        const   size_t          max_sub_blocks,
        const   dim3&           currentblock,
        const   dim3&           currentgrid,
        const   cudaStream_t&   stream);

    void tune_single_pass(
                dim3&       bestBlock, 
                dim3&       bestGrid,
        const   size_t&     shared_element_bytes, 
        const   size_t&     data_size_in_x, 
        const   size_t&     data_size_in_y,
                word_std_t* block_intermediate_prefix_z, 
                word_std_t* block_intermediate_prefix_x,
        const   size_t      num_chunks,
        const   size_t      num_words_minor,
        const   size_t      max_blocks);

    void tune_prefix_pass_1(
                dim3&       bestBlock, 
                dim3&       bestGrid,
        const   size_t&     shared_element_bytes, 
        const   size_t&     data_size_in_x, 
        const   size_t&     data_size_in_y,
                word_std_t* block_intermediate_prefix_z,
                word_std_t* block_intermediate_prefix_x,
                word_std_t* subblocks_prefix_z, 
                word_std_t* subblocks_prefix_x,
        const   size_t&     num_blocks,
        const   size_t&     num_words_minor,
        const   size_t&     max_blocks,
        const   size_t&     max_sub_blocks);

	void call_injectcx_pass_1_kernel(
                Tableau<DeviceAllocator>& targets, 
                Tableau<DeviceAllocator>& input,
                word_std_t *        block_intermediate_prefix_z,
                word_std_t *        block_intermediate_prefix_x,
        const   Commutation *       commutations,
        const   uint32              pivot,
        const   size_t              total_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);

	void call_injectcx_pass_2_kernel(
                Tableau<DeviceAllocator>& targets, 
                Tableau<DeviceAllocator>& input,
        const   word_std_t *        block_intermediate_prefix_z,
        const   word_std_t *        block_intermediate_prefix_x,
        const   Commutation *       commutations,
        const   uint32              pivot,
        const   size_t              total_targets,
        const   size_t              num_words_major,
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded,
        const   size_t              max_blocks,
        const   size_t              pass_1_blocksize,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);

    void tune_inject_pass_1(
		        dim3&           bestBlock, 
                dim3&           bestGrid,
		const   size_t&         shared_element_bytes, 
		const   size_t&         data_size_in_x, 
		const   size_t&         data_size_in_y,
		        Tableau<DeviceAllocator>& targets, 
		        Tableau<DeviceAllocator>& input, 
                word_std_t *    block_intermediate_prefix_z,
                word_std_t *    block_intermediate_prefix_x,
		const   Commutation*    commutations,
		const   uint32&         pivot,
		const   size_t&         total_targets,
		const   size_t&         num_words_major,
		const   size_t&         num_words_minor,
		const   size_t&         num_qubits_padded,
		const   size_t&         max_blocks);

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

	void tune_inject_pass_2(
				dim3& 			bestBlock, 
				dim3& 			bestGrid,
		const 	size_t& 		shared_element_bytes, 
		const 	size_t& 		data_size_in_x, 
		const 	size_t& 		data_size_in_y,
				Tableau<DeviceAllocator>& targets, 
				Tableau<DeviceAllocator>& input, 
        const 	word_std_t *	block_intermediate_prefix_z,
        const 	word_std_t *	block_intermediate_prefix_x,
		const 	Commutation* 	commutations,
		const 	uint32& 		pivot,
		const 	size_t& 		total_targets,
		const 	size_t& 		num_words_major,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		num_qubits_padded,
		const 	size_t& 		max_blocks,
		const 	size_t& 		pass_1_blocksize);
	
}

#endif