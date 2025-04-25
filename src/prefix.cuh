
#pragma once

#include "definitions.cuh"
#include "datatypes.cuh"
#include "shared.cuh"
#include "word.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "tableau.cuh"
#include "prefixintra.cuh"
#include "measurecheck.cuh"
#include "options.hpp"
#include "kernelconfig.hpp"

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

	INLINE_DEVICE 
    void compute_local_sign_per_block(                
                sign_t&     local_sign, 
                word_std_t& t_stab_word_ref,
		const   word_std_t& zc_xor_prefix,
        const   word_std_t& c_stab_word,
        const   word_std_t& t_destab_word) 
    {
        const word_std_t t_stab_word = t_stab_word_ref;
        const word_std_t not_zc_xor_prefix = ~(zc_xor_prefix ^ t_stab_word);
        local_sign ^= (c_stab_word & t_destab_word & not_zc_xor_prefix);
        t_stab_word_ref = t_stab_word ^ c_stab_word;
    }

	class Prefix {

		DeviceAllocator& allocator;
		MeasurementChecker& checker;

		#if PREFIX_INTERLEAVE
		PrefixCell* global_prefix;
		PrefixCell* intermediate_prefix;
		PrefixCell* subblock_prefix;
		#else
		Tableau targets;
		word_std_t* intermediate_prefix_z;
		word_std_t* intermediate_prefix_x;
		word_std_t* subblocks_prefix_z;
		word_std_t* subblocks_prefix_x;
		#endif

		size_t max_intermediate_blocks;
		size_t max_sub_blocks;

		size_t num_qubits;
		size_t config_qubits;
		size_t num_words_major;
		size_t num_words_minor;

		size_t prev_active_targets;
		uint32 prev_yblock_size;
		uint32 prev_ygrid_size;
		uint32 min_yblock_size;

		static constexpr size_t AT_LEAST_TUNED_MULTIPASS = 2048;
		static constexpr size_t AT_LEAST_TUNED_SINGLE_PASS = 1024;

	public:

		Prefix(DeviceAllocator& allocator, MeasurementChecker& checker) : 
			allocator(allocator)
		,	checker(checker)
		#if PREFIX_INTERLEAVE
		,	global_prefix(nullptr)
		,	intermediate_prefix(nullptr)
		,	subblock_prefix(nullptr)
		#else
		,	targets(allocator)
		,	intermediate_prefix_z(nullptr)
		,	intermediate_prefix_x(nullptr)
		,	subblocks_prefix_z(nullptr)
		,	subblocks_prefix_x(nullptr)
		#endif
		,	max_intermediate_blocks(0)
		,	max_sub_blocks(0)
		,   num_qubits(0)
		,	config_qubits(0)
		,   num_words_major(0)
		,   num_words_minor(0)
		,	prev_active_targets(0)
		,	prev_yblock_size(0)
		,	prev_ygrid_size(0)
		,	min_yblock_size(2)
		{}

		#if PREFIX_INTERLEAVE
		PrefixCell* global_prefixes	() { assert(global_prefix != nullptr); return global_prefix; }
		PrefixCell* block_prefixes	() { assert(intermediate_prefix != nullptr); return intermediate_prefix; }
		#else
		word_std_t* zblocks			() { assert(intermediate_prefix_z != nullptr); return intermediate_prefix_z; }
		word_std_t* xblocks			() { assert(intermediate_prefix_x != nullptr); return intermediate_prefix_x; }
		#endif
		void 		alloc			(const Tableau& input, const size_t& config_qubits, const size_t& max_window_bytes);
		void 		resize			(const Tableau& input, const size_t& max_window_bytes);
		void 		scan_blocks		(const size_t& num_blocks, const size_t& inject_pass_1_blocksize, const cudaStream_t& stream);
		void 		scan_warp 		(Tableau& input, const pivot_t* pivots, const size_t& active_targets, const cudaStream_t& stream);
		void 		scan_block 		(Tableau& input, const pivot_t* pivots, const size_t& active_targets, const cudaStream_t& stream);
		void 		scan_large		(Tableau& input, const pivot_t* pivots, const size_t& active_targets, const cudaStream_t& stream);
		void		tune_inject_cx	(Tableau& input, const pivot_t* pivots, const size_t& max_active_targets);
		void 		tune_scan_blocks();
		void 		tune_grid_size	(dim3& currentblock, dim3& currentgrid, const size_t& pow2_active_targets) {
			assert(num_words_minor);
			assert(maxGPUThreads);
			currentblock.x = pow2_active_targets;
			if (currentblock.x < 1024 &&
				(prev_active_targets == pow2_active_targets || 
				ROUNDUP(num_words_minor, min_yblock_size) == prev_ygrid_size)) {
				currentblock.y = prev_yblock_size;
				currentgrid.y = prev_ygrid_size;
			}
			else {
				currentblock.y = 1024 / currentblock.x;
				currentgrid.y = ROUNDUP(num_words_minor, currentblock.y);
				while (currentblock.y > min_yblock_size &&
						currentgrid.y < maxGPUBlocks) {
					currentblock.y >>= 1;
					currentgrid.y = ROUNDUP(num_words_minor, currentblock.y);
				}
				currentgrid.y = MIN(currentgrid.y, maxGPUBlocks);\
				prev_active_targets = pow2_active_targets;
				prev_yblock_size = currentblock.y;
				prev_ygrid_size = currentgrid.y;
			}
		}

	};

	void call_single_pass_kernel(
                SINGLE_PASS_ARGS,
        const   size_t&             num_chunks,
        const   size_t&             num_words_minor,
        const   size_t&             max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);

    void call_scan_blocks_pass_1_kernel(
                PASS_1_ARGS_PREFIX, 
        const   size_t&         	num_blocks,
        const   size_t&         	num_words_minor,
        const   size_t&         	max_blocks,
        const   size_t&         	max_sub_blocks,
        const   dim3&           	currentblock,
        const   dim3&           	currentgrid,
        const   cudaStream_t&   	stream);

	void call_injectcx_pass_1_kernel(
                CALL_ARGS_GLOBAL_PREFIX,
				Tableau& 			input,
        const   pivot_t*            pivots,
        const   size_t&             active_targets,
        const   size_t&             num_words_major,
        const   size_t&             num_words_minor,
        const   size_t&             num_qubits_padded,
        const   size_t&             max_blocks,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);

	void call_injectcx_pass_2_kernel(
                CALL_ARGS_GLOBAL_PREFIX, 
                Tableau& 			input,
        const   pivot_t*            pivots,
        const   size_t&             active_targets,
        const   size_t&             num_words_major,
        const   size_t&             num_words_minor,
        const   size_t&             num_qubits_padded,
        const   size_t&             max_blocks,
        const   size_t&             pass_1_blocksize,
        const   dim3&               currentblock,
        const   dim3&               currentgrid,
        const   cudaStream_t&       stream);


    void tune_single_pass(
                dim3&       	bestBlock, 
                dim3&       	bestGrid,
        const   size_t&     	shared_element_bytes, 
        const   size_t&     	data_size_in_x, 
        const   size_t&     	data_size_in_y,
                SINGLE_PASS_ARGS,
        const   size_t&     	num_chunks,
        const   size_t&     	num_words_minor,
        const   size_t&     	max_blocks);

    void tune_prefix_pass_1(
                dim3&       	bestBlock, 
                dim3&       	bestGrid,
        const   size_t&     	shared_element_bytes, 
        const   size_t&     	data_size_in_x, 
        const   size_t&     	data_size_in_y,
                PASS_1_ARGS_PREFIX,
        const   size_t&     	num_blocks,
        const   size_t&     	num_words_minor,
        const   size_t&     	max_blocks,
        const   size_t&     	max_sub_blocks);

	void tune_prefix_pass_2(
		void (*kernel)(
				PASS_2_ARGS_PREFIX, 
		const 	size_t, 
		const 	size_t, 
		const 	size_t, 
		const 	size_t, 
		const 	size_t),
				dim3& 			bestBlock, 
				dim3& 			bestGrid,
		const	size_t& 		data_size_in_x, 
		const 	size_t& 		data_size_in_y,
				PASS_2_ARGS_PREFIX,
		const 	size_t& 		num_blocks,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		max_blocks,
		const 	size_t& 		max_sub_blocks,
		const 	size_t& 		pass_1_blocksize);

    void tune_inject_pass_1(
		        dim3&           bestBlock, 
                dim3&           bestGrid,
		const   size_t&         shared_element_bytes, 
		const   size_t&         data_size_in_x, 
		const   size_t&         data_size_in_y,
		        CALL_ARGS_GLOBAL_PREFIX,
				Tableau& 		input, 
		const 	pivot_t* 		pivots,
		const 	size_t& 		active_targets,
		const   size_t&         num_words_major,
		const   size_t&         num_words_minor,
		const   size_t&         num_qubits_padded,
		const   size_t&         max_blocks);

	void tune_inject_pass_2(
				dim3& 			bestBlock, 
				dim3& 			bestGrid,
		const 	size_t& 		shared_element_bytes, 
		const 	size_t& 		data_size_in_x, 
		const 	size_t& 		data_size_in_y,
				CALL_ARGS_GLOBAL_PREFIX,
				Tableau& 		input,
		const 	pivot_t* 		pivots,
		const 	size_t& 		active_targets,
		const 	size_t& 		num_words_major,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		num_qubits_padded,
		const 	size_t& 		max_blocks,
		const 	size_t& 		pass_1_blocksize);
	
}