
#ifndef __CU_PREFIX_H
#define __CU_PREFIX_H

#include "definitions.hpp"
#include "definitions.cuh"
#include "shared.cuh"
#include "word.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "tableau.cuh"

namespace QuaSARQ {

	#define NUM_BANKS 16
    #define LOG_NUM_BANKS 4
    #ifdef ZERO_BANK_CONFLICTS
    #define CONFLICT_FREE_OFFSET(n) (((n) >> NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))
    #else
    #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
    #endif

	#define MIN_BLOCK_INTERMEDIATE_SIZE 32
	#define MIN_SUB_BLOCK_SIZE 16
	#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
	#define MIN_SINGLE_PASS_THRESHOLD 512
	#else
	#define MIN_SINGLE_PASS_THRESHOLD 1024
	#endif


    __device__ word_std_t scan_block_exclusive(word_std_t* data, const int& n);

	__global__ void scan_blocks_single_pass(word_std_t* block_intermediate_prefix_z, 
                                         word_std_t* block_intermediate_prefix_x,
                                         const size_t num_blocks,
                                         const size_t num_words_minor);

	__global__
	void scan_blocks_pass_1(
		word_std_t* block_intermediate_prefix_z,
		word_std_t* block_intermediate_prefix_x,
		word_std_t* subblocks_prefix_z, 
		word_std_t* subblocks_prefix_x, 
		const size_t num_blocks,
		const size_t num_words_minor
	);

	__global__ 
	void scan_blocks_pass_2(
        word_std_t* block_intermediate_prefix_z,
        word_std_t* block_intermediate_prefix_x,
        const word_std_t* subblocks_prefix_z,
        const word_std_t* subblocks_prefix_x,
        size_t num_blocks,
        size_t num_words_minor,
        size_t pass_1_blocksize
    );

	class Prefix {

		DeviceAllocator& allocator;

		Tableau<DeviceAllocator> targets;

		word_std_t* block_intermediate_prefix_z;
		word_std_t* block_intermediate_prefix_x;
		word_std_t* subblocks_prefix_z;
		word_std_t* subblocks_prefix_x;

		size_t max_intermediate_blocks;
		size_t max_sub_blocks;
		size_t min_blocksize_y;

		size_t num_qubits;
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
		,   num_words_major(0)
		,   num_words_minor(0)
		,   pivot(0)
		{}

		word_std_t* zblocks() { assert(block_intermediate_prefix_z != nullptr); return block_intermediate_prefix_z; }
		word_std_t* xblocks() { assert(block_intermediate_prefix_x != nullptr); return block_intermediate_prefix_x; }


		void alloc(const Tableau<DeviceAllocator>& input, const size_t& max_window_bytes);
		void scan_blocks(const size_t& num_blocks, const cudaStream_t& stream);
		void inject_CX(Tableau<DeviceAllocator>& input, const uint32& pivot, const qubit_t& qubit, const cudaStream_t& stream);

	};
	
}

#endif