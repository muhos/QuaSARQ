#pragma once

#include "datatypes.cuh"
#include "tableau.cuh"
#include "access.cuh"

namespace QuaSARQ {


	struct PrefixCell {
        word_std_t x, z;

		INLINE_ALL 
		PrefixCell() : x(0), z(0) {}

		INLINE_ALL
		PrefixCell(const word_std_t& x, const word_std_t& z) : x(x), z(z) {}

		INLINE_ALL 
		void print() const {
			printf("PrefixCell: x = %016llx, z = %016llx\n", x, z);
		}
    };

	#if PREFIX_INTERLEAVE
		typedef const PrefixCell *  __restrict__ const_prefixcells_t;
		#define PASS_1_ARGS_GLOBAL_PREFIX \
			PrefixCell *        global_prefix, \
			PrefixCell *        intermediate_prefix

		#define PASS_2_ARGS_GLOBAL_PREFIX \
			const_prefixcells_t global_prefix, \
			const_prefixcells_t intermediate_prefix

		#define CHECK_PREFIX_SINGLE_PASS_ARGS \
			const   PrefixCell*     other_intermediate
		
		#define CHECK_PREFIX_PASS_1_ARGS \
			const 	PrefixCell*     other_targets, \
			CHECK_PREFIX_SINGLE_PASS_ARGS

		#define CHECK_PREFIX_SINGLE_PASS_INPUT intermediate_prefix

		#define CHECK_PREFIX_PASS_1_INPUT \
			global_prefix, \
			CHECK_PREFIX_SINGLE_PASS_INPUT

		#define CALL_ARGS_GLOBAL_PREFIX PASS_1_ARGS_GLOBAL_PREFIX

		#define KERNEL_INPUT_GLOBAL_PREFIX \
			global_prefix, \
			intermediate_prefix

		#define CALL_INPUT_GLOBAL_PREFIX KERNEL_INPUT_GLOBAL_PREFIX

		#define WRITE_GLOBAL_PREFIX(WORD_IDX, Z, X) \
			PrefixCell& cell = global_prefix[WORD_IDX]; \
            cell.z = (Z); \
            cell.x = (X);

		#define READ_GLOBAL_PREFIX(WORD_IDX, Z, X) \
			const PrefixCell& cell = global_prefix[WORD_IDX]; \
            Z = cell.z; \
            X = cell.x;

		#define SINGLE_PASS_ARGS \
			PrefixCell * intermediate_prefix
		
		#define PASS_1_ARGS_PREFIX \
			SINGLE_PASS_ARGS, \
			PrefixCell * subblock_prefix

		#define PASS_2_ARGS_PREFIX \
			SINGLE_PASS_ARGS, \
			const_prefixcells_t subblock_prefix

		#define SINGLE_PASS_INPUT \
			intermediate_prefix

		#define SINGLE_PASS_SUBINPUT \
			subblock_prefix

		#define MULTI_PASS_INPUT \
			SINGLE_PASS_INPUT, \
			SINGLE_PASS_SUBINPUT

		#define READ_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			const PrefixCell& icell = intermediate_prefix[WORD_IDX]; \
			Z = icell.z; \
			X = icell.x;

		#define WRITE_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			PrefixCell& icell = intermediate_prefix[WORD_IDX]; \
			icell.z = (Z); \
			icell.x = (X);

		#define XOR_TO_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			PrefixCell& icell = intermediate_prefix[WORD_IDX]; \
			icell.z ^= (Z); \
			icell.x ^= (X);

		#define XOR_FROM_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			const PrefixCell& icell = intermediate_prefix[WORD_IDX]; \
			Z ^= icell.z; \
			X ^= icell.x;

		#define READ_SUBBLOCK_PREFIX(WORD_IDX, Z, X) \
			const PrefixCell& scell = subblock_prefix[WORD_IDX]; \
			Z = scell.z; \
			X = scell.x;

		#define WRITE_SUBBLOCK_PREFIX(WORD_IDX, Z, X) \
			PrefixCell& scell = subblock_prefix[WORD_IDX]; \
			scell.z = (Z); \
			scell.x = (X);

		#define D_PREFIX_XS(WORD_IDX) d_prefix[WORD_IDX].x
		#define D_PREFIX_ZS(WORD_IDX) d_prefix[WORD_IDX].z

		#define D_INTERMEDIATE_PREFIX_X(WORD_IDX) d_intermediate_prefix[WORD_IDX].x
		#define D_INTERMEDIATE_PREFIX_Z(WORD_IDX) d_intermediate_prefix[WORD_IDX].z
		
	#else
		#define PASS_1_ARGS_GLOBAL_PREFIX \
			Table *             prefix_xs, \
            Table *             prefix_zs, \
			word_std_t *        intermediate_prefix_x, \
			word_std_t *        intermediate_prefix_z

		#define PASS_2_ARGS_GLOBAL_PREFIX \
			const_table_t       prefix_xs, \
			const_table_t       prefix_zs, \
			const_words_t       intermediate_prefix_x, \
			const_words_t       intermediate_prefix_z

		#define CHECK_PREFIX_SINGLE_PASS_ARGS \
			const   word_std_t*     other_xs, \
			const   word_std_t*     other_zs
		
		#define CHECK_PREFIX_PASS_1_ARGS \
			const 	Tableau&        other_targets, \
			CHECK_PREFIX_SINGLE_PASS_ARGS

		#define CHECK_PREFIX_SINGLE_PASS_INPUT \
			xblocks(), \
            zblocks()

		#define CHECK_PREFIX_PASS_1_INPUT \
			targets, \
			CHECK_PREFIX_SINGLE_PASS_INPUT

		#define CALL_ARGS_GLOBAL_PREFIX \
			Tableau&            targets, \
			word_std_t *        intermediate_prefix_x, \
			word_std_t *        intermediate_prefix_z

		#define CALL_INPUT_GLOBAL_PREFIX \
			targets, \
			intermediate_prefix_x, \
			intermediate_prefix_z

		#define KERNEL_INPUT_GLOBAL_PREFIX \
			XZ_TABLE(targets), \
			intermediate_prefix_x, \
			intermediate_prefix_z

		#define WRITE_GLOBAL_PREFIX(WORD_IDX, Z, X) \
			(*prefix_zs)[WORD_IDX] = (Z); \
            (*prefix_xs)[WORD_IDX] = (X);

		#define READ_GLOBAL_PREFIX(WORD_IDX, Z, X) \
			Z = (*prefix_zs)[WORD_IDX]; \
            X = (*prefix_xs)[WORD_IDX];

		#define SINGLE_PASS_ARGS \
			word_std_t * intermediate_prefix_x, \
			word_std_t * intermediate_prefix_z
		
		#define PASS_1_ARGS_PREFIX \
			SINGLE_PASS_ARGS, \
			word_std_t * subblocks_prefix_x, \
			word_std_t * subblocks_prefix_z

		#define PASS_2_ARGS_PREFIX \
			SINGLE_PASS_ARGS, \
			const_words_t subblocks_prefix_x, \
			const_words_t subblocks_prefix_z

		#define SINGLE_PASS_INPUT \
			intermediate_prefix_x, \
			intermediate_prefix_z

		#define SINGLE_PASS_SUBINPUT \
			subblocks_prefix_x, \
			subblocks_prefix_z

		#define MULTI_PASS_INPUT \
			SINGLE_PASS_INPUT, \
			SINGLE_PASS_SUBINPUT

		#define READ_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			Z = intermediate_prefix_z[WORD_IDX]; \
			X = intermediate_prefix_x[WORD_IDX];

		#define WRITE_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			intermediate_prefix_z[WORD_IDX] = (Z); \
			intermediate_prefix_x[WORD_IDX] = (X);

		#define XOR_TO_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			intermediate_prefix_z[WORD_IDX] ^= (Z); \
			intermediate_prefix_x[WORD_IDX] ^= (X);

		#define XOR_FROM_INTERMEDIATE_PREFIX(WORD_IDX, Z, X) \
			Z ^= intermediate_prefix_z[WORD_IDX]; \
			X ^= intermediate_prefix_x[WORD_IDX];

		#define READ_SUBBLOCK_PREFIX(WORD_IDX, Z, X) \
			Z = subblocks_prefix_z[WORD_IDX]; \
			X = subblocks_prefix_x[WORD_IDX];

		#define WRITE_SUBBLOCK_PREFIX(WORD_IDX, Z, X) \
			subblocks_prefix_z[WORD_IDX] = (Z); \
			subblocks_prefix_x[WORD_IDX] = (X);

		#define D_PREFIX_XS(WORD_IDX) d_prefix_xs[WORD_IDX]
		#define D_PREFIX_ZS(WORD_IDX) d_prefix_zs[WORD_IDX]
		#define D_INTERMEDIATE_PREFIX_X(WORD_IDX) d_intermediate_prefix_x[WORD_IDX]
		#define D_INTERMEDIATE_PREFIX_Z(WORD_IDX) d_intermediate_prefix_z[WORD_IDX]
			
	#endif
	
}