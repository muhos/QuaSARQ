
#ifndef __CU_ACCESS_H
#define __CU_ACCESS_H

namespace QuaSARQ {

	#define ROW_MAJOR 0
	#define PREFIX_ROW_MAJOR 1

	#if PREFIX_ROW_MAJOR
		#define PREFIX_TABLEAU_INDEX(WORD_IDX, TID) ((TID) + (WORD_IDX) * active_targets)
		#define PREFIX_INTERMEDIATE_INDEX(WORD_IDX, BX) ((BX) + (WORD_IDX) * max_blocks)
		#define PREFIX_SUBINTERMEDIATE_INDEX(WORD_IDX, BX) ((BX) + (WORD_IDX) * max_sub_blocks)
	#else
		#define PREFIX_TABLEAU_INDEX(WORD_IDX, TID) ((TID) * num_words_minor + (WORD_IDX))
		#define PREFIX_INTERMEDIATE_INDEX(WORD_IDX, BX) ((BX) * num_words_minor + (WORD_IDX))
		#define PREFIX_SUBINTERMEDIATE_INDEX(WORD_IDX, BX) PREFIX_INTERMEDIATE_INDEX(WORD_IDX, BX)
	#endif

	#if ROW_MAJOR
		#define TABLEAU_INDEX(WORD_IDX, QUBIT_IDX) ((QUBIT_IDX) + (WORD_IDX) * (2 * num_qubits_padded))
		#define TABLEAU_STAB_OFFSET (num_qubits_padded)
	#else
		#define TABLEAU_INDEX(WORD_IDX, QUBIT_IDX) ((QUBIT_IDX) * num_words_major + (WORD_IDX))
		#define TABLEAU_STAB_OFFSET (num_words_minor)
	#endif
	
}

#endif