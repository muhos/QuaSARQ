
#ifndef __CU_ACCESS_H
#define __CU_ACCESS_H

namespace QuaSARQ {

	#define ROW_MAJOR 1

	#if ROW_MAJOR
		#define TABLEAU_INDEX(WORD_IDX, QUBIT_IDX) ((QUBIT_IDX) + (WORD_IDX) * (2 * num_qubits_padded))
		#define TABLEAU_STAB_OFFSET (num_qubits_padded)
	#else
		#define TABLEAU_INDEX(WORD_IDX, QUBIT_IDX) ((QUBIT_IDX) * num_words_major + (WORD_IDX))
		#define TABLEAU_STAB_OFFSET (num_words_minor)
	#endif
	
}

#endif