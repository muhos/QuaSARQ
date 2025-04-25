#pragma once

#include "tableau.cuh"

namespace QuaSARQ {

	#define CHECK_IDENTITY_ARGS \
		const   size_t offset, \
        const   size_t num_qubits, \
        const   size_t num_words_major, \
                Table* xs, \
                Table* zs

	bool check_identity(
        const Tableau&  tableau, 
        const size_t&   offset_per_partition, 
        const size_t&   num_qubits_per_partition,
        const bool&     measuring);

}