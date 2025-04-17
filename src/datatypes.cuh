#pragma once

#include "table.cuh"
#include "signs.cuh"
#include "gate.cuh"
#include "word.cuh"

namespace QuaSARQ {

    typedef uint32 pivot_t;
	typedef const Table* __restrict__ const_table_t;
    typedef const Signs* __restrict__ const_signs_t;
    typedef const pivot_t* __restrict__ const_pivots_t;
	typedef const bucket_t* __restrict__ const_buckets_t;
    typedef const gate_ref_t* __restrict__ const_refs_t;
    typedef const word_std_t* __restrict__ const_words_t;

}