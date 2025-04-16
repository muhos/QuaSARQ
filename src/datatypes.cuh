#pragma once

#include "table.cuh"
#include "signs.cuh"
#include "pivot.cuh"
#include "gate.cuh"
#include "word.cuh"

namespace QuaSARQ {

	typedef const Table* __restrict__ ConstTablePointer;
    typedef const Signs* __restrict__ ConstSignsPointer;
    typedef const pivot_t* __restrict__ CPivotsPtr;
	typedef const bucket_t* __restrict__ ConstBucketsPointer;
    typedef const gate_ref_t* __restrict__ ConstRefsPointer;
    typedef const word_std_t* __restrict__ ConstWordsPointer;

}