
#ifndef __CU_DATATYPES_H
#define __CU_DATATYPES_H

#include "table.cuh"
#include "signs.cuh"
#include "pivot.cuh"
#include "gate.cuh"
#include "word.cuh"
#include "commutation.cuh"

namespace QuaSARQ {

	typedef const Table* __restrict__ ConstTablePointer;
    typedef const Signs* __restrict__ ConstSignsPointer;
    typedef const pivot_t* __restrict__ ConstPivotsPointer;
	typedef const bucket_t* __restrict__ ConstBucketsPointer;
    typedef const gate_ref_t* __restrict__ ConstRefsPointer;
    typedef const word_std_t* __restrict__ ConstWordsPointer;
    typedef const Commutation* __restrict__ ConstCommsPointer;


}

#endif