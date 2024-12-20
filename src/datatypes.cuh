
#ifndef __CU_DATATYPES_H
#define __CU_DATATYPES_H

#include "table.cuh"
#include "signs.cuh"
#include "pivot.cuh"
#include "gate.cuh"

namespace QuaSARQ {

	typedef const Table* __restrict__ ConstTablePointer;
    typedef const Signs* __restrict__ ConstSignsPointer;
    typedef const Pivot* __restrict__ ConstPivotsPointer;
	typedef const bucket_t* __restrict__ ConstBucketsPointer;
    typedef const gate_ref_t* __restrict__ ConstRefsPointer;

}

#endif