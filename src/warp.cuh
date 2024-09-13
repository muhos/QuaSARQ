#ifndef __CU_WARP_H
#define __CU_WARP_H

#include "definitions.cuh"

namespace QuaSARQ {

	INLINE_DEVICE unsigned laneID() {
        unsigned ret; 
        asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
        return ret;
    }

    INLINE_DEVICE unsigned laneMaskLess() {
        unsigned ret; 
        asm volatile ("mov.u32 %0, %%lanemask_lt;" : "=r"(ret));
        return ret;
    }
}

#endif