
#include "atomic.cuh"

namespace QuaSARQ {

    NOINLINE_DEVICE uint32 atomicAggInc(uint32* counter) {
        const uint32 mask = __activemask(), total = __popc(mask);
        uint32 lane_mask = 0;
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask));
        const uint32 prefix = (uint32)__popc(mask & lane_mask);
        const int lowest_lane = __ffs(mask) - 1;
        uint32 warpRes = prefix ? 0 : atomicAdd(counter, total);
        warpRes = __shfl_sync(mask, warpRes, lowest_lane);
        return (prefix + warpRes);
    }

    NOINLINE_DEVICE void lock(int* mutex) {
        while (atomicCAS(mutex, 0, 1) != 0);
    }

    NOINLINE_DEVICE void unlock(int* mutex) {
        atomicExch(mutex, 0);
    }

}