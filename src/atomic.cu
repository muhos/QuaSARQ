
#include "atomic.cuh"
#include "warp.cuh"

#include <cooperative_groups.h>

using namespace cooperative_groups;

namespace QuaSARQ {

    NOINLINE_DEVICE uint32 atomicAggInc(uint32* counter) {
        coalesced_group g = coalesced_threads();
        uint32 prev;
        if (g.thread_rank() == 0) {
            prev = atomicAdd(counter, g.num_threads());
        }
        prev = g.thread_rank() + g.shfl(prev, 0);
        return prev;
    }

    NOINLINE_DEVICE uint32 atomicAggMin(uint32* min, const uint32& val) {
        const uint32 activemask = __activemask(), min_id = __ffs(activemask) - 1;
        if (laneID() == min_id)
            atomicMin(min, val);
    }

}