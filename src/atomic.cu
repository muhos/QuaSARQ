
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

    #define EXTRACT_BYTE_FROM_ADDR(ADDR,VAL) \
	    uint64 addr_val = (uint64)ADDR; \
        uint32 al_offset = uint32(addr_val & 3) << 3; \
        uint32* byte_addr = reinterpret_cast<uint32*> (addr_val & (0xFFFFFFFFFFFFFFFCULL)); \
        uint32 byte = (VAL << al_offset) \

#if defined(WORD_SIZE_8)
    #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
    NOINLINE_DEVICE word_std_t
    #else
    NOINLINE_DEVICE void
    #endif
    atomicXOR(word_std_t* addr, const uint32& value) {
        assert(value <= WORD_MAX);
		EXTRACT_BYTE_FROM_ADDR(addr, value);
        #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
        return word_std_t((atomicXor(byte_addr, byte) >> al_offset) & 0xFF);
        #else
        atomicXor(byte_addr, byte);
        #endif
    }
#else
    NOINLINE_DEVICE word_std_t atomicXOR(word_std_t* addr, const word_std_t& value) {
        return atomicXor(addr, value);
    }
#endif

    NOINLINE_DEVICE void atomicByteXOR(byte_t* addr, const uint32& value) {
    	EXTRACT_BYTE_FROM_ADDR(addr, value);
        atomicXor(byte_addr, byte);
    }

}