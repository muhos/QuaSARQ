#ifndef __CU_PIVOT_H
#define __CU_PIVOT_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "logging.hpp"

namespace QuaSARQ {

    typedef uint32 pivot_t;

    constexpr pivot_t INVALID_PIVOT = UINT32_MAX;

    struct Pivoting {

        DeviceAllocator& allocator;

        pivot_t* pivots;
        pivot_t* host_pivots;
        uint32* d_active_pivots;
        uint32* h_active_pivots;
        byte_t* auxiliary;

        size_t num_qubits;
        size_t auxiliary_bytes;
        
        Pivoting(DeviceAllocator& allocator) :
                allocator(allocator)
            ,	pivots(nullptr)
		    ,   auxiliary(nullptr)
            ,	host_pivots(nullptr)
            ,	d_active_pivots(nullptr)
            ,	h_active_pivots(nullptr)
            ,   num_qubits(0)
            ,   auxiliary_bytes(0) {}

        DEVICE __forceinline__
        bool operator()(const pivot_t &a) const {
            return a != INVALID_PIVOT;
        }

        inline
        void alloc(const size_t &num_qubits) {
            this->num_qubits = num_qubits;
            pivots = allocator.allocate<pivot_t>(num_qubits + 1); // extra pivot for marking commutations.
			allocator.resize_pinned<pivot_t>(host_pivots, num_qubits + 1);
            d_active_pivots = allocator.allocate<uint32>(1);
            allocator.resize_pinned<pivot_t>(h_active_pivots, 1);
        }

        inline
		void copypivots(const cudaStream_t& stream, const size_t& num_pivots) {
            assert(num_pivots <= num_qubits);
			LOGN2(2, " Copying %lld of initial pivots to host asynchroneously.. ", int64(num_pivots));
			CHECK(cudaMemcpyAsync(host_pivots, pivots, sizeof(pivot_t) * num_pivots, cudaMemcpyDeviceToHost, stream));
			LOGDONE(2, 4);
		}

        void compact_pivots(const cudaStream_t& stream);
    };

}

#endif