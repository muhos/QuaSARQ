#pragma once

#include "definitions.cuh"
#include "datatypes.cuh"
#include "datatypes.hpp"
#include "logging.hpp"

namespace QuaSARQ {

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

	__global__ 
    void reset_all_pivots(pivot_t* pivots, const size_t num_gates);

	__global__ 
    void anti_commuting_pivots(
                pivot_t*            scatter,
                const_table_t       inv_xs, 
        const   qubit_t             qubit, 
        const   size_t              num_qubits, 
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded);

    void tune_reset_pivots(
		void (*kernel)(
				pivot_t*, 
		const 	size_t),
				dim3& 		bestBlock,
				dim3& 		bestGrid,
				pivot_t* 	pivots,
		const 	size_t 		size);

    void tune_finding_new_pivots(
		void (*kernel)(
				pivot_t*,
				const_table_t,
		const 	qubit_t,
		const 	size_t,
		const 	size_t,
		const 	size_t,
		const 	size_t),
				dim3& 				bestBlock,
				dim3& 				bestGrid,
		const 	size_t& 			shared_element_bytes,
				pivot_t* 			pivots,
				const_table_t 	    inv_xs,
		const 	qubit_t& 			qubit,
		const 	size_t& 			size,
		const 	size_t 				num_words_major,
		const 	size_t 				num_words_minor,
		const 	size_t 				num_qubits_padded);

    void tune_finding_all_pivots(
		void (*kernel)(
				pivot_t*,
				const_buckets_t,
				const_refs_t,
				const_table_t,
		const 	size_t,
		const 	size_t,
		const 	size_t,
		const 	size_t,
		const 	size_t),
				dim3& 				bestBlock,
				dim3& 				bestGrid,
		const 	size_t& 			shared_element_bytes,
		const 	bool& 				shared_size_yextend,
		const 	size_t& 			data_size_in_x,
		const 	size_t& 			data_size_in_y,
				pivot_t* 			pivots,
				const_buckets_t     measurements,
				const_refs_t 	    refs,
				const_table_t 	    inv_xs,
		const 	size_t 				num_gates,
		const 	size_t 				num_qubits,
		const 	size_t 				num_words_major,
		const 	size_t 				num_words_minor,
		const 	size_t 				num_qubits_padded);

}