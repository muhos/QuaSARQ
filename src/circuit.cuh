
#pragma once

#include "circuit.hpp"
#include "statistics.hpp"
#include "definitions.cuh"
#include "datatypes.cuh"
#include "pivot.cuh"
#include "timer.cuh"

namespace QuaSARQ {

	class DeviceCircuit {

		DeviceAllocator& allocator;

		bucket_t* _buckets;
		gate_ref_t* _references;

		bucket_t* _pinned_buckets;
		gate_ref_t* _pinned_references;

		curand_algorithm_t* _noise_states;
		uint32* _noise_paulis;
		size_t  _max_noise_gates;

		size_t max_references;
		size_t max_buckets;
		size_t num_gates;
		size_t max_qubits;

		gate_ref_t buckets_offset;

		template <class T>
		void copyhost(T* dest, const T* src, const size_t& size, const T& off) {
			if (!off) {
				std::memcpy(dest, src, size * sizeof(T));
			}
			else {
				for (size_t i = 0; i < size; ++i) {
					assert(src[i] >= off);
					dest[i] = src[i] - off;
				}
			}
		}

	public:

		INLINE_ALL DeviceCircuit(DeviceAllocator& allocator) :
			allocator(allocator)
		,	_buckets(nullptr)
		,	_references(nullptr)
		,	_pinned_buckets(nullptr)
		,	_pinned_references(nullptr)
		,	_noise_states(nullptr)
		,	_noise_paulis(nullptr)
		,	_max_noise_gates(0)
		,	max_references(0)
		,	max_buckets(0)
		,	num_gates(0)
		,	max_qubits(0)
		,	buckets_offset(0)
		{ }

		inline 
		void reset_circuit_offset		(const gate_ref_t& buckets_offset) { 
			LOG2(2, "");
		    LOGN2(2, "Initializing offset of window buckets to %lld.. ", int64(buckets_offset));
			assert(buckets_offset < NO_REF);
			this->buckets_offset = buckets_offset;
			LOGDONE(2, 4);
			LOG2(2, "");
		} 

		void initiate					(
											const size_t& 		max_qubits, 
											const size_t& 		max_references, 
											const size_t& 		max_buckets);
		void init_noise_states			(
											const uint64& 		seed, 
											const size_t& 		max_gates, 
											const cudaStream_t& stream);
		void copyto 					(
											Circuit& 			circuit, 
											const depth_t& 		depth_level);
		void copyfrom 					(	
											Statistics& 		stats, 
											Circuit& 			circuit, 
											const depth_t& 		depth_level, 
											const bool& 		reversed, 
											const bool& 		sync, 
											const cudaStream_t& s1, 
											const cudaStream_t& s2);
		

		inline
		bucket_t*    		gates				() { return _buckets; }
		inline const
		bucket_t*    		gates				() const { return _buckets; }
		inline
		gate_ref_t*  		references			() { return _references; }
		inline const
		gate_ref_t*  		references			() const { return _references; }
		inline 
		gate_ref_t 	 		get_buckets_offset	() const { return buckets_offset; }
		inline
		curand_algorithm_t* noise_states		() { return _noise_states; }
		inline
		uint32*             noise_paulis		() { return _noise_paulis; }
		inline
		size_t              max_noise_gates		() const { return _max_noise_gates; }

	};
}