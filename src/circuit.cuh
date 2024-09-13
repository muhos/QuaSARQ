
#ifndef __CU_CIRCUIT_H
#define __CU_CIRCUIT_H

#include "definitions.cuh"
#include "vector.cuh"
#include "timer.cuh"
#include "circuit.hpp"
#include "statistics.hpp"

namespace QuaSARQ {

	template <class ALLOCATOR>
	class DeviceCircuit {

		ALLOCATOR& allocator;

		bucket_t* _buckets;
		gate_ref_t* _references;

		bucket_t* cached_buckets;
		gate_ref_t* cached_references;

		bucket_t* pinned_buckets;
		gate_ref_t* pinned_references;

		size_t max_references;
		size_t max_buckets;

		gate_ref_t num_references_prev;
		gate_ref_t num_buckets_prev;

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

		INLINE_ALL DeviceCircuit(ALLOCATOR& allocator) : 
			allocator(allocator)
		,	_buckets(nullptr)
		,	_references(nullptr)
		,	cached_buckets(nullptr)
		,	cached_references(nullptr)
		,	pinned_buckets(nullptr)
		,	pinned_references(nullptr)
		,	max_references(0)
		,	max_buckets(0)
		,	num_references_prev(0)
		,	num_buckets_prev(0)
		,	buckets_offset(0)

		{ }

		inline 
		void reset_circuit_offsets (const gate_ref_t& references_offset, const gate_ref_t& buckets_offset) { 
			LOG2(2, "");
		    LOGN2(2, "Initializing window offsets to %lld and %lld respectively.. ", int64(references_offset), int64(buckets_offset));
			assert(references_offset < NO_REF);
			assert(buckets_offset < NO_REF);
			num_references_prev = 0;
			num_buckets_prev = 0;
			this->buckets_offset = buckets_offset;
			cached_references = _references;
			cached_buckets = _buckets;
			LOGDONE(2, 3);
			LOG2(2, "");
		} 

		inline
		void 		initiate	(const size_t& max_references, const size_t& max_buckets) {
			if (!max_references || max_references > MAX_QUBITS)
				LOGERROR("maximum number of references per window is invalid.");
			if (!max_buckets || max_buckets > NO_REF)
				LOGERROR("maximum number of buckets per window is invalid.");
			if (this->max_references < max_references) {
				LOGN2(2, "Resizing a (pinned) window for %lld references.. ", int64(max_references));
				this->max_references = max_references;
				_references = allocator.template allocate<gate_ref_t>(max_references);
				cached_references = _references;
				allocator.template resize_pinned<gate_ref_t>(pinned_references, max_references);
				LOGDONE(2, 3);
			}
			if (this->max_buckets < max_buckets) {
				LOGN2(2, "Resizing a (pinned) window for %lld buckets.. ", int64(max_buckets));
				this->max_buckets = max_buckets;
				_buckets = allocator.template allocate<bucket_t>(max_buckets);
				cached_buckets = _buckets;
				allocator.template resize_pinned<bucket_t>(pinned_buckets, max_buckets);
				LOGDONE(2, 3);
			}
		}

		inline
		void 		copyfrom 	(Statistics& stats, const Circuit& circuit, const depth_t& depth_level
								, const bool& reversed, const bool& sync, const cudaStream_t& s1, const cudaStream_t& s2) {
			if (_references == nullptr)
                LOGERROR("cannot copy empty references to device.");
			if (_buckets == nullptr)
                LOGERROR("cannot copy empty gates to device.");
			if (buckets_offset >= circuit.num_buckets()) 
				LOGERROR("buckets offset overflow during gates transfer to GPU.");
			const auto curr_num_references = circuit[depth_level].size();
			const auto curr_num_buckets = circuit.num_buckets(depth_level);
			assert(curr_num_references <= max_references);
			assert(curr_num_buckets <= max_buckets);
			const auto* window = circuit[depth_level].data();
			const auto* buckets = circuit.data(buckets_offset);
			double ttime = 0;
			if (sync) cutimer.start(s1);
			LOGN2(1, "Copying %lld references and %lld buckets (offset by %c%lld) per depth level %lld %ssynchroneously.. ", 
				int64(curr_num_references), 
				int64(curr_num_buckets), 
				reversed ? '-' : '+' , 
				int64(buckets_offset), 
				int64(depth_level), 
				sync ? "" : "a");
			copyhost(pinned_references, window, curr_num_references, buckets_offset);
			CHECK(cudaMemcpyAsync(_references + num_references_prev, pinned_references, sizeof(gate_ref_t) * curr_num_references, cudaMemcpyHostToDevice, s1));
			if (sync) { 
				cutimer.stop(s1); 
				ttime += cutimer.time();
				cutimer.start(s2);
			}
			copyhost(pinned_buckets, buckets, curr_num_buckets, bucket_t(0));
			CHECK(cudaMemcpyAsync(_buckets + num_buckets_prev, pinned_buckets, BUCKETSIZE * curr_num_buckets, cudaMemcpyHostToDevice, s2));
			if (sync) {
				cutimer.stop(s2);
				ttime += cutimer.time();
				stats.time.transfer += ttime;
				LOG2(1, "done in %f ms.", ttime);
			}
			if (reversed) {
				const size_t num_buckets_prev = depth_level ? circuit.num_buckets(depth_level - 1) : 0;
				assert(buckets_offset >= num_buckets_prev);
				buckets_offset -= (gate_ref_t) num_buckets_prev;
			}
			else {
				buckets_offset += (gate_ref_t) curr_num_buckets;
			}
			if (!sync) LOGDONE(1, 3);
		}

		inline
		bucket_t*    gates		() { return _buckets; }

		inline const
		bucket_t*    gates		() const { return _buckets; }

		inline
		gate_ref_t*  references	() { return cached_references; }

		inline const
		gate_ref_t*  references	() const { return cached_references; }

		inline
		void advance_references() { 
			cached_references = _references + num_references_prev;
		}

		inline
		void advance_references(const gate_ref_t& curr_num_references, const bool& reversed) { 
			if (reversed) {
				assert(num_references_prev >= curr_num_references);
				num_references_prev -= curr_num_references;
			}
			else {
				num_references_prev += curr_num_references;
			}
			LOGN2(1, "Offsetting references by %c%lld.. ", reversed ? '-' : '+', int64(num_references_prev));
			cached_references = _references + num_references_prev;
			LOGDONE(1, 3);
		}

	};
}

#endif