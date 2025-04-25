
#pragma once

#include "definitions.cuh"
#include "pivot.cuh"
#include "timer.cuh"
#include "circuit.hpp"
#include "statistics.hpp"

namespace QuaSARQ {

	class DeviceCircuit {

		DeviceAllocator& allocator;

		bucket_t* _buckets;
		gate_ref_t* _references;

		bucket_t* _pinned_buckets;
		gate_ref_t* _pinned_references;

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

		inline
		void 		initiate			(const size_t& max_qubits, const size_t& max_references, const size_t& max_buckets) {
			if (!max_qubits || max_qubits > MAX_QUBITS)
				LOGERROR("maximum number of qubits per window is invalid.");
			if (!max_references || max_references > MAX_QUBITS)
				LOGERROR("maximum number of references per window is invalid.");
			if (!max_buckets || max_buckets > NO_REF)
				LOGERROR("maximum number of buckets per window is invalid.");		
			if (this->max_references < max_references) {
				LOGN2(2, "Resizing a (pinned) window for %lld references.. ", int64(max_references));
				this->max_references = max_references;
				_references = allocator.allocate<gate_ref_t>(max_references);
				allocator.resize_pinned<gate_ref_t>(_pinned_references, max_references);
				LOGDONE(2, 4);
			}
			if (this->max_buckets < max_buckets) {
				LOGN2(2, "Resizing a (pinned) window for %lld buckets.. ", int64(max_buckets));
				this->max_buckets = max_buckets;
				_buckets = allocator.allocate<bucket_t>(max_buckets);
				allocator.resize_pinned<bucket_t>(_pinned_buckets, max_buckets);
				LOGDONE(2, 4);
			}
			this->max_qubits = max_qubits;
		}

		inline
		void 		copyfrom 			(Statistics& stats, const Circuit& circuit, const depth_t& depth_level, 
								const bool& reversed, const bool& sync, const cudaStream_t& s1, const cudaStream_t& s2) {
			if (_references == nullptr)
                LOGERROR("cannot copy empty references to device.");
			if (_buckets == nullptr)
                LOGERROR("cannot copy empty gates to device.");
			if (buckets_offset >= circuit.num_buckets()) 
				LOGERROR("buckets offset overflow during gates transfer to GPU.");
			num_gates = circuit[depth_level].size();
			assert(num_gates <= max_qubits);
			const auto curr_num_buckets = circuit.num_buckets(depth_level);
			assert(num_gates <= max_references);
			assert(curr_num_buckets <= max_buckets);
			const auto* window = circuit[depth_level].data();
			const auto* buckets = circuit.data(buckets_offset);
			double ttime = 0;
			if (sync) cutimer.start(s1);
			LOGN2(2, "Copying %lld references and %lld buckets (offset by %c%lld) per depth level %lld %ssynchroneously.. ", 
				int64(num_gates), 
				int64(curr_num_buckets), 
				reversed ? '-' : '+' , 
				int64(buckets_offset), 
				int64(depth_level), 
				sync ? "" : "a");
			copyhost(_pinned_references, window, num_gates, buckets_offset);
			CHECK(cudaMemcpyAsync(_references, _pinned_references, sizeof(gate_ref_t) * num_gates, cudaMemcpyHostToDevice, s1));
			if (sync) { 
				cutimer.stop(s1); 
				ttime += cutimer.time();
				cutimer.start(s2);
			}
			copyhost(_pinned_buckets, buckets, curr_num_buckets, bucket_t(0));
			CHECK(cudaMemcpyAsync(_buckets, _pinned_buckets, BUCKETSIZE * curr_num_buckets, cudaMemcpyHostToDevice, s2));
			if (sync) {
				cutimer.stop(s2);
				ttime += cutimer.time();
				stats.time.transfer += ttime;
				LOG2(2, "done in %f ms.", ttime);
			}
			if (reversed) {
				const size_t num_buckets_prev = depth_level ? circuit.num_buckets(depth_level - 1) : 0;
				assert(buckets_offset >= num_buckets_prev);
				buckets_offset -= (gate_ref_t) num_buckets_prev;
			}
			else {
				buckets_offset += (gate_ref_t) curr_num_buckets;
			}
			if (!sync) LOGDONE(2, 4);
		}

		inline
		void 		copyto 			(Circuit& circuit, const depth_t& depth_level) {
			const auto curr_num_buckets = circuit.num_buckets(depth_level);
			const gate_ref_t prev_buckets_offset = buckets_offset - curr_num_buckets;
			if (prev_buckets_offset >= circuit.num_buckets()) 
				LOGERROR("buckets offset overflow during gates transfer to host.");
			LOGN2(2, "Copying back %lld buckets to host per depth level %lld synchroneously.. ", int64(curr_num_buckets), int64(depth_level));
			CHECK(cudaMemcpy(circuit.data(prev_buckets_offset), _buckets, BUCKETSIZE * curr_num_buckets, cudaMemcpyDeviceToHost));
			LOGDONE(2, 4);
		}

		inline
		bucket_t*    gates			() { return _buckets; }

		inline const
		bucket_t*    gates			() const { return _buckets; }

		inline
		gate_ref_t*  references		() { return _references; }

		inline const
		gate_ref_t*  references		() const { return _references; }

		inline gate_ref_t 	
				get_buckets_offset	() const { return buckets_offset; }

	};
}