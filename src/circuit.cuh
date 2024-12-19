
#ifndef __CU_CIRCUIT_H
#define __CU_CIRCUIT_H

#include "definitions.cuh"
#include "pivot.cuh"
#include "timer.cuh"
#include "circuit.hpp"
#include "statistics.hpp"

namespace QuaSARQ {

	template <class ALLOCATOR>
	class DeviceCircuit {

		ALLOCATOR& allocator;

		Pivot* _pivots;
		bucket_t* _buckets;
		gate_ref_t* _references;

		Pivot* _pinned_pivots;
		bucket_t* _pinned_buckets;
		gate_ref_t* _pinned_references;

		size_t max_references;
		size_t max_buckets;

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
		,	_pivots(nullptr)
		,	_buckets(nullptr)
		,	_references(nullptr)
		,	_pinned_pivots(nullptr)
		,	_pinned_buckets(nullptr)
		,	_pinned_references(nullptr)
		,	max_references(0)
		,	max_buckets(0)
		,	buckets_offset(0)
		{ }

		inline 
		void reset_circuit_offset		(const gate_ref_t& buckets_offset) { 
			LOG2(2, "");
		    LOGN2(2, "Initializing offset of window buckets to %lld.. ", int64(buckets_offset));
			assert(buckets_offset < NO_REF);
			this->buckets_offset = buckets_offset;
			LOGDONE(2, 3);
			LOG2(2, "");
		} 

		inline
		void 		initiate			(const size_t& max_references, const size_t& max_buckets) {
			if (!max_references || max_references > MAX_QUBITS)
				LOGERROR("maximum number of references per window is invalid.");
			if (!max_buckets || max_buckets > NO_REF)
				LOGERROR("maximum number of buckets per window is invalid.");
			if (this->max_references < max_references) {
				LOGN2(2, "Resizing a (pinned) window for %lld references.. ", int64(max_references));
				this->max_references = max_references;
				_pivots = allocator.template allocate<Pivot>(max_references);
				allocator.template resize_pinned<Pivot>(_pinned_pivots, max_references);
				_references = allocator.template allocate<gate_ref_t>(max_references);
				allocator.template resize_pinned<gate_ref_t>(_pinned_references, max_references);
				LOGDONE(2, 3);
			}
			if (this->max_buckets < max_buckets) {
				LOGN2(2, "Resizing a (pinned) window for %lld buckets.. ", int64(max_buckets));
				this->max_buckets = max_buckets;
				_buckets = allocator.template allocate<bucket_t>(max_buckets);
				allocator.template resize_pinned<bucket_t>(_pinned_buckets, max_buckets);
				LOGDONE(2, 3);
			}
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
			const auto curr_num_references = circuit[depth_level].size();
			const auto curr_num_buckets = circuit.num_buckets(depth_level);
			assert(curr_num_references <= max_references);
			assert(curr_num_buckets <= max_buckets);
			const auto* window = circuit[depth_level].data();
			const auto* buckets = circuit.data(buckets_offset);
			double ttime = 0;
			if (sync) cutimer.start(s1);
			LOGN2(2, "Copying %lld references and %lld buckets (offset by %c%lld) per depth level %lld %ssynchroneously.. ", 
				int64(curr_num_references), 
				int64(curr_num_buckets), 
				reversed ? '-' : '+' , 
				int64(buckets_offset), 
				int64(depth_level), 
				sync ? "" : "a");
			copyhost(_pinned_references, window, curr_num_references, buckets_offset);
			CHECK(cudaMemcpyAsync(_references, _pinned_references, sizeof(gate_ref_t) * curr_num_references, cudaMemcpyHostToDevice, s1));
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
			if (!sync) LOGDONE(2, 3);
		}

		inline
		void 		copyto 			(Circuit& circuit, const depth_t& depth_level) {
			const auto curr_num_buckets = circuit.num_buckets(depth_level);
			const gate_ref_t prev_buckets_offset = buckets_offset - curr_num_buckets;
			if (prev_buckets_offset >= circuit.num_buckets()) 
				LOGERROR("buckets offset overflow during gates transfer to host.");
			LOGN2(2, "Copying back %lld buckets to host per depth level %lld synchroneously.. ", int64(curr_num_buckets), int64(depth_level));
			CHECK(cudaMemcpy(circuit.data(prev_buckets_offset), _buckets, BUCKETSIZE * curr_num_buckets, cudaMemcpyDeviceToHost));
			LOGDONE(2, 3);
		}

		inline
		void 		copypivots		(const cudaStream_t& stream, const size_t& num_gates) {
			LOGN2(2, "Copying back %lld pivots to host asynchroneously.. ", int64(num_gates));
			CHECK(cudaMemcpyAsync(_pinned_pivots, _pivots, sizeof(Pivot) * num_gates, cudaMemcpyDeviceToHost, stream));
			LOGDONE(2, 3);
		}

		inline
		void 		copygateto 		(Circuit& circuit, const gate_ref_t& host_ref, const depth_t& depth_level, const cudaStream_t& stream) {
			const size_t prev_buckets_offset = circuit.reference(depth_level, 0);
			assert(host_ref >= prev_buckets_offset);
			const gate_ref_t device_ref = host_ref - prev_buckets_offset;
			const Gate* host_gate = circuit.gateptr(host_ref);
			const size_t num_buckets = NBUCKETS(host_gate->size);
			LOGN2(2, "Copying back gate");
			if (options.verbose >= 2) host_gate->print(true);
			LOGN2(2, " to host asynchroneously.. ");
			CHECK(cudaMemcpyAsync(circuit.data(host_ref), _buckets + device_ref, BUCKETSIZE * num_buckets, cudaMemcpyDeviceToHost, stream));
			LOGDONE(2, 3);
		}

		inline
		void 		copypivotto 	(Pivot& pivot, const uint32& gate_index, const cudaStream_t& stream) {
			LOGN2(2, "Copying gate pivot to host asynchroneously.. ");
			
			CHECK(cudaMemcpyAsync(&(pivot), _pivots + gate_index, sizeof(Pivot), cudaMemcpyDeviceToHost, stream));
			LOGDONE(2, 3);
		}

		inline
		Pivot*    	 pivots			() { return _pivots; }

		inline const
		Pivot*    	 pivots			() const { return _pivots; }

		inline
		Pivot*    	 host_pivots	() { return _pinned_pivots; }

		inline const
		Pivot*    	 host_pivots	() const { return _pinned_pivots; }

		inline
		bucket_t*    gates			() { return _buckets; }

		inline const
		bucket_t*    gates			() const { return _buckets; }

		inline
		gate_ref_t*  references		() { return _references; }

		inline const
		gate_ref_t*  references		() const { return _references; }

		inline 
		gate_ref_t 	get_buckets_offset() const { return buckets_offset; }

	};
}

#endif