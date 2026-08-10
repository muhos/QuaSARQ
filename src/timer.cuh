#pragma once

#include "definitions.cuh"
#include "constants.hpp"

namespace QuaSARQ {

	class cuTimer {
	private:
		cudaEvent_t _start = nullptr, _stop = nullptr;
		float _gpuTime = 0;

		// Moves CUDA event creation to the first start() call,
		// to avoid CUDA context creation on using quasarq as a library.
		inline void ensure() {
			if (_start == nullptr) {
				cudaEventCreate(&_start);
				cudaEventCreate(&_stop);
			}
		}
		
	public:
		cuTimer() = default;
		inline void  start  (const cudaStream_t& _s = 0) { ensure(); cudaEventRecord(_start, _s); }
		inline void  stop   (const cudaStream_t& _s = 0) { ensure(); cudaEventRecord(_stop, _s); }
		// Return kernel time in milliseconds.
		inline float elapsed() {
			_gpuTime = 0;
			if (_stop == nullptr) return _gpuTime;
			cudaEventSynchronize(_stop);
			cudaEventElapsedTime(&_gpuTime, _start, _stop);
			return _gpuTime;
		}
	};

	extern cuTimer cutimer;

}