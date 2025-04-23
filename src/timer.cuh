#pragma once

#include "definitions.cuh"
#include "constants.hpp"

namespace QuaSARQ {

	class cuTimer {
	private:
		cudaEvent_t _start, _stop;
		float _gpuTime;
	public:
		cuTimer() {
			RESETSTRUCT(this);
			cudaEventCreate(&_start);
			cudaEventCreate(&_stop);
		}
		~cuTimer() {
			cudaEventDestroy(_start);
			cudaEventDestroy(_stop);
			_start = nullptr, _stop = nullptr;
		}
		inline void  start  (const cudaStream_t& _s = 0) { cudaEventRecord(_start, _s); }
		inline void  stop   (const cudaStream_t& _s = 0) { cudaEventRecord(_stop, _s); }
		// Return kernel time in milliseconds.
		inline float time	() {
			_gpuTime = 0;
			cudaEventSynchronize(_stop);
			cudaEventElapsedTime(&_gpuTime, _start, _stop);
			return _gpuTime;
		}
	};

	extern cuTimer cutimer;

}