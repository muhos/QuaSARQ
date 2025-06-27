#pragma once

#include <chrono>
#include "constants.hpp"

namespace QuaSARQ {

	class Timer {

	private:

		double _time;
		std::chrono::steady_clock::time_point _start, _end;

    public:

		Timer() {
			RESETSTRUCT(this);
		}

		inline void  start  () { _start = std::chrono::steady_clock::now(); }
		inline void  stop   () { _end = std::chrono::steady_clock::now(); }
		// Report time in milliseconds.
		inline double elapsed () {
			_time = double(std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count());
			return _time / 1000.0;
		}
	};

	extern Timer timer;

	// Benchmark a 'cpu' function up to NSAMPLES times 
    // and record the time in AVGTIME per ms.
	#define BENCHMARK_CPU(FUN, AVGTIME, NSAMPLES, ...) \
	do { \
		double runtime = 0; \
		for (size_t sample = 0; sample < NSAMPLES; sample++) { \
			timer.start(); \
			FUN ( __VA_ARGS__ ); \
			timer.stop(); \
			runtime += timer.elapsed(); \
		} \
		AVGTIME = (runtime / NSAMPLES); \
	} while(0)

}