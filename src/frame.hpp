#pragma once

#include "simulator.hpp"
#include "record.cuh"

namespace QuaSARQ {

	class Framing : public Simulator {

		size_t num_shots;
        Recorder recorder;
        
	public:

		Framing(const size_t& num_shots) : Simulator() {}
		Framing(const string& path, const size_t& num_shots);
        void randomize(word_std_t *data, const size_t& num_words, const cudaStream_t& stream);
        void shot(const depth_t& depth_level, const cudaStream_t& stream);
        void step(const depth_t& depth_level);
        void print();
        void report();
        void sample();

	};

}