#pragma once

#include <curand_kernel.h>
#include "simulator.hpp"
#include "samples.cuh"

namespace QuaSARQ {

	class Framing : public Simulator {

		size_t              num_shots;
        Samples             samples_record;
        curand_algorithm_t* rand_states;
        size_t              rand_states_size;

	public:

		Framing(const size_t& num_shots) : Simulator(), rand_states(nullptr), rand_states_size(0) {}
		Framing(const string& path, const size_t& num_shots);
        void init_rand_states(const uint64& seed, 
                              const size_t& num_words_per_table,
                              const cudaStream_t& stream);
        void randomize(word_std_t *data, const size_t& num_words, const cudaStream_t& stream);
        void shot(const depth_t& depth_level, const cudaStream_t& stream);
        void step(const depth_t& depth_level);
        void print();
        void report();
        void sample();

	};

}