#pragma once

#include "simulator.hpp"
#include "samples.cuh"

namespace QuaSARQ {

	class Framing : public Simulator {

		size_t              num_shots;
        size_t              total_shots;
        size_t              chunk_start = 0;
        size_t              chunk_index = 0;
        size_t              max_chunk_shots;
        size_t              measurement_offset = 0; // Measurements depth offset in the samples table.
        Samples             samples_record;
        curand_algorithm_t* rand_states = nullptr;
        size_t              rand_states_size = 0;

	public:

		Framing(const size_t& num_shots) :
            Simulator()
            , num_shots(num_shots)
            , total_shots(num_shots)
            , max_chunk_shots(num_shots) {}
		Framing(const string& path, const size_t& num_shots);
		Framing(char* data, const size_t& length, const size_t& num_shots);
        size_t choose_chunk_shots() const;
        void init_rand_states(const uint64& seed,
                              const size_t& num_words_per_table,
                              const size_t& total_words_minor,
                              const size_t& chunk_word_offset,
                              const cudaStream_t& stream);
        void randomize(word_std_t *data, const size_t& num_words, const cudaStream_t& stream);
        void shot(const depth_t& depth_level, const cudaStream_t& stream);
        void step(const depth_t& depth_level);
        void print_observables_sampled(FILE* out = stdout, const cudaStream_t& stream = 0);
        void print_detectors_sampled(FILE* out = stdout, const cudaStream_t& stream = 0);
        size_t sample_device_bytes() const override;
        size_t sample_host_bytes() const override;
        void print(const cudaStream_t& stream = 0);
        void sample();

	};

}
