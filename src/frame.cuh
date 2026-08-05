#pragma once

#include "simulator.hpp"
#include "samples.cuh"

namespace QuaSARQ {

	constexpr size_t COLLECT_STAGING_BYTES = 32 * MB;

	struct FrameResults {
		uint8*  detectors;
		uint8*  observables;
		size_t  detectors_stride;
		size_t  observables_stride;
		bool    bit_packed;

		FrameResults() :
			detectors(nullptr)
			, observables(nullptr)
			, detectors_stride(0)
			, observables_stride(0)
			, bit_packed(false) {}

		static size_t stride_of(const size_t& num_units, const bool& bit_packed) {
			return bit_packed ? (num_units + 7) / 8 : num_units;
		}
	};

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
        FrameResults*       results = nullptr;
        bool                sample_host_required = false;
        bool                reference_ready = false;

	public:

		Framing(const size_t& num_shots) :
            Simulator()
            , num_shots(num_shots)
            , total_shots(num_shots)
            , max_chunk_shots(num_shots) {}

		Framing(const string& path, const size_t& num_shots);
		Framing(char* data, const size_t& length, const size_t& num_shots, const bool& require_detectors = false);

        size_t choose_chunk_shots() const;
        void randomize(word_std_t *data, const size_t& num_words, const cudaStream_t& stream);
        void shot(const depth_t& depth_level, const cudaStream_t& stream);
        void step(const depth_t& depth_level);
        void print_observables_sampled(FILE* out = stdout, const cudaStream_t& stream = 0);
        void print_detectors_sampled(FILE* out = stdout, const cudaStream_t& stream = 0);
        bool needs_sample_host() const;
        size_t sample_device_bytes() const override;
        size_t sample_host_bytes() const override;
        void print(const cudaStream_t& stream = 0);
        void sample();
        void collect_frame_refs(uint8* dest,
                                const size_t& dest_stride,
                                const uint32& n,
                                const RecordRefs& refs,
                                const cudaStream_t& stream,
                                const char* label);
        void init_rand_states(const uint64& seed,
                                const size_t& num_words_per_table,
                                const size_t& total_words_minor,
                                const size_t& chunk_word_offset,
                                const cudaStream_t& stream);

        inline size_t shots() const { return total_shots; }
        inline void require_sample_host(const bool& on) { sample_host_required = on; }
        inline void set_results(FrameResults* target) { results = target; }
        inline void set_shots(const size_t& shots) {
            num_shots = shots;
            total_shots = shots;
            max_chunk_shots = shots;
        }

	};

}
