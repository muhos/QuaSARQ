#include "frame.cuh"
#include "random.cuh"

namespace QuaSARQ {

    __global__
    void setup_rand_k(
        curand_algorithm_t* states,
        uint64              seed,
        size_t              total_states)
    {
        for_parallel_x(i, total_states) {
            curand_init(seed, i, 0, &states[i]);
        }
    }

    __global__
    void randomize_kernel(
                curand_algorithm_t* states,
                word_std_t*         data,
        const   size_t              num_words)
    {
        for_parallel_x(w, num_words) {
            curand_algorithm_t local = states[w];
            data[w] = curand_word(&local);
            states[w] = local;
        }
    }

    void Framing::init_rand_states(const uint64& seed, const size_t& num_words_per_table, const cudaStream_t& stream) {
        const size_t sample_states = winfo.max_parallel_gates * tableau.num_words_minor();
        const size_t needed = MAX(num_words_per_table, sample_states);
        if (rand_states_size < needed) {
            rand_states_size = needed;
            rand_states = gpu_allocator.allocate<curand_algorithm_t>(needed, Region::Stable);
        }
        dim3 currentblock(1, 1), currentgrid(1, 1);
        currentblock = bestblockreset;
        OPTIMIZEBLOCKS(currentgrid.x, needed, currentblock.x);
        setup_rand_k<<<currentgrid, currentblock, 0, stream>>>(rand_states, seed, needed);
    }

    void Framing::randomize(word_std_t *data, const size_t& num_words, const cudaStream_t& stream) {
        dim3 currentblock(1, 1), currentgrid(1, 1);
        currentblock = bestblockreset;
        OPTIMIZEBLOCKS(currentgrid.x, num_words, currentblock.x);
        LOGN2(2, "Randomizing %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ",
            num_words, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        double elapsed = 0;
        if (options.sync) cutimer.start(stream);
        randomize_kernel<<<currentgrid, currentblock, 0, stream>>>(rand_states, data, num_words);
        if (options.sync) {
            LASTERR("failed to launch randomize kernel");
            cutimer.stop(stream);
            elapsed = cutimer.elapsed();
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
    }

}

