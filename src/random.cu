#include "frame.hpp"
#include "random.cuh"

namespace QuaSARQ {

    __global__
    void randomize_kernel(
                word_std_t* data,
        const   size_t      num_words,
        const   uint64      seed)
    {
        curandStatePhilox4_32_10_t st;
        for_parallel_x(w, num_words) {
            randomize_word(data[w], st, seed, global_tx);
        }
    }

    void Framing::randomize(word_std_t *data, const size_t& num_words, const cudaStream_t& stream) {
        dim3 currentblock(1, 1), currentgrid(1, 1);
        currentblock = bestblockreset;
        OPTIMIZEBLOCKS(currentgrid.x, num_words, currentblock.x);
        LOGN2(2, "Randomizing %lld words with block(x:%u, y:%u) and grid(x:%u, y:%u).. ",
            num_words, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        double elapsed = 0;
        if (options.sync) cutimer.start(stream);
        randomize_kernel <<< currentgrid, currentblock, 0, stream >>> (data, num_words, options.seed);
        if (options.sync) {
            LASTERR("failed to launch randomize kernel");
            cutimer.stop(stream);
            elapsed = cutimer.elapsed();
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
    }

}

