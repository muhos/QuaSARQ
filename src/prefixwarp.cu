#include "simulator.hpp"
#include "prefix.cuh"
#include "collapse.cuh"
#include "templatedim.cuh"
#include "warp.cuh"

namespace QuaSARQ {

    __global__
    void inject_cx_warp_1(
                Table*          inv_xs,
                Table*          inv_zs,
                Signs*          inv_ss,
                const_pivots_t  pivots,
        const   size_t          active_targets,
        const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded)
    {
        assert(active_targets == 1);
        assert(blockDim.y == 1);
        assert(gridDim.y == 1);
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();
        sign_t *ss = inv_ss->data();
        for_parallel_x(w, num_words_minor) {
            const unsigned mask = __activemask();
            pivot_t pivot, t;
            if (!laneID()) {
                pivot = pivots[0];
                t = pivots[1];
            }
            pivot = __shfl_sync(mask, pivot, 0, blockDim.x);
            t = __shfl_sync(mask, t, 0, blockDim.x);
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t t_destab = TABLEAU_INDEX(w, t);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
            const size_t signs_stab_idx = w + num_words_minor;
            inject_cx_half(xs[c_destab], zs[c_destab], xs[t_destab], zs[t_destab], ss[w]);
            inject_cx_half(xs[c_stab], zs[c_stab], xs[t_stab], zs[t_stab], ss[signs_stab_idx]);
        }
    }

    __global__
    void inject_cx_warp(
                Table*          inv_xs,
                Table*          inv_zs,
                Signs*          inv_ss,
                const_pivots_t  pivots,
        const   size_t          active_targets,
        const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded)
    {
        assert(active_targets > 1);

        word_std_t * __restrict__ xs = inv_xs->words();
        word_std_t * __restrict__ zs = inv_zs->words();
        sign_t     * __restrict__ ss = inv_ss->data();

        const pivot_t pivot = pivots[0];

        for_parallel_x(w, num_words_minor) {

            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab   = c_destab + TABLEAU_STAB_OFFSET;

            const word_std_t cx_destab = xs[c_destab];
            const word_std_t cx_stab   = xs[c_stab];
            word_std_t cz_destab = zs[c_destab];
            word_std_t cz_stab   = zs[c_stab];
            sign_t s_destab = 0, s_stab = 0;

            #pragma unroll 4
            for (int i = 0; i < active_targets; i++) {
                const size_t t_destab = TABLEAU_INDEX(w, pivots[i + 1]);
                const size_t t_stab   = t_destab + TABLEAU_STAB_OFFSET;
                const word_std_t tx_destab = xs[t_destab], tz_destab = zs[t_destab];
                s_destab      ^= cx_destab & tz_destab & ~(tx_destab ^ cz_destab);
                cz_destab     ^= tz_destab;
                xs[t_destab]   = tx_destab ^ cx_destab;
                const word_std_t tx_stab = xs[t_stab], tz_stab = zs[t_stab];
                s_stab        ^= cx_stab & tz_stab & ~(tx_stab ^ cz_stab);
                cz_stab       ^= tz_stab;
                xs[t_stab]     = tx_stab ^ cx_stab;
            }

            zs[c_destab]             = cz_destab;
            zs[c_stab]               = cz_stab;
            ss[w]                   ^= s_destab;
            ss[w + num_words_minor] ^= s_stab;
        }
    }

    #define CALL_INJECT_CX_WARP(KERNEL) \
        KERNEL <<<currentgrid, currentblock, 0, stream>>> ( \
            XZ_TABLE(input), \
            input.signs(), \
            pivots, \
            active_targets, \
            num_words_major, \
            num_words_minor, \
            num_qubits_padded);

    double Prefix::scan_warp(Tableau& input, const pivot_t* pivots, const size_t& active_targets, const cudaStream_t& stream) {
        const size_t num_qubits_padded = input.num_qubits_padded();
        const size_t pow2_active_targets = nextPow2(active_targets);
        if (pow2_active_targets > 32) {
            LOGERROR("power-of-2 active targets %d exceeds maximum warp size of 32", pow2_active_targets);
        }

        dim3 currentblock(1, 1), currentgrid(1, 1);
        tune_grid_size(currentblock, currentgrid, pow2_active_targets);
        double elapsed = 0;
        if (options.sync) cutimer.start(stream);
        currentblock.x = currentblock.y;
        currentgrid.x  = currentgrid.y;
        currentblock.y = 1;
        currentgrid.y  = 1;
        LOGN2(2, "Injecting CX for %d targets using warp(x:%u) and grid(x:%u).. ",
            active_targets, currentblock.x, currentgrid.x);
        if (options.sync) cutimer.start(stream);
        if (active_targets == 1) {
            CALL_INJECT_CX_WARP(inject_cx_warp_1);
        } else {
            CALL_INJECT_CX_WARP(inject_cx_warp);
        }
        if (options.sync) {
            LASTERR("failed to launch inject_cx_warp kernel");
            cutimer.stop(stream);
            elapsed = cutimer.elapsed();
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            checker.check_inject_cx(input);
        }
        return elapsed;
    }

}
