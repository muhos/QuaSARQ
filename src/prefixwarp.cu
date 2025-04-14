#include "simulator.hpp"
#include "prefix.cuh"
#include "collapse.cuh"
#include "templatedim.cuh"

namespace QuaSARQ {

    template<int B>
    INLINE_DEVICE 
    word_std_t warp_exclusive_xor(
                word_std_t      target, 
                word_std_t&     initial_control, 
        const   uint32&         active_targets) 
    {
        unsigned mask = __activemask();
        word_std_t prev, in = target;
        if (B >= 2) {
            prev = __shfl_up_sync(mask, target, 1); if (threadIdx.x >= 1) target ^= prev;
        }
        if (B >= 4) {
            prev = __shfl_up_sync(mask, target, 2); if (threadIdx.x >= 2) target ^= prev;
        }
        if (B >= 8) {
            prev = __shfl_up_sync(mask, target, 4); if (threadIdx.x >= 4) target ^= prev;
        }
        if (B >= 16) {
            prev = __shfl_up_sync(mask, target, 8); if (threadIdx.x >= 8) target ^= prev;
        }
        if (B >= 32) {
            prev = __shfl_up_sync(mask, target, 16); if (threadIdx.x >= 16) target ^= prev;
        }
        word_std_t warp_prefix = !threadIdx.x ? initial_control : initial_control ^ target ^ in;
        if (threadIdx.x == active_targets - 1) {
            initial_control ^= target;
        }
        return warp_prefix;
    }

     __global__ 
    void inject_cx_warp_1(
        Table* inv_xs, 
        Table* inv_zs, 
        Signs* inv_ss, 
        const pivot_t* pivots,
        const size_t active_targets, 
        const size_t num_words_major, 
        const size_t num_words_minor,
        const size_t num_qubits_padded) 
    {
        assert(active_targets == 1);
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();
        sign_t *ss = inv_ss->data();
        for_parallel_x(w, num_words_minor) { 
            const pivot_t pivot = pivots[0];
            assert(pivot != INVALID_PIVOT);
            const size_t t = pivots[1];
            assert(t != pivot);
            assert(t != INVALID_PIVOT);
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t t_destab = TABLEAU_INDEX(w, t);
            word_std_t zt_destab = zs[t_destab];
            word_std_t xt_destab = xs[t_destab];
            word_std_t prefix_zc = zs[c_destab];
            word_std_t prefix_xc = xs[c_destab];
            zs[c_destab] ^= zt_destab;
            xs[c_destab] ^= xt_destab;
            sign_t local_destab_s = 0;
            sign_t local_stab_s = 0;
            const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            compute_local_sign_per_block(local_destab_s, zs[t_stab], prefix_zc, zs[c_stab], zt_destab);
            compute_local_sign_per_block(local_stab_s, xs[t_stab], prefix_xc, xs[c_stab], xt_destab);
            ss[w] ^= local_destab_s;
            ss[w + num_words_minor] ^= local_stab_s;
        }
    }

    template<int B>
    __global__ 
    void inject_cx_warp(
        Table* inv_xs, 
        Table* inv_zs, 
        Signs* inv_ss, 
        const pivot_t* pivots,
        const size_t active_targets, 
        const size_t num_words_major, 
        const size_t num_words_minor,
        const size_t num_qubits_padded) 
    {
        assert(active_targets > 0);
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();
        sign_t *ss = inv_ss->data();
        for_parallel_y(w, num_words_minor) { 
            const pivot_t pivot = pivots[0];
            assert(pivot != INVALID_PIVOT);
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            word_std_t xt_destab = 0;
            word_std_t zt_destab = 0;
            grid_t tid = threadIdx.x;
            if (tid < active_targets) {
                const size_t t = pivots[tid + 1];
                assert(t != pivot);
                assert(t != INVALID_PIVOT);
                const size_t t_destab = TABLEAU_INDEX(w, t);
                zt_destab = zs[t_destab];
                xt_destab = xs[t_destab];
            }
            word_std_t prefix_zc = warp_exclusive_xor<B>(zt_destab, zs[c_destab], active_targets);
            word_std_t prefix_xc = warp_exclusive_xor<B>(xt_destab, xs[c_destab], active_targets);
            sign_t local_destab_s = 0;
            sign_t local_stab_s = 0;
            if (tid < active_targets) {
                const size_t t = pivots[tid + 1];
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
                compute_local_sign_per_block(local_destab_s, zs[t_stab], prefix_zc, zs[c_stab], zt_destab);
                compute_local_sign_per_block(local_stab_s, xs[t_stab], prefix_xc, xs[c_stab], xt_destab);
            }
            collapse_warp_dual_template<B>(local_destab_s, local_stab_s);
            if (!tid) {
                ss[w] ^= local_destab_s;
                ss[w + num_words_minor] ^= local_stab_s;
            }
        }
    }

    #define CALL_INJECT_CX_WARP(B, YDIM) \
        inject_cx_warp<B> <<<currentgrid, currentblock, 0, stream>>> ( \
            XZ_TABLE(input), \
            input.signs(), \
            pivots, \
            active_targets, \
            num_words_major, \
            num_words_minor, \
            num_qubits_padded);

    void Prefix::scan_warp(Tableau& input, const pivot_t* pivots, const size_t& active_targets, const cudaStream_t& stream) {
        const size_t num_qubits_padded = input.num_qubits_padded();
        const size_t pow2_active_targets = nextPow2(active_targets);
        if (pow2_active_targets > 32) {
            LOGERROR("power-of-2 active targets %d exceeds maximum warp size of 32", pow2_active_targets);
        }
        dim3 currentblock(1, 1), currentgrid(1, 1);
        tune_grid_size(currentblock, currentgrid, pow2_active_targets);
        if (active_targets == 1) {
            currentblock.x = currentblock.y;
            currentgrid.x = currentgrid.y;
            currentblock.y = 1;
            currentgrid.y = 1;
            LOGN2(2, "Injecting CX for 1 targets using warp(x:%u) and grid(x:%u).. ",
                currentblock.x, currentgrid.x);
            if (options.sync) cutimer.start(stream);
            inject_cx_warp_1<<<currentgrid, currentblock, 0, stream>>> (
                XZ_TABLE(input),
                input.signs(),
                pivots,
                active_targets,
                num_words_major,
                num_words_minor,
                num_qubits_padded);
        } else {
            LOGN2(2, "Injecting CX for %d targets using warp(x:%u, y:%u) and grid(x:%u, y:%u).. ",
            active_targets, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
            if (options.sync) cutimer.start(stream);
            switch (currentblock.x) {
                FOREACH_X_DIM_MAX_32(CALL_INJECT_CX_WARP, currentblock.y);
                default:
                    break;
            }
        }
        if (options.sync) {
            LASTERR("failed to launch inject_cx_warp kernel");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            checker.check_inject_cx(input);
        }
    }


}