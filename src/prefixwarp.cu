#include "simulator.hpp"
#include "prefix.cuh"
#include "collapse.cuh"
#include "templatedim.cuh"

namespace QuaSARQ {

    template<int B>
    INLINE_DEVICE 
    word_std_t warp_exclusive_xor(
        const   word_std_t&     target,
                word_std_t&     prefix, 
        const   word_std_t&     initial_control, 
        const   uint32&         active_targets) 
    {
        prefix = target;
        unsigned mask = __activemask();
        word_std_t prev;
        if (B >= 2) {
            prev = __shfl_up_sync(mask, prefix, 1); if (threadIdx.x >= 1) prefix ^= prev;
        }
        if (B >= 4) {
            prev = __shfl_up_sync(mask, prefix, 2); if (threadIdx.x >= 2) prefix ^= prev;
        }
        if (B >= 8) {
            prev = __shfl_up_sync(mask, prefix, 4); if (threadIdx.x >= 4) prefix ^= prev;
        }
        if (B >= 16) {
            prev = __shfl_up_sync(mask, prefix, 8); if (threadIdx.x >= 8) prefix ^= prev;
        }
        if (B >= 32) {
            prev = __shfl_up_sync(mask, prefix, 16); if (threadIdx.x >= 16) prefix ^= prev;
        }
        word_std_t sum = prefix;
        prefix ^= (initial_control ^ target);
        return sum;
    }

     __global__ 
    void inject_cx_warp_1(
                Table*      inv_xs, 
                Table*      inv_zs, 
                Signs*      inv_ss, 
                CPivotsPtr  pivots,
        const   size_t      active_targets, 
        const   size_t      num_words_major, 
        const   size_t      num_words_minor,
        const   size_t      num_qubits_padded) 
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
            const word_std_t zt_destab = zs[t_destab];
            const word_std_t xt_destab = xs[t_destab];
            const word_std_t prefix_zc = zs[c_destab];
            const word_std_t prefix_xc = xs[c_destab];
            zs[c_destab] ^= zt_destab;
            xs[c_destab] ^= xt_destab;
            const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            compute_local_sign_per_block(ss[w], zs[t_stab], prefix_zc, zs[c_stab], zt_destab);
            compute_local_sign_per_block(ss[w + num_words_minor], xs[t_stab], prefix_xc, xs[c_stab], xt_destab);
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
            word_std_t prefix_zc = 0;
            word_std_t prefix_xc = 0;
            word_std_t init_z = 0;
            word_std_t init_x = 0;
            word_std_t z = 0;
            word_std_t x = 0;
            int tid = threadIdx.x;
            if (tid < int(active_targets)) {
                const size_t t = pivots[tid + 1];
                assert(t != pivot);
                assert(t != INVALID_PIVOT);
                const size_t t_destab = TABLEAU_INDEX(w, t);
                z = zs[t_destab];
                x = xs[t_destab];             
                init_z = zs[c_destab];
                init_x = xs[c_destab];
            }
            word_std_t warpsum_z = warp_exclusive_xor<B>(z, prefix_zc, init_z, active_targets);
            word_std_t warpsum_x = warp_exclusive_xor<B>(x, prefix_xc, init_x, active_targets);
            if (tid == active_targets - 1) {
                zs[c_destab] ^= warpsum_z;
                xs[c_destab] ^= warpsum_x;
            }
            sign_t local_destab_s = 0;
            sign_t local_stab_s = 0;
            if (tid < active_targets) {
                const size_t t = pivots[tid + 1];
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
                compute_local_sign_per_block(local_destab_s, zs[t_stab], prefix_zc, zs[c_stab], z);
                compute_local_sign_per_block(local_stab_s, xs[t_stab], prefix_xc, xs[c_stab], x);
            }
            collapse_warp_dual<B, sign_t>(local_destab_s, local_stab_s, tid);
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