#include "simulator.hpp"
#include "prefix.cuh"
#include "collapse.cuh"
#include "templatedim.cuh"
#include "warp.cuh"

namespace QuaSARQ {

    template<int B>
    INLINE_DEVICE 
    word_std_t warp_exclusive_xor(
        const   word_std_t&     target,
                word_std_t&     prefix, 
        const   word_std_t&     initial_control) 
    {
        prefix = target;
        unsigned mask = __activemask();
        word_std_t prev;
        if constexpr (B >= 2) {
            prev = __shfl_up_sync(mask, prefix, 1); if (threadIdx.x >= 1) prefix ^= prev;
        }
        if constexpr (B >= 4) {
            prev = __shfl_up_sync(mask, prefix, 2); if (threadIdx.x >= 2) prefix ^= prev;
        }
        if constexpr (B >= 8) {
            prev = __shfl_up_sync(mask, prefix, 4); if (threadIdx.x >= 4) prefix ^= prev;
        }
        if constexpr (B >= 16) {
            prev = __shfl_up_sync(mask, prefix, 8); if (threadIdx.x >= 8) prefix ^= prev;
        }
        if constexpr (B >= 32) {
            prev = __shfl_up_sync(mask, prefix, 16); if (threadIdx.x >= 16) prefix ^= prev;
        }
        word_std_t sum = prefix;
        prefix ^= (initial_control ^ target);
        return sum;
    }

    template<int B>
    INLINE_DEVICE 
    PrefixCell dual_warp_exclusive_xor(
        const   word_std_t&     target_x,
                word_std_t&     prefix_x,
        const   word_std_t&     target_z,
                word_std_t&     prefix_z, 
        const   word_std_t&     initial_x,
        const   word_std_t&     initial_z) 
    {
        prefix_x = target_x, prefix_z = target_z;
        unsigned mask = __activemask();
        word_std_t prev_x, prev_z;
        if constexpr (B >= 2) {
            prev_x = __shfl_up_sync(mask, prefix_x, 1);
            prev_z = __shfl_up_sync(mask, prefix_z, 1);
            if (threadIdx.x >= 1) prefix_x ^= prev_x, prefix_z ^= prev_z;
        }
        if constexpr (B >= 4) {
            prev_x = __shfl_up_sync(mask, prefix_x, 2);
            prev_z = __shfl_up_sync(mask, prefix_z, 2);
            if (threadIdx.x >= 2) prefix_x ^= prev_x, prefix_z ^= prev_z;
        }
        if constexpr (B >= 8) {
            prev_x = __shfl_up_sync(mask, prefix_x, 4);
            prev_z = __shfl_up_sync(mask, prefix_z, 4);
            if (threadIdx.x >= 4) prefix_x ^= prev_x, prefix_z ^= prev_z;
        }
        if constexpr (B >= 16) {
            prev_x = __shfl_up_sync(mask, prefix_x, 8);
            prev_z = __shfl_up_sync(mask, prefix_z, 8);
            if (threadIdx.x >= 8) prefix_x ^= prev_x, prefix_z ^= prev_z;
        }
        if constexpr (B >= 32) {
            prev_x = __shfl_up_sync(mask, prefix_x, 16);
            prev_z = __shfl_up_sync(mask, prefix_z, 16);
            if (threadIdx.x >= 16) prefix_x ^= prev_x, prefix_z ^= prev_z;
        }
        PrefixCell sum(prefix_x, prefix_z);
        prefix_x ^= (initial_x ^ target_x);
        prefix_z ^= (initial_z ^ target_z);
        return sum;
    }


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

        word_std_t * __restrict__ xs = inv_xs->words();
        word_std_t * __restrict__ zs = inv_zs->words();
        sign_t * __restrict__ ss = inv_ss->data();

        int tx = threadIdx.x, ty = threadIdx.y;

        for_parallel_y_tiled(by, num_words_minor) { 
            const grid_t w = ty + by * blockDim.y;

            bool active = (w < num_words_minor && tx < active_targets); 
        
            pivot_t pivot, t;
            word_std_t init_z = 0, init_x = 0;
            if (!tx && active) {
                pivot = pivots[0];
                const size_t c_destab = TABLEAU_INDEX(w, pivot);
                init_z = zs[c_destab];
                init_x = xs[c_destab];
            }

            word_std_t z = 0, x = 0;
            if (active) {
                t = pivots[tx + 1];
                const unsigned mask = __activemask();
                pivot = __shfl_sync(mask, pivot, 0, B);
                init_z = __shfl_sync(mask, init_z, 0, B);
                init_x = __shfl_sync(mask, init_x, 0, B);
                const size_t t_destab = TABLEAU_INDEX(w, t);
                z = zs[t_destab]; 
                x = xs[t_destab];
            }

            word_std_t prefix_zc, prefix_xc;
            PrefixCell warpsum;
            #if PREFIX_INTERLEAVE
            warpsum = dual_warp_exclusive_xor<B>(x, prefix_xc, z, prefix_zc, init_x, init_z);
            #else
            warpsum.z = warp_exclusive_xor<B>(z, prefix_zc, init_z);
            warpsum.x = warp_exclusive_xor<B>(x, prefix_xc, init_x);
            #endif
            

            sign_t local_destab_s = 0;
            sign_t local_stab_s = 0;
            if (active) {
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                const size_t c_destab = TABLEAU_INDEX(w, pivot);
                const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
                compute_local_sign_per_block(local_destab_s, zs[t_stab], prefix_zc, zs[c_stab], z);
                compute_local_sign_per_block(local_stab_s, xs[t_stab], prefix_xc, xs[c_stab], x);
                if (tx == active_targets - 1) {
                    zs[c_destab] = init_z ^ warpsum.z;
                    xs[c_destab] = init_x ^ warpsum.x;
                }
            }

            collapse_warp_dual<B, sign_t>(local_destab_s, local_stab_s, tx);

            if (!tx) {
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