#include "simulator.hpp"
#include "prefix.cuh"
#include "collapse.cuh"

namespace QuaSARQ {

    INLINE_DEVICE 
    word_std_t block_exclusive_xor(
                word_std_t*     data, 
        const   word_std_t&     target, 
                word_std_t&     initial_control, 
        const   uint32&         active_targets) 
    {
        int tid = threadIdx.x;
        if (tid < active_targets)
            data[tid] = target;
        __syncthreads();
        if (blockDim.x > 1) {
            if (tid >= 1 && tid < active_targets)
                data[tid] ^= data[tid - 1];
            __syncthreads();
        }
        if (blockDim.x > 2) {
            if (tid >= 2 && tid < active_targets)
                data[tid] ^= data[tid - 2];
            __syncthreads();
        }
        if (blockDim.x > 4) {
            if (tid >= 4 && tid < active_targets)
                data[tid] ^= data[tid - 4];
            __syncthreads();
        }
        if (blockDim.x > 8) {
            if (tid >= 8 && tid < active_targets)
                data[tid] ^= data[tid - 8];
            __syncthreads();
        }
        if (blockDim.x > 16) {
            if (tid >= 16 && tid < active_targets)
                data[tid] ^= data[tid - 16];
            __syncthreads();
        }
        if (blockDim.x > 32) {
            if (tid >= 32 && tid < active_targets)
                data[tid] ^= data[tid - 32];
            __syncthreads();
        }
        if (blockDim.x > 64) {
            if (tid >= 64 && tid < active_targets)
                data[tid] ^= data[tid - 64];
            __syncthreads();
        }
        if (blockDim.x > 128) {
            if (tid >= 128 && tid < active_targets)
                data[tid] ^= data[tid - 128];
            __syncthreads();
        }
        if (blockDim.x > 256) {
            if (tid >= 256 && tid < active_targets)
                data[tid] ^= data[tid - 256];
            __syncthreads();
        }
        if (blockDim.x > 512) {
            if (tid >= 512 && tid < active_targets)
                data[tid] ^= data[tid - 512];
            __syncthreads();
        }
        word_std_t prefix = !tid ? initial_control : (initial_control ^ data[tid - 1]);
        if (tid == active_targets - 1) {
            initial_control ^= data[active_targets - 1];
        }
        return prefix;
    }

    // make this a template for different block sizes.

    __global__ 
    void inject_cx_block(
        Table* inv_xs, 
        Table* inv_zs, 
        Signs* inv_ss, 
        const pivot_t* pivots,
        const size_t active_targets, 
        const size_t num_words_major, 
        const size_t num_words_minor,
        const size_t num_qubits_padded) {
        assert(active_targets > 0);
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();
        sign_t *ss = inv_ss->data();
        word_std_t* smem = SharedMemory<word_std_t>();
        const size_t slice = 2 * blockDim.x;
        word_std_t* base = smem + threadIdx.y * slice;
        word_std_t* prefix_zs = base;
        word_std_t* prefix_xs = base + blockDim.x;
        word_std_t* destab_ss = prefix_zs;
        word_std_t*   stab_ss = prefix_xs;
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
            word_std_t prefix_zc = block_exclusive_xor(prefix_zs, zt_destab, zs[c_destab], active_targets);
            word_std_t prefix_xc = block_exclusive_xor(prefix_xs, xt_destab, xs[c_destab], active_targets);
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
            collapse_load_shared_dual(destab_ss, local_destab_s, stab_ss, local_stab_s, tid, active_targets);
            collapse_shared_dual(destab_ss, local_destab_s, stab_ss, local_stab_s, tid);
            collapse_warp_dual(local_destab_s, local_stab_s);
            if (tid == 0) {
                ss[w] ^= local_destab_s;
                ss[w + num_words_minor] ^= local_stab_s;
            }
        }
    }

    void Prefix::scan_block(Tableau& input, const pivot_t* pivots, const size_t& active_targets, const cudaStream_t& stream) {
        const size_t num_qubits_padded = input.num_qubits_padded();
        const size_t pow2_active_targets = nextPow2(active_targets);
        if (pow2_active_targets > 1024) {
            LOGERROR("power-of-2 active targets %d exceeds maximum block size of 1024", pow2_active_targets);
        }
        dim3 currentblock(1, 1), currentgrid(1, 1);
        tune_grid_size(currentblock, currentgrid, pow2_active_targets);
        OPTIMIZESHARED(smem_size, currentblock.x * currentblock.y, 2 * sizeof(word_std_t));
        LOGN2(2, "Injecting CX for %d targets with block(x:%u, y:%u) and grid(x:%u, y:%u).. ",
            active_targets, currentblock.x, currentblock.y, currentgrid.x, currentgrid.y);
        if (options.sync) cutimer.start(stream);
        inject_cx_block <<<currentgrid, currentblock, smem_size, stream>>> (
            XZ_TABLE(input),
            input.signs(),
            pivots,
            active_targets,
            num_words_major,
            num_words_minor,
            num_qubits_padded);
        if (options.sync) {
            LASTERR("failed to launch inject_cx_block kernel");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            checker.check_inject_cx(input);
        }
    }

}

