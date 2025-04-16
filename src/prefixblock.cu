#include "simulator.hpp"
#include "prefix.cuh"
#include "collapse.cuh"
#include "templatedim.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

namespace QuaSARQ {

    template <int BLOCKX, int BLOCKY>
    __global__
    void inject_cx_block(
                Table *         inv_xs,
                Table *         inv_zs,
                Signs *         inv_ss,
                CPivotsPtr      pivots,
        const   size_t          active_targets,
        const   size_t          num_words_major,
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded)
    {
        word_std_t *xs = inv_xs->words();
        word_std_t *zs = inv_zs->words();
        sign_t *ss = inv_ss->data();

        using ScanType   = cub::BlockScan<word_std_t, BLOCKX, cub::BLOCK_SCAN_RAKING>;
        using ReduceType = cub::BlockReduce<word_std_t, BLOCKX>;

        __shared__ union {
            typename ScanType::TempStorage   prefix_zs[BLOCKY];
            typename ScanType::TempStorage   prefix_xs[BLOCKY];
            typename ReduceType::TempStorage destab_ss[BLOCKY];
            typename ReduceType::TempStorage   stab_ss[BLOCKY];
        } smem;

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

            word_std_t blocksum_z, blocksum_x; 
            ScanType(smem.prefix_zs[threadIdx.y]).ExclusiveScan(z, prefix_zc, 0, XOROP(), blocksum_z);
            ScanType(smem.prefix_xs[threadIdx.y]).ExclusiveScan(x, prefix_xc, 0, XOROP(), blocksum_x);

            if (tid == active_targets - 1) {
                zs[c_destab] ^= blocksum_z;
                xs[c_destab] ^= blocksum_x;
            }

            sign_t local_destab_s = 0;
            sign_t local_stab_s   = 0;
            if (tid < active_targets) {
                const size_t t = pivots[tid + 1];
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
                prefix_zc ^= init_z;
                prefix_xc ^= init_x;
                compute_local_sign_per_block(local_destab_s, zs[t_stab], prefix_zc, zs[c_stab], z);
                compute_local_sign_per_block(local_stab_s  , xs[t_stab], prefix_xc, xs[c_stab], x);
            }

            local_destab_s = ReduceType(smem.destab_ss[threadIdx.y]).Reduce(local_destab_s, XOROP());
            local_stab_s   = ReduceType(smem.  stab_ss[threadIdx.y]).Reduce(local_stab_s,   XOROP());

            if (!tid) {
                ss[w] ^= local_destab_s;
                ss[w + num_words_minor] ^= local_stab_s;
            }
        }
    }

    #define CALL_INJECT_CX_BLOCK(X, Y) \
        inject_cx_block<X, Y> <<<currentgrid, currentblock, smem_size, stream>>> ( \
            XZ_TABLE(input), \
            input.signs(), \
            pivots, \
            active_targets, \
            num_words_major, \
            num_words_minor, \
            num_qubits_padded);

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
        GENERATE_SWITCH_FOR_CALL(CALL_INJECT_CX_BLOCK);
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

