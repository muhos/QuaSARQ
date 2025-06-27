#include "simulator.hpp"
#include "injectx.cuh"
#include "print.cuh"

namespace QuaSARQ {
    
    __global__ 
    void inject_x_k(
                Table*              inv_xs, 
                Table*              inv_zs,
                Signs*              inv_ss, 
                const_pivots_t      pivots,
        const   size_t              num_words_major, 
        const   size_t              num_words_minor,
        const   size_t              num_qubits_padded) 
    {
        for_parallel_x(w, num_words_minor) {
            const pivot_t do_measurement = SIGN_FLAG;
            assert(do_measurement == 0 || do_measurement == 1);
            if (do_measurement) {
                word_t* xs = inv_xs->data();
                word_t* zs = inv_zs->data();
                sign_t* ss = inv_ss->data();
                const pivot_t pivot = pivots[0];
                assert(pivot != INVALID_PIVOT);
                const size_t c_destab = TABLEAU_INDEX(w, pivot);
                inject_X(zs[c_destab], ss[w]);
                inject_X(xs[c_destab], ss[w + num_words_minor]);
            }
        }

    }

    void Simulator::inject_x(const qubit_t& qubit, const sign_t& rbit, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockinjectx, bestgridinjectx, num_words_minor, 0);
        dim3 currentblock = bestblockinjectx, currentgrid = bestgridinjectx;
        TRIM_GRID_IN_1D(num_words_minor, x);
        LOGN2(2, "Running inject-x kernel with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", \
            currentblock.x, currentblock.y, currentgrid.x, currentgrid.y); \
        if (options.sync) cutimer.start(stream);
        inject_x_k<<<currentgrid, currentblock, 0, stream>>> (
            XZ_TABLE(tableau),
            tableau.signs(),
            pivoting.pivots,
            num_words_major,
            num_words_minor,
            num_qubits_padded);
        if (options.sync) {
            LASTERR("failed to inject X");
            cutimer.stop(stream);
            double elapsed = cutimer.elapsed();
            if (options.profile) stats.profile.time.injectx += elapsed;
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            mchecker.check_inject_x(tableau, pivoting.pivots, 3, rbit);
        }
    }

	void MeasurementChecker::inject_x_cpu() {
        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }
        if (pivot == INVALID_PIVOT) {
            LOGERROR("pivot unknown");
        }
        if (qubit == INVALID_QUBIT) {
            LOGERROR("qubit not set");
        }

        for (size_t w = 0; w < num_words_minor; w++) { 
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            assert(c_destab < h_zs.size());
            assert(c_destab < h_xs.size());
            inject_X(h_zs[c_destab], h_ss[w]);
            inject_X(h_xs[c_destab], h_ss[w + num_words_minor]);
        }
    }

    void MeasurementChecker::check_inject_x(
        const Tableau&  other_input, 
        const pivot_t*  other_pivots, 
        const size_t&   num_pivots, 
        const sign_t&   random_bit) {
        SYNCALL;

        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }
        if (pivot == INVALID_PIVOT) {
            LOGERROR("pivot unknown");
        }
        if (qubit == INVALID_QUBIT) {
            LOGERROR("qubit not set");
        }

        LOGN2(2, "  Checking inject-x for qubit %d and pivot %d.. ", qubit, pivot);

        copy_input(other_input, true);

        const size_t q_w = WORD_OFFSET(qubit);
        const sign_t q_bitpos = qubit & WORD_MASK;
        const sign_t q_mask = sign_t(1) << q_bitpos;
        const sign_t sign_word = h_ss[q_w + num_words_minor];
        const sign_t q_sign = (sign_word & q_mask) >> q_bitpos;
        assert(q_sign <= 1);
        assert(random_bit <= 1);
        sign_t do_measurement = q_sign ^ random_bit;

        copy_pivots(other_pivots, num_pivots);

        assert(num_pivots > 1);

        if (pivot != d_compact_pivots[0]) {
            LOGERROR("Pivot %lld is not the minimum", d_compact_pivots[0]);
        }

        bool other_measurement_bit = D_SIGN_FLAG;

        if (do_measurement != other_measurement_bit) {
            LOGERROR("Measurement bit not identical at pivot(%lld)", pivot);
        }

        if (do_measurement) 
            inject_x_cpu();

        for (size_t w = 0; w < num_words_minor; w++) { 
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            if (h_xs[c_destab] != d_xs[c_destab]) {
                LOGERROR("X-Destabilizer failed at w(%lld), pivot(%lld)", w, pivot);
            }
            if (h_zs[c_destab] != d_zs[c_destab]) {
                LOGERROR("Z-Destabilizer failed at w(%lld), pivot(%lld)", w, pivot);
            }
            if (h_ss[w] != d_ss[w]) {
                LOGERROR("Destabilizer signs failed at w(%lld)", w);
            }
            if (h_ss[w + num_words_minor] != d_ss[w + num_words_minor]) {
                LOGERROR("Stabilizer signs failed at w(%lld)", w + num_words_minor);
            }
        }

        LOG2(2, "%sPASSED.%s", CGREEN, CNORMAL);
    }
	
}