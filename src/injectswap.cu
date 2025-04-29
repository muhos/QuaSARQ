#include "simulator.hpp"
#include "injectswap.cuh"
#include "print.cuh"

namespace QuaSARQ {

     __global__ 
    void check_x_destab(
                pivot_t*        pivots,
        const   Table*          inv_xs, 
        const   qubit_t         qubit,
        const   size_t          num_words_major,
        const   size_t          num_qubits_padded)
    {
        const pivot_t pivot = pivots[0];
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        const size_t word_idx = TABLEAU_INDEX(q_w, pivot);
        const word_std_t qubit_word = (*inv_xs)[word_idx];
        COMMUTING_FLAG = (qubit_word & q_mask) ? 1 : 0;
    }

    __global__ 
    void inject_swap_k(
                Table*          inv_xs, 
                Table*          inv_zs,
                Signs*          inv_ss, 
                pivot_t*        pivots,
        const   qubit_t         qubit,
        const   sign_t          random_bit,
        const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded) 
    {
        for_parallel_x(w, num_words_minor) { 
            const pivot_t pivot = pivots[0];
            assert(pivot != INVALID_PIVOT);
            const pivot_t is_commuting = COMMUTING_FLAG;
            assert(is_commuting == 0 || is_commuting == 1);
            word_t* xs = inv_xs->data();
            word_t* zs = inv_zs->data();
            sign_t* ss = inv_ss->data();
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            assert(c_destab < inv_zs->size());
            assert(c_stab < inv_zs->size());
            assert(c_destab < inv_xs->size());
            assert(c_stab < inv_xs->size());
            const size_t signs_stab_idx = w + num_words_minor;
            if (is_commuting) {
                do_YZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_YZ_Swap(xs[c_stab], xs[c_destab], ss[signs_stab_idx]);
            }
            else {
                do_XZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_XZ_Swap(xs[c_stab], xs[c_destab], ss[signs_stab_idx]);
            }
            
            // Wait for the thread that updated the q's word.
            const size_t q_w = WORD_OFFSET(qubit);
            if (w == q_w) {
                const sign_t bitpos = qubit & WORD_MASK;
                const sign_t sign_word = ss[signs_stab_idx];
                assert(random_bit <= 1);
                SIGN_FLAG = ((sign_word >> bitpos) & 1) ^ random_bit;
            }
        }
    }

    void Simulator::inject_swap(const qubit_t& qubit, const sign_t& rbit, const cudaStream_t& stream) {
        if (rbit > 1)
            LOGERROR("random sign %lld cannot be greater than 1.", int64(rbit));
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        check_x_destab<<<1, 1, 0, stream>>> (
            pivoting.pivots,
            tableau.xtable(),
            qubit,
            num_words_major,
            num_qubits_padded);
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockinjectswap, bestgridinjectswap, num_words_minor, 0);
        dim3 currentblock = bestblockinjectswap, currentgrid = bestgridinjectswap;
        TRIM_GRID_IN_1D(num_words_minor, x);
        LOGN2(2, "Running inject-swap kernel with block(x:%u, y:%u) and grid(x:%u, y:%u).. ", \
            currentblock.x, currentblock.y, currentgrid.x, currentgrid.y); \
        if (options.sync) cutimer.start(stream);
        inject_swap_k<<<currentgrid, currentblock, 0, stream>>> (
            XZ_TABLE(tableau),
            tableau.signs(),
            pivoting.pivots,
            qubit, 
            rbit,
            num_words_major,
            num_words_minor,
            num_qubits_padded);
        if (options.sync) {
            LASTERR("failed to inject swap");
            cutimer.stop(stream);
            LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            mchecker.check_inject_swap(tableau, pivoting.pivots, 2);
        }
    }

    bool is_commuting_cpu(
		const 	Table&          h_xs, 
		const   qubit_t         qubit,
		const   pivot_t         pivot,
		const   size_t          num_words_major, 
        const   size_t          num_words_minor,
        const   size_t          num_qubits_padded) 
	{
		assert(pivot != INVALID_PIVOT);
        const size_t q_w = WORD_OFFSET(qubit);
        const word_std_t q_mask = BITMASK_GLOBAL(qubit);
        const size_t word_idx = TABLEAU_INDEX(q_w, pivot);
        const word_std_t qubit_word = h_xs[word_idx];
        return bool(qubit_word & q_mask);
	}

	void MeasurementChecker::inject_swap_cpu() {
        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }
        if (pivot == INVALID_PIVOT) {
            LOGERROR("pivot unknown");
        }
        if (qubit == INVALID_QUBIT) {
            LOGERROR("qubit not set");
        }

		const 
		bool commuting = is_commuting_cpu(
			h_xs,
			qubit,
			pivot,
			num_words_major,
			num_words_minor,
			num_qubits_padded
		);

        for (size_t w = 0; w < num_words_minor; w++) { 
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            assert(c_destab < h_zs.size());
            assert(c_stab < h_zs.size());
            assert(c_destab < h_xs.size());
            assert(c_stab < h_xs.size());
            if (commuting) {
                do_YZ_Swap(h_zs[c_stab], h_zs[c_destab], h_ss[w]);
                do_YZ_Swap(h_xs[c_stab], h_xs[c_destab], h_ss[w + num_words_minor]);
            }
            else {
                do_XZ_Swap(h_zs[c_stab], h_zs[c_destab], h_ss[w]);
                do_XZ_Swap(h_xs[c_stab], h_xs[c_destab], h_ss[w + num_words_minor]);
            }
        }
    }

    void MeasurementChecker::check_inject_swap(const Tableau& other_input, const pivot_t* other_pivots, const size_t& num_pivots) {
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

        LOGN2(2, "  Checking inject-swap for qubit %d and pivot %d.. ", qubit, pivot);

        copy_input(other_input, true);

        const 
		bool commuting = is_commuting_cpu(
			h_xs,
			qubit,
			pivot,
			num_words_major,
			num_words_minor,
			num_qubits_padded
		);

        copy_pivots(other_pivots, num_pivots);

        assert(num_pivots > 1);

        if (pivot != d_compact_pivots[0]) {
            LOGERROR("Pivot %lld is not the minimum", d_compact_pivots[0]);
        }

        bool other_commuting_bit = D_COMMUTING_FLAG;

        if (commuting != other_commuting_bit) {
            LOGERROR("Commuting bit not identical at pivot(%lld)", pivot);
        }

        inject_swap_cpu();

        for (size_t w = 0; w < num_words_minor; w++) { 
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            if (h_xs[c_stab] != d_xs[c_stab]) {
                LOGERROR("X-Stabilizer failed at w(%lld), pivot(%lld)", w, pivot);
            }
            if (h_zs[c_stab] != d_zs[c_stab]) {
                LOGERROR("Z-Stabilizer failed at w(%lld), pivot(%lld)", w, pivot);
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