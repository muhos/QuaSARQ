#include "simulator.hpp"
#include "prefix.cuh"
#include "tuner.cuh"
#include "print.cuh"
#include "measurement.cuh"

namespace QuaSARQ {

    __global__ 
    void inject_cx_1D(
        Table* inv_xs, 
        Table* inv_zs, 
        Signs* inv_ss, 
        const pivot_t* pivots,
        const size_t active_targets, 
        const size_t num_words_major, 
        const size_t num_words_minor,
        const size_t num_qubits_padded) {
        assert(active_targets > 0);
        const pivot_t pivot = pivots[0];
        assert(pivot != INVALID_PIVOT);
        word_t *xs = inv_xs->data();
        word_t *zs = inv_zs->data();
        sign_t *ss = inv_ss->data();
        for_parallel_x(w, num_words_minor) { // Update all words in both destabs and stabs.
            for (size_t q = 0; q < active_targets; q++) { // anti-commuting targets: pivot + 1, ..., num_qubits - 1.
                const size_t t = pivots[q + 1];
                assert(t != pivot);
                assert(t != INVALID_PIVOT);  
                const size_t c_destab = TABLEAU_INDEX(w, pivot);
                const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
                const size_t t_destab = TABLEAU_INDEX(w, t);
                const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                assert(c_destab < inv_zs->size());
                assert(t_destab < inv_zs->size());
                assert(c_stab < inv_zs->size());
                assert(t_stab < inv_zs->size());
                do_CX_sharing_control(zs[c_stab], zs[c_destab], zs[t_stab], zs[t_destab], ss[w]);
                assert(c_destab < inv_xs->size());
                assert(t_destab < inv_xs->size());
                assert(c_stab < inv_xs->size());
                assert(t_stab < inv_xs->size());
                do_CX_sharing_control(xs[c_stab], xs[c_destab], xs[t_stab], xs[t_destab], ss[w + num_words_minor]);
            }
        }
    }

    void Simulator::inject_cx(const uint32& active_targets, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_qubits_padded = tableau.num_qubits_padded();
        prefix.inject_CX(tableau, commuting.pivots, active_targets, stream);
        // uint32 currentblock = 512, currentgrid;
        // OPTIMIZEBLOCKS(currentgrid, num_words_minor, currentblock);
        // LOGN2(2, "Injecting CX for %d targets using %u threads and %u blocks.. ", active_targets, currentblock, currentgrid);
        // if (options.sync) cutimer.start(stream);
        // inject_cx_1D<<<currentblock, currentgrid, 0, stream>>> (
        //     XZ_TABLE(tableau),
        //     tableau.signs(),
        //     commuting.pivots,
        //     active_targets,
        //     num_words_major,
        //     num_words_minor,
        //     num_qubits_padded);
        // if (options.sync) {
        //     LASTERR("failed to launch inject_cx_1D kernel");
        //     cutimer.stop(stream);
        //     LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
        // } else LOGDONE(2, 4);
        if (options.check_measurement) {
            mchecker.check_inject_cx(tableau);
        }
    }

    void MeasurementChecker::check_inject_cx(const Tableau& other_input) {
        SYNCALL;

        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }
        if (qubit == INVALID_QUBIT) {
            LOGERROR("qubit not set");
        }

        copy_input(other_input, true);

        find_min_pivot(qubit);

        if (pivot == INVALID_PIVOT) 
            return;

        LOGN2(2, " Checking inject-cx for qubit %d and pivot %d.. ", qubit, pivot);

        anticommuting.resize(num_qubits, 0);
        for (size_t t = pivot + 1; t < num_qubits; t++) {
            anticommuting[t] = is_anti_commuting_cpu(
                h_xs,
                qubit,
                t,
                num_words_major,
                num_words_minor,
                num_qubits_padded
            );
        }

        for (size_t w = 0; w < num_words_minor; w++) {
            const size_t c_destab = TABLEAU_INDEX(w, pivot);
            const size_t c_stab = c_destab + TABLEAU_STAB_OFFSET;
            for (size_t t = pivot + 1; t < num_qubits; t++) {
                if (anticommuting[t]) {
                    const size_t t_destab = TABLEAU_INDEX(w, t);
                    const size_t t_stab = t_destab + TABLEAU_STAB_OFFSET;
                    assert(c_destab < h_zs.size());
                    assert(t_destab < h_zs.size());
                    assert(c_stab < h_zs.size());
                    assert(t_stab < h_zs.size());
                    do_CX_sharing_control(h_zs[c_stab], h_zs[c_destab], h_zs[t_stab], h_zs[t_destab], h_ss[w]);
                    assert(c_destab < h_xs.size());
                    assert(t_destab < h_xs.size());
                    assert(c_stab < h_xs.size());
                    assert(t_stab < h_xs.size());
                    do_CX_sharing_control(h_xs[c_stab], h_xs[c_destab], h_xs[t_stab], h_xs[t_destab], h_ss[w + num_words_minor]);
                    if (h_xs[t_stab] != d_xs[t_stab]) {
                        LOGERROR("injecting CX FAILED at stab-x[w: %lld, t: %lld]", w, t);
                    }
                    if (h_zs[t_stab] != d_zs[t_stab]) {
                        LOGERROR("injecting CX FAILED at stab-z[w: %lld, t: %lld]", w, t);
                    }
                }
            }
            if (h_xs[c_destab] != d_xs[c_destab]) {
                LOGERROR("injecting CX FAILED at destab-x[w: %lld, pivot: %lld]", w, pivot);
            }
            if (h_zs[c_destab] != d_zs[c_destab]) {
                LOGERROR("injecting CX FAILED at destab-z[w: %lld, pivot: %lld]", w, pivot);
            }
            if (h_ss[w] != d_ss[w]) {
                LOGERROR("injecting CX FAILED at destab-s[w: %lld]", w);
            }
            if (h_ss[w + num_words_minor] != d_ss[w + num_words_minor]) {
                LOGERROR("injecting CX FAILED at   stab-s[w: %lld]", w + num_words_minor);
            }
        }

        LOG2(2, "PASSED");
    }
}

