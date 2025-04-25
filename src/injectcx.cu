#include "simulator.hpp"
#include "prefix.cuh"
#include "print.cuh"
#include "collapse.cuh"
#include "measurecheck.cuh"

namespace QuaSARQ {

    void Simulator::inject_cx(const uint32& active_targets, const cudaStream_t& stream) {
        if (active_targets <= 32)
            prefix.scan_warp(tableau, pivoting.pivots, active_targets, stream);   
        else if (active_targets > 32 && active_targets <= 1024)
            prefix.scan_block(tableau, pivoting.pivots, active_targets, stream);
        else
            prefix.scan_large(tableau, pivoting.pivots, active_targets, stream);
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

        LOG2(2, "%sPASSED.%s", CGREEN, CNORMAL);
    }
}

