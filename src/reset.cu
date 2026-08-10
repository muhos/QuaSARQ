#include "simulator.hpp"
#include "step.cuh"
#include "print.cuh"
#include "access.cuh"
#include "pivot.cuh"

namespace QuaSARQ {

    __global__
    void reset_signs_k(
                Signs*              inv_ss,
        const_refs_t                refs,
        const_buckets_t             gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor)
    {
        sign_t* ss = inv_ss->data();
        for_parallel_x(i, num_gates) {
            const Gate& gate = (Gate&) gates[refs[i]];
            if (!isReset(int(gate.type)) && gate.type != byte_t(MR))
                continue;
            const size_t q = gate.wires[0];
            const size_t w = WORD_OFFSET(q);
            const word_std_t mask = ~BITMASK_GLOBAL(q);
            atomicAND(ss + w, mask);
            atomicAND(ss + w + num_words_minor, mask);
        }
    }

    void Simulator::reset_signs(const size_t& num_gates, const depth_t& depth_level, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        dim3 currentblock(256, 1), currentgrid;
        OPTIMIZEBLOCKS(currentgrid.x, num_gates, currentblock.x);
        LOGN2(2, "Resetting signs after collapsing over %lld gates.. ", int64(num_gates));
        if (options.sync) cutimer.start(stream);
        reset_signs_k <<<currentgrid, currentblock, 0, stream>>> (
            tableau.signs(),
            gpu_circuit.references(),
            gpu_circuit.gates(),
            num_gates,
            num_words_minor);
        if (options.sync) {
            LASTERR("failed to reset signs");
            cutimer.stop(stream);
            double elapsed = cutimer.elapsed();
            if (options.profile) stats.profile.time.resetsigns += elapsed;
            LOGENDING(2, 4, "(time %.3f ms)", elapsed);
        } else LOGDONE(2, 4);
        if (options.check_measurement) {
            mchecker.check_reset_signs(tableau, circuit, depth_level);
        }
    }

    void MeasurementChecker::check_reset_signs(const Tableau& other_input, const Circuit& circuit, const depth_t& depth_level) {
        SYNCALL;

        if (!input_copied) {
            LOGERROR("device input not copied to the checker");
        }

        LOGN2(2, "  Checking resetting signs at depth level %d.. ", depth_level);

        copy_input(other_input, true);

        const auto num_gates = circuit[depth_level].size();

        for (auto i = 0; i < num_gates; i++) {
            const Gate& m = circuit.gate(depth_level, i);
            if (!isMeasurement(m.type))
                LOGERROR("host gate %d at depth level %d is not a measurement gate", i, depth_level);
            if (!isReset(int(m.type)) && m.type != MR)
                continue;
            const size_t q = m.wires[0];
            const size_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = ~BITMASK_GLOBAL(q);
            h_ss[q_w] &= q_mask;
            h_ss[q_w + num_words_minor] &= q_mask;
        }

        for (size_t w = 0; w < num_words_minor; w++) { 
            if (h_ss[w] != d_ss[w]) {
                LOGERROR("Destabilizer signs failed at w(%lld)", w);
            }
            if (h_ss[w + num_words_minor] != d_ss[w + num_words_minor]) {
                LOGERROR("Stabilizer signs failed at w(%lld)", w + num_words_minor);
            }
        }

        LOGPASSED(2);
    }

}

