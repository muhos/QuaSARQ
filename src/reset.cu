#include "simulator.hpp"
#include "print.cuh"
#include "access.cuh"
#include "pivot.cuh"

namespace QuaSARQ {

    __global__
    void reset_collect_masks_k(
              word_std_t*           cmasks,
        const bool*   __restrict__  flags,
        const_refs_t                refs,
        const_buckets_t             gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor)
    {
        for_parallel_x(i, num_gates) {
            const Gate& gate = (Gate&) gates[refs[i]];
            if (!isReset(int(gate.type)) && gate.type != byte_t(MR))
                continue;
            if (!flags[i])
                continue;
            const size_t q = gate.wires[0];
            const bool x_class = (gate.type == byte_t(RX) || gate.type == byte_t(RY));
            const size_t base = x_class ? 0 : num_words_minor;
            atomicXOR(cmasks + base + WORD_OFFSET(q), BITMASK_GLOBAL(q));
        }
    }

    __global__
    void reset_correct_k(
                Signs*              inv_ss,
                const_table_t       inv_xs,
                const_table_t       inv_zs,
        const_words_t               cmasks,
        const   size_t              num_qubits,
        const   size_t              num_words_major,
        const   size_t              num_words_minor)
    {
        sign_t* ss = inv_ss->data();
        const word_std_t* mask_x = cmasks;
        const word_std_t* mask_z = cmasks + num_words_minor;
        for_parallel_y(g, num_qubits) {
            uint32 par_destab = 0, par_stab = 0;
            for_parallel_x(w, num_words_minor) {
                const size_t d_idx = g * num_words_major + w;
                const size_t s_idx = d_idx + num_words_minor;
                const word_std_t xd = (*inv_xs)[d_idx], zd = (*inv_zs)[d_idx];
                const word_std_t xs = (*inv_xs)[s_idx], zs = (*inv_zs)[s_idx];
                par_destab += popcount_word((xd & mask_x[w]) | (zd & mask_z[w]));
                par_stab   += popcount_word((xs & mask_x[w]) | (zs & mask_z[w]));
            }
            const size_t g_w = WORD_OFFSET(g);
            const word_std_t g_mask = BITMASK_GLOBAL(g);
            if (par_destab & 1)
                atomicXOR(ss + g_w, g_mask);
            if (par_stab & 1)
                atomicXOR(ss + g_w + num_words_minor, g_mask);
        }
    }

    void Simulator::reset_signs(const size_t& num_gates, const depth_t& depth_level, const cudaStream_t& stream) {
        const size_t num_words_minor = tableau.num_words_minor();
        LOGN2(2, "Resetting signs after collapsing over %lld gates.. ", int64(num_gates));
        if (options.sync) cutimer.start(stream);
        CHECK(cudaMemsetAsync(selector.device_cmasks(), 0, selector.cmask_bytes(), stream));
        dim3 setupblock(128, 1), setupgrid;
        OPTIMIZEBLOCKS(setupgrid.x, num_gates, setupblock.x);
        reset_collect_masks_k <<<setupgrid, setupblock, 0, stream>>> (
            selector.device_cmasks(),
            selector.device_flags(),
            gpu_circuit.references(),
            gpu_circuit.gates(),
            num_gates,
            num_words_minor);
        dim3 correctblock(32, 8), correctgrid;
        OPTIMIZEBLOCKS2D(correctgrid.x, num_words_minor, correctblock.x);
        OPTIMIZEBLOCKS2D(correctgrid.y, num_qubits, correctblock.y);
        reset_correct_k <<<correctgrid, correctblock, 0, stream>>> (
            tableau.signs(),
            tableau.xtable(),
            tableau.ztable(),
            selector.device_cmasks(),
            num_qubits,
            tableau.num_words_major(),
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

        Vec<bool> negative(num_gates, false);
        for (auto i = 0; i < num_gates; i++) {
            const Gate& m = circuit.gate(depth_level, i);
            if (!isMeasurement(m.type))
                LOGERROR("host gate %d at depth level %d is not a measurement gate", i, depth_level);
            if (isReset(int(m.type)) || m.type == MR)
                negative[i] = measured_value(m);
        }

        for (auto i = 0; i < num_gates; i++) {
            if (!negative[i])
                continue;
            const Gate& m = circuit.gate(depth_level, i);
            const byte_t mtype = (m.type == MR) ? byte_t(M) : m.type;
            const size_t q = m.wires[0];
            const size_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = BITMASK_GLOBAL(q);
            for (size_t g = 0; g < num_qubits; g++) {
                const size_t d_idx = g * num_words_major + q_w;
                const size_t s_idx = d_idx + num_words_minor;
                const word_std_t g_mask = BITMASK_GLOBAL(g);
                const size_t g_w = WORD_OFFSET(g);
                if (select_conjugate_word(h_xs[d_idx], h_zs[d_idx], mtype) & q_mask)
                    h_ss[g_w] ^= g_mask;
                if (select_conjugate_word(h_xs[s_idx], h_zs[s_idx], mtype) & q_mask)
                    h_ss[g_w + num_words_minor] ^= g_mask;
            }
        }
        negative.clear(true);

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

