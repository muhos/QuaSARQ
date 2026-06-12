#include "frame.cuh"
#include "step.cuh"
#include "operators.cuh"
#include "random.cuh"


namespace QuaSARQ {

    INLINE_DEVICE
    word_std_t sample_frame_error_mask(curand_algorithm_t& state, const float& probability) {
        word_std_t mask = 0;
        #pragma unroll
        for (uint32 b = 0; b < WORD_BITS; b++) {
            if (curand_uniform(&state) < probability)
                mask |= (word_std_t(1) << b);
        }
        return mask;
    }

    INLINE_DEVICE
    void apply_frame_pauli(
                word_t&       x_q1,
                word_t&       z_q1,
                word_t*       x_q2,
                word_t*       z_q2,
        const   word_std_t&   mask,
        const   uint32&       pauli)
    {
        if (pauli & 1u) x_q1 ^= mask;
        if (pauli & 2u) z_q1 ^= mask;
        if (x_q2 != nullptr && z_q2 != nullptr) {
            if (pauli & 4u) *x_q2 ^= mask;
            if (pauli & 8u) *z_q2 ^= mask;
        }
    }

    INLINE_DEVICE
    void apply_frame_noise(
                word_t&                 x_q1,
                word_t&                 z_q1,
                word_t*                 x_q2,
                word_t*                 z_q2,
        const   Gate&                   gate,
                curand_algorithm_t&     state)
    {
        if (gate.type == X_ERROR) {
            const word_std_t mask = sample_frame_error_mask(state, gate.get_prob(0));
            x_q1 ^= mask;
            return;
        }
        if (gate.type == Z_ERROR) {
            const word_std_t mask = sample_frame_error_mask(state, gate.get_prob(0));
            z_q1 ^= mask;
            return;
        }
        if (gate.type == Y_ERROR) {
            const word_std_t mask = sample_frame_error_mask(state, gate.get_prob(0));
            x_q1 ^= mask;
            z_q1 ^= mask;
            return;
        }

        #pragma unroll
        for (uint32 b = 0; b < WORD_BITS; b++) {
            uint32 pauli = 0;
            const float prob = curand_uniform(&state);
            if (gate.type == PAULI_CHANNEL_1) {
                const float px = gate.get_prob(0), py = gate.get_prob(1), pz = gate.get_prob(2);
                pauli = prob < px ? 1u : prob < px + py ? 3u : prob < px + py + pz ? 2u : 0u;
            }
            else if (gate.type == PAULI_CHANNEL_2) {
                constexpr uint32 pc2_pauli[15] = {4, 12, 8, 1, 5, 13, 9, 3, 7, 15, 11, 2, 6, 14, 10};
                float acc = 0.0f;
                for (uint32 k = 0; k < 15u; k++) {
                    acc += gate.get_prob(k);
                    if (prob < acc) { pauli = pc2_pauli[k]; break; }
                }
            }
            else if (prob < gate.get_prob(0)) {
                pauli = gate.type == DEPOLARIZE1 ? 1u + (curand(&state) % 3u) :
                        gate.type == DEPOLARIZE2 ? 1u + (curand(&state) % 15u) : 0u;
            }
            if (pauli)
                apply_frame_pauli(x_q1, z_q1, x_q2, z_q2, word_std_t(1) << b, pauli);
        }
    }

    INLINE_DEVICE
    void update_forall_gate(
                word_t*             x_gens_word,
                word_t*             z_gens_word,
                const_refs_t        refs,
                const_buckets_t     gates,
        const   size_t&             num_gates,
        const   size_t&             num_words_minor,
                curand_algorithm_t* rand_states,
        const   size_t              w_offset
    )
    {
        for_parallel_x(i, num_gates) {

            const gate_ref_t r = refs[i];

            assert(r < NO_REF);

            const Gate& gate = (Gate&) gates[r];

            assert(gate.size <= 2);

            const size_t q1 = gate.wires[0];
            assert(q1 != INVALID_QUBIT);
            const size_t q1_word_idx = q1 * num_words_minor;

            #if DEBUG_STEP
            LOGGPU("  word(%-4lld): Gate(%-5s, r:%-4u, s:%d), qubits(%-3lld, %-3lld)\n",
                w, G2S[gate.type], r, gate.size, q1, gate.wires[gate.size - 1]);
            #endif

            switch (gate.type) {
            case Z:
            case X:
            case Y:
            case I: { break; }
            case DEPOLARIZE1:
            case X_ERROR:
            case Y_ERROR:
            case Z_ERROR:
            case PAULI_CHANNEL_1: {
                LOAD_Q1_WORDS;
                curand_algorithm_t local = rand_states[q1_word_idx + w_offset];
                apply_frame_noise(x_words_q1, z_words_q1, nullptr, nullptr, gate, local);
                rand_states[q1_word_idx + w_offset] = local;
                break;
            }
            case DEPOLARIZE2:
            case PAULI_CHANNEL_2: {
                LOAD_Q2_WORDS(num_words_minor);
                curand_algorithm_t local = rand_states[q1_word_idx + w_offset];
                apply_frame_noise(x_words_q1, z_words_q1, &x_words_q2, &z_words_q2, gate, local);
                rand_states[q1_word_idx + w_offset] = local;
                break;
            }
            case H: { 
                LOAD_Q1_WORDS;
                update_H(words_q1); 
                break; 
            }
            case S_DAG:
            case S: {
                LOAD_Q1_WORDS;
                update_S(words_q1);
                break;
            }
            case SQRT_X_DAG:
            case SQRT_X: {
                LOAD_Q1_WORDS;
                update_SQRT_X(words_q1);
                break;
            }
            case SQRT_Y_DAG:
            case SQRT_Y: {
                LOAD_Q1_WORDS;
                update_H(words_q1);
                break;
            }
            case CX: { 
                LOAD_Q2_WORDS(num_words_minor);
                update_CX(q1, q2); break; 
            }
            case CZ: { 
                LOAD_Q2_WORDS(num_words_minor);
                update_CZ(q1, q2); break; 
            }
            case CY: { 
                LOAD_Q2_WORDS(num_words_minor);
                update_CY(q1, q2); break; 
            }
            case SWAP: { 
                LOAD_Q2_WORDS(num_words_minor);
                update_SWAP(x_words_q1, x_words_q2); 
                update_SWAP(z_words_q1, z_words_q2); 
                break; 
            }
            case ISWAP_DAG:
            case ISWAP: {
                LOAD_Q2_WORDS(num_words_minor);
                update_iSWAP(q1, q2);
                break;
            }
            default: break;
            }
        }
    }

    __global__
    void frame_step_2D(
                const_refs_t        refs,
                const_buckets_t     gates,
        const   size_t              num_gates,
        const   size_t              num_words_minor,
                Table *             xs,
                Table *             zs,
                curand_algorithm_t* rand_states)
    {
        for_parallel_y(w, num_words_minor) {
            update_forall_gate(
                xs->data() + w,
                zs->data() + w,
                refs,
                gates,
                num_gates,
                num_words_minor,
                rand_states,
                w
            );
        }
    }

    void Framing::step(const depth_t& depth_level) {
        assert(options.streams >= 3);
        const cudaStream_t copy_stream1 = copy_streams[0];
        const cudaStream_t copy_stream2 = copy_streams[1];
        const cudaStream_t kernel_stream = kernel_streams[0];
        const size_t num_gates_per_window = circuit[depth_level].size();
        const size_t num_words_minor = tableau.num_words_minor();

        // Sync previous kernel streams before copying new gates.
        if (options.progress_en)
            progress_timer.start();
        else if (depth_level) { 
            SYNC(kernel_streams[0]);
            SYNC(kernel_streams[1]);
        }

        // Copy current window to GPU memory.
        gpu_circuit.copyfrom(stats, circuit, depth_level, false, options.sync, copy_stream1, copy_stream2);

        print_gates(gpu_circuit, num_gates_per_window, depth_level);

        // Sync copy streams before launching kernel.
        SYNC(copy_stream1);
        SYNC(copy_stream2);

        if (!circuit.is_measuring(depth_level)) {

            #if DEBUG_STEP
            SYNCALL;
            LOG1(" Debugging at depth %2d:", depth_level);
            frame_step_2D << < dim3(1, 1), dim3(1, 1) >> > (
                gpu_circuit.references(),
                gpu_circuit.gates(),
                num_gates_per_window,
                num_words_major,
                XZ_TABLE(tableau),
                rand_states);
            LASTERR("failed to launch step kernel");
            SYNCALL;
            #else

            TRIM_BLOCK_IN_DEBUG_MODE(bestblockstep, bestgridstep, num_gates_per_window, num_words_minor);

            LOGN2(2, "Running frame-step with block(x:%u, y:%u) and grid(x:%u, y:%u) per depth level %d %s.. ",
                bestblockstep.x, bestblockstep.y, bestgridstep.x, bestgridstep.y,
                depth_level,
                sync ? "synchroneously" : "asynchroneously");

            // Run simulation.
            if (options.sync) cutimer.start(kernel_stream);

            double elapsed = 0;

            frame_step_2D <<< bestgridstep, bestblockstep, 0, kernel_stream >>> (
                gpu_circuit.references(),
                gpu_circuit.gates(),
                num_gates_per_window,
                num_words_minor,
                XZ_TABLE(tableau),
                rand_states);

            if (options.sync) { 
                LASTERR("failed to launch step kernel");
                cutimer.stop(kernel_stream);
                elapsed = cutimer.elapsed();
                if (options.profile) stats.profile.time.gaterules += elapsed;
                LOGENDING(2, 4, "(time %.3f ms)", elapsed);
            } else LOGDONE(2, 4);

            #endif // DEBUG MACRO.

            if (options.print_steptableau)
                print_tableau(tableau, depth_level, false, true);
        } // END of non-measuring simulation.
        else {
            shot(depth_level, kernel_stream);
        }

        if (options.progress_en || options.check_tableau) {
            SYNC(kernel_streams[0]);
            SYNC(kernel_streams[1]);
        }

        if (options.progress_en && !options.check_tableau) {
            print_progress(chunk_index, depth_level, true);
        }

    }

}
