#include "frame.hpp"
#include "step.cuh"
#include "operators.cuh"
#include "random.cuh"


namespace QuaSARQ {

    INLINE_DEVICE
    void update_forall_gate(
                word_t*         x_gens_word,
                word_t*         z_gens_word,
                const_refs_t    refs,
                const_buckets_t gates,
        const   size_t&         num_gates,
        const   size_t&         num_words_minor
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
                const_refs_t 	refs,
                const_buckets_t gates,
        const 	size_t 			num_gates,
        const 	size_t 			num_words_minor,
                Table *			xs, 
                Table *			zs) 
    {
        for_parallel_y(w, num_words_minor) {
            update_forall_gate(
                xs->data() + w,
                zs->data() + w,
                refs,
                gates,
                num_gates,
                num_words_minor
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

        if (!circuit.is_measuring(depth_level)) {

            #if DEBUG_STEP
            SYNCALL;
            LOG1(" Debugging at depth %2d:", depth_level);
            frame_step_2D << < dim3(1, 1), dim3(1, 1) >> > (
                gpu_circuit.references(), 
                gpu_circuit.gates(), 
                num_gates_per_window, 
                num_words_major, 
                XZ_TABLE(tableau));
            LASTERR("failed to launch step kernel");
            SYNCALL;
            #else

            TRIM_BLOCK_IN_DEBUG_MODE(bestblockstep, bestgridstep, num_gates_per_window, num_words_minor);

            // sync data transfer.
            SYNC(copy_stream1);
            SYNC(copy_stream2);

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
                XZ_TABLE(tableau));

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
            print_progress(0, depth_level, true);
        }

    }

}