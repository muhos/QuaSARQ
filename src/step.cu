#include "simulator.hpp"
#include "step.cuh"
#include "operators.cuh"
#include "collapse.cuh"
#include "templatedim.cuh"


namespace QuaSARQ {

    #define LOAD_X_WORDS(Q) \
        word_t& x_words_ ## Q = x_gens_word[Q ## _word_idx]

    #define LOAD_Z_WORDS(Q) \
        word_t& z_words_ ## Q = z_gens_word[Q ## _word_idx]

    #define LOAD_Q1_WORDS \
        LOAD_X_WORDS(q1); \
        LOAD_Z_WORDS(q1)

    #define LOAD_Q2_WORDS \
        const size_t q2 = gate.wires[1]; \
        assert(q2 != INVALID_QUBIT); \
        const size_t q2_word_idx = q2 * num_words_major; \
        LOAD_Q1_WORDS; \
        LOAD_X_WORDS(q2); \
        LOAD_Z_WORDS(q2)

    INLINE_DEVICE
    void update_forall_gate(
                sign_t&         signs_word,
                word_t*         x_gens_word,
                word_t*         z_gens_word,
                const_refs_t    refs,
                const_buckets_t gates,
        const   size_t&         num_gates,
        const   size_t&         num_words_major
    ) {
        for_parallel_x(i, num_gates) {

            const gate_ref_t r = refs[i];

            assert(r < NO_REF);

            const Gate& gate = (Gate&) gates[r];

            assert(gate.size <= 2);

            const size_t q1 = gate.wires[0];
            assert(q1 != INVALID_QUBIT);
            const size_t q1_word_idx = q1 * num_words_major;

            #if DEBUG_STEP
            LOGGPU("  word(%-4lld): Gate(%-5s, r:%-4u, s:%d), qubits(%-3lld, %-3lld)\n", 
                w, G2S[gate.type], r, gate.size, q1, gate.wires[gate.size - 1]);
            #endif

            switch (gate.type) {
            case I: { break; }
            case H: { 
                LOAD_Q1_WORDS;
                do_H(signs_word, words_q1); 
                break; 
            }
            case S: { 
                LOAD_Q1_WORDS;
                do_S(signs_word, words_q1); 
                break; 
            }
            case S_DAG: { 
                LOAD_Q1_WORDS;
                do_Sdg(signs_word, words_q1); 
                break; 
            }
            case Z: { 
                LOAD_X_WORDS(q1);
                sign_update_X_or_Z(signs_word, x_words_q1); 
                break; 
            }
            case X: { 
                LOAD_Z_WORDS(q1);
                sign_update_X_or_Z(signs_word, z_words_q1); 
                break; 
            }
            case Y: { 
                LOAD_Q1_WORDS;
                sign_update_Y(signs_word, x_words_q1, z_words_q1); 
                break; 
            }
            case CX: { 
                LOAD_Q2_WORDS;
                do_CX(signs_word, q1, q2); break; 
            }
            case CZ: { 
                LOAD_Q2_WORDS;
                do_CZ(signs_word, q1, q2); break; 
            }
            case CY: { 
                LOAD_Q2_WORDS;
                do_CY(signs_word, q1, q2); break; 
            }
            case SWAP: { 
                LOAD_Q2_WORDS;
                do_SWAP(x_words_q1, x_words_q2); do_SWAP(z_words_q1, z_words_q2); break; 
            }
            case ISWAP: { 
                LOAD_Q2_WORDS;
                do_iSWAP(signs_word, q1, q2); break; 
            }
            default: break;
            }
        }
    }

    __global__ 
    void step_2D_atomic(
                const_refs_t 	refs,
                const_buckets_t gates,
        const 	size_t 			num_gates,
        const 	size_t 			num_words_major,
                Table *			xs, 
                Table *			zs,
                Signs *			ss) 
    {
        sign_t* signs = ss->data();
        for_parallel_y(w, num_words_major) {
            sign_t signs_word = signs[w];
            update_forall_gate(
                signs_word,
                xs->data() + w,
                zs->data() + w,
                refs,
                gates,
                num_gates,
                num_words_major
            );
            if (signs_word) {
                atomicXOR(signs + w, signs_word);
            }
        }
    }

    template<int B>
    __global__ 
    void step_2D(
                        const_refs_t 	refs,
                        const_buckets_t gates,
                const 	size_t 			num_gates,
                const 	size_t 			num_words_major,
                        Table *			xs, 
                        Table *			zs,
                        Signs *			ss) 
    {
        uint32 tx = threadIdx.x;
        sign_t* smem = SharedMemory<sign_t>();
        sign_t* shared_signs = smem + threadIdx.y * B;
        sign_t* signs = ss->data();
        for_parallel_y(w, num_words_major) {
            sign_t signs_word = signs[w];
            update_forall_gate(
                signs_word,
                xs->data() + w,
                zs->data() + w,
                refs,
                gates,
                num_gates,
                num_words_major
            );
            collapse_load_shared(shared_signs, signs_word, tx, num_gates);
            collapse_shared<B, sign_t>(shared_signs, signs_word, tx);
            collapse_warp<B, sign_t>(signs_word, tx);
            if (!tx && signs_word) {
                atomicXOR(signs + w, signs_word);
            }
        }
    }

    template<int B>
    __global__ 
    void step_2D_warped(
                const_refs_t 	refs,
                const_buckets_t gates,
        const 	size_t 			num_gates,
        const 	size_t 			num_words_major,
                Table *			xs, 
                Table *			zs,
                Signs *			ss) 
    {
        assert(B <= 32);
        uint32 tx = threadIdx.x;
        sign_t* signs = ss->data();
        for_parallel_y(w, num_words_major) {
            sign_t signs_word = signs[w];
            word_t* x_gens_word = (!tx) ? xs->data() + w : nullptr;
            x_gens_word = (word_t*)__shfl_sync(0xFFFFFFFF, uint64(x_gens_word), 0, B);
            word_t* z_gens_word = (!tx) ? zs->data() + w : nullptr;
            z_gens_word = (word_t*)__shfl_sync(0xFFFFFFFF, uint64(z_gens_word), 0, B);
            update_forall_gate(
                signs_word,
                x_gens_word,
                z_gens_word,
                refs,
                gates,
                num_gates,
                num_words_major
            );
            collapse_warp<B, sign_t>(signs_word, tx);
            if (!tx && signs_word) {
                atomicXOR(signs + w, signs_word);
            }
        }
    }

    #define CALL_STEP_2D_WARPED(B, YDIM) \
        step_2D_warped<B> <<<currentgrid, currentblock, 0, stream>>> ( \
            refs, \
            gates, \
            num_gates_per_window, \
            num_words_major, \
            XZ_TABLE(tableau), \
            tableau.signs() \
        );

    #define CALL_STEP_2D(B, YDIM) \
        step_2D<B> <<<currentgrid, currentblock, shared_size, stream>>> ( \
            refs, \
            gates, \
            num_gates_per_window, \
            num_words_major, \
            XZ_TABLE(tableau), \
            tableau.signs() \
        );

    void call_step_2D(
                const_refs_t    refs,
                const_buckets_t gates,
                Tableau&        tableau,
        const   size_t          num_gates_per_window,
        const   size_t          num_words_major,
        const   dim3&           currentblock,
        const   dim3&           currentgrid,
        const   size_t          shared_size,
        const   cudaStream_t&   stream)
    {
        if (currentblock.x == 1) {
            step_2D_atomic << < currentgrid, currentblock, 0, stream >> > (
                refs, 
                gates, 
                num_gates_per_window, 
                num_words_major, 
                XZ_TABLE(tableau), 
                tableau.signs());
        }
        else if (currentblock.x > 1 && currentblock.x <= maxWarpSize) {
            switch (currentblock.x) {
                FOREACH_X_DIM_MAX_32(CALL_STEP_2D_WARPED, currentblock.y);
                default:
                    break;
            }
        }
        else {
            switch (currentblock.x) {
                FOREACH_X_DIM_MAX_1024(CALL_STEP_2D, currentblock.y);
                default:
                    break;
            }
        }
    }

    void Simulator::step(const size_t& p, const depth_t& depth_level, const bool& reversed) {
        assert(options.streams >= 3);
        const cudaStream_t copy_stream1 = copy_streams[0];
        const cudaStream_t copy_stream2 = copy_streams[1];
        const cudaStream_t kernel_stream = kernel_streams[0];
        const size_t num_gates_per_window = circuit[depth_level].size();
        const size_t num_words_major = tableau.num_words_major();
        const size_t shared_element_bytes = sizeof(word_std_t);

        // Sync previous kernel streams before copying new gates.
        if (options.progress_en)
            progress_timer.start();
        else if (depth_level) { 
            SYNC(kernel_streams[0]);
            SYNC(kernel_streams[1]);
        }

        // Copy current window to GPU memory.
        gpu_circuit.copyfrom(stats, circuit, depth_level, reversed, options.sync, copy_stream1, copy_stream2);
        
        print_gates(gpu_circuit, num_gates_per_window, depth_level);

        if (!circuit.is_measuring(depth_level)) {

            #if DEBUG_STEP
            SYNCALL;
            LOG1(" Debugging at %sdepth %2d:", reversed ? "reversed " : "", depth_level);
            OPTIMIZESHARED(reduce_smem_size, 1, shared_element_bytes);
            step_2D_atomic << < dim3(1, 1), dim3(1, 1) >> > (
                gpu_circuit.references(), 
                gpu_circuit.gates(), 
                num_gates_per_window, 
                num_words_major, 
                XZ_TABLE(tableau), 
                tableau.signs());
            LASTERR("failed to launch step kernel");
            SYNCALL;
            #else

            if (options.tune_step) {
                tune_step(
                    // best kernel config to be found. 
                    bestblockstep, bestgridstep
                    // shared memory size.
                    , shared_element_bytes, true
                    // data length.         
                    , num_gates_per_window, num_words_major
                    // kernel arguments.
                    , gpu_circuit.references(), gpu_circuit.gates(), tableau
                );
            }

            TRIM_BLOCK_IN_DEBUG_MODE(bestblockstep, bestgridstep, num_gates_per_window, num_words_major);

            OPTIMIZESHARED(reduce_smem_size, bestblockstep.y * bestblockstep.x, shared_element_bytes);

            // sync data transfer.
            SYNC(copy_stream1);
            SYNC(copy_stream2);

            LOGN2(2, "Running step with block(x:%u, y:%u) and grid(x:%u, y:%u) per depth level %d %s.. ", bestblockstep.x, bestblockstep.y, bestgridstep.x, bestgridstep.y, depth_level, sync ? "synchroneously" : "asynchroneously");

            // Run simulation.
            if (options.sync) cutimer.start(kernel_stream);

            double elapsed = 0;
            call_step_2D(
                gpu_circuit.references(), 
                gpu_circuit.gates(), 
                tableau, 
                num_gates_per_window, 
                num_words_major, 
                bestblockstep, 
                bestgridstep, 
                reduce_smem_size, 
                kernel_stream);

            if (options.sync) { 
                LASTERR("failed to launch step kernel");
                cutimer.stop(kernel_stream);
                elapsed = cutimer.elapsed();
                if (options.profile) stats.profile.time.gaterules += elapsed;
                LOGENDING(2, 4, "(time %.3f ms)", elapsed);
            } else LOGDONE(2, 4);

            #endif // DEBUG MACRO.

            if (options.print_steptableau)
                print_tableau(tableau, depth_level, reversed);
            if (options.print_stepstate)
                print_paulis(tableau, depth_level, reversed);
        } // END of non-measuring simulation.
        else {
            measure(p, depth_level, reversed);
        }

        if (options.progress_en || options.check_tableau) {
            SYNC(kernel_streams[0]);
            SYNC(kernel_streams[1]);
        }

        if (options.progress_en && !options.check_tableau) {
            print_progress(p, depth_level, true);
        }

        //print_measurements(gpu_circuit, num_gates_per_window, depth_level);

    } // End of function.

}