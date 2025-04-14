#include "simulator.hpp"
#include "step.cuh"
#include "tuner.cuh"
#include "operators.cuh"
#include "macros.cuh"
#include "templatedim.cuh"


namespace QuaSARQ {

    __global__ void step_2D_atomic(ConstRefsPointer refs, ConstBucketsPointer gates, const size_t num_gates, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss) {
        sign_t* signs = ss->data();

        for_parallel_y(w, num_words_major) {

            sign_t signs_word = signs[w];

            #ifdef INTERLEAVE_XZ
                #ifdef INTERLEAVE_WORDS
                word_t* generators = ps->data() + X_OFFSET(w);
                #else
                word_t* generators = ps->data() + w;
                #endif
            #else
            word_t* x_gens_word = xs->data() + w;
            word_t* z_gens_word = zs->data() + w;
            #endif

            for_parallel_x(i, num_gates) {

                const gate_ref_t r = refs[i];

                assert(r < NO_REF);

                const Gate& gate = (Gate&) gates[r];

                assert(gate.size <= 2);

                const size_t q1 = gate.wires[0];
                const size_t q2 = gate.wires[gate.size - 1];

                assert(q1 != INVALID_QUBIT);
                assert(q2 != INVALID_QUBIT);

                const size_t q1_x_word_idx = X_OFFSET(q1) * num_words_major;
                const size_t q2_x_word_idx = X_OFFSET(q2) * num_words_major;
                #ifdef INTERLEAVE_WORDS
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_major + 1;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_major + 1;
                #else
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_major;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_major;
                #endif

                #ifdef INTERLEAVE_XZ
                assert(q1_x_word_idx + w < ps->size()), assert(q1_z_word_idx + w < ps->size());
                assert(q2_x_word_idx + w < ps->size()), assert(q2_z_word_idx + w < ps->size());
                word_t& x_words_q1 = generators[q1_x_word_idx];
                word_t& x_words_q2 = generators[q2_x_word_idx];
                word_t& z_words_q1 = generators[q1_z_word_idx];
                word_t& z_words_q2 = generators[q2_z_word_idx];
                #else
                assert(q1_x_word_idx + w < xs->size()), assert(q1_z_word_idx + w < zs->size());
                assert(q2_x_word_idx + w < xs->size()), assert(q2_z_word_idx + w < zs->size());
                word_t& x_words_q1 = x_gens_word[q1_x_word_idx];
                word_t& x_words_q2 = x_gens_word[q2_x_word_idx];
                word_t& z_words_q1 = z_gens_word[q1_z_word_idx];
                word_t& z_words_q2 = z_gens_word[q2_z_word_idx];
                #endif

                #if DEBUG_STEP
                LOGGPU("  word(%-4lld): Gate(%-5s, r:%-4u, s:%d), qubits(%-3lld, %-3lld)\n", w, G2S[gate.type], r, gate.size, q1, q2);
                #endif

                switch (gate.type) {
                case I: { break; }
                case H: { do_H(signs_word, words_q1); break; }
                case S: { do_S(signs_word, words_q1); break; }
                case S_DAG: { do_Sdg(signs_word, words_q1); break; }
                case Z: { sign_update_X_or_Z(signs_word, x_words_q1); break; }
                case X: { sign_update_X_or_Z(signs_word, z_words_q1); break; }
                case Y: { sign_update_Y(signs_word, x_words_q1, z_words_q1); break; }
                case CX: { do_CX(signs_word, q1, q2); break; }
                case CZ: { do_CZ(signs_word, q1, q2); break; }
                case CY: { do_CY(signs_word, q1, q2); break; }
                case SWAP: { do_SWAP(x_words_q1, x_words_q2); do_SWAP(z_words_q1, z_words_q2); break; }
                case ISWAP: { do_iSWAP(signs_word, q1, q2); break; }
                default: break;
                }
            }

            if (signs_word) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    __global__ void step_2D(ConstRefsPointer refs, ConstBucketsPointer gates, const size_t num_gates, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss) {
        grid_t tx = threadIdx.x;
        grid_t BX = blockDim.x;
        grid_t global_offset = blockIdx.x * BX;
        grid_t collapse_tid = threadIdx.y * BX + tx;
        sign_t* shared_signs = SharedMemory<sign_t>();
        sign_t* signs = ss->data();

        for_parallel_y(w, num_words_major) {

            sign_t signs_word = signs[w];

            #ifdef INTERLEAVE_XZ
                #ifdef INTERLEAVE_WORDS
                word_t* generators = ps->data() + X_OFFSET(w);
                #else
                word_t* generators = ps->data() + w;
                #endif
            #else
            word_t* x_gens_word = xs->data() + w;
            word_t* z_gens_word = zs->data() + w;
            #endif

            for_parallel_x(i, num_gates) {

                const gate_ref_t r = refs[i];

                assert(r < NO_REF);

                const Gate& gate = (Gate&) gates[r];

                assert(gate.size <= 2);

                const size_t q1 = gate.wires[0];
                const size_t q2 = gate.wires[gate.size - 1];

                assert(q1 != INVALID_QUBIT);
                assert(q2 != INVALID_QUBIT);

                const size_t q1_x_word_idx = X_OFFSET(q1) * num_words_major;
                const size_t q2_x_word_idx = X_OFFSET(q2) * num_words_major;
                #ifdef INTERLEAVE_WORDS
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_major + 1;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_major + 1;
                #else
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_major;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_major;
                #endif

                #ifdef INTERLEAVE_XZ
                assert(q1_x_word_idx + w < ps->size()), assert(q1_z_word_idx + w < ps->size());
                assert(q2_x_word_idx + w < ps->size()), assert(q2_z_word_idx + w < ps->size());
                word_t& x_words_q1 = generators[q1_x_word_idx];
                word_t& x_words_q2 = generators[q2_x_word_idx];
                word_t& z_words_q1 = generators[q1_z_word_idx];
                word_t& z_words_q2 = generators[q2_z_word_idx];
                #else
                assert(q1_x_word_idx + w < xs->size()), assert(q1_z_word_idx + w < zs->size());
                assert(q2_x_word_idx + w < xs->size()), assert(q2_z_word_idx + w < zs->size());
                word_t& x_words_q1 = x_gens_word[q1_x_word_idx];
                word_t& x_words_q2 = x_gens_word[q2_x_word_idx];
                word_t& z_words_q1 = z_gens_word[q1_z_word_idx];
                word_t& z_words_q2 = z_gens_word[q2_z_word_idx];
                #endif

                #if DEBUG_STEP
                LOGGPU("  word(%-4lld): Gate(%-5s, r:%-4u, s:%d), qubits(%-3lld, %-3lld)\n", w, G2S[gate.type], r, gate.size, q1, q2);
                #endif

                switch (gate.type) {
                case I: { break; }
                case H: { do_H(signs_word, words_q1); break; }
                case S: { do_S(signs_word, words_q1); break; }
                case S_DAG: { do_Sdg(signs_word, words_q1); break; }
                case Z: { sign_update_X_or_Z(signs_word, x_words_q1); break; }
                case X: { sign_update_X_or_Z(signs_word, z_words_q1); break; }
                case Y: { sign_update_Y(signs_word, x_words_q1, z_words_q1); break; }
                case CX: { do_CX(signs_word, q1, q2); break; }
                case CZ: { do_CZ(signs_word, q1, q2); break; }
                case CY: { do_CY(signs_word, q1, q2); break; }
                case SWAP: { do_SWAP(x_words_q1, x_words_q2); do_SWAP(z_words_q1, z_words_q2); break; }
                case ISWAP: { do_iSWAP(signs_word, q1, q2); break; }
                default: break;
                }
            }

            // Only blocks >= 64 use shared memory, < 64 use warp communications.
            if (BX >= 64) {
			    shared_signs[collapse_tid] = (tx < num_gates) ? signs_word : 0;
			    __syncthreads();
		    }

            collapse_load_shared(shared_signs, signs_word, collapse_tid, num_gates);
            collapse_shared(shared_signs, signs_word, collapse_tid);
            collapse_warp(shared_signs, signs_word, collapse_tid);

            // Atomically collapse all blocks.
            if (!tx && global_offset < num_gates && signs_word) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    template<int B>
    __global__ 
    void step_2D_warped(ConstRefsPointer refs, ConstBucketsPointer gates, const size_t num_gates, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss) {
        grid_t tx = threadIdx.x;
        grid_t BX = blockDim.x;
        grid_t global_offset = blockIdx.x * BX;
        sign_t* signs = ss->data();

        for_parallel_y(w, num_words_major) {

            sign_t signs_word = signs[w];

            #ifdef INTERLEAVE_XZ
                #ifdef INTERLEAVE_WORDS
                word_t* generators = (!tx) ? ps->data() + X_OFFSET(w) : nullptr;
                generators = (word_t*)__shfl_sync(FULL_WARP, uint64(generators), 0, BX);
                #else
                word_t* generators = (!tx) ? ps->data() + w : nullptr;
                generators = (word_t*)__shfl_sync(FULL_WARP, uint64(generators), 0, BX);
                #endif
            #else
            word_t* x_gens_word = (!tx) ? xs->data() + w : nullptr;
            x_gens_word = (word_t*)__shfl_sync(FULL_WARP, uint64(x_gens_word), 0, BX);
            word_t* z_gens_word = (!tx) ? zs->data() + w : nullptr;
            z_gens_word = (word_t*)__shfl_sync(FULL_WARP, uint64(z_gens_word), 0, BX);
            #endif

            for_parallel_x(i, num_gates) {

                const gate_ref_t r = refs[i];

                assert(r < NO_REF);

                const Gate& gate = (Gate&) gates[r];

                assert(gate.size <= 2);

                const size_t q1 = gate.wires[0];
                const size_t q2 = gate.wires[gate.size - 1];

                assert(q1 != INVALID_QUBIT);
                assert(q2 != INVALID_QUBIT);

                const size_t q1_x_word_idx = X_OFFSET(q1) * num_words_major;
                const size_t q2_x_word_idx = X_OFFSET(q2) * num_words_major;
                #ifdef INTERLEAVE_WORDS
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_major + 1;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_major + 1;
                #else
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_major;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_major;
                #endif

                #ifdef INTERLEAVE_XZ
                assert(q1_x_word_idx + w < ps->size()), assert(q1_z_word_idx + w < ps->size());
                assert(q2_x_word_idx + w < ps->size()), assert(q2_z_word_idx + w < ps->size());
                word_t& x_words_q1 = generators[q1_x_word_idx];
                word_t& x_words_q2 = generators[q2_x_word_idx];
                word_t& z_words_q1 = generators[q1_z_word_idx];
                word_t& z_words_q2 = generators[q2_z_word_idx];
                #else
                assert(q1_x_word_idx + w < xs->size()), assert(q1_z_word_idx + w < zs->size());
                assert(q2_x_word_idx + w < xs->size()), assert(q2_z_word_idx + w < zs->size());
                word_t& x_words_q1 = x_gens_word[q1_x_word_idx];
                word_t& x_words_q2 = x_gens_word[q2_x_word_idx];
                word_t& z_words_q1 = z_gens_word[q1_z_word_idx];
                word_t& z_words_q2 = z_gens_word[q2_z_word_idx];
                #endif

                #if DEBUG_STEP
                LOGGPU("  word(%-4lld): Gate(%-5s, r:%-4u, s:%d), qubits(%-3lld, %3lld)\n", w, G2S[gate.type], r, gate.size, gate.wires[0], gate.wires[gate.size - 1]);
                #endif

                switch (gate.type) {
                case I: { break; }
                case H: { do_H(signs_word, words_q1); break; }
                case S: { do_S(signs_word, words_q1); break; }
                case S_DAG: { do_Sdg(signs_word, words_q1); break; }
                case Z: { sign_update_X_or_Z(signs_word, x_words_q1); break; }
                case X: { sign_update_X_or_Z(signs_word, z_words_q1); break; }
                case Y: { sign_update_Y(signs_word, x_words_q1, z_words_q1); break; }
                case CX: { do_CX(signs_word, q1, q2); break; }
                case CZ: { do_CZ(signs_word, q1, q2); break; }
                case CY: { do_CY(signs_word, q1, q2); break; }
                case SWAP: { do_SWAP(x_words_q1, x_words_q2); do_SWAP(z_words_q1, z_words_q2); break; }
                case ISWAP: { do_iSWAP(signs_word, q1, q2); break; }
                default: break;
                }
            }

            assert(BX <= 32);
            collapse_warp_only<B>(signs_word);

            // Atomically collapse all blocks.
            if (!tx && global_offset < num_gates && signs_word) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    #define CALL_STEP_2D_WARPED(B, YDIM) \
        step_2D_warped<B> <<<bestgridstep, bestblockstep, 0, kernel_stream>>> ( \
            gpu_circuit.references(), \
            gpu_circuit.gates(), \
            num_gates_per_window, \
            num_words_major, \
            XZ_TABLE(tableau), \
            tableau.signs() \
        );

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

            LOG1(" Debugging at %sdepth %2d:", reversed ? "reversed " : "", depth_level);
            OPTIMIZESHARED(reduce_smem_size, 1, shared_element_bytes);
            step_2D << < dim3(1, 1), dim3(1, 1), reduce_smem_size >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, tableau.xtable(), tableau.ztable(), tableau.signs());
            LASTERR("failed to launch step kernel");
            SYNCALL;
            #else

            if (options.tune_step) {
                tune_kernel
                (
                #if TUNE_WARPED_VERSION
                    step_2D_warped, "step warped"
                #else
                    step_2D, "step"
                #endif
                    // best kernel config to be found. 
                    , bestblockstep, bestgridstep
                    // shared memory size.
                    , shared_element_bytes, true
                    // data length.         
                    , num_gates_per_window, num_words_major
                    // kernel arguments.
                    , gpu_circuit.references(), gpu_circuit.gates(), XZ_TABLE(tableau), tableau.signs()
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

            if (bestblockstep.x == 1) {
                step_2D_atomic << < bestgridstep, bestblockstep, 0, kernel_stream >> > (
                    gpu_circuit.references(), 
                    gpu_circuit.gates(), 
                    num_gates_per_window, 
                    num_words_major, 
                    XZ_TABLE(tableau), 
                    tableau.signs());
            }
            else if (bestblockstep.x <= maxWarpSize) {
                switch (bestblockstep.x) {
                    FOREACH_X_DIM_MAX_32(CALL_STEP_2D_WARPED, bestblockstep.y);
                    default:
                        break;
                }
            }
            else {
                step_2D << < bestgridstep, bestblockstep, reduce_smem_size, kernel_stream >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, XZ_TABLE(tableau), tableau.signs());
            }

            if (options.sync) { 
                LASTERR("failed to launch step kernel");
                cutimer.stop(kernel_stream);
                LOGENDING(2, 4, "(time %.3f ms)", cutimer.time());
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

        if (options.progress_en) {
            SYNC(kernel_streams[0]);
            SYNC(kernel_streams[1]);
            print_progress(p, depth_level);
        }

        print_measurements(gpu_circuit, num_gates_per_window, depth_level);

    } // End of function.

}