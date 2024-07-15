#include "simulator.hpp"
#include "step.cuh"
#include "tuner.cuh"
#include "collapse.cuh"
#include "operators.cuh"
#include "macros.cuh"

namespace QuaSARQ {

    dim3 bestBlockStep(2, 128), bestGridStep(103, 52);

    __global__ void step_2D(const gate_ref_t* refs, const bucket_t* gates, const size_t num_gates, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss) {
        grid_t tx = threadIdx.x;
        grid_t bx = blockDim.x;
        grid_t global_offset = blockIdx.x * bx;
        grid_t collapse_tid = threadIdx.y * bx + tx;
        word_std_t* shared_signs = SharedMemory<word_std_t>();
        sign_t* signs = ss->data();

        for_parallel_y(w, num_words_major) {

            word_std_t signs_word = 0;

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

                assert(q1 < MAX_QUBITS);
                assert(q2 < MAX_QUBITS);

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
                case Sdg: { do_Sdg(signs_word, words_q1); break; }
                case Z: { sign_update_X_or_Z(signs_word, x_words_q1); break; }
                case X: { sign_update_X_or_Z(signs_word, z_words_q1); break; }
                case Y: { sign_update_Y(signs_word, x_words_q1, z_words_q1); break; }
                case CX: { do_CX(signs_word, q1, q2); break; }
                case CZ: { do_CZ(signs_word, q1, q2); break; }
                case CY: { do_CY(signs_word, q1, q2); break; }
                case Swap: { do_SWAP(x_words_q1, x_words_q2); do_SWAP(z_words_q1, z_words_q2); break; }
                case iSwap: { do_iSWAP(signs_word, q1, q2); break; }
                default: break;
                }
            }

            load_shared(shared_signs, signs_word, collapse_tid, tx, num_gates);
            collapse_shared(shared_signs, signs_word, collapse_tid, bx, tx);
            collapse_warp(shared_signs, signs_word, collapse_tid, bx, tx);

            // Atomically collapse all blocks.
            if (!tx && global_offset < num_gates) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    __global__ void step_2D_warped(const gate_ref_t* refs, const bucket_t* gates, const size_t num_gates, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps, 
    #else
    Table* xs, Table* zs,
    #endif
    Signs* ss) {
        grid_t tx = threadIdx.x;
        grid_t bx = blockDim.x;
        grid_t global_offset = blockIdx.x * bx;
        word_std_t* shared_signs = SharedMemory<word_std_t>();
        sign_t* signs = ss->data();

        for_parallel_y(w, num_words_major) {

            word_std_t signs_word = 0;

            #ifdef INTERLEAVE_XZ
                #ifdef INTERLEAVE_WORDS
                word_t* generators = (!tx) ? ps->data() + X_OFFSET(w) : nullptr;
                generators = (word_t*)__shfl_sync(FULL_WARP, uint64(generators), 0, bx);
                #else
                word_t* generators = (!tx) ? ps->data() + w : nullptr;
                generators = (word_t*)__shfl_sync(FULL_WARP, uint64(generators), 0, bx);
                #endif
            #else
            word_t* x_gens_word = (!tx) ? xs->data() + w : nullptr;
            x_gens_word = (word_t*)__shfl_sync(FULL_WARP, uint64(x_gens_word), 0, bx);
            word_t* z_gens_word = (!tx) ? zs->data() + w : nullptr;
            z_gens_word = (word_t*)__shfl_sync(FULL_WARP, uint64(z_gens_word), 0, bx);
            #endif

            for_parallel_x(i, num_gates) {

                const gate_ref_t r = refs[i];

                assert(r < NO_REF);

                const Gate& gate = (Gate&) gates[r];

                assert(gate.size <= 2);

                const size_t q1 = gate.wires[0];
                const size_t q2 = gate.wires[gate.size - 1];

                assert(q1 < MAX_QUBITS);
                assert(q2 < MAX_QUBITS);

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
                case Sdg: { do_Sdg(signs_word, words_q1); break; }
                case Z: { sign_update_X_or_Z(signs_word, x_words_q1); break; }
                case X: { sign_update_X_or_Z(signs_word, z_words_q1); break; }
                case Y: { sign_update_Y(signs_word, x_words_q1, z_words_q1); break; }
                case CX: { do_CX(signs_word, q1, q2); break; }
                case CZ: { do_CZ(signs_word, q1, q2); break; }
                case CY: { do_CY(signs_word, q1, q2); break; }
                case Swap: { do_SWAP(x_words_q1, x_words_q2); do_SWAP(z_words_q1, z_words_q2); break; }
                case iSwap: { do_iSWAP(signs_word, q1, q2); break; }
                default: break;
                }
            }

            assert(bx <= 32);
            collapse_warp_only(signs_word);

            // Atomically collapse all blocks.
            if (!tx && global_offset < num_gates) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    void Simulator::step(const size_t& p, const depth_t& depth_level, const cudaStream_t* streams, const bool& reversed) {

        double stime = 0;
        cudaStream_t copy_stream1 = cudaStream_t(0);
        cudaStream_t copy_stream2 = cudaStream_t(0);
        cudaStream_t kernel_stream = cudaStream_t(0);

        if (options.overlap) {
            copy_stream1 = streams[COPY_STREAM1];
            copy_stream2 = streams[COPY_STREAM2];
            kernel_stream = streams[KERNEL_STREAM];
        }

        // First level copy.
        if (!options.overlap || (!p && ((!reversed && !depth_level) || (reversed && depth_level == depth - 1)))) {
            LOGN2(1, "Partition %zd: ", p);
            gpu_circuit.copyfrom(stats, circuit, depth_level, reversed, options.sync, options.overlap, copy_stream1, copy_stream2);
        }

        const size_t num_gates_per_window = circuit[depth_level].size();
        const size_t num_words_major = tableau.num_words_major();
        const size_t shared_element_bytes = sizeof(word_std_t);

        print_gates_step(gpu_circuit, num_gates_per_window, depth_level);

    #if DEBUG_STEP

        LOG1(" Debugging at %sdepth %2d:", reversed ? "reversed " : "", depth_level);
        OPTIMIZESHARED(reduce_smem_size, bestBlockStep.y * bestBlockStep.x, shared_element_bytes);
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
                , bestBlockStep, bestGridStep
                // shared memory size.
                , shared_element_bytes, true
                // data length.         
                , num_gates_per_window, num_words_major
                // kernel arguments.
                , gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, XZ_TABLE(tableau), tableau.signs()
            );
        }

        LOGN2(1, "Partition %zd: Simulating the %d-time step %s using grid(%d, %d) and block(%d, %d).. ", 
            p, depth_level, !options.sync ? "asynchronously" : "",
            bestGridStep.x, bestGridStep.y, bestBlockStep.x, bestBlockStep.y);

        if (options.sync) cutimer.start();

        OPTIMIZESHARED(reduce_smem_size, bestBlockStep.y * bestBlockStep.x, shared_element_bytes);
        // sync data transfer.
        if (!options.overlap || !p) {
            SYNC(copy_stream1);
            SYNC(copy_stream2);
        }
        if (bestBlockStep.x > maxWarpSize)
            step_2D << < bestGridStep, bestBlockStep, reduce_smem_size, kernel_stream >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, XZ_TABLE(tableau), tableau.signs());
        else
            step_2D_warped << < bestGridStep, bestBlockStep, reduce_smem_size, kernel_stream >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, XZ_TABLE(tableau), tableau.signs());

        if (options.sync) { 
            LASTERR("failed to launch step kernel");
            cutimer.stop();
            stime = cutimer.time();
        }

        if (options.sync) {
            LOG2(1, "done in %f ms", stime);
        }
        else LOGDONE(1, 3);

    #endif // End of debug/release mode.

        print_tableau_step(tableau, depth_level);

        // Overlap next copy with current kernel execution.
        if (options.overlap) {
            if (!p) {
                LOG2(1, "");
                gpu_circuit.advance_references();
                if (!reversed && depth_level < depth - 1) {
                    LOGN2(1, "Partition %zd: ", p);
                    gpu_circuit.copyfrom(stats, circuit, depth_level + 1, reversed, options.sync, options.overlap, copy_stream1, copy_stream2);
                }
                else if (reversed && depth_level > 0) {
                    LOGN2(1, "Partition %zd: ", p);
                    gpu_circuit.copyfrom(stats, circuit, depth_level - 1, reversed, options.sync, options.overlap, copy_stream1, copy_stream2);
                }
            }
            else {
                LOG2(1, "");
                gpu_circuit.advance_references(num_gates_per_window, reversed);
            }
        }

    } // End of function.

}