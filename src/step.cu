#include "simulator.hpp"
#include "step.cuh"
#include "tuner.cuh"
#include "operators.cuh"
#include "macros.cuh"
#include "locker.cuh"
#include "warp.cuh"

namespace QuaSARQ {

    dim3 bestBlockStep(2, 128), bestGridStep(103, 52);

    __global__ void step_2D(const gate_ref_t* refs, const bucket_t* gates, const size_t num_gates, const size_t num_words_per_column, 
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

        for_parallel_y(w, num_words_per_column) {

            word_std_t signs_word = signs[w];

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

                const size_t q1_x_word_idx = X_OFFSET(q1) * num_words_per_column;
                const size_t q2_x_word_idx = X_OFFSET(q2) * num_words_per_column;
                #ifdef INTERLEAVE_WORDS
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_per_column + 1;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_per_column + 1;
                #else
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_per_column;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_per_column;
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

            load_shared(shared_signs, signs_word, collapse_tid, tx, num_gates);
            collapse_shared(shared_signs, signs_word, collapse_tid, bx, tx);
            collapse_warp(shared_signs, signs_word, collapse_tid, bx, tx);

            // Atomically collapse all blocks.
            if (!tx && global_offset < num_gates) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    __global__ void step_2D_warped(const gate_ref_t* refs, const bucket_t* gates, const size_t num_gates, const size_t num_words_per_column, 
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

        for_parallel_y(w, num_words_per_column) {

            word_std_t signs_word = signs[w];

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

                const size_t q1_x_word_idx = X_OFFSET(q1) * num_words_per_column;
                const size_t q2_x_word_idx = X_OFFSET(q2) * num_words_per_column;
                #ifdef INTERLEAVE_WORDS
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_per_column + 1;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_per_column + 1;
                #else
                const size_t q1_z_word_idx = Z_OFFSET(q1) * num_words_per_column;
                const size_t q2_z_word_idx = Z_OFFSET(q2) * num_words_per_column;
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

            assert(bx <= 32);
            collapse_warp_only(signs_word);

            // Atomically collapse all blocks.
            if (!tx && global_offset < num_gates) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    __global__ void is_indeterminate_outcome(const gate_ref_t* refs, bucket_t* measurements, Table* xs, Table* zs, Signs* ss, DeviceLocker* dlocker, const size_t num_qubits, const size_t num_gates, const size_t num_words_per_column) {

        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            const size_t col = m.wires[0] * num_words_per_column;
            for_parallel_x(j, num_words_per_column) {
                // Check stabilizers if Xgq has 1, if so save the minimum row index.
                const word_std_t word = (*xs)[col + j];
                if (word) {
                    const int64 generator_index = GET_GENERATOR_INDEX(word, j);
                    //printf("(%lld, %lld): ffs in word(" B2B_STR ") = %d, generator index = %lld\n", i, j, RB2B(word), bit_index_per_word, generator_index);
                    // Find the minimum generator.
                    if (generator_index >= num_qubits)
                        atomicMin(&(m.pivot), uint32(generator_index - num_qubits));
                }
            }
        }

    }

    __global__ void measure_determinate(const gate_ref_t* refs, bucket_t* measurements, Table* xs, Table* zs, Signs* ss, uint32* aux_sign, byte_t* aux, DeviceLocker* dlocker, const size_t num_qubits, const size_t num_gates, const size_t num_words_per_column) {
        
        uint32* shared_signs = SharedMemory<uint32>();

        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            // Consdier only determinate measures.
            if (m.pivot == MAX_QUBITS) {
                assert(m.size == 1);
                measure_determinate_qubit(*dlocker, m, *xs, *zs, *ss, shared_signs, aux, num_qubits, num_words_per_column);
            }
        }
    }

    __global__ void measure_indeterminate(const gate_ref_t* refs, bucket_t* measurements, Table* xs, Table* zs, Signs* ss, byte_t* aux, DeviceLocker* dlocker, const size_t num_qubits, const size_t num_gates, const size_t num_words_per_column) {
        
        for(size_t i = 0; i < num_gates; i++) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            if (m.pivot != MAX_QUBITS) {
                assert(m.size == 1);
                const size_t col = m.wires[0] * num_words_per_column;
                word_std_t word = 0;
                // Check again if this gate changed to determinate.
                bool is_determinate = true;
                for_parallel_x(j, num_words_per_column) {
                    const word_std_t word = (*xs)[col + j];
                    printf("(%lld, %lld): word(" B2B_STR ")\n", i, j, RB2B(word));
                }
                // for (size_t j = 0; j < num_words_per_column; j++) {
                //     // Check if all stabilizers are zero.
                //     const word_std_t word = (*xs)[col + j];
                //     if (word) {
                //         const int64 generator_index = GET_GENERATOR_INDEX(word, j);
                //         //printf("(%lld, %lld): ffs in word(" B2B_STR ") = %d, generator index = %lld\n", i, j, RB2B(word), bit_index_per_word, generator_index);
                //         if (generator_index >= num_qubits) {
                //             is_determinate = false;
                //             break;
                //         }
                //     }
                // }
                // if (is_determinate) {
                //     measure_determinate_qubit(*dlocker, *xs, *zs, *ss, aux, m.wires[0], num_qubits, num_words_per_column);
                // }
            }
        }
    }

    void Simulator::step(const size_t& p, const depth_t& depth_level, const cudaStream_t* streams, const bool& reversed) {

        double stime = 0;
        cudaStream_t copy_stream1 = cudaStream_t(0);
        cudaStream_t copy_stream2 = cudaStream_t(0);
        cudaStream_t kernel_stream = cudaStream_t(0);

        // Copy current window to GPU memory.
        LOGN2(1, "Partition %zd, step %d: ", p, depth_level);
        gpu_circuit.copyfrom(stats, circuit, depth_level, reversed, options.sync, copy_stream1, copy_stream2);

        const size_t num_gates_per_window = circuit[depth_level].size();
        const size_t num_words_per_column = tableau.num_words_per_column();
        const size_t shared_element_bytes = sizeof(word_std_t);

        print_gates(gpu_circuit, num_gates_per_window, depth_level);

        if (!circuit.is_measuring(depth_level)) {

            #if DEBUG_STEP

            LOG1(" Debugging at %sdepth %2d:", reversed ? "reversed " : "", depth_level);
            OPTIMIZESHARED(reduce_smem_size, 1, shared_element_bytes);
            step_2D << < dim3(1, 1), dim3(1, 1), reduce_smem_size >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_per_column, tableau.xtable(), tableau.ztable(), tableau.signs());
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
                    , num_gates_per_window, num_words_per_column
                    // kernel arguments.
                    , gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_per_column, XZ_TABLE(tableau), tableau.signs()
                );
            }

            LOGN2(1, "Partition %zd, step %d: Simulating %s using grid(%d, %d) and block(%d, %d).. ", 
                p, depth_level, !options.sync ? "asynchronously" : "",
                bestGridStep.x, bestGridStep.y, bestBlockStep.x, bestBlockStep.y);

            OPTIMIZESHARED(reduce_smem_size, bestBlockStep.y * bestBlockStep.x, shared_element_bytes);

            // sync data transfer.
            SYNC(copy_stream1);
            SYNC(copy_stream2);

            if (options.sync) cutimer.start();

            // Run simulation.
            if (bestBlockStep.x > maxWarpSize)
                step_2D << < bestGridStep, bestBlockStep, reduce_smem_size, kernel_stream >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_per_column, XZ_TABLE(tableau), tableau.signs());
            else
                step_2D_warped << < bestGridStep, bestBlockStep, reduce_smem_size, kernel_stream >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_per_column, XZ_TABLE(tableau), tableau.signs());

            if (options.sync) { 
                LASTERR("failed to launch step kernel");
                cutimer.stop();
                stime = cutimer.time();
            }
            if (options.sync) {
                LOG2(1, "done in %f ms", stime);
            }
            else LOGDONE(1, 3);

            #endif 

            if (options.print_step_tableau)
                print_tableau(tableau, depth_level, reversed);
            if (options.print_step_state)
                print_paulis(tableau, depth_level, reversed);
        } // END of non-measuring simulation.
        else {
            SYNCALL; // just for debugging
            printf("\ndepth %d has %d measurements\n", depth_level, num_gates_per_window);
            fflush(stdout);
            if (!depth_level) locker.reset(0);
            is_indeterminate_outcome<<<bestGridStep, bestBlockStep>>>(gpu_circuit.references(), gpu_circuit.gates(), XZ_TABLE(tableau), tableau.signs(), locker.deviceLocker(), num_qubits, num_gates_per_window, num_words_per_column);
            printf("After checking determinism: "), print_gates(gpu_circuit, num_gates_per_window, depth_level);
            SYNCALL;
            OPTIMIZESHARED(reduce_smem_size, bestBlockStep.y * bestBlockStep.x, sizeof(uint32));
            measure_determinate<<<bestGridStep, bestBlockStep, reduce_smem_size, 0>>>(gpu_circuit.references(), gpu_circuit.gates(), XZ_TABLE(tableau), tableau.signs(), tableau.auxiliary_sign(), tableau.auxiliary(), locker.deviceLocker(), num_qubits, num_gates_per_window, num_words_per_column);
            printf("After determinate measuring: "), print_gates(gpu_circuit, num_gates_per_window, depth_level);
            SYNCALL; // just for debugging
            //measure_indeterminate<<<1, 4>>>(gpu_circuit.references(), gpu_circuit.gates(), XZ_TABLE(tableau), tableau.signs(), tableau.auxiliary(), locker.deviceLocker(), num_qubits, num_gates_per_window, num_words_per_column);
            //SYNCALL; // just for debugging
        }

    } // End of function.

}