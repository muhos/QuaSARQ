#include "simulator.hpp"
#include "measurement.cuh"
#include "tuner.cuh"
#include "locker.cuh"
#include "transpose.cuh"

namespace QuaSARQ {

    dim3 bestBlockMeasure(2, 128), bestGridMeasure(103, 52);

    // Let threads in x-dim find the minimum (de)stabilizer generator commuting.
    INLINE_DEVICE void find_min_pivot(Pivot& p, const qubit_t& q, const Table& inv_xs, const size_t num_qubits, const size_t num_words_minor) {
        const qubit_t q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        const grid_t stab_offset = num_qubits * num_words_minor;
        for_parallel_x(g, num_qubits) {
            const grid_t word_idx = g * num_words_minor + q_w;
            word_std_t qubit_word = inv_xs[stab_offset + word_idx];
            if (qubit_word & q_mask)
                atomicMin(&p.indeterminate, g);
            else {
                qubit_word = inv_xs[word_idx];
                if (qubit_word & q_mask)   
                    atomicMin(&p.determinate, g);
            }
        }
    }

    __global__ void reset_pivots(Pivot* pivots, const size_t num_gates) {
        for_parallel_x(i, num_gates) {
            pivots[i].reset();
        }
    }

    __global__ void find_pivots_initial(Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, const Table* inv_xs, 
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor) {
        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            find_min_pivot(pivots[i], m.wires[0], *inv_xs, num_qubits, num_words_minor);
        }
    }

    __global__ void initialize_determinate_measurements(Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Signs* inv_ss,
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor) {
        for_parallel_x(i, num_gates) {
            // Check if the current gate is determinate
            // Indeterminate has higher priority here.
            if (pivots[i].indeterminate == INVALID_PIVOT) {
                // Determinate pivot must be there if measurement is not indeterminate. CAN we prove this?
                assert(pivots[i].determinate != INVALID_PIVOT);
                const gate_ref_t r = refs[i];
                assert(r < NO_REF);
                Gate& m = (Gate&) measurements[r];
                assert(m.size == 1);
                assert(pivots[i].determinate < num_qubits);
                const grid_t pivot = pivots[i].determinate;
                const grid_t stab_pivot = pivot + num_qubits;
                m.measurement = inv_ss->get_unpacked_sign(stab_pivot);
                assert(m.measurement != UNMEASURED);
                const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
                const word_std_t q_mask = BITMASK_GLOBAL(q);
                for (grid_t des_pivot = pivot + 1; des_pivot < num_qubits; des_pivot++) {
                    const word_std_t qubit_word = (*inv_xs)[des_pivot * num_words_minor + q_w];
                    if (qubit_word & q_mask) {
                        CHECK_SIGN_OVERFLOW(des_pivot, m.measurement, inv_ss->get_unpacked_sign(des_pivot + num_qubits));
                        //printf("M(%3d): adding sign[%lld]: %d to measurement: %d -> measurement: %d\n", q, des_pivot + num_qubits, inv_ss->get_unpacked_sign(des_pivot + num_qubits), m.measurement, inv_ss->get_unpacked_sign(des_pivot + num_qubits) + m.measurement);
                        m.measurement += inv_ss->get_unpacked_sign(des_pivot + num_qubits);
                    }
                }
            }
            // Mark determinate pivot invalid if measurement is indeterminate.
            else if (pivots[i].determinate != INVALID_PIVOT) {
                pivots[i].determinate = INVALID_PIVOT;
            }
        }
    }

        // Tile a tableau row into shared memory.
    INLINE_DEVICE void row_to_aux(word_std_t* aux, const Table& inv_xs, const Table& inv_zs, const grid_t& stab_pivot, const size_t& num_words_minor) {
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        const grid_t w = blockIdx.x * BX + tx;
        word_std_t* aux_xs = aux;
        word_std_t* aux_zs = aux_xs + blockDim.x;
        if (w < num_words_minor && tx < num_words_minor) {
            const grid_t qubits_word_idx = stab_pivot * num_words_minor + w;
            aux_xs[shared_tid] = inv_xs[qubits_word_idx];
            aux_zs[shared_tid] = inv_zs[qubits_word_idx];
        }
        else {
            aux_xs[shared_tid] = 0;
            aux_zs[shared_tid] = 0;
        }
        __syncthreads();
    }

    // Multiply a tableau row to a tiled-row in shared memory.
    INLINE_DEVICE void row_aux_mul(Gate& m, word_std_t* aux, int* aux_power, const Table& inv_xs, const Table& inv_zs, const grid_t& des_idx, const size_t& num_words_minor) {
        word_std_t* aux_xs = aux;
        word_std_t* aux_zs = aux + blockDim.x;
        int* pos_is = aux_power;
        int* neg_is = aux_power + blockDim.x;
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        const grid_t w = blockIdx.x * BX + tx;
        int pos_i = 0, neg_i = 0;    
        if (w < num_words_minor && tx < num_words_minor) {
            const grid_t qubits_word_idx = des_idx * num_words_minor + w;
            const word_std_t x     = inv_xs[qubits_word_idx], z     = inv_zs[qubits_word_idx];
            const word_std_t aux_x = aux_xs[shared_tid]     , aux_z = aux_zs[shared_tid];
            aux_xs[shared_tid] ^= x;
            aux_zs[shared_tid] ^= z;
            COMPUTE_POWER_I(pos_i, neg_i, x, z, aux_x, aux_z);
        }
        ACCUMULATE_POWER_I(m.measurement);
    }

    // Measure a determinate qubit.
    INLINE_DEVICE void measure_determinate_qubit(DeviceLocker& dlocker, Gate& m, word_std_t* aux, int* aux_power, const Table& inv_xs, const Table& inv_zs, const size_t& src_pivot, const size_t num_qubits, const size_t num_words_minor) {
        row_to_aux(aux, inv_xs, inv_zs, src_pivot + num_qubits, num_words_minor);
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        for (grid_t des_pivot = src_pivot + 1; des_pivot < num_qubits; des_pivot++) {
            word_std_t qubit_word = inv_xs[des_pivot * num_words_minor + q_w];
            if (qubit_word & q_mask) {
                row_aux_mul(m, aux, aux_power, inv_xs, inv_zs, des_pivot + num_qubits, num_words_minor);
            }
        }
    }

    __global__ void measure_determinate(DeviceLocker* dlocker, Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Table* inv_zs, 
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor) {
        word_std_t* aux = SharedMemory<word_std_t>();
        int* aux_power = reinterpret_cast<int*>(aux + blockDim.y * blockDim.x * 2);
        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            // Consider only determinate measures.
            if (pivots[i].indeterminate == INVALID_PIVOT) {
                assert(pivots[i].determinate != INVALID_PIVOT);
                measure_determinate_qubit(*dlocker, m, aux, aux_power, *inv_xs, *inv_zs, pivots[i].determinate, num_qubits, num_words_minor);
            }
        }
    }

    // __global__ void measure_indeterminate(Table* inv_xs, Table* inv_zs, Signs *inv_ss, DeviceLocker* dlocker, 
    //                                     bucket_t* measurements, const gate_ref_t* refs, const size_t gate_index, 
    //                                     const size_t num_gates, const size_t num_qubits, const size_t num_words_minor) {
    //     word_std_t* aux = SharedMemory<word_std_t>();
    //     int* aux_power = reinterpret_cast<int*>(aux + blockDim.y * blockDim.x * 2);
    //     int* unpacked_ss = inv_ss->unpacked_data();
    //     const gate_ref_t r = refs[gate_index];
    //     Gate& m = (Gate&) measurements[r];
    //     if (m.pivot != MAX_QUBITS) {
    //         const uint32 destab_pivot = m.pivot;
    //         const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
    //         const word_std_t q_mask = BITMASK_GLOBAL(q);
    //         const grid_t stab_pivot = destab_pivot + num_qubits;
    //         word_std_t qubit_word = (*inv_xs)[stab_pivot * num_words_minor + q_w];
    //         // If pivot is still valid in the current quantum state.
    //         if (qubit_word & q_mask) {
    //             row_to_row((*inv_xs), (*inv_zs), unpacked_ss, destab_pivot, stab_pivot, num_words_minor);
    //             row_set((*inv_xs), (*inv_zs), unpacked_ss, stab_pivot, q, num_words_minor);
    //             if (!global_tx) {
    //                 const int rand_measure = 2; //2 * (rand() % 2);
    //                 m.measurement = rand_measure;
    //                 unpacked_ss[stab_pivot] = rand_measure;
    //             }
    //             for (grid_t j = 0; j < 2 * num_qubits; j++) {
    //                 const word_std_t qubit_word = (*inv_xs)[j * num_words_minor + q_w];
    //                 // (j != stab_pivot) is defensive against data racing over qubit_word.
    //                 if ((j != destab_pivot) && (j != stab_pivot) && (qubit_word & q_mask)) {
    //                     row_mul(*dlocker, m, aux_power, *inv_xs, *inv_zs, unpacked_ss, j, destab_pivot, q_w, num_words_minor);
    //                 }
    //             }
    //         }
    //         // Pivot changed due to previous indeterminate measurement.
    //         else {
    //             if (!global_tx) m.pivot = MAX_QUBITS;
    //             find_min_pivot(m, *inv_xs, num_qubits, num_words_minor);
    //             //if (!global_tx) printf("-----------> pivot changed! <-------------\n");
    //         }
    //     }
    //     else {
    //         measure_determinate_qubit(*dlocker, m, *inv_xs, *inv_zs, unpacked_ss, aux, aux_power, num_qubits, num_words_minor);
    //     }
    // }

    // create separate kernels for this:

    // There is data racing ofr some unknown reason when setting measurement to the sign.

    // __global__ void is_indeterminate_outcome_single(bucket_t* measurements, const gate_ref_t* refs, const Table* inv_xs, 
    //                                                 const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
    //     const gate_ref_t r = refs[gate_index];
    //     assert(r < NO_REF);
    //     Gate& m = (Gate&) measurements[r];
    //     assert(m.size == 1);
    //     find_min_pivot(m, *inv_xs, num_qubits, num_words_minor);
    // }

    // __global__ void measure_indeterminate_single(Table* inv_xs, Table* inv_zs, Signs *inv_ss, DeviceLocker* dlocker, 
    //                                     bucket_t* measurements, const gate_ref_t* refs, const size_t gate_index, 
    //                                     const size_t num_qubits, const size_t num_words_minor) {
    //     word_std_t* aux = SharedMemory<word_std_t>();
    //     int* aux_power = reinterpret_cast<int*>(aux + blockDim.x * 2);
    //     int* unpacked_ss = inv_ss->unpacked_data();
    //     const gate_ref_t r = refs[gate_index];
    //     Gate& m = (Gate&) measurements[r];
    //     assert(m.pivot != MAX_QUBITS);
    //     const grid_t destab_pivot = m.pivot;
    //     const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
    //     const word_std_t q_mask = BITMASK_GLOBAL(q);
    //     const grid_t stab_pivot = destab_pivot + num_qubits;
    //     const grid_t stab_row = stab_pivot * num_words_minor;
    //     const word_std_t qubit_stab_word = (*inv_xs)[stab_row + q_w];
    //     // If pivot is still valid in the current quantum state.
    //     if (qubit_stab_word & q_mask) {
    //         const grid_t destab_row = destab_pivot * num_words_minor;
    //         for_parallel_x(w, num_words_minor) {
    //             const grid_t src_word_idx = stab_row + w;
    //             const grid_t des_word_idx = destab_row + w;
    //             (*inv_xs)[des_word_idx] = (*inv_xs)[src_word_idx];
    //             (*inv_zs)[des_word_idx] = (*inv_zs)[src_word_idx];
    //             if (w != q_w) {
    //                 (*inv_xs)[src_word_idx] = 0;
    //                 (*inv_zs)[src_word_idx] = 0; 
    //             }
    //             else {
    //                 (*inv_xs)[src_word_idx] = 0;
    //                 (*inv_zs)[src_word_idx] = q_mask;
    //                 const int rand_measure = 2; //2 * (rand() % 2);
    //                 m.measurement = rand_measure;
    //                 unpacked_ss[destab_pivot] = unpacked_ss[stab_pivot];
    //                 unpacked_ss[stab_pivot] = rand_measure;
    //             }
    //         }
    //         for (grid_t j = 0; j < 2 * num_qubits; j++) {
    //             const word_std_t j_qubit_word = (*inv_xs)[j * num_words_minor + q_w];
    //             // (j != stab_pivot) is defensive against data racing over qubit_word.
    //             if ((j != destab_pivot) && (j != stab_pivot) && (j_qubit_word & q_mask)) {
    //                 row_mul(*dlocker, m, aux_power, *inv_xs, *inv_zs, unpacked_ss, j, destab_pivot, q_w, num_words_minor);
    //             }
    //         }
    //     }
    //     // Pivot changed due to previous indeterminate measurement.
    //     else if (!global_tx) { 
    //         // Reset pivot.
    //         m.pivot = MAX_QUBITS;
    //     }
    // }

    // __global__ void measure_determinate_single(Table* inv_xs, Table* inv_zs, Signs *inv_ss, DeviceLocker* dlocker, 
    //                                     bucket_t* measurements, const gate_ref_t* refs, const size_t gate_index, 
    //                                     const size_t num_qubits, const size_t num_words_minor) {
    //     word_std_t* aux = SharedMemory<word_std_t>();
    //     int* aux_power = reinterpret_cast<int*>(aux + blockDim.x * 2);
    //     int* unpacked_ss = inv_ss->unpacked_data();
    //     const gate_ref_t r = refs[gate_index];
    //     Gate& m = (Gate&) measurements[r];
    //     assert(m.pivot == MAX_QUBITS);
    //     measure_determinate_qubit(*dlocker, m, *inv_xs, *inv_zs, unpacked_ss, aux, aux_power, num_qubits, num_words_minor);
    // }

    void Simulator::measure(const size_t& p, const depth_t& depth_level, const cudaStream_t* streams, const bool& reversed) {

        cudaStream_t copy_stream1 = cudaStream_t(0);
        cudaStream_t copy_stream2 = cudaStream_t(0);
        cudaStream_t kernel_stream = cudaStream_t(0);

        if (!depth_level) locker.reset(copy_stream1);

        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_words_major = inv_tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();

        uint32 nThreads_gates = 32;
        uint32 nBlocks_gates = ROUNDUPBLOCKS(num_gates_per_window, nThreads_gates);

        // Reset pivots.
        reset_pivots <<<nBlocks_gates, nThreads_gates, 0, copy_stream1>>> (gpu_circuit.pivots(), num_gates_per_window);

        // Transpose the tableau into row-major format.
        transpose_to_rowmajor<<< bestGridMeasure, bestBlockMeasure, 0, kernel_stream >>>(XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
        if (options.sync) {
            LASTERR("failed to launch transpose_to_rowmajor kernel");
            SYNC(kernel_stream);
        }

        // Sync streams.
        SYNC(copy_stream1);
        SYNC(copy_stream2);

        find_pivots_initial <<<bestGridMeasure, bestBlockMeasure, 0, kernel_stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch find_pivots_initial kernel");
            SYNC(kernel_stream);
        }

        //print_tableau(inv_tableau, depth_level, false);

        initialize_determinate_measurements <<<nBlocks_gates, nThreads_gates, 0, kernel_stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch initialize_determinate_measurements kernel");
            SYNC(kernel_stream);
        }

        print_gates(gpu_circuit, num_gates_per_window, depth_level);

        // This kernel cannot use grid-stride loops in
        // x-dim. Nr. of blocks must be always sufficient
        // as we use shraed memory as scratch space.
        dim3 nThreads_det(32, 2);
        uint32 nBlocksX_det = ROUNDUPBLOCKS(num_words_minor, nThreads_det.x);
        OPTIMIZEBLOCKS(nBlocksY_det, num_gates_per_window, nThreads_det.y);
        dim3 nBlocks_det(nBlocksX_det, nBlocksY_det);
        OPTIMIZESHARED(smem_size_det, nThreads_det.y * (nThreads_det.x * 2), sizeof(int) + sizeof(word_std_t));
        measure_determinate <<<nBlocks_det, nThreads_det, smem_size_det, kernel_stream>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch measure_determinate kernel");
            SYNC(kernel_stream);
        }

        // ========> REPLACE this with pivots copy not gates.

        //gpu_circuit.copyto(circuit, depth_level);

        //circuit.print_window(depth_level);

        //printf("--> gates before measuring\n"), print_gates(gpu_circuit, num_gates_per_window, depth_level);

        // uint32 nThreads_indet = 32;
        // uint32 nBlocks_indet = ROUNDUPBLOCKS(num_words_minor, nThreads_indet);
        // OPTIMIZESHARED(smem_size_indet, (nThreads_indet * 2), sizeof(int) + sizeof(word_std_t));
        // Window& window = circuit[depth_level];
        // for(size_t i = 0; i < num_gates_per_window; i++) {
        //     const gate_ref_t r = window[i];
        //     assert(r < NO_REF);
        //     Gate* m = circuit.gateptr(r);
        //     qubit_t curr_pivot = m->pivot;
        //     if (curr_pivot != MAX_QUBITS) {
        //         //printf("--> before measuring\n"), print_tableau(inv_tableau, depth_level, reversed);
        //         measure_indeterminate_single<<<nBlocks_indet, nThreads_indet, smem_size_indet, kernel_stream>>>(XZ_TABLE(inv_tableau), inv_tableau.signs(), locker.deviceLocker(), gpu_circuit.gates(), gpu_circuit.references(), i, num_qubits, num_words_minor);
        //         is_indeterminate_outcome_single<<<nBlocks_indet, nThreads_indet, 0, kernel_stream>>>(gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), i, num_qubits, num_words_minor);
        //         gpu_circuit.copygateto(circuit, r, depth_level, kernel_stream);
        //         SYNC(kernel_stream);
        //         if (m->pivot != curr_pivot) {
        //             printf("pivot %d changed to %d\n", curr_pivot, m->pivot);
        //             if (m->pivot == MAX_QUBITS)
        //                 measure_determinate_single<<<nBlocks_indet, nThreads_indet, smem_size_indet, kernel_stream>>>(XZ_TABLE(inv_tableau), inv_tableau.signs(), locker.deviceLocker(), gpu_circuit.gates(), gpu_circuit.references(), i, num_qubits, num_words_minor);
        //             else
        //                 measure_indeterminate_single<<<nBlocks_indet, nThreads_indet, smem_size_indet, kernel_stream>>>(XZ_TABLE(inv_tableau), inv_tableau.signs(), locker.deviceLocker(), gpu_circuit.gates(), gpu_circuit.references(), i, num_qubits, num_words_minor);
        //         }
        //         //printf("\n--> after measuring"), m->print(true), printf(":"), print_tableau(inv_tableau, depth_level, false);
        //     }
        // }

        // Transpose the tableau back into column-major format.
        transpose_to_colmajor<<< bestGridMeasure, bestBlockMeasure, 0, kernel_stream >>>(XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
        if (options.sync) {
            LASTERR("failed to launch transpose_to_rowmajor kernel");
            SYNC(kernel_stream);
        }

        //print_gates(gpu_circuit, num_gates_per_window, depth_level);
        print_measurements(gpu_circuit, num_gates_per_window, depth_level);
        //print_tableau(inv_tableau, depth_level, false);

    } // End of function.

}