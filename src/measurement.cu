#include "simulator.hpp"
#include "measurement.cuh"
#include "tuner.cuh"
#include "locker.cuh"
#include "transpose.cuh"

namespace QuaSARQ {;

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

    __global__ void reset_pivot(Pivot* pivots, const size_t gate_index) {
        pivots[gate_index].reset();
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
                assert(pivots[i].determinate < num_qubits);
                const gate_ref_t r = refs[i];
                assert(r < NO_REF);
                Gate& m = (Gate&) measurements[r];
                m.measurement = inv_ss->get_unpacked_sign(pivots[i].determinate + num_qubits);
                assert(m.measurement != UNMEASURED);
            }
            // Mark determinate pivot invalid if measurement is indeterminate.
            else if (pivots[i].determinate != INVALID_PIVOT) {
                pivots[i].determinate = INVALID_PIVOT;
            }
        }
    }

    // Measure a determinate qubit.
    INLINE_DEVICE void measure_determinate_qubit(Gate& m, word_std_t* aux, int* aux_power, const Table& inv_xs, const Table& inv_zs, const Signs& inv_ss, const size_t& src_pivot, const size_t num_qubits, const size_t num_words_minor) {
        const grid_t src_idx = src_pivot + num_qubits;
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        const grid_t w = blockIdx.x * BX + tx;
        word_std_t* aux_xs = aux;
        word_std_t* aux_zs = aux_xs + blockDim.x;
        int* pos_is = aux_power;
        int* neg_is = aux_power + blockDim.x;
        if (w < num_words_minor) {
            const grid_t qubits_word_idx = src_idx * num_words_minor + w;
            aux_xs[shared_tid] = inv_xs[qubits_word_idx];
            aux_zs[shared_tid] = inv_zs[qubits_word_idx];
        }
        else {
            aux_xs[shared_tid] = 0;
            aux_zs[shared_tid] = 0;
        }
        __syncthreads();
        for (grid_t des_pivot = src_pivot + 1; des_pivot < num_qubits; des_pivot++) {
            word_std_t qubit_word = inv_xs[des_pivot * num_words_minor + q_w];
            if (qubit_word & q_mask) {
                const grid_t des_idx = des_pivot + num_qubits;
                int pos_i = 0, neg_i = 0;    
                if (w < num_words_minor) {
                    const grid_t qubits_word_idx = des_idx * num_words_minor + w;
                    const word_std_t x     = inv_xs[qubits_word_idx], z     = inv_zs[qubits_word_idx];
                    const word_std_t aux_x = aux_xs[shared_tid]     , aux_z = aux_zs[shared_tid];
                    aux_xs[shared_tid] ^= x;
                    aux_zs[shared_tid] ^= z;
                    COMPUTE_POWER_I(pos_i, neg_i, x, z, aux_x, aux_z);
                }
                ACCUMULATE_POWER_I_OFFSET(m.measurement, inv_ss.get_unpacked_sign(des_idx));
            }
        }
    }

    __global__ void measure_determinate(const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Table* inv_zs, const Signs* inv_ss, 
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
                measure_determinate_qubit(m, aux, aux_power, *inv_xs, *inv_zs, *inv_ss, pivots[i].determinate, num_qubits, num_words_minor);
            }
        }
    }

    __global__ void measure_indeterminate_phase1(DeviceLocker* dlocker, Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        int* unpacked_ss = inv_ss->unpacked_data();
        int* pos_is = SharedMemory<int>();
        int* neg_is = pos_is + blockDim.x;
        const grid_t destab_pivot = pivots[gate_index].indeterminate;
        assert(pivots[gate_index].determinate == INVALID_PIVOT);
        assert(destab_pivot != INVALID_PIVOT);
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t w = blockIdx.x * BX + tx;
        const grid_t stab_pivot = destab_pivot + num_qubits;
        const grid_t stab_row = stab_pivot * num_words_minor;
        const gate_ref_t r = refs[gate_index];
        const Gate& m = (Gate&) measurements[r];
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        const word_std_t qubit_stab_word = (*inv_xs)[stab_row + q_w];
        // If pivot is still valid in the current quantum state.
        if (qubit_stab_word & q_mask) {
            inv_xs->set_stab(true);
            const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
            if (w < num_words_minor) {
                const grid_t src_word_idx = stab_row + w;
                const grid_t des_word_idx = destab_pivot * num_words_minor + w;
                (*inv_xs)[des_word_idx] = (*inv_xs)[src_word_idx];
                (*inv_zs)[des_word_idx] = (*inv_zs)[src_word_idx];
                if (w != q_w) {
                    (*inv_xs)[src_word_idx] = 0;
                    (*inv_zs)[src_word_idx] = 0; 
                }
                else {
                    (*inv_zs)[src_word_idx] = q_mask;
                    unpacked_ss[destab_pivot] = unpacked_ss[stab_pivot];
                }
            }
            for (grid_t des_idx = 0; des_idx < 2 * num_qubits; des_idx++) {
                const word_std_t des_qubit_word = (*inv_xs)[des_idx * num_words_minor + q_w];
                if ((des_idx != destab_pivot) && (des_idx != stab_pivot) && (des_qubit_word & q_mask)) {
                    int pos_i = 0, neg_i = 0; 
                    if (w < num_words_minor) {
                        const grid_t src_word_idx = destab_pivot * num_words_minor + w;
                        const grid_t des_word_idx = des_idx * num_words_minor + w;
                        const word_std_t src_x = (*inv_xs)[src_word_idx], src_z = (*inv_zs)[src_word_idx];
                        const word_std_t des_x = (*inv_xs)[des_word_idx], des_z = (*inv_zs)[des_word_idx];
                        if (w != q_w) 
                            (*inv_xs)[des_word_idx] = des_x ^ src_x;
                        (*inv_zs)[des_word_idx] = des_z ^ src_z;
                        COMPUTE_POWER_I(pos_i, neg_i, src_x, src_z, des_x, des_z);
                    }
                    ACCUMULATE_POWER_I(unpacked_ss[des_idx]);
                }
            }
        }
        // Pivot changed due to previous indeterminate measurement.
        else if (!w) { 
            // Reset pivot.
            inv_xs->set_stab(false);
            pivots[gate_index].reset();
        }
    }


    __global__ void measure_indeterminate_copy(const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                            Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                            const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const grid_t destab_pivot = pivots[gate_index].indeterminate;
        assert(destab_pivot != INVALID_PIVOT);  
        const grid_t stab_pivot = destab_pivot + num_qubits;
        const gate_ref_t r = refs[gate_index];
        Gate& m = (Gate&) measurements[r];
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        int* unpacked_ss = inv_ss->unpacked_data();
        for_parallel_x (w, num_words_minor) {
            const grid_t src_word_idx = stab_pivot * num_words_minor + w;
            const grid_t des_word_idx = destab_pivot * num_words_minor + w;
            (*inv_xs)[des_word_idx] = (*inv_xs)[src_word_idx];
            (*inv_zs)[des_word_idx] = (*inv_zs)[src_word_idx];
            (*inv_xs)[src_word_idx] = 0;
            (*inv_zs)[src_word_idx] = 0; 
            if (w == q_w) {
                (*inv_zs)[src_word_idx] = q_mask;
                unpacked_ss[destab_pivot] = unpacked_ss[stab_pivot];
                const int rand_measure = 2; //2 * (rand() % 2);
                m.measurement = rand_measure;
                unpacked_ss[stab_pivot] = rand_measure;
            }
        }
    }

    __global__ void measure_indeterminate_mul_phase1(const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const grid_t destab_pivot = pivots[gate_index].indeterminate;
        assert(destab_pivot != INVALID_PIVOT);  
        const grid_t stab_pivot = destab_pivot + num_qubits;
        const gate_ref_t r = refs[gate_index];
        const Gate& m = (Gate&) measurements[r];
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        int* unpacked_ss = inv_ss->unpacked_data();
        int* pos_is = SharedMemory<int>();
        int* neg_is = pos_is + blockDim.x;
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        for_parallel_y(des_idx, 2 * num_qubits) {
            const word_std_t des_qubit_word = (*inv_xs)[des_idx * num_words_minor + q_w];
            if ((des_idx != destab_pivot) && (des_idx != stab_pivot) && (des_qubit_word & q_mask)) {
                int pos_i = 0, neg_i = 0; 
                for_parallel_x(w, num_words_minor) {
                    const grid_t src_word_idx = destab_pivot * num_words_minor + w;
                    const grid_t des_word_idx = des_idx * num_words_minor + w;
                    const word_std_t src_x = (*inv_xs)[src_word_idx], src_z = (*inv_zs)[src_word_idx];
                    const word_std_t des_x = (*inv_xs)[des_word_idx], des_z = (*inv_zs)[des_word_idx];
                    COMPUTE_POWER_I(pos_i, neg_i, src_x, src_z, des_x, des_z);
                    if (w != q_w) (*inv_xs)[des_word_idx] = des_x ^ src_x;
                    (*inv_zs)[des_word_idx] = des_z ^ src_z;
                }
                ACCUMULATE_POWER_I(unpacked_ss[des_idx]);
            }
        }
    }

    __global__ void measure_indeterminate_mul_phase2(const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const gate_ref_t r = refs[gate_index];
        const Gate& m = (Gate&) measurements[r];
        const grid_t destab_pivot = pivots[gate_index].indeterminate;
        assert(destab_pivot != INVALID_PIVOT);
        const grid_t stab_pivot = destab_pivot + num_qubits;
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        int* unpacked_ss = inv_ss->unpacked_data();
        for_parallel_x(des_idx, 2 * num_qubits) {
            const word_std_t des_qubit_word = (*inv_xs)[des_idx * num_words_minor + q_w];
            if ((des_idx != destab_pivot) && (des_idx != stab_pivot) && (des_qubit_word & q_mask)) {
                (*inv_xs)[des_idx * num_words_minor + q_w] ^= (*inv_xs)[destab_pivot * num_words_minor + q_w];
                CHECK_SIGN_OVERFLOW(des_idx, unpacked_ss[des_idx], unpacked_ss[destab_pivot]);
                unpacked_ss[des_idx] += unpacked_ss[destab_pivot];
            }
        }
    }

    __global__ void find_new_pivots(Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, const Table* inv_xs, 
                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        assert(m.size == 1);
        find_min_pivot(pivots[gate_index], m.wires[0], *inv_xs, num_qubits, num_words_minor);
    }

    __global__ void initialize_single_determinate_measurement(const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Signs* inv_ss,
                                        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        assert(pivots[gate_index].indeterminate == INVALID_PIVOT);
        assert(pivots[gate_index].determinate < num_qubits);
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        m.measurement = inv_ss->get_unpacked_sign(pivots[gate_index].determinate + num_qubits);
    }

    __global__ void measure_single_determinate(const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Table* inv_zs, const Signs* inv_ss, 
                                        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        assert(pivots[gate_index].determinate != INVALID_PIVOT);
        word_std_t* aux = SharedMemory<word_std_t>();
        int* aux_power = reinterpret_cast<int*>(aux + blockDim.y * blockDim.x * 2);
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        assert(m.size == 1);
        measure_determinate_qubit(m, aux, aux_power, *inv_xs, *inv_zs, *inv_ss, pivots[gate_index].determinate, num_qubits, num_words_minor);
    }

    void Simulator::measure(const size_t& p, const depth_t& depth_level, const bool& reversed) {

        //bool enable_measurement_tuner = options.tune_reset;

        //if (enable_measurement_tuner)
        //    tune_measurement(p, depth_level, reversed);

        assert(options.streams >= 4);
        cudaStream_t copy_stream1 = copy_streams[0];
        cudaStream_t copy_stream2 = copy_streams[1];
        cudaStream_t kernel_stream1 = kernel_streams[0];
        cudaStream_t kernel_stream2 = kernel_streams[1];

        if (!depth_level) locker.reset(kernel_stream1);

        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_words_major = inv_tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();

        // Reset pivots.
        if (options.tune_reset) {
            SYNCALL;
            tune_kernel_m(reset_pivots, "Resetting pivots", bestblockreset, bestgridreset, gpu_circuit.pivots(), num_gates_per_window);
        }
        reset_pivots <<<bestgridreset, bestblockreset, 0, kernel_stream2>>> (gpu_circuit.pivots(), num_gates_per_window);

        // Transpose the tableau into row-major format.
        if (options.tune_transpose2r) {
            SYNCALL;
            tune_kernel_m(transpose_to_rowmajor, "Transposing to row-major", 
            bestblocktranspose2r, bestgridtranspose2r, 
            0, false,        // shared size, extend?
            num_words_major, // x-dim
            2 * num_qubits,  // y-dim 
            XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
        }
        transpose_to_rowmajor<<< bestgridtranspose2r, bestblocktranspose2r, 0, kernel_stream1 >>>(XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
        if (options.sync) {
            LASTERR("failed to launch transpose_to_rowmajor kernel");
            SYNC(kernel_stream1);
        }

        // Sync copying gates to device.
        SYNC(copy_stream1);
        SYNC(copy_stream2);
        // Sync resetting pivots.
        SYNC(kernel_stream2);

        // Find all pivots if exist.
        if (options.tune_allpivots) {
            SYNCALL;
            tune_kernel_m(find_pivots_initial, "Find all pivots", 
            bestblockallpivots, bestgridallpivots, 
            0, false,        // shared size, extend?
            num_qubits,      // x-dim
            num_gates_per_window,  // y-dim 
            gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), num_gates_per_window, num_qubits, num_words_minor);
            reset_pivots <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), num_gates_per_window);
            SYNCALL;
        }
        find_pivots_initial <<<bestgridallpivots, bestblockallpivots, 0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch find_pivots_initial kernel");
            SYNC(kernel_stream1);
        }

        // Initialize determinate measurements with tableau signs.
        initialize_determinate_measurements <<<bestgridreset, bestblockreset, 0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch initialize_determinate_measurements kernel");
            SYNC(kernel_stream1);
        }

        // Sync modifying pivots.
        SYNC(kernel_stream1);

        // Copy pivots to host.
        gpu_circuit.copypivots(copy_stream1, num_gates_per_window);
        
        // Measure all determinate.
        if (options.tune_multdeterminate) {
            SYNCALL;
            const size_t shared_bytes = 2 * (sizeof(int) + sizeof(word_std_t));
            tune_determinate(measure_determinate, "measure all determinate", 
            bestblockmultdeterminate, bestgridmultdeterminate, 
            shared_bytes, true,     // shared size, extend?
            num_words_minor,        // x-dim
            num_gates_per_window,   // y-dim 
            gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
            initialize_determinate_measurements <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
            SYNCALL;
        }
        if (!bestgridmultdeterminate.x) {
            bestblockmultdeterminate = dim3(32, 2);
            bestgridmultdeterminate.x = ROUNDUPBLOCKS(num_words_minor, bestblockmultdeterminate.x);
            OPTIMIZEBLOCKS2D(nBlocksY, num_gates_per_window, bestblockmultdeterminate.y);
            bestgridmultdeterminate.y = nBlocksY;
        }
        OPTIMIZESHARED(smem_multdeterminate, bestblockmultdeterminate.y * (bestblockmultdeterminate.x * 2), sizeof(int) + sizeof(word_std_t));
        measure_determinate <<<bestgridmultdeterminate, bestblockmultdeterminate, smem_multdeterminate, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch measure_determinate kernel");
            SYNC(kernel_stream1);
        }

        // Sync copying pivots.
        SYNC(copy_stream1);

        measure_indeterminate(depth_level, kernel_stream1);

        // Transpose the tableau back into column-major format.
        if (options.tune_transpose2c) {
            SYNCALL;
            tune_kernel_m(transpose_to_colmajor, "Transposing to column-major", 
            bestblocktranspose2c, bestgridtranspose2c, 
            0, false,        // shared size, extend?
            num_words_major, // x-dim
            num_qubits,      // y-dim 
            XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
        }
        transpose_to_colmajor<<< bestgridtranspose2c, bestblocktranspose2c, 0, kernel_stream1 >>>(XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
        if (options.sync) {
            LASTERR("failed to launch transpose_to_rowmajor kernel");
            SYNC(kernel_stream1);
        }

        //print_gates(gpu_circuit, num_gates_per_window, depth_level);
        print_measurements(gpu_circuit, num_gates_per_window, depth_level);
        //print_tableau(inv_tableau, depth_level, false);

    } // End of function.

    void Simulator::measure_indeterminate(const size_t& gate_index, const size_t& smem_size, const cudaStream_t& stream) {
        const size_t num_words_minor = inv_tableau.num_words_minor();
        if (options.tune_copyindeterminate || options.tune_phase1indeterminate || options.tune_phase2indeterminate) {
            printf("before tuning:\n"), print_tableau(inv_tableau, -1, false);
            SYNCALL;
            ts.recover = true;
            ts.set_original_pointers(inv_tableau.xdata(), inv_tableau.zdata(), inv_tableau.num_words());
            ts.set_saving_pointers(tableau.xdata(), tableau.zdata());
            tune_indeterminate(measure_indeterminate_copy, measure_indeterminate_mul_phase1, measure_indeterminate_mul_phase2,
                bestblockcopyindeterminate, bestgridcopyindeterminate, 
                bestblockphase1indeterminate, bestgridphase1indeterminate,
                bestblockphase2indeterminate, bestgridphase2indeterminate,
                2 * sizeof(int), true, 
                gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
            ts.recover = false;
            inv_tableau.reset_signs();
            SYNCALL;
            printf("after tuning:\n"), print_tableau(inv_tableau, -1, false);
        }
        measure_indeterminate_copy        <<<bestgridcopyindeterminate,   bestblockcopyindeterminate,   0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
        measure_indeterminate_mul_phase1  <<<bestgridphase1indeterminate, bestblockphase1indeterminate, smem_size, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
        measure_indeterminate_mul_phase2  <<<bestgridphase2indeterminate, bestblockphase2indeterminate, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
    }

    void Simulator::measure_indeterminate(const depth_t& depth_level, const cudaStream_t& stream) {

        //printf("--> gates before indeterminate measuring\n"), print_gates(gpu_circuit, num_gates_per_window, depth_level);
        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_gates_per_window = circuit[depth_level].size();
        bool tuning_indeterminate = options.tune_copyindeterminate | options.tune_phase1indeterminate | options.tune_phase2indeterminate;

        OPTIMIZESHARED(smem_indeterminate, bestblockphase1indeterminate.y * (bestblockphase1indeterminate.x * 2), sizeof(int));

        uint32 nThreads_det = 256;
        uint32 nBlocks_det = ROUNDUPBLOCKS(num_words_minor, nThreads_det);
        OPTIMIZESHARED(smem_size_det, (nThreads_det * 2), sizeof(int) + sizeof(word_std_t));

        Pivot* host_pivots = gpu_circuit.host_pivots();
        Pivot new_pivot;
        for(size_t i = 0; i < num_gates_per_window; i++) {
            Pivot curr_pivot = host_pivots[i];
            if (curr_pivot.indeterminate != INVALID_PIVOT) {
                assert(curr_pivot.determinate == INVALID_PIVOT);
                //printf("--> before measuring\n"), circuit.gateptr(depth_level, i)->print(true), printf(":\n"), print_tableau(inv_tableau, depth_level, false);
                //measure_indeterminate_phase1 <<<nBlocks_indet, nThreads_indet, smem_size_indet, stream>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), i, num_qubits, num_words_minor);
                const Gate& m = circuit.gate(depth_level, i);
                if (inv_tableau.is_xstab_valid(m.wires[0], curr_pivot.indeterminate, stream)) {
                    measure_indeterminate(i, smem_indeterminate, stream);
                }
                else {
                    // Put this in a function called fund_new_pivots.
                    reset_pivot <<<1, 1, 0, stream>>> (gpu_circuit.pivots(), i);
                    find_new_pivots <<<bestgridnewpivots, bestblocknewpivots, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), i, num_qubits, num_words_minor);
                    gpu_circuit.copypivotto(new_pivot, i, stream);
                    SYNC(stream);
                    assert(new_pivot.indeterminate != curr_pivot.indeterminate);
                    //printf("pivot %d changed to %d\n", curr_pivot.indeterminate, new_pivot.indeterminate);
                    if (new_pivot.indeterminate == INVALID_PIVOT) {
                        initialize_single_determinate_measurement <<<1, 1, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), i, num_qubits, num_words_minor);
                        measure_single_determinate <<<nBlocks_det, nThreads_det, smem_size_det, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), i, num_qubits, num_words_minor);
                    }
                    else {
                        measure_indeterminate(i, smem_indeterminate, stream);
                    }
                }
                //printf("\n--> after measuring"), circuit.gateptr(depth_level, i)->print(true), printf(":\n"), print_tableau(inv_tableau, depth_level, false);
            }
        }
    }
}

