#include "simulator.hpp"
#include "measurement.cuh"
#include "tuner.cuh"
namespace QuaSARQ {;

    __global__ void initialize_determinate_measurements(Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs,
                                        ConstTablePointer inv_xs, ConstSignsPointer inv_ss,
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

    __global__ void measure_all_determinate(ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs,
                                        ConstTablePointer inv_xs, ConstTablePointer inv_zs, ConstSignsPointer inv_ss, 
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

    __global__ void measure_indeterminate_copy(ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs, 
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

    __global__ void measure_indeterminate_mul_phase1(ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs, 
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

    __global__ void measure_indeterminate_mul_phase2(ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs, 
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

    __global__ void initialize_single_determinate_measurement(ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs,
                                        ConstTablePointer inv_xs, ConstSignsPointer inv_ss,
                                        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        assert(pivots[gate_index].indeterminate == INVALID_PIVOT);
        assert(pivots[gate_index].determinate < num_qubits);
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        m.measurement = inv_ss->get_unpacked_sign(pivots[gate_index].determinate + num_qubits);
    }

    __global__ void measure_single_determinate(ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs,
                                        ConstTablePointer inv_xs, ConstTablePointer inv_zs, ConstSignsPointer inv_ss, 
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
        assert(options.streams >= 4);
        cudaStream_t copy_stream1 = copy_streams[0];
        cudaStream_t copy_stream2 = copy_streams[1];
        cudaStream_t kernel_stream1 = kernel_streams[0];
        cudaStream_t kernel_stream2 = kernel_streams[1];

        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_words_major = inv_tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();

        // Reset all pivots.
        reset_pivots(num_gates_per_window, kernel_stream2);

        transpose(true, kernel_stream1);

        // Sync copying gates to device.
        SYNC(copy_stream1);
        SYNC(copy_stream2);
        // Sync resetting pivots.
        SYNC(kernel_stream2);

        // Find all pivots if exist.
        bestblockallpivots.x = 32;
        bestblockallpivots.y = 16;
        find_pivots(tableau, num_gates_per_window, true, kernel_stream1);

        //// Initialize determinate measurements with tableau signs.
        //initialize_determinate_measurements <<<bestgridreset, bestblockreset, 0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
        //if (options.sync) {
        //    LASTERR("failed to launch initialize_determinate_measurements kernel");
        //    SYNC(kernel_stream1);
        //}

        // Sync finding pivots.
        SYNC(kernel_stream1);

        print_tableau(tableau, depth_level, false);

        // Copy pivots to host.
        gpu_circuit.copypivots(copy_stream1, num_gates_per_window);
        if (options.sync) {
           LASTERR("failed to copy pivots");
           SYNC(copy_stream1);
        }


        SYNC(copy_stream1); gpu_circuit.print_pivots();
        //
        //// Measure all determinate.
        //measure_determinate(num_gates_per_window, true, kernel_stream1);

        //// Sync copying pivots.
        //SYNC(copy_stream1);

        //Measures& measure_stats = stats.circuit.measure_stats;
        //measure_stats.random_per_window = measure_indeterminate(depth_level, kernel_stream1);
        //measure_stats.random += measure_stats.random_per_window;
        //measure_stats.definite += num_gates_per_window - measure_stats.random_per_window;

        // Transpose the tableau back into column-major format.
        transpose(false, kernel_stream1);
    }

    void Simulator::measure_determinate(const size_t& num_gates_or_index, const bool& bulky, const cudaStream_t& stream) {
        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_words_major = inv_tableau.num_words_major();
        dim3 currentblock, currentgrid;
        if (bulky) {
            if (options.tune_multdeterminate) {
                SYNCALL;
                const size_t shared_bytes = 2 * (sizeof(int) + sizeof(word_std_t));
                tune_determinate(measure_all_determinate, "measure all determinate", 
                bestblockmultdeterminate, bestgridmultdeterminate, 
                shared_bytes, true,   // shared size, extend?
                num_words_minor,      // x-dim
                num_gates_or_index,   // y-dim 
                gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_gates_or_index, num_qubits, num_words_minor);
                initialize_determinate_measurements <<<bestgridreset, bestblockreset>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), num_gates_or_index, num_qubits, num_words_minor);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblockmultdeterminate, bestgridmultdeterminate, num_words_minor, num_gates_or_index);
            // Make sure there are sufficient threads in x-dim.
            // Grid-stride loop cannot be used here.
            if (size_t(bestgridmultdeterminate.x) * size_t(bestblockmultdeterminate.x) < num_words_minor) {
                bestblockmultdeterminate.x = 32;
                bestgridmultdeterminate.x = ROUNDUPBLOCKS(num_words_minor, bestblockmultdeterminate.x);
            }
            currentblock = bestblockmultdeterminate, currentgrid = bestgridmultdeterminate;
            TRIM_GRID_IN_2D(num_gates_or_index, y);
            OPTIMIZESHARED(smem_multdeterminate, currentblock.y * (currentblock.x * 2), sizeof(int) + sizeof(word_std_t));
            measure_all_determinate <<<currentgrid, currentblock, smem_multdeterminate, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_gates_or_index, num_qubits, num_words_minor);
            if (options.sync) {
                LASTERR("failed to launch measure_all_determinate kernel");
                SYNC(stream);
            }
        }
        else {
            const size_t gate_index = num_gates_or_index;
            initialize_single_determinate_measurement <<<1, 1, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
            if (options.tune_singdeterminate) {
                SYNCALL;
                const size_t shared_bytes = 2 * (sizeof(int) + sizeof(word_std_t));
                tune_single_determinate(measure_single_determinate, "Single determinate", 
                    bestblocksingdeterminate, bestgridsingdeterminate, shared_bytes,
                    gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
                initialize_single_determinate_measurement <<<1, 1, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
                SYNCALL;
            }
            TRIM_BLOCK_IN_DEBUG_MODE(bestblocksingdeterminate, bestgridsingdeterminate, num_words_minor, 0);
            // Make sure there are sufficient threads in x-dim.
            // Grid-stride loop cannot be used here.
            if (size_t(bestgridsingdeterminate.x) * size_t(bestblocksingdeterminate.x) < num_words_minor) {
                bestblocksingdeterminate.x = 256;
                bestgridsingdeterminate.x = ROUNDUPBLOCKS(num_words_minor, bestblocksingdeterminate.x);
            }
            currentblock = bestblocksingdeterminate, currentgrid = bestgridsingdeterminate;
            TRIM_GRID_IN_1D(num_words_minor, x);
            OPTIMIZESHARED(smem_singdeterminate, (currentblock.x * 2), sizeof(int) + sizeof(word_std_t));
            measure_single_determinate <<<currentgrid, currentblock, smem_singdeterminate, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
            if (options.sync) {
                LASTERR("failed to launch measure_single_determinate kernel");
                SYNC(stream);
            }
        }
    }

    void Simulator::measure_indeterminate(const size_t& gate_index, const cudaStream_t& stream) {
        const size_t num_words_minor = inv_tableau.num_words_minor();
        dim3 currentblock, currentgrid;
        if (options.tune_copyindeterminate || options.tune_phase1indeterminate || options.tune_phase2indeterminate) {
            SYNCALL;
            ts.recover = true;
            ts.set_original_pointers(inv_tableau.xdata(), inv_tableau.zdata(), inv_tableau.num_words_per_table());
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
        }
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockcopyindeterminate, bestgridcopyindeterminate, num_words_minor, 0);
        currentblock = bestblockcopyindeterminate, currentgrid = bestgridcopyindeterminate;
        TRIM_GRID_IN_1D(num_words_minor, x);
        measure_indeterminate_copy <<<currentgrid, currentblock, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch measure_indeterminate_copy kernel");
            SYNC(stream);
        }
        //
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockphase1indeterminate, bestgridphase1indeterminate, num_words_minor, 2 * num_qubits);
        currentblock = bestblockphase1indeterminate, currentgrid = bestgridphase1indeterminate;     
        TRIM_GRID_IN_XY(num_words_minor, 2 * num_qubits);
        OPTIMIZESHARED(smem_indeterminate, currentblock.y * (currentblock.x * 2), sizeof(int));
        measure_indeterminate_mul_phase1  <<<currentgrid, currentblock, smem_indeterminate, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch measure_indeterminate_mul_phase1 kernel");
            SYNC(stream);
        }
        //
        TRIM_BLOCK_IN_DEBUG_MODE(bestblockphase2indeterminate, bestgridphase2indeterminate, 2 * num_qubits, 0);
        currentblock = bestblockphase2indeterminate, currentgrid = bestgridphase2indeterminate;
        TRIM_GRID_IN_1D(2 * num_qubits, x);
        measure_indeterminate_mul_phase2  <<<currentgrid, currentblock, 0, stream>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), gate_index, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch measure_indeterminate_mul_phase2 kernel");
            SYNC(stream);
        }    
    }

    int64 Simulator::measure_indeterminate(const depth_t& depth_level, const cudaStream_t& stream) {
        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_gates_per_window = circuit[depth_level].size();
        Pivot* host_pivots = gpu_circuit.host_pivots();
        Pivot new_pivot;
        int64 random_measures = 0;
        for(size_t i = 0; i < num_gates_per_window; i++) {
            Pivot curr_pivot = host_pivots[i];
            if (curr_pivot.indeterminate != INVALID_PIVOT) {
                assert(curr_pivot.determinate == INVALID_PIVOT);
                const Gate& m = circuit.gate(depth_level, i);
                if (inv_tableau.is_xstab_valid(m.wires[0], curr_pivot.indeterminate, stream)) {
                    measure_indeterminate(i, stream);
                    random_measures++;
                }
                // Find new pivot.
                else {
                    find_pivots(inv_tableau, i, false, stream);
                    gpu_circuit.copypivotto(new_pivot, i, stream);
                    SYNC(stream);
                    assert(new_pivot.indeterminate != curr_pivot.indeterminate);
                    if (new_pivot.indeterminate == INVALID_PIVOT) {
                        measure_determinate(i, false, stream);
                    }
                    else {
                        measure_indeterminate(i, stream);
                        random_measures++;
                    }
                }
            }
        }
        return random_measures;
    }
}

