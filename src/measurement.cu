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

    INLINE_DEVICE void initialize_determinate_measurement_qubit(Gate& m, const size_t& pivot, const Table& inv_xs, const Signs& inv_ss, const size_t num_qubits, const size_t num_words_minor) {
        // Determinate pivot must be there if measurement is not indeterminate. CAN we prove this?
        assert(pivot != INVALID_PIVOT);
        assert(m.size == 1);
        assert(pivot < num_qubits);
        const grid_t stab_pivot = pivot + num_qubits;
        m.measurement = inv_ss.get_unpacked_sign(stab_pivot);
        assert(m.measurement != UNMEASURED);
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        for (grid_t des_pivot = pivot + 1; des_pivot < num_qubits; des_pivot++) {
            const word_std_t qubit_word = inv_xs[des_pivot * num_words_minor + q_w];
            if (qubit_word & q_mask) {
                CHECK_SIGN_OVERFLOW(des_pivot, m.measurement, inv_ss.get_unpacked_sign(des_pivot + num_qubits));
                //printf("M(%3d): adding sign[%lld]: %d to measurement: %d -> measurement: %d\n", q, des_pivot + num_qubits, inv_ss->get_unpacked_sign(des_pivot + num_qubits), m.measurement, inv_ss->get_unpacked_sign(des_pivot + num_qubits) + m.measurement);
                m.measurement += inv_ss.get_unpacked_sign(des_pivot + num_qubits);
            }
        }
    }

    __global__ void initialize_determinate_measurements(Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Signs* inv_ss,
                                        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor) {
        for_parallel_x(i, num_gates) {
            // Check if the current gate is determinate
            // Indeterminate has higher priority here.
            if (pivots[i].indeterminate == INVALID_PIVOT) {
                const gate_ref_t r = refs[i];
                assert(r < NO_REF);
                Gate& m = (Gate&) measurements[r];
                initialize_determinate_measurement_qubit(m, pivots[i].determinate, *inv_xs, *inv_ss, num_qubits, num_words_minor);
            }
            // Mark determinate pivot invalid if measurement is indeterminate.
            else if (pivots[i].determinate != INVALID_PIVOT) {
                pivots[i].determinate = INVALID_PIVOT;
            }
        }
    }

    // Measure a determinate qubit.
    INLINE_DEVICE void measure_determinate_qubit(DeviceLocker& dlocker, Gate& m, word_std_t* aux, int* aux_power, const Table& inv_xs, const Table& inv_zs, const size_t& src_pivot, const size_t num_qubits, const size_t num_words_minor) {
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
                ACCUMULATE_POWER_I(m.measurement);
            }
        }
    }

    __global__ void measure_determinate(DeviceLocker* dlocker, const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
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

    // TODO: possible optimizations:
    //  - try to load source row into shared memory.
    //  - use 2D threads however, this kernel has to be split into two kernels
    //    one to copy the row in 1D, the other to do the multiplcations in 2D.
    //    this will be quite effective as #muls is high w.r.t num_qubits

    __global__ void measure_indeterminate_phase1(DeviceLocker* dlocker, Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t w = blockIdx.x * BX + tx;
        const grid_t destab_pivot = pivots[gate_index].indeterminate;
        assert(destab_pivot != INVALID_PIVOT);  
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
            int* unpacked_ss = inv_ss->unpacked_data();
            word_std_t* aux = SharedMemory<word_std_t>();
            int* pos_is = reinterpret_cast<int*>(aux + blockDim.x * 2);
            int* neg_is = pos_is + blockDim.x;
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

    __global__ void measure_indeterminate_phase1_shared(DeviceLocker* dlocker, Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t w = blockIdx.x * BX + tx;
        const grid_t destab_pivot = pivots[gate_index].indeterminate;
        assert(destab_pivot != INVALID_PIVOT);  
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
            int* unpacked_ss = inv_ss->unpacked_data();
            word_std_t* aux = SharedMemory<word_std_t>();
            word_std_t* aux_xs = aux;
            word_std_t* aux_zs = aux_xs + blockDim.x;
            int* pos_is = reinterpret_cast<int*>(aux + blockDim.x * 2);
            int* neg_is = pos_is + blockDim.x;
            const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
            if (w < num_words_minor) {
                const grid_t src_word_idx = stab_row + w;
                const grid_t des_word_idx = destab_pivot * num_words_minor + w;
                word_std_t src_x = (*inv_xs)[src_word_idx];
                word_std_t src_z = (*inv_zs)[src_word_idx];
                (*inv_xs)[des_word_idx] = src_x;
                (*inv_zs)[des_word_idx] = src_z;
                aux_xs[shared_tid] = src_x;
                aux_zs[shared_tid] = src_z;
                if (w != q_w) {
                    (*inv_xs)[src_word_idx] = 0;
                    (*inv_zs)[src_word_idx] = 0; 
                }
                else {
                    (*inv_zs)[src_word_idx] = q_mask;
                    unpacked_ss[destab_pivot] = unpacked_ss[stab_pivot];
                }
            }
            else {
                aux_xs[shared_tid] = 0;
                aux_zs[shared_tid] = 0;
            }
            __syncthreads();
            for (grid_t des_idx = 0; des_idx < 2 * num_qubits; des_idx++) {
                const word_std_t des_qubit_word = (*inv_xs)[des_idx * num_words_minor + q_w];
                if ((des_idx != destab_pivot) && (des_idx != stab_pivot) && (des_qubit_word & q_mask)) {
                    int pos_i = 0, neg_i = 0; 
                    if (w < num_words_minor) {
                        const grid_t des_word_idx = des_idx * num_words_minor + w;
                        const word_std_t src_x = aux_xs[shared_tid], src_z = aux_zs[shared_tid];
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

    __global__ void measure_indeterminate_phase2(DeviceLocker* dlocker, Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs, 
                                                Table* inv_xs, Table* inv_zs, Signs *inv_ss,
                                                const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        // Update X and set a ranfom measurement.
        if (inv_xs->is_stab_valid()) {
            const gate_ref_t r = refs[gate_index];
            Gate& m = (Gate&) measurements[r];
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
            if (!global_tx) {
                const int rand_measure = 2; //2 * (rand() % 2);
                m.measurement = rand_measure;
                unpacked_ss[stab_pivot] = rand_measure;
                (*inv_xs)[stab_pivot * num_words_minor + q_w] = 0;
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
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        initialize_determinate_measurement_qubit(m, pivots[gate_index].determinate, *inv_xs, *inv_ss, num_qubits, num_words_minor);
    }

    __global__ void measure_single_determinate(DeviceLocker* dlocker, const Pivot* pivots, bucket_t* measurements, const gate_ref_t* refs,
                                        const Table* inv_xs, const Table* inv_zs, 
                                        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor) {
        assert(pivots[gate_index].determinate != INVALID_PIVOT);
        word_std_t* aux = SharedMemory<word_std_t>();
        int* aux_power = reinterpret_cast<int*>(aux + blockDim.y * blockDim.x * 2);
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        assert(m.size == 1);
        measure_determinate_qubit(*dlocker, m, aux, aux_power, *inv_xs, *inv_zs, pivots[gate_index].determinate, num_qubits, num_words_minor);
    }

    void Simulator::measure(const size_t& p, const depth_t& depth_level, const bool& reversed) {

        assert(options.streams >= 4);
        cudaStream_t copy_stream1 = copy_streams[0];
        cudaStream_t copy_stream2 = copy_streams[1];
        cudaStream_t kernel_stream1 = kernel_streams[0];
        cudaStream_t kernel_stream2 = kernel_streams[1];

        if (!depth_level) locker.reset(kernel_stream1);

        const size_t num_words_minor = inv_tableau.num_words_minor();
        const size_t num_words_major = inv_tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();

        uint32 nThreads_gates = 32;
        uint32 nBlocks_gates = ROUNDUPBLOCKS(num_gates_per_window, nThreads_gates);

        // Reset pivots.
        reset_pivots <<<nBlocks_gates, nThreads_gates, 0, kernel_stream2>>> (gpu_circuit.pivots(), num_gates_per_window);

        // Transpose the tableau into row-major format.
        transpose_to_rowmajor<<< bestGridMeasure, bestBlockMeasure, 0, kernel_stream1 >>>(XZ_TABLE(inv_tableau), inv_tableau.signs(), XZ_TABLE(tableau), tableau.signs(), num_words_major, num_words_minor, num_qubits);
        if (options.sync) {
            LASTERR("failed to launch transpose_to_rowmajor kernel");
            SYNC(kernel_stream1);
        }

        // Sync copying gates to device.
        SYNC(copy_stream1);
        SYNC(copy_stream2);
        // Sync resetting pivots.
        SYNC(kernel_stream2);

        find_pivots_initial <<<bestGridMeasure, bestBlockMeasure, 0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch find_pivots_initial kernel");
            SYNC(kernel_stream1);
        }

        //print_tableau(inv_tableau, depth_level, false);

        initialize_determinate_measurements <<<nBlocks_gates, nThreads_gates, 0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch initialize_determinate_measurements kernel");
            SYNC(kernel_stream1);
        }

        // Sync modifying pivots.
        SYNC(kernel_stream1);
        // Copy pivots to host.
        gpu_circuit.copypivots(copy_stream1);
        

        // This kernel cannot use grid-stride loops in
        // x-dim. Nr. of blocks must be always sufficient
        // as we use shraed memory as scratch space.
        dim3 nThreads_det(32, 2);
        uint32 nBlocksX_det = ROUNDUPBLOCKS(num_words_minor, nThreads_det.x), nBlocksY_det = 0;
        OPTIMIZEBLOCKS(nBlocksY_det, num_gates_per_window, nThreads_det.y);
        dim3 nBlocks_det(nBlocksX_det, nBlocksY_det);
        OPTIMIZESHARED(smem_size_det, nThreads_det.y * (nThreads_det.x * 2), sizeof(int) + sizeof(word_std_t));
        measure_determinate <<<nBlocks_det, nThreads_det, smem_size_det, kernel_stream1>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), num_gates_per_window, num_qubits, num_words_minor);
        if (options.sync) {
            LASTERR("failed to launch measure_determinate kernel");
            SYNC(kernel_stream1);
        }

        // Sync copying pivots.
        SYNC(copy_stream1);

        //printf("--> gates before indeterminate measuring\n"), print_gates(gpu_circuit, num_gates_per_window, depth_level);

        uint32 nThreads_indet = 256, nBlocks_indet = 0;
        OPTIMIZESHARED(smem_size_indet, (nThreads_indet * 2), sizeof(int) + sizeof(word_std_t));
        Window& window = circuit[depth_level];
        Pivot* host_pivots = gpu_circuit.host_pivots();
        Pivot new_pivot;
        for(size_t i = 0; i < num_gates_per_window; i++) {
            Pivot curr_pivot = host_pivots[i];
            if (curr_pivot.indeterminate != INVALID_PIVOT) {
                assert(curr_pivot.determinate == INVALID_PIVOT);
                //printf("--> before measuring\n"), circuit.gateptr(depth_level, i)->print(true), printf(":\n"), print_tableau(inv_tableau, depth_level, reversed);
                nBlocks_indet = ROUNDUPBLOCKS(num_words_minor, nThreads_indet);
                measure_indeterminate_phase1 <<<nBlocks_indet, nThreads_indet, smem_size_indet, kernel_stream1>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), i, num_qubits, num_words_minor);
                
                // TODO: try communicating is_stab() before launching phase2, could save time
                // as in case of false no need to run phase2.
                OPTIMIZEBLOCKS(nBlocks_indet, 2 * num_qubits, nThreads_indet);
                measure_indeterminate_phase2 <<<nBlocks_indet, nThreads_indet,               0, kernel_stream1>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), i, num_qubits, num_words_minor);

                OPTIMIZEBLOCKS(nBlocks_indet, num_words_minor, nThreads_indet);
                find_new_pivots              <<<nBlocks_indet, nThreads_indet,               0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), i, num_qubits, num_words_minor);
                gpu_circuit.copypivotto(new_pivot, i, kernel_stream1);
                SYNC(kernel_stream1);
                if (new_pivot.indeterminate != curr_pivot.indeterminate) {
                    //printf("pivot %d changed to %d\n", curr_pivot.indeterminate, new_pivot.indeterminate);
                    if (new_pivot.indeterminate == INVALID_PIVOT) {
                        initialize_single_determinate_measurement <<<1, 1, 0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), inv_tableau.xtable(), inv_tableau.signs(), i, num_qubits, num_words_minor);
                        nBlocks_indet = ROUNDUPBLOCKS(num_words_minor, nThreads_indet);
                        measure_single_determinate <<<nBlocks_indet, nThreads_indet, smem_size_indet, kernel_stream1>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), i, num_qubits, num_words_minor);
                    }
                    else {
                        nBlocks_indet = ROUNDUPBLOCKS(num_words_minor, nThreads_indet);
                        measure_indeterminate_phase1 <<<nBlocks_indet, nThreads_indet, smem_size_indet, kernel_stream1>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), i, num_qubits, num_words_minor);
                        OPTIMIZEBLOCKS(nBlocks_indet, 2 * num_qubits, nThreads_indet);
                        measure_indeterminate_phase2 <<<nBlocks_indet, nThreads_indet,               0, kernel_stream1>>> (locker.deviceLocker(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), XZ_TABLE(inv_tableau), inv_tableau.signs(), i, num_qubits, num_words_minor);
                    }
                }
                //printf("\n--> after measuring"), circuit.gateptr(depth_level, i)->print(true), printf(":\n"), print_tableau(inv_tableau, depth_level, reversed);
            }
        }

        // Transpose the tableau back into column-major format.
        transpose_to_colmajor<<< bestGridMeasure, bestBlockMeasure, 0, kernel_stream1 >>>(XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(inv_tableau), inv_tableau.signs(), num_words_major, num_words_minor, num_qubits);
        if (options.sync) {
            LASTERR("failed to launch transpose_to_rowmajor kernel");
            SYNC(kernel_stream1);
        }

        //print_gates(gpu_circuit, num_gates_per_window, depth_level);
        print_measurements(gpu_circuit, num_gates_per_window, depth_level);
        //print_tableau(inv_tableau, depth_level, false);

    } // End of function.

}