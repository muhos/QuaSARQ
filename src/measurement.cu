#include "simulator.hpp"
#include "measurement.cuh"
#include "tuner.cuh"
#include "operators.cuh"

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

    INLINE_DEVICE void do_CX_sharing_control(word_t& Xc, word_t& Zc, word_t& Xt, word_t& Zt, sign_t& s) {
        const word_std_t xc = Xc, zc = Zc, xt = Xt, zt = Zt;
        // Update X and Z words.
        Xt = xt ^ xc;
        Zc = zc ^ zt;
        // Update Sign words.
        const word_std_t xc_and_zt = xc & zt;
        const word_std_t not_xt_xor_zc = ~(xt ^ zc);
        //s ^= (xc_and_zt & not_xt_xor_zc);
        s ^= (xt ^ zc);
    }

    INLINE_DEVICE void do_YZ_Swap(word_t& X, word_t& Z, sign_t& s) {
        const word_std_t x = X, z = Z;
        X = x ^ z;
        s ^= (x & ~z);
    }

    INLINE_DEVICE void do_XZ_Swap(word_t& X, word_t& Z, sign_t& s) {
        do_SWAP(X, Z);
        s ^= word_std_t(X & Z);
    }

    __global__ void inject_CX(Table* inv_xs, Table* inv_zs, Signs* inv_ss, 
                            ConstPivotsPointer pivots, ConstBucketsPointer measurements, ConstRefsPointer refs, 
                            const size_t gate_index, 
                            const size_t num_qubits, 
                            const size_t num_words_major, const size_t num_words_minor) {
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        const uint32 c = pivots[gate_index].indeterminate; // control = pivot
        assert(c != INVALID_PIVOT);
        const size_t c_row = c * num_words_major;
        word_t* xs = inv_xs->data();
        word_t* zs = inv_zs->data();
        sign_t* ss = inv_ss->data();

        // for debugging.
        ss[0] = 44;

        for (size_t t = c + 1; t < num_qubits; t++) { // targets: pivot + 1, ..., num_qubits - 1.
            if (t > c) {
                const size_t t_row = t * num_words_major;
                // create a boolean array for all qubits ana initial them to q's position in stabilizers.
                const size_t q_word_idx = t_row + q_w + num_words_minor;
                const word_std_t qubit_word = xs[q_word_idx];
                //if (qubit_word & q_mask) {
                    for_parallel_x(w, num_words_minor) { // Update all words in both destabs and stabs.
                        const size_t c_destab = c_row + w;
                        const size_t t_destab = t_row + w;
                        const size_t c_stab = c_destab + num_words_minor;
                        const size_t t_stab = t_destab + num_words_minor;
                        assert(c_destab < inv_zs->size());
                        assert(t_destab < inv_zs->size());
                        assert(c_stab < inv_zs->size());
                        assert(t_stab < inv_zs->size());
                        do_CX_sharing_control(zs[c_stab], zs[c_destab], zs[t_stab], zs[t_destab], ss[w]);
                        assert(c_destab < inv_xs->size());
                        assert(t_destab < inv_xs->size());
                        assert(c_stab < inv_xs->size());
                        assert(t_stab < inv_xs->size());
                        do_CX_sharing_control(xs[c_stab], xs[c_destab], xs[t_stab], xs[t_destab], ss[w + num_words_minor]);
                    }
                //}

                // for debugging.
                // printf("CX(%lld, %lld):\n", c, t);
                // print_tables(*inv_xs, *inv_zs, inv_ss, 0);
            }
        }
    }

    __global__ void check_x_destab(Table* inv_xs, Pivot* pivots, ConstBucketsPointer measurements, ConstRefsPointer refs, 
                                    const size_t gate_index, const size_t num_words_major) {
        const gate_ref_t r = refs[gate_index];
        assert(r < NO_REF);
        Gate& m = (Gate&) measurements[r];
        const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
        const word_std_t q_mask = BITMASK_GLOBAL(q);
        Pivot& pivot = pivots[gate_index];
        assert(pivot.indeterminate != INVALID_PIVOT);
        const word_std_t qubit_word = (*inv_xs)[pivot.indeterminate * num_words_major + q_w];
        pivot.destab = bool(qubit_word & q_mask);
    }

    __global__ void inject_Swap(Table* inv_xs, Table* inv_zs, Signs* inv_ss, 
                            ConstPivotsPointer pivots,
                            const size_t gate_index, 
                            const size_t num_words_major, const size_t num_words_minor) {
        const uint32 c = pivots[gate_index].indeterminate; // control = pivot
        assert(c != INVALID_PIVOT);
        const size_t c_row = c * num_words_major;
        word_t* xs = inv_xs->data();
        word_t* zs = inv_zs->data();
        sign_t* ss = inv_ss->data();

        for_parallel_x(w, num_words_minor) { // Update all words in both destabs and stabs.
            const size_t c_destab = c_row + w;
            const size_t c_stab = c_destab + num_words_minor;
            assert(c_destab < inv_zs->size());
            assert(c_stab < inv_zs->size());
            assert(c_destab < inv_xs->size());
            assert(c_stab < inv_xs->size());
            // Load this word to a temp location before calling this kernel.
            if (pivots[gate_index].destab) {
                do_YZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_YZ_Swap(xs[c_stab], xs[c_destab], ss[w + num_words_minor]);
            }
            else {
                do_XZ_Swap(zs[c_stab], zs[c_destab], ss[w]);
                do_XZ_Swap(xs[c_stab], xs[c_destab], ss[w + num_words_minor]);
            }
        }
    }

    INLINE_DEVICE void do_CX_sharing_control_p1(const word_t& Xc, const word_t& Zc, word_std_t& t_expr, word_t& Xt, const word_t& Zt) {
        // word_std_t zc = atomicXOR(reinterpret_cast<word_std_t*>(&Zc), zt); // Zc = zc ^ zt
        // Update Sign words.
        //const word_std_t xc_and_zt = xc & zt;
        //const word_std_t not_xt_xor_zc = ~(xt ^ zc);
        //s ^= (xc_and_zt & not_xt_xor_zc);
        //s ^= (xt ^ zc);
        //return (Xt ^ Zc);
    }

    __device__ void blockPrefixXorInclusive(sign_t* arr, int n)
    {
        int tid = threadIdx.x;
        // We'll do log2(n) passes; each pass "XORs" the offset element
        for (int offset = 1; offset < n; offset <<= 1) {
            if (tid >= offset) {
                arr[tid] = arr[tid] ^ arr[tid - offset];
            }
            __syncthreads();
        }
        
    }

    __global__ void inject_CX_p1(Table* prefix_xs, Table* prefix_zs, Table* inv_xs, Table* inv_zs, Signs* inv_ss,
                            ConstPivotsPointer pivots, ConstBucketsPointer measurements, ConstRefsPointer refs, 
                            const size_t gate_index, 
                            const size_t num_qubits, 
                            const size_t num_words_major, const size_t num_words_minor) {
        word_std_t* shared = SharedMemory<word_std_t>();
        word_std_t* t_prefix_z = shared;
        word_std_t* t_prefix_x = t_prefix_z + blockDim.x;
        grid_t prefix_tid = threadIdx.y * 2 * blockDim.x + threadIdx.x;
        const uint32 c = pivots[gate_index].indeterminate; // control = pivot
        assert(c != INVALID_PIVOT);
        const size_t c_row = c * num_words_major;
        word_t* xs = inv_xs->data();
        word_t* zs = inv_zs->data();
        sign_t* ss = inv_ss->data();

        // for debugging.
        ss[0] = 44;

        //for_parallel_y(w, num_words_minor) { // Will be parallel in y-dim later.
        for (size_t w = 0; w < num_words_minor; w++) { // Update all words in both destabs and stabs.
            const size_t c_destab = c_row + w;
            const size_t c_stab = c_destab + num_words_minor;

            // Include the first word at step 0 in prefix.
            word_std_t t_delta_z = 0;
            word_std_t t_delta_x = 0;

            grid_t tid_x = blockIdx.x * blockDim.x + threadIdx.x; 
            size_t t = tid_x + c + 1; 

            if (t < num_qubits) { // targets: pivot + 1, ..., num_qubits - 1.
                assert(t > c);
                const gate_ref_t r = refs[gate_index];
                assert(r < NO_REF);
                Gate& m = (Gate&) measurements[r];
                const qubit_t q = m.wires[0], q_w = WORD_OFFSET(q);
                const word_std_t q_mask = BITMASK_GLOBAL(q);
                const size_t t_row = t * num_words_major;
                // create a boolean array for all qubits ana initial them to q's position in stabilizers.
                const size_t q_word_idx = t_row + q_w + num_words_minor;
                const word_std_t qubit_word = xs[q_word_idx];
                //if (qubit_word & q_mask) {
                    const size_t t_destab = t_row + w;              
                    const size_t t_stab = t_destab + num_words_minor;
                    assert(c_destab < inv_zs->size());
                    assert(t_destab < inv_zs->size());
                    printf("w(%lld), t(%lld): Zc(" B2B_STR ") ^ Zt(" B2B_STR ") = " B2B_STR "\n", w, t, 
                    RB2B((word_std_t)zs[c_destab]), RB2B((word_std_t)zs[t_destab]), RB2B((word_std_t)(zs[t_destab] ^ zs[c_destab])));   
                    // We need to compute prefix-xor of t-th destabilizer in X,Z for t = c+1, c+2, ... c+n-1
                    // so that later we can xor every prefix-xor with controlled destabilizer.
                    // However, with Xt, we don't need prefix-xor as Xt is keept in 'tableau' untouched untill we compute the signs.
                    t_delta_z = zs[t_destab];            
                    t_delta_x = xs[t_destab];
                //}

            }

            t_prefix_z[prefix_tid] = t_delta_z;
            t_prefix_x[prefix_tid] = t_delta_x;

            __syncthreads();


            // In-place inclusive XOR prefix-scan
            blockPrefixXorInclusive(t_prefix_z + threadIdx.y * 4 * blockDim.x, blockDim.x);
            blockPrefixXorInclusive(t_prefix_x + threadIdx.y * 4 * blockDim.x, blockDim.x);

            // Write each thread's final prefix-scan value to output
            if (t < num_qubits) {
                printf("w(%lld), t(%lld): prefix-xor (tz) = " B2B_STR "\n", w, tid_x, RB2B(t_prefix_z[prefix_tid]));
                printf("w(%lld), t(%lld): prefix-xor (tx) = " B2B_STR "\n", w, tid_x, RB2B(t_prefix_x[prefix_tid]));
                (*prefix_zs)[tid_x * num_words_minor + w] = t_prefix_z[prefix_tid];
                (*prefix_xs)[tid_x * num_words_minor + w] = t_prefix_x[prefix_tid];
            }
            
        }
    }

    void Simulator::measure(const size_t& p, const depth_t& depth_level, const bool& reversed) {
        assert(options.streams >= 4);
        cudaStream_t copy_stream1 = copy_streams[0];
        cudaStream_t copy_stream2 = copy_streams[1];
        cudaStream_t kernel_stream1 = kernel_streams[0];
        cudaStream_t kernel_stream2 = kernel_streams[1];

        const size_t num_words_minor = tableau.num_words_minor();
        const size_t num_words_major = tableau.num_words_major();
        const size_t num_gates_per_window = circuit[depth_level].size();

        // Reset all pivots.
        reset_pivots(num_gates_per_window, kernel_stream2);

        transpose(true, kernel_stream1);

        // Sync copying gates to device.
        SYNC(copy_stream1);
        SYNC(copy_stream2);
        // Sync resetting pivots.
        SYNC(kernel_stream2);

        cutimer.start(kernel_stream1);

        // Find all pivots if exist.
        bestblockallpivots.x = 32;
        bestblockallpivots.y = 16;
        find_pivots(tableau, num_gates_per_window, true, kernel_stream1);

        //// Initialize determinate measurements with tableau signs.
        //initialize_determinate_measurements <<<bestgridreset, bestblockreset, 0, kernel_stream1>>> (gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), tableau.xtable(), tableau.signs(), num_gates_per_window, num_qubits, num_words_minor);
        //if (options.sync) {
        //    LASTERR("failed to launch initialize_determinate_measurements kernel");
        //    SYNC(kernel_stream1);
        //}

        // Sync finding pivots.
        SYNC(kernel_stream1);

        cutimer.stop(kernel_stream1);

        printf("Finding pviots time = %.3f\n", cutimer.time());

        print_tableau(tableau, depth_level, false);

        // // Copy pivots to host.
        gpu_circuit.copypivots(copy_stream1, num_gates_per_window);
        if (options.sync) {
           LASTERR("failed to copy pivots");
           SYNC(copy_stream1);
        }


        SYNC(copy_stream1); gpu_circuit.print_pivots();

        Pivot* host_pivots = gpu_circuit.host_pivots();
        Pivot new_pivot;
        int64 random_measures = 0;
        for(size_t i = 0; i < num_gates_per_window; i++) {
            Pivot curr_pivot = host_pivots[i];
            if (curr_pivot.indeterminate != INVALID_PIVOT) {
                dim3 blocksize(num_qubits, 1);
                OPTIMIZESHARED(smem_size, blocksize.y * blocksize.x, 2 * sizeof(word_std_t));
                inject_CX_p1 <<<1, blocksize, smem_size, kernel_stream1>>> (XZ_TABLE(prefix_tableau), XZ_TABLE(tableau), tableau.signs(), 
                            gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(),
                            i, num_qubits, num_words_major, num_words_minor);
                LASTERR("failed to inject_CX_p1");
                SYNC(kernel_stream1);
                // inject_CX <<<8, 2, smem_size, kernel_stream1>>> (XZ_TABLE(tableau), tableau.signs(), 
                //             gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(),
                //             i, num_qubits, num_words_major, num_words_minor);
                // printf("qubit(%d), pivot(%d):\n", circuit.gate(depth_level, i).wires[0], curr_pivot.indeterminate), print_tableau(tableau, depth_level, false);
                // check_x_destab<<<1, 1, 0, kernel_stream1>>>(tableau.xtable(), gpu_circuit.pivots(), gpu_circuit.gates(), gpu_circuit.references(), i, num_words_major);
                // inject_Swap<<<1, 1, 0, kernel_stream1>>>(XZ_TABLE(tableau), tableau.signs(), gpu_circuit.pivots(), i, num_words_major, num_words_minor);
            
                printf("After inject_CX:\n"), print_tableau(prefix_tableau, depth_level, false, true);
            }
        }

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
            TRIM_GRID_IN_2D(currentblock, currentgrid, num_gates_or_index, y);
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

