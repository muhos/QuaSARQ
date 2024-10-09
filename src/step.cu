#include "simulator.hpp"
#include "step.cuh"
#include "tuner.cuh"
#include "operators.cuh"
#include "macros.cuh"
#include "locker.cuh"
#include "warp.cuh"
#include "sum.cuh"

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
        grid_t BX = blockDim.x;
        grid_t global_offset = blockIdx.x * BX;
        grid_t collapse_tid = threadIdx.y * BX + tx;
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

            // Only blocks >= 64 use shared memory, < 64 use warp communications.
            if (BX >= 64) {
			    shared_signs[collapse_tid] = (tx < num_gates) ? signs_word : 0;
			    __syncthreads();
		    }

            collapse_load_shared(shared_signs, signs_word, collapse_tid, tx, num_gates);
            collapse_shared(shared_signs, signs_word, collapse_tid, BX, tx);
            collapse_warp(shared_signs, signs_word, collapse_tid, BX, tx);

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
        grid_t BX = blockDim.x;
        grid_t global_offset = blockIdx.x * BX;
        word_std_t* shared_signs = SharedMemory<word_std_t>();
        sign_t* signs = ss->data();

        for_parallel_y(w, num_words_per_column) {

            word_std_t signs_word = signs[w];

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

            assert(BX <= 32);
            collapse_warp_only(signs_word);

            // Atomically collapse all blocks.
            if (!tx && global_offset < num_gates) {
                atomicXOR(signs + w, signs_word);
            }

        }

    }

    INLINE_DEVICE uint32 find_min_pivot(Gate& m, uint32& new_pivot, const word_std_t& word, const size_t& word_idx, const size_t num_qubits) {
        uint32 old_pivot = MAX_QUBITS;
        new_pivot = old_pivot;
        if (word) {
            const int64 bit_offset = word_idx << WORD_POWER;
            const int64 first_one_pos = int64(__ffsll(word)) - 1;
            assert(first_one_pos >= 0);
            int64 generator_index = 0;
            // Find the first high generator per word.
            #pragma unroll
            for (int64 bit_index = first_one_pos; bit_index < WORD_BITS; ++bit_index) {
                if (word & BITMASK(bit_index)) {
                    generator_index = bit_index + bit_offset;
                    if (generator_index >= num_qubits) {
                        new_pivot = generator_index - num_qubits;
                        break;
                    }
                }
            }
            if (new_pivot != MAX_QUBITS)
                old_pivot = atomicMin(&(m.pivot), new_pivot);
        }
        return old_pivot;
    }

    __global__ void is_indeterminate_outcome(bucket_t* measurements, const gate_ref_t* refs, const Table* xs, const size_t num_qubits, const size_t num_gates, const size_t num_words_per_column) {

        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            assert(m.size == 1);
            const size_t col = m.wires[0] * num_words_per_column;
            for_parallel_x(j, num_words_per_column) {
                // Check stabilizers if Xgq has 1, if so save the minimum row index.
                uint32 new_pivot;
                find_min_pivot(m, new_pivot, (*xs)[col + j], j, num_qubits);
            }
        }

    }


    INLINE_DEVICE void row_aux_mul(DeviceLocker& dlocker, Gate& m, byte_t* aux, int* aux_power, const Table& xs, const Table& zs, const Signs& ss, const size_t& src_idx, const size_t& aux_idx, const size_t& num_qubits, const size_t& num_words_per_column) {
        byte_t* aux_xs = aux;
        byte_t* aux_zs = aux + blockDim.x;
        int* pos_is = aux_power;
        int* neg_is = aux_power + blockDim.x;
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t global_offset = blockIdx.x * BX;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        const grid_t q = blockIdx.x * BX + tx;
        const word_std_t generator_mask = BITMASK_GLOBAL(src_idx);
        const word_std_t shifts = (src_idx & WORD_MASK);
        const int s = (ss[WORD_OFFSET(src_idx)] & generator_mask) >> shifts;
        assert(s <= 1);

        // track power of i.
        int pos_i = 0, neg_i = 0;    
        if (!q) {
            // First add s.
            m.measurement += s * 2; // s = {0, 1} * 2
        }
        if (q < num_qubits && tx < num_qubits) {
            const size_t word_idx = q * num_words_per_column + WORD_OFFSET(src_idx);
            byte_t x = ((word_std_t(xs[word_idx]) & generator_mask) >> shifts);
            byte_t z = ((word_std_t(zs[word_idx]) & generator_mask) >> shifts);
            assert(x <= 1);
            assert(z <= 1);
            const byte_t aux_x = aux_xs[shared_tid];
            const byte_t aux_z = aux_zs[shared_tid];
            assert(aux_x <= 1);
            assert(aux_z <= 1);
            aux_xs[shared_tid] ^= x;
            aux_zs[shared_tid] ^= z;
            // We don't need to sync shared memory here.
            // It would be done in collapse_load_shared.
            int x_only = x & ~z;  // X 
            int y = x & z;        // Y 
            int z_only = ~x & z;  // Z 
            int aux_x_only = aux_x & ~aux_z;  // aux_X
            int aux_y = aux_x & aux_z;        // aux_Y 
            int aux_z_only = ~aux_x & aux_z;  // aux_Z 
            assert(x_only <= 1);
            assert(y <= 1);
            assert(z_only <= 1);
            assert(aux_x_only <= 1);
            assert(aux_y <= 1);
            assert(aux_z_only <= 1);
            // XY=iZ, YZ=iX, ZX=iY
            pos_i = (x_only & aux_y) + (y & aux_z_only) + (z_only & aux_x_only);
            assert(pos_i <= 1);
            // XZ=-iY, YX=-iZ, ZY=-iX     
            neg_i = (x_only & aux_z_only) + (y & aux_x_only) + (z_only & aux_y); 
            assert(neg_i <= 1);
        }

        // accumulate thread-local values in shared memory.
        load_shared(pos_is, pos_i, neg_is, neg_i, shared_tid, tx, num_qubits);
        sum_shared(pos_is, pos_i, neg_is, neg_i, shared_tid, BX, tx);
        sum_warp(pos_is, pos_i, neg_is, neg_i, shared_tid, BX, tx);
        
        if (!tx) {
            int old_measurement = atomicAdd(&m.measurement, (pos_i - neg_i));
            assert(old_measurement < UNMEASURED);
        }
    }

    INLINE_DEVICE void row_to_aux(int& measurement, byte_t* aux, const Table& xs, const Table& zs, const Signs& ss, const size_t& src_idx, const size_t& num_qubits, const size_t& num_words_per_column) {
        const word_std_t generator_mask = BITMASK_GLOBAL(src_idx);
        const word_std_t shifts = (src_idx & WORD_MASK);
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        const grid_t q = blockIdx.x * BX + tx;
        if (!q) {
            assert(measurement == UNMEASURED);
            measurement = int((ss[WORD_OFFSET(src_idx)] & generator_mask) >> shifts) * 2;
            assert(measurement >= 0 && measurement <= 2);
        }
        byte_t* aux_xs = aux;
        byte_t* aux_zs = aux + blockDim.x;
        if (q < num_qubits && tx < num_qubits) {
            const size_t word_idx = q * num_words_per_column + WORD_OFFSET(src_idx);
            aux_xs[shared_tid] = (word_std_t(xs[word_idx]) & generator_mask) >> shifts;
            aux_zs[shared_tid] = (word_std_t(zs[word_idx]) & generator_mask) >> shifts;
        }
        else {
            aux_xs[shared_tid] = 0;
            aux_zs[shared_tid] = 0;
        }
        __syncthreads();
    }


    INLINE_DEVICE void measure_determinate_qubit(DeviceLocker& dlocker, Gate& m, Table& xs, Table& zs, Signs& ss, byte_t* smem, const size_t num_qubits, const size_t num_words_per_column) {
        const size_t col = m.wires[0] * num_words_per_column;
        word_std_t word = 0;
        byte_t* aux = smem;
        int* aux_power = (int*)(aux + blockDim.y * blockDim.x * 2);
        for (size_t j = 0; j < num_words_per_column; j++) {
            word = xs[col + j];
            if (word) {
                const int64 generator_index = GET_GENERATOR_INDEX(word, j);
                // GET_GENERATOR_INDEX will work here as we check destabilizers,
                // which are stored first in bit-packing. Thus, we don't need to
                // loop over bits inside each word.
                if (generator_index < num_qubits) {
                    // TODO: transform the rows involved here into words. This could improve the rowmul operation.                 
                    row_to_aux(m.measurement, aux, xs, zs, ss, generator_index + num_qubits, num_qubits, num_words_per_column);
                    //print_shared_aux(dlocker, m, aux, num_qubits, generator_index + num_qubits, generator_index + num_qubits);
                    //if (!global_tx) print_row(*dlocker, m, *xs, *zs, *ss, generator_index + num_qubits, num_qubits, num_words_per_column);
                    for (size_t k = generator_index + 1; k < num_qubits; k++) {
                        word = xs[col + WORD_OFFSET(k)];
                        if (word & BITMASK_GLOBAL(k)) {
                            //if (!global_tx) print_row(*dlocker, m, *xs, *zs, *ss, k + num_qubits, num_qubits, num_words_per_column);
                            row_aux_mul(dlocker, m, aux, aux_power, xs, zs, ss, k + num_qubits, generator_index + num_qubits, num_qubits, num_words_per_column);
                            //print_shared_aux(dlocker, m, aux, num_qubits, generator_index + num_qubits, k + num_qubits);
                        }
                    }
                    break;
                }
            }
        }
    }

    __global__ void measure_determinate(const gate_ref_t* refs, bucket_t* measurements, Table* xs, Table* zs, Signs* ss, uint32* aux_sign, byte_t* aux, DeviceLocker* dlocker, const size_t num_qubits, const size_t num_gates, const size_t num_words_per_column) {
        byte_t* smem = SharedMemory<byte_t>();
        for_parallel_y(i, num_gates) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            // Consdier only determinate measures.
            if (m.pivot == MAX_QUBITS) {
                assert(m.size == 1);
                measure_determinate_qubit(*dlocker, m, *xs, *zs, *ss, smem, num_qubits, num_words_per_column);
            }
        }
    }

    INLINE_DEVICE void row_to_row(Table& xs, Table& zs, Signs& ss, const size_t& dest_idx, const size_t& src_idx, const size_t& num_qubits, const size_t& num_words_per_column) {
        const word_std_t src_bit_pos = src_idx & WORD_MASK;
        const word_std_t dest_bit_pos = dest_idx & WORD_MASK;
        const word_std_t dest_reset_mask = ~BITMASK_GLOBAL(dest_idx);
        for_parallel_x(q, num_qubits) {
            const size_t src_row = WORD_OFFSET(src_idx);
            const size_t dest_row = WORD_OFFSET(dest_idx);
            const size_t src_word_idx = q * num_words_per_column + src_row;
            const size_t dest_word_idx = q * num_words_per_column + dest_row;
            word_std_t x_src = xs[src_word_idx];
            word_std_t z_src = zs[src_word_idx];
            x_src = (x_src >> src_bit_pos) & 1;
            z_src = (z_src >> src_bit_pos) & 1;
            assert(x_src <= 1);
            assert(z_src <= 1);
            word_std_t x = xs[dest_word_idx];
            word_std_t z = zs[dest_word_idx];
            x &= dest_reset_mask;
            z &= dest_reset_mask;  
            xs[dest_word_idx] = (x | (x_src << dest_bit_pos));
            zs[dest_word_idx] = (z | (z_src << dest_bit_pos));
            if (!q) {
                word_std_t s_src = ss[src_row];
                word_std_t s = ss[dest_row];
                s &= dest_reset_mask;
                s_src = (s_src >> src_bit_pos) & 1;
                assert(s_src <= 1);
                ss[dest_row] = (s | (s_src << dest_bit_pos));
            }
        }
    }

    INLINE_DEVICE void row_set(Table& xs, Table& zs, Signs& ss, const size_t& dest_idx, const size_t& qubit, const size_t& num_qubits, const size_t& num_words_per_column) {
        const word_std_t dest_bit_pos = dest_idx & WORD_MASK;
        const word_std_t dest_reset_mask = ~BITMASK_GLOBAL(dest_idx);
        for_parallel_x(q, num_qubits) {
            const size_t dest_row = WORD_OFFSET(dest_idx);
            const size_t dest_word_idx = q * num_words_per_column + dest_row;
            xs[dest_word_idx] &= dest_reset_mask;
            zs[dest_word_idx] &= dest_reset_mask; 
            if (q == qubit) {
                ss[dest_row] &= dest_reset_mask;
                zs[dest_word_idx] |= (word_std_t(1) << dest_bit_pos);
            }
        }
    }

    __global__ void measure_indeterminate(const gate_ref_t* refs, bucket_t* measurements, Table* xs, Table* zs, Signs* ss, byte_t* aux, DeviceLocker* dlocker, const size_t num_qubits, const size_t num_gates, const size_t num_words_per_column) {
        
        for(size_t i = 0; i < num_gates; i++) {
            const gate_ref_t r = refs[i];
            assert(r < NO_REF);
            Gate& m = (Gate&) measurements[r];
            uint32 destab_pivot = m.pivot;
            if (destab_pivot != MAX_QUBITS) {
                assert(m.size == 1);
                // First we need to test the validity of this pivot,
                // by checking if the corresponding generator is HIGH.
                const size_t stab_pivot = destab_pivot + num_qubits;
                const size_t col = m.wires[0] * num_words_per_column;
                const word_std_t generator_word = (*xs)[col + WORD_OFFSET(stab_pivot)];
                const word_std_t generator_bit = generator_word & BITMASK_GLOBAL(stab_pivot);
                if (generator_bit) {
                    // We are good to go to do measurement.

                    // rowcopy(q, p, p + q->n);				// Set Xbar_p := Zbar_p
                    row_to_row(*xs, *zs, *ss, destab_pivot, stab_pivot, num_qubits, num_words_per_column);

                    // rowset(q, p + q->n, b + q->n);			// Set Zbar_p := Z_b
                    row_set(*xs, *zs, *ss, stab_pivot, m.wires[0], num_qubits, num_words_per_column);

                    // q->r[p + q->n] = 2 * (rand() % 2);		// moment of quantum randomness
                    // Outcome of measurement.
                    m.measurement = 2 * (i % 2); //2 * (rand() % 2);		// moment of quantum randomness
                    // for (i = 0; i < 2 * q->n; i++)			// Now update the Xbar's and Zbar's that don't commute with
                    //      if ((i != p) && (q->x[i][b5] & pw)) // Z_b
                    //          rowmult(q, i, p);
                    for (size_t j = 0; j < 2 * num_qubits; j++) {
                        word_std_t word = (*xs)[col + WORD_OFFSET(j)];
                        if (word & BITMASK_GLOBAL(j)) {
                            if (!global_tx) print_row(*dlocker, m, *xs, *zs, *ss, destab_pivot, num_qubits, num_words_per_column);
                            row_aux_mul(dlocker, m, xs, zs, ss, k + num_qubits, generator_index + num_qubits, num_qubits, num_words_per_column);
                            if (!global_tx) print_row(*dlocker, m, *xs, *zs, *ss, stab_pivot, num_qubits, num_words_per_column);
                        }
                    }
                    // if (q->r[p + q->n])
                    //     return 3;
                    // else
                    //     return 2;
                }
                else {
                    // In this situation, we have to possible scenarios:
                    // 1) the pivot has changed to a different value.
                    // 2) There is no pivot and this gate became determinate.
                    // In either case, we need to check again for a pivot, 
                    // but this time we don't have to find the minimum, as
                    // we need the pivot immediately.

                    // The idea is to check with atomicMin, as we did above 
                    // with the exception that we check the old value of atomicMin, 
                    // if the current thread found a pivot and the old value is MAX_QUBITS,
                    // this this pivot is new and we are good to go, if the old value, 
                    // already have been set to some pivot, we don't do anything and skip.
                    // By the end of this kernel, we could have some determinate gates (non-set pivots).
                    // Thus we need to run the measure_determinate kernel again after this one, we have to do this
                    // in a iteratively as long as we have determinate gates.

                    // Actually no, we can do also determinate gates here, by checking if are at the last thread ID and the old value, 
                    // of the atomicMin is still non-pivot, that means we couldn't find any pivot. We should test both approaches or maybe do some heuristic like counting how many
                    // determinate gates found, if they are suffiently large, do them in parallel, otherwise do them here.

                    // Reset current pivot.
                    if (!global_tx)
                        m.pivot = MAX_QUBITS;
                    // Check stabilizers if Xgq has 1, if so save the minimum row index.
                    for_parallel_x(j, num_words_per_column) {    
                        uint32 old_pivot = find_min_pivot(m, destab_pivot, (*xs)[col + j], j, num_qubits);
                        if (old_pivot == MAX_QUBITS && destab_pivot != MAX_QUBITS) {
                            
                        }
                    }
                }
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

        if (!circuit.is_measuring(depth_level)) {

            print_gates(gpu_circuit, num_gates_per_window, depth_level);

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
            // sync data transfer.
            SYNC(copy_stream1);
            SYNC(copy_stream2);

            if (!depth_level) locker.reset(0);
            is_indeterminate_outcome<<<bestGridStep, bestBlockStep, 0, kernel_stream>>>(gpu_circuit.gates(), gpu_circuit.references(), tableau.xtable(), num_qubits, num_gates_per_window, num_words_per_column);

            // This kernel cannot use grid-stride loops in
            // x-dim. Nr. of blocks must be always sufficient
            // as we use shraed memory as scratch space.
            dim3 nThreads(256, 2);
            uint32 nBlocksX = ROUNDUPBLOCKS(num_qubits, nThreads.x);
            OPTIMIZEBLOCKS(nBlocksY, num_gates_per_window, nThreads.y);
            OPTIMIZESHARED(smem_size, nThreads.y * (nThreads.x * 2), sizeof(int) + sizeof(byte_t));

            measure_determinate<<<dim3(nBlocksX, nBlocksY), nThreads, smem_size, kernel_stream>>>(gpu_circuit.references(), gpu_circuit.gates(), XZ_TABLE(tableau), tableau.signs(), tableau.auxiliary_sign(), tableau.auxiliary(), locker.deviceLocker(), num_qubits, num_gates_per_window, num_words_per_column);
            
            print_measurements(gpu_circuit, num_gates_per_window, depth_level);

            print_gates(gpu_circuit, num_gates_per_window, depth_level);
            
            measure_indeterminate<<<1, 16>>>(gpu_circuit.references(), gpu_circuit.gates(), XZ_TABLE(tableau), tableau.signs(), tableau.auxiliary(), locker.deviceLocker(), num_qubits, num_gates_per_window, num_words_per_column);
            SYNCALL; // just for debugging
            
            print_tableau(tableau, depth_level, false);

            print_gates(gpu_circuit, num_gates_per_window, depth_level);
        }

    } // End of function.

}