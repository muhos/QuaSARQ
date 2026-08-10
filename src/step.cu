#include "simulator.hpp"
#include "step.cuh"
#include "noise.cuh"
#include "collapse.cuh"
#include "operators.cuh"
#include "templatedim.cuh"
#include "pivot.cuh"
#include "atomic.cuh"
#include "sum.cuh"

namespace QuaSARQ {

    INLINE_DEVICE
    void do_forall_gate(
                sign_t&         signs_word,
                word_t*         x_gens_word,
                word_t*         z_gens_word,
                const_refs_t    refs,
                const_buckets_t gates,
        const   uint32*         noise_paulis,
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
            case SQRT_X: {
                LOAD_Q1_WORDS;
                do_SQRT_X(signs_word, words_q1);
                break;
            }
            case SQRT_X_DAG: {
                LOAD_Q1_WORDS;
                do_SQRT_X_DAG(signs_word, words_q1);
                break;
            }
            case SQRT_Y: {
                LOAD_Q1_WORDS;
                do_SQRT_Y(signs_word, words_q1);
                break;
            }
            case SQRT_Y_DAG: {
                LOAD_Q1_WORDS;
                do_SQRT_Y_DAG(signs_word, words_q1);
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
            case DEPOLARIZE1:
            case X_ERROR:
            case Y_ERROR:
            case Z_ERROR:
            case PAULI_CHANNEL_1: {
                LOAD_Q1_WORDS;
                const uint32 pauli = (noise_paulis != nullptr) ? noise_paulis[i] : 0;
                do_depolarize1(signs_word, x_words_q1, z_words_q1, pauli);
                break;
            }
            case DEPOLARIZE2:
            case PAULI_CHANNEL_2: {
                LOAD_Q2_WORDS(num_words_major);
                const uint32 pauli = (noise_paulis != nullptr) ? noise_paulis[i] : 0;
                do_depolarize2(signs_word, x_words_q1, z_words_q1, x_words_q2, z_words_q2, pauli);
                break;
            }
            case CX: {
                LOAD_Q2_WORDS(num_words_major);
                do_CX(signs_word, q1, q2); break;
            }
            case CZ: {
                LOAD_Q2_WORDS(num_words_major);
                do_CZ(signs_word, q1, q2); break;
            }
            case CY: {
                LOAD_Q2_WORDS(num_words_major);
                do_CY(signs_word, q1, q2); break;
            }
            case SWAP: {
                LOAD_Q2_WORDS(num_words_major);
                do_SWAP(x_words_q1, x_words_q2); do_SWAP(z_words_q1, z_words_q2); break;
            }
            case ISWAP: {
                LOAD_Q2_WORDS(num_words_major);
                do_iSWAP(signs_word, q1, q2); break;
            }
            case ISWAP_DAG: {
                LOAD_Q2_WORDS(num_words_major);
                do_iSWAPdg(signs_word, q1, q2); break;
            }
            default: break;
            }
        }
    }

    __global__
    void step_append_atomic(
                const_refs_t 	refs,
                const_buckets_t gates,
        const   uint32*         noise_paulis,
        const 	size_t 			num_gates,
        const 	size_t 			num_words_major,
                Table *			xs,
                Table *			zs,
                Signs *			ss)
    {
        sign_t* signs = ss->data();
        for_parallel_y(w, num_words_major) {
            sign_t signs_word = 0;
            do_forall_gate(
                signs_word,
                xs->data() + w,
                zs->data() + w,
                refs,
                gates,
                noise_paulis,
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
    void step_append(
                    const_refs_t 	refs,
                    const_buckets_t gates,
            const   uint32*         noise_paulis,
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
            sign_t signs_word = 0;
            do_forall_gate(
                signs_word,
                xs->data() + w,
                zs->data() + w,
                refs,
                gates,
                noise_paulis,
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
    void step_append_warped(
                const_refs_t 	refs,
                const_buckets_t gates,
        const   uint32*         noise_paulis,
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
            sign_t signs_word = 0;
            word_t* x_gens_word = (!tx) ? xs->data() + w : nullptr;
            x_gens_word = (word_t*)__shfl_sync(0xFFFFFFFF, uint64(x_gens_word), 0, B);
            word_t* z_gens_word = (!tx) ? zs->data() + w : nullptr;
            z_gens_word = (word_t*)__shfl_sync(0xFFFFFFFF, uint64(z_gens_word), 0, B);
            do_forall_gate(
                signs_word,
                x_gens_word,
                z_gens_word,
                refs,
                gates,
                noise_paulis,
                num_gates,
                num_words_major
            );
            collapse_warp<B, sign_t>(signs_word, tx);
            if (!tx && signs_word) {
                atomicXOR(signs + w, signs_word);
            }
        }
    }

    #define CALL_STEP_APPEND_WARPED(B, YDIM) \
        step_append_warped<B> <<<currentgrid, currentblock, 0, stream>>> ( \
            refs, \
            gates, \
            noise_paulis, \
            num_gates_per_window, \
            num_words_major, \
            XZ_TABLE(tableau), \
            tableau.signs() \
        );

    #define CALL_STEP_APPEND(B, YDIM) \
        step_append<B> <<<currentgrid, currentblock, shared_size, stream>>> ( \
            refs, \
            gates, \
            noise_paulis, \
            num_gates_per_window, \
            num_words_major, \
            XZ_TABLE(tableau), \
            tableau.signs() \
        );

        void call_step_append(
                const_refs_t                refs,
                const_buckets_t             gates,
                Tableau &                   tableau,
        const   size_t &                    num_gates_per_window,
        const   size_t &                    num_words_major,
                curand_algorithm_t*         noise_states,
                uint32*                     noise_paulis,
        const   dim3 &                      currentblock,
        const   dim3 &                      currentgrid,
        const   size_t &                    shared_size,
        const   cudaStream_t &              stream)
    {
        // Sample noise.
        if (noise_states != nullptr && noise_paulis != nullptr) {
            dim3 sblock(256), sgrid;
            OPTIMIZEBLOCKS(sgrid.x, num_gates_per_window, sblock.x);
            sample_noise_k<<<sgrid, sblock, 0, stream>>>(
                noise_states, 
                noise_paulis, 
                refs, 
                gates, 
                num_gates_per_window);
        }
        // Apply gates.
        if (currentblock.x == 1) {
            step_append_atomic<<<currentgrid, currentblock, 0, stream>>>(
                refs,
                gates,
                noise_paulis,
                num_gates_per_window,
                num_words_major,
                XZ_TABLE(tableau),
                tableau.signs());
        }
        else if (currentblock.x > 1 && currentblock.x <= maxWarpSize) {
            switch (currentblock.x) {
                FOREACH_X_DIM_MAX_32(CALL_STEP_APPEND_WARPED, currentblock.y);
                default:
                    break;
            }
        }
        else {
            switch (currentblock.x) {
                FOREACH_X_DIM_MAX_1024(CALL_STEP_APPEND, currentblock.y);
                default:
                    break;
            }
        }
    }

    struct RowRef {
        word_std_t* x;
        word_std_t* z;
        sign_t*     s;
        word_std_t  smask;

        INLINE_DEVICE bool sign() const { return (*s & smask) != 0; }

        // Only this warp owns this bit (gates in a window are qubit-disjoint), but the word
        // is shared with up to WORD_BITS-1 other qubits, so the update must be atomic.
        INLINE_DEVICE void flip_sign() const { atomicXOR(s, smask); }
    };

    INLINE_DEVICE
    RowRef make_row(
                word_std_t*     xs,
                word_std_t*     zs,
                sign_t*         ss,
        const   size_t&         q,
        const   size_t&         half,
        const   size_t&         num_words_major,
        const   size_t&         num_words_minor)
    {
        RowRef r;
        const size_t base = q * num_words_major + half * num_words_minor;
        r.x = xs + base;
        r.z = zs + base;
        r.s = ss + half * num_words_minor + WORD_OFFSET(q);
        r.smask = BITMASK_GLOBAL(q);
        return r;
    }

    INLINE_DEVICE
    uint32 row_mul(
        const   RowRef&         lhs,
        const   RowRef&         rhs,
        const   size_t&         num_words_minor,
        const   uint32&         lane)
    {
        const bool rsign = rhs.sign();
        word_std_t phase_lo = 0, phase_hi = 0;
        for (size_t w = lane; w < num_words_minor; w += 32) {
            word_std_t x1 = lhs.x[w], z1 = lhs.z[w];
            const word_std_t x2 = rhs.x[w], z2 = rhs.z[w];
            const word_std_t ox1 = x1, oz1 = z1;
            x1 ^= x2; z1 ^= z2;
            const word_std_t x1z2 = ox1 & z2;
            const word_std_t anti = (x2 & oz1) ^ x1z2;
            phase_hi ^= (phase_lo ^ x1 ^ z1 ^ x1z2) & anti;
            phase_lo ^= anti;
            lhs.x[w] = x1; lhs.z[w] = z1;
        }
        uint32 log_i = (popcount_word(phase_lo) + 2u * popcount_word(phase_hi)) & 3u;
        log_i = warp_reduce_mod4(log_i);
        return (log_i ^ (uint32(rsign) << 1)) & 3u;
    }

    INLINE_DEVICE
    void multiply_into(
        const   RowRef&         lhs,
        const   RowRef&         rhs,
        const   size_t&         num_words_minor,
        const   uint32&         lane)
    {
        const uint32 log_i = row_mul(lhs, rhs, num_words_minor, lane);
        if (!lane && (log_i & 2))
            lhs.flip_sign();
        __syncwarp();
    }

    INLINE_DEVICE
    void swap_rows(
        const   RowRef&         a,
        const   RowRef&         b,
        const   size_t&         num_words_minor,
        const   uint32&         lane)
    {
        for (size_t w = lane; w < num_words_minor; w += 32) {
            word_std_t t;
            t = a.x[w]; a.x[w] = b.x[w]; b.x[w] = t;
            t = a.z[w]; a.z[w] = b.z[w]; b.z[w] = t;
        }
        if (!lane && a.sign() != b.sign()) {
            a.flip_sign();
            b.flip_sign();
        }
        __syncwarp();
    }

    #define DESTAB 0
    #define STAB   1
    #define ROW(Q, HALF) make_row(xs, zs, ss, Q, HALF, num_words_major, num_words_minor)

    #define PREPEND_X(Q) do { if (!lane) ROW(Q, STAB).flip_sign();   __syncwarp(); } while (0)
    #define PREPEND_Z(Q) do { if (!lane) ROW(Q, DESTAB).flip_sign(); __syncwarp(); } while (0)

    __global__
    void step_prepend_k(
                const_refs_t    refs,
                const_buckets_t gates,
        const   uint32*         noise_paulis,
        const   size_t          num_gates,
        const   size_t          num_words_major,
        const   size_t          num_words_minor,
                Table*          xs_table,
                Table*          zs_table,
                Signs*          ss_signs)
    {
        word_std_t* xs = xs_table->words();
        word_std_t* zs = zs_table->words();
        sign_t*     ss = ss_signs->data();
        const uint32 lane = threadIdx.x;

        for (size_t i = blockIdx.x; i < num_gates; i += gridDim.x) {
            const Gate& gate = (Gate&) gates[refs[i]];
            const size_t q1 = gate.wires[0];
            const size_t q2 = (gate.size == 2) ? size_t(gate.wires[1]) : q1;

            switch (gate.type) {
            case I: break;

            case Z: PREPEND_Z(q1); break;
            case X: PREPEND_X(q1); break;
            case Y: PREPEND_X(q1); PREPEND_Z(q1); break;

            case H:
                swap_rows(ROW(q1, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                break;

            case S:
                multiply_into(ROW(q1, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                break;
            case S_DAG:
                multiply_into(ROW(q1, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                PREPEND_Z(q1);
                break;

            case SQRT_X:
                multiply_into(ROW(q1, STAB), ROW(q1, DESTAB), num_words_minor, lane);
                break;
            case SQRT_X_DAG:
                multiply_into(ROW(q1, STAB), ROW(q1, DESTAB), num_words_minor, lane);
                PREPEND_X(q1);
                break;

            case SQRT_Y:
                swap_rows(ROW(q1, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                PREPEND_X(q1);
                break;
            case SQRT_Y_DAG:
                PREPEND_X(q1);
                swap_rows(ROW(q1, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                break;

            case DEPOLARIZE1:
            case X_ERROR:
            case Y_ERROR:
            case Z_ERROR:
            case PAULI_CHANNEL_1: {
                const uint32 pauli = (noise_paulis != nullptr) ? noise_paulis[i] : 0;
                if (pauli & 1u) PREPEND_X(q1);
                if (pauli & 2u) PREPEND_Z(q1);
                break;
            }
            case DEPOLARIZE2:
            case PAULI_CHANNEL_2: {
                const uint32 pauli = (noise_paulis != nullptr) ? noise_paulis[i] : 0;
                if (pauli & 1u) PREPEND_X(q1);
                if (pauli & 2u) PREPEND_Z(q1);
                if (pauli & 4u) PREPEND_X(q2);
                if (pauli & 8u) PREPEND_Z(q2);
                break;
            }

            case CX:
                multiply_into(ROW(q2, STAB),   ROW(q1, STAB),   num_words_minor, lane);
                multiply_into(ROW(q1, DESTAB), ROW(q2, DESTAB), num_words_minor, lane);
                break;

            case CZ:
                multiply_into(ROW(q2, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                multiply_into(ROW(q1, DESTAB), ROW(q2, STAB), num_words_minor, lane);
                break;

            case CY:
                multiply_into(ROW(q2, STAB), ROW(q2, DESTAB), num_words_minor, lane);
                PREPEND_Z(q2);
                multiply_into(ROW(q2, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                multiply_into(ROW(q1, DESTAB), ROW(q2, STAB), num_words_minor, lane);
                multiply_into(ROW(q2, STAB), ROW(q2, DESTAB), num_words_minor, lane);
                PREPEND_Z(q2);
                break;

            case SWAP:
                swap_rows(ROW(q1, STAB),   ROW(q2, STAB),   num_words_minor, lane);
                swap_rows(ROW(q1, DESTAB), ROW(q2, DESTAB), num_words_minor, lane);
                break;

            case ISWAP:
            case ISWAP_DAG:
                swap_rows(ROW(q1, STAB),   ROW(q2, STAB),   num_words_minor, lane);
                swap_rows(ROW(q1, DESTAB), ROW(q2, DESTAB), num_words_minor, lane);
                multiply_into(ROW(q2, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                multiply_into(ROW(q1, DESTAB), ROW(q2, STAB), num_words_minor, lane);
                multiply_into(ROW(q1, DESTAB), ROW(q1, STAB), num_words_minor, lane);
                multiply_into(ROW(q2, DESTAB), ROW(q2, STAB), num_words_minor, lane);
                if (gate.type == ISWAP_DAG) { PREPEND_Z(q1); PREPEND_Z(q2); }
                break;

            default: break;
            }
            __syncwarp();
        }
    }

    void call_step(
                const_refs_t        refs,
                const_buckets_t     gates,
                curand_algorithm_t* noise_states,
                uint32*             noise_paulis,
        const   size_t&             num_gates,
        const   size_t&             num_words_major,
        const   size_t&             num_words_minor,
                Table*              xs,
                Table*              zs,
                Signs*              ss,
        const   cudaStream_t&       stream)
    {
        if (!num_gates) return;
        if (noise_states != nullptr && noise_paulis != nullptr) {
            dim3 sblock(256), sgrid;
            OPTIMIZEBLOCKS(sgrid.x, num_gates, sblock.x);
            sample_noise_k<<<sgrid, sblock, 0, stream>>>(
                noise_states, noise_paulis, refs, gates, num_gates);
        }
        const dim3 block(32, 1, 1);
        dim3 grid;
        OPTIMIZEBLOCKS(grid.x, num_gates, 1);
        step_prepend_k <<<grid, block, 0, stream>>> (
            refs, gates, noise_paulis, num_gates,
            num_words_major, num_words_minor, xs, zs, ss);
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
            {
                if (gpu_circuit.noise_states() != nullptr) {
                    dim3 sblock(256), sgrid;
                    OPTIMIZEBLOCKS(sgrid.x, num_gates_per_window, sblock.x);
                    sample_noise_k<<<sgrid, sblock>>>(
                        gpu_circuit.noise_states(), 
                        gpu_circuit.noise_paulis(),
                        gpu_circuit.references(), 
                        gpu_circuit.gates(), 
                        num_gates_per_window);
                }
                step_append_atomic<<<dim3(1, 1), dim3(1, 1)>>>(
                    gpu_circuit.references(),
                    gpu_circuit.gates(),
                    gpu_circuit.noise_paulis(),
                    num_gates_per_window,
                    num_words_major,
                    XZ_TABLE(tableau),
                    tableau.signs());
            }
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
                    // noise state buffers.
                    , gpu_circuit.noise_states(), gpu_circuit.noise_paulis()
                    // kernel arguments.
                    , gpu_circuit.references(), gpu_circuit.gates(), tableau
                );
            }

            TRIM_BLOCK_IN_DEBUG_MODE(bestblockstep, bestgridstep, num_gates_per_window, num_words_major);

            // sync data transfer.
            SYNC(copy_stream1);
            SYNC(copy_stream2);

            LOGN2(2, "Running step per depth level %d %s.. ", depth_level, sync ? "synchroneously" : "asynchroneously");

            // Run simulation.
            if (options.sync) cutimer.start(kernel_stream);

            double elapsed = 0;
            call_step(
                gpu_circuit.references(),
                gpu_circuit.gates(),
                gpu_circuit.noise_states(),
                gpu_circuit.noise_paulis(),
                num_gates_per_window,
                num_words_major,
                tableau.num_words_minor(),
                XZ_TABLE(tableau),
                tableau.signs(),
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
            if (options.print_steptableau)
                print_tableau(tableau, depth_level, reversed);
            if (options.print_stepstate)
                print_paulis(tableau, depth_level, reversed);
        }

        if (options.progress_en || options.check_tableau) {
            SYNC(kernel_streams[0]);
            SYNC(kernel_streams[1]);
        }

        if (options.progress_en && !options.check_tableau) {
            print_progress(p, depth_level, true);
        }

        if (circuit.is_measuring(depth_level)) {
            recorder.print();
        }

    } // End of function.

}
