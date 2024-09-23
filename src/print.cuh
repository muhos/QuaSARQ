#ifndef __CU_PRINT_H
#define __CU_PRINT_H

#include "table.cuh"
#include "signs.cuh"
#include "vector.cuh"
#include "locker.cuh"
#include "circuit.cuh"

namespace QuaSARQ {

    INLINE_DEVICE void REPCH_GPU(const char* ch, const size_t& size, const size_t& off = 0) {
        for (size_t i = off; i < size; i++) LOGGPU("%s", ch);
    }

    // Print bit-packed bits in matrix format.
    template <class T>
    INLINE_ALL void print_table(const T& t) {
        const word_t* data = t.data();
        const size_t size = t.size();
        const size_t major_end = t.num_words_per_column();
        const size_t minor_end = t.num_words_per_column() * WORD_BITS;
        #if defined(INTERLEAVE_XZ)
        const size_t interleaving_offset = major_end * INTERLEAVE_COLS * 2;
        #else
        const size_t interleaving_offset = major_end;
        #endif
        size_t j = 0;
        for (size_t i = 0; i < size; i++) {
            if (i > 0 && i % major_end == 0)
                LOGGPU("\n");
            if (i > 0 && i % minor_end == 0)
                LOGGPU("\n");
            LOGGPU("  ");
            
            if (i > 0 && i % interleaving_offset == 0) 
                j++;
            if (i % major_end == 0)
                LOGGPU("%-2lld ", j);
            #if defined(WORD_SIZE_64)
            LOGGPU(B2B_STR, RB2B(uint32(word_std_t(data[i]) & 0xFFFFFFFFUL)));
            LOGGPU(B2B_STR, RB2B(uint32((word_std_t(data[i]) >> 32) & 0xFFFFFFFFUL)));
            #else
            LOGGPU(B2B_STR, RB2B(word_std_t(data[i])));
            #endif
        }
        LOGGPU("\n");
    }

    INLINE_ALL void print_table_signs(const Signs& ss, const size_t& offset = 0) {
        LOGGPU("     ");
        for (size_t i = offset; i < ss.size(); i++) {
             #if defined(WORD_SIZE_64)
            LOGGPU(B2B_STR, RB2B(uint32(word_std_t(ss[i]) & 0xFFFFFFFFUL)));
            LOGGPU(B2B_STR, RB2B(uint32((word_std_t(ss[i]) >> 32) & 0xFFFFFFFFUL)));
            #else
            LOGGPU(B2B_STR, RB2B(word_std_t(ss[i])));
            #endif
            LOGGPU("  ");
        }
        LOGGPU("\n");
    }

    INLINE_ALL void print_tables(const Table& xs, const Table& zs, const Signs& ss, const int64& level, const bool& measuring) {
        if (measuring)
            LOGGPU(" ---[ Destab/stab X-Table at (%-2lld)-step ]---------------------\n", level);
        else
            LOGGPU(" ---[ X-Table at (%-2lld)-step ]---------------------\n", level);
        print_table(xs);
        if (measuring)
            LOGGPU(" ---[ Destab/stab Z-Table at (%-2lld)-step ]---------------------\n", level);
        else
            LOGGPU(" ---[ Z-Table at (%-2lld)-step ]---------------------\n", level);
        print_table(zs);
        LOGGPU(" ---[ Signs at (%-2lld)-step ]-----------------------\n", level);
        print_table_signs(ss);
    }

    INLINE_ALL void print_tables(const Table& ps, const Signs& ss, const int64& level, const bool& measuring) {
        LOGGPU(" ---[ XZ bits at (%-2lld)-step ]---------------------\n", level);
        print_table(ps);
        LOGGPU(" ---[ Signs at (%-2lld)-step   ]---------------------\n", level);
        print_table_signs(ss);
    }

    INLINE_ALL void print_state(const Table& xs, const Table& zs, const Signs& ss, 
                                const size_t& start, const size_t& end, 
                                const size_t& num_qubits, const size_t& num_words_per_column) {
        for (size_t w = start; w < end; w++) {
            const word_t pow2 = BITMASK_GLOBAL(w);
            if (ss[WORD_OFFSET(w)] & sign_t(pow2)) {
                LOGGPU("-");
            }
            else {
                LOGGPU("+");
            }
            for (size_t q = 0; q < num_qubits; q++) {
                const size_t word_idx = q * num_words_per_column + WORD_OFFSET(w);
                if ((!(xs[word_idx] & pow2)) && (!(zs[word_idx] & pow2)))
                    LOGGPU("I");
                if ((xs[word_idx] & pow2) && (!(zs[word_idx] & pow2)))
                    LOGGPU("X");
                if ((!(xs[word_idx] & pow2)) && (zs[word_idx] & pow2))
                    LOGGPU("Z");
                if ((xs[word_idx] & pow2) && (zs[word_idx] & pow2))
                    LOGGPU("Y");
            }
            LOGGPU("\n");
        }
    }

    INLINE_DEVICE void print_column(DeviceLocker& dlocker, const Table& xs, const Table& zs, const Signs& ss, const size_t& q, const size_t& num_qubits, const size_t& num_words_per_column) {
        dlocker.lock();
        LOGGPU("   X(%-2lld)   Z(%-2lld)   S\n", q, q, q);
        for (size_t i = 0; i < 2 * num_qubits; i++) {
            if (i == num_qubits) {
                REPCH_GPU("-", 20);
                LOGGPU("\n");
            }
            LOGGPU("%-2lld   %-2d     %-2d     %-2d\n", i,
                bool(word_std_t(xs[q * num_words_per_column + WORD_OFFSET(i)]) & BITMASK_GLOBAL(i)),
                bool(word_std_t(zs[q * num_words_per_column + WORD_OFFSET(i)]) & BITMASK_GLOBAL(i)),
                bool(word_std_t(ss[WORD_OFFSET(i)]) & BITMASK_GLOBAL(i)));
        }
        dlocker.unlock();
    }

    INLINE_DEVICE void print_row(DeviceLocker& dlocker, const Gate& m, const Table& xs, const Table& zs, const Signs& ss, const size_t& src_idx, const size_t& num_qubits, const size_t& num_words_per_column) {
        dlocker.lock();
        m.print(true);
        LOGGPU("--> Row(%lld):   ", src_idx);
        for (size_t i = 0; i < num_qubits; i++)
            LOGGPU("%d", bool(word_std_t(xs[i * num_words_per_column + WORD_OFFSET(src_idx)]) & BITMASK_GLOBAL(src_idx)));
        LOGGPU(" ");
        for (size_t i = 0; i < num_qubits; i++)
            LOGGPU("%d", bool(word_std_t(zs[i * num_words_per_column + WORD_OFFSET(src_idx)]) & BITMASK_GLOBAL(src_idx)));
        LOGGPU(" ");
        LOGGPU("%d\n", bool(word_std_t(ss[WORD_OFFSET(src_idx)]) & BITMASK_GLOBAL(src_idx)));
        dlocker.unlock();
    }

    INLINE_DEVICE void print_row(DeviceLocker& dlocker, const Gate& m, const byte_t* aux, const size_t& num_qubits) {
        dlocker.lock();
        m.print(true);
        LOGGPU("--> Auxiliary: ");
        for (size_t i = 0; i < num_qubits; i++)
            LOGGPU("%d", aux[i]);
        LOGGPU(" ");
        for (size_t i = 0; i < num_qubits; i++)
            LOGGPU("%d", aux[i + num_qubits]);
        LOGGPU("\n");
        dlocker.unlock();
    }

    INLINE_DEVICE void print_shared_aux(DeviceLocker& dlocker, const Gate& m, const uint32* aux, const size_t& num_qubits, const size_t& copied_row, const size_t& multiplied_row) {
        dlocker.lock();
        if (!threadIdx.x) {
            uint32 offset = threadIdx.y * 3 * blockDim.x;
            LOGGPU("offset(%d), qubit(%d), row(%lld) x row(%lld):\n", offset, m.wires[0], copied_row, multiplied_row);
            for (uint32 i = 0; i < blockDim.x; i++)
                LOGGPU(" offset(%d): aux_xs[b:%d, t:%d] = %d\n", offset, blockIdx.x, i, aux[offset + i]);
            for (uint32 i = blockDim.x; i < 2 * blockDim.x; i++)
                LOGGPU(" offset(%d): aux_zs[b:%d, t:%d] = %d\n", offset, blockIdx.x, i, aux[offset + i]);
            for (uint32 i = 2 * blockDim.x; i < 3 * blockDim.x; i++)
                LOGGPU(" offset(%d): aux_ss[b:%d, t:%d] = %d\n", offset, blockIdx.x, i, aux[offset + i]);
        }
        dlocker.unlock();
    }

    INLINE_DEVICE void print_measurement(DeviceLocker& dlocker, const Gate& m) {
        dlocker.lock();
        LOGGPU("Outcome of qubit %d is %d\n", m.wires[0], m.measurement);
        dlocker.unlock();
    }

    // Print the tableau in binary format (generators are columns).
    __global__ void print_tableau_k(const Table* xs, const Table* zs, const Signs* ss, const depth_t level, const bool measuring);
    __global__ void print_tableau_k(const Table* ps, const Signs* ss, const depth_t level, const bool measuring);

    // Print the tableau's Pauli strings.
    __global__ void print_paulis_k(const Table* xs, const Table* zs, const Signs* ss, const size_t num_words_per_column, const size_t num_qubits, const bool extended);
    __global__ void print_paulis_k(const Table* ps, const Signs* ss, const size_t num_words_per_column, const size_t num_qubits, const bool extended);

    // Print gates.
    __global__ void print_gates_k(const gate_ref_t* refs, const bucket_t* gates, const gate_ref_t num_gates);

}

#endif