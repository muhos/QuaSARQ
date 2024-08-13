#ifndef __CU_PRINT_H
#define __CU_PRINT_H

#include "table.cuh"
#include "signs.cuh"
#include "vector.cuh"
#include "circuit.cuh"

namespace QuaSARQ {

    // Print bit-packed bits in matrix format.
    template <class T>
    INLINE_ALL void print_table(const T& t) {
        const word_t* data = t.data();
        const size_t size = t.size();
        const size_t major_end = t.num_words_per_column();
        const size_t minor_end = t.num_words_per_column() * WORD_BITS;
        size_t j = 0;
        for (size_t i = 0; i < size; i++) {
            if (i > 0 && i % major_end == 0)
                LOGGPU("\n");
            if (i > 0 && i % minor_end == 0)
                LOGGPU("\n");
            LOGGPU("  ");
            if (i > 0 && i % (major_end * 2) == 0) 
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

    INLINE_ALL void print_signs(const Signs& ss, const size_t& offset = 0) {
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

    INLINE_ALL void print_gates_per_window(const gate_ref_t* refs, const bucket_t* gates, const gate_ref_t& num_gates) {
        for (gate_ref_t i = 0; i < num_gates; i++) {
            const gate_ref_t r = refs[i];
            LOGGPU(" Gate(%d, r:%d):", i, r);
            const Gate& gate = (Gate&)gates[r];
            gate.print();
        }
    }

    INLINE_ALL void print_tables(const Table& xs, const Table& zs, const Signs& ss, const int64& level) {
        LOGGPU(" ---[ X-Table at (%-2lld)-step ]---------------------\n", level);
        print_table(xs);
        LOGGPU(" ------------------------------------------------\n");
        LOGGPU(" ---[ Z-Table at (%-2lld)-step ]---------------------\n", level);
        print_table(zs);
        LOGGPU(" ------------------------------------------------\n");
        LOGGPU(" ---[ Signs at (%-2lld)-step ]-----------------------\n", level);
        print_signs(ss);
        LOGGPU(" ------------------------------------------------\n");
    }

    INLINE_ALL void print_tables(const Table& ps, const Signs& ss, const int64& level) {
        LOGGPU(" ---[ XZ bits at (%-2lld)-step ]---------------------\n", level);
        print_table(ps);
        LOGGPU(" ------------------------------------------------\n");
        LOGGPU(" ---[ Signs at (%-2lld)-step   ]---------------------\n", level);
        print_signs(ss);
        LOGGPU(" ------------------------------------------------\n");
    }

    // Print the tableau in binary format (generators are columns).
    __global__ void print_tableau(const Table* xs, const Table* zs, const Signs* ss, const depth_t level);

    // Print gates.
    __global__ void print_gates(const gate_ref_t* refs, const bucket_t* gates, const gate_ref_t num_gates);

    // Print signs.
    __global__ void print_signs(const Signs* ss);

    #define PRINT_TABLEAU(TABLEAU, STEP) \
    do { \
        if (!options.sync) SYNCALL; \
        print_tableau << <1, 1 >> > (XZ_TABLE(TABLEAU), TABLEAU.signs(), STEP); \
        LASTERR("failed to launch print-tableau kernel"); \
        SYNCALL; \
        fflush(stdout); \
    } while(0)

}

#endif