#pragma once

#include "definitions.cuh"
#include "datatypes.hpp"
#include "gatetypes.hpp"
#include "logging.hpp"

namespace QuaSARQ {

    typedef byte_t input_size_t;
    typedef uint32 gate_ref_t;
    typedef uint32 bucket_t;

    constexpr size_t BUCKETSIZE = sizeof(bucket_t);
    constexpr gate_ref_t NO_REF = UINT32_MAX;
    constexpr int UNMEASURED = INT_MAX;

    // Must follow the same order of 'Gatetypes'.
    __constant__ 
    constexpr arg_t G2S[NR_GATETYPES] = { 
        FOREACH_GATE(GATE2STR)
    };

    constexpr arg_t H_G2S[NR_GATETYPES] = { 
        FOREACH_GATE(GATE2STR)
    };

    /**
     * Gate structure.
     * Stores its type and dynamic number of inputs.
     * The 'wires' in 'Gate' cannot be used directly.
     * Because there is not enough memory will be 
     * allocated for 'Gate'. 
    */
    struct Gate {

        byte_t type;
        input_size_t size;
        qubit_t wires[0];

        INLINE_ALL 
        Gate() : type(I), size(1) { }

        INLINE_ALL 
        explicit Gate(const input_size_t& size) : 
            type(I), size(size) { }

		INLINE_ALL 
        size_t capacity() const { assert(size); return size_t(size) * sizeof(qubit_t) + sizeof(*this); }

        inline
        void print_host(const bool& nonl = false) const {
            if (type < NR_GATETYPES) {
                PRINT(" %5s(", H_G2S[type]);
            }
            else {
                PRINT("  Unknown(");
            }
            for (input_size_t i = 0; i < size; i++) { 
                PRINT("%6d", wires[i]);
                if (i < size - 1)
                    PRINT(" , ");
            }
            PRINT(")%s", nonl ? "" : "\n");
        }

        INLINE_DEVICE
        void print(const bool& nonl = false) const {
            if (type < NR_GATETYPES) {
                LOGGPU(" %5s(", G2S[type]);
            }
            else {
                LOGGPU("  Unknown(");
            }
            for (input_size_t i = 0; i < size; i++) { 
                LOGGPU("%6d", wires[i]);
                if (i < size - 1)
                    LOGGPU(" , ");
            }
            LOGGPU(")%s", nonl ? "" : "\n");
        }
 
    };

    constexpr size_t GATESIZE = sizeof(Gate);

}