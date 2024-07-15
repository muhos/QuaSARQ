#ifndef __GATE_H
#define __GATE_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "gatetypes.hpp"
#include "logging.hpp"

namespace QuaSARQ {

    /**
     * Gate structure.
     * Stores its type and dynamic number of inputs.
     * The 'wires' in 'Gate' cannot be used directly.
     * Because there is not enough memory will be 
     * allocated for 'Gate'. 
    */

    typedef byte_t input_size_t;
    typedef uint32 gate_ref_t;
    typedef uint32 bucket_t;

    constexpr size_t BUCKETSIZE = sizeof(bucket_t);
    constexpr gate_ref_t NO_REF     = -1;

    // Must follow the same order of 'Gatetypes'.
    __constant__ constexpr arg_t G2S[NR_GATETYPES] = 
    { 
        "I",
        "Z",
        "X",
        "Y",
        "H",
        "S",
        "Sdg",
        "CX",
        "CY",
        "CZ",
        "Swap",
        "iSwap"
    };

    struct Gate {
        byte_t type;
        input_size_t size;
        qubit_t wires[0];

        INLINE_ALL 
        Gate() : type(I), size(1) { }

        INLINE_ALL 
        explicit Gate(const input_size_t& size) : 
            type(I), size(size) { }

		INLINE_ALL size_t capacity() const { assert(size); return size_t(size) * sizeof(qubit_t) + sizeof(*this); }

        INLINE_ALL
        void print() const {
            if (type < NR_GATETYPES) {
                LOGGPU("  %s(", G2S[type]);
            }
            else {
                LOGGPU("  Unknown(");
            }
            for (input_size_t i = 0; i < size; i++) { 
                LOGGPU("%d", wires[i]);
                if (i < size - 1)
                    LOGGPU(",");
            }
            LOGGPU(")\n");
        }
 
    };

    constexpr size_t GATESIZE = sizeof(Gate);

    typedef Vec<qubit_t, input_size_t> Gate_inputs;

}

#endif