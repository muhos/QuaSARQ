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

    // Returns true for 2-qubit gates (including DEPOLARIZE2).
    INLINE_ALL 
    bool isGate2(const int& type) {
        return type >= int(NR_GATETYPES_1) && type < int(NR_GATETYPES);
    }

    // Returns true for measurement gates.
    INLINE_ALL 
    bool isMeasurement(const int& type) {
        return type == int(R) || type == int(M) || type == int(MR);
    }

    INLINE_ALL
    bool isReset(const int& type) {
        return type == int(R);
    }

    // Returns true for depolarizing noise gates.
    INLINE_ALL 
    bool isDepolarize(const int& type) {
        return type == int(DEPOLARIZE1) || type == int(DEPOLARIZE2);
    }

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
        Gate() : type(I), size(1) { 
            assert(sizeof(qubit_t) >= sizeof(float));
        }

        INLINE_ALL 
        explicit Gate(const input_size_t& size) : 
            type(I), size(size) { 
                assert(sizeof(qubit_t) >= sizeof(float));
            }

		INLINE_ALL
        size_t capacity() const {
            assert(size);
            const size_t extra = isDepolarize(int(type)) ? 1 : 0;
            return (size_t(size) + extra) * sizeof(qubit_t) + sizeof(*this);
        }

        // Store depolarizing probability in the last wire.
        // Only valid when isDepolarize(type) and sizeof(float) <= sizeof(qubit_t).
        INLINE_ALL
        void set_prob(const float& p) {
            memcpy(&wires[size], &p, sizeof(float));
        }

        INLINE_ALL
        float get_prob() const {
            float p;
            memcpy(&p, &wires[size], sizeof(float));
            return p;
        }

        INLINE_ALL
        void dagger() {
            if (type < NR_GATETYPES) {
                if (type == S_DAG) type = S;
                else if (type == S) type = S_DAG;
                else if (type == ISWAP) type = ISWAP_DAG;
                else if (type == ISWAP_DAG) type = ISWAP;
            }
            else {
                LOGGPU("Unknown gate type %d cannot be adjoined.\n", type);
            }
        }

        inline
        void print_host(const bool& nonl = false) const {
            if (type < NR_GATETYPES) {
                PRINT(" %-12s", H_G2S[type]);
            }
            else {
                PRINT("  Unknown");
            }
            if (isDepolarize(int(type))) {
                PRINT("(p=%.3f)", get_prob());
            }
            PRINT("(");
            for (input_size_t i = 0; i < size; i++) { 
                PRINT("%8d", wires[i]);
                if (i < size - 1)
                    PRINT(" , ");
            }
            PRINT(")%s", nonl ? "" : "\n");
        }

        INLINE_DEVICE
        void print(const bool& nonl = false) const {
            if (type < NR_GATETYPES) {
                LOGGPU(" %-12s", G2S[type]);
            }
            else {
                LOGGPU("  Unknown");
            }
            if (isDepolarize(int(type))) {
                LOGGPU("(p=%.3f)", get_prob());
            }
            LOGGPU("(");
            for (input_size_t i = 0; i < size; i++) { 
                LOGGPU("%8d", wires[i]);
                if (i < size - 1)
                    LOGGPU(" , ");
            }
            LOGGPU(")%s", nonl ? "" : "\n");
        }
 
    };

    constexpr size_t GATESIZE = sizeof(Gate);

    // Special structure for measurement operator.
    struct M_OP {
        qubit_t qubit;
        byte_t type;

        INLINE_ALL
        M_OP() : qubit(0), type(0) { }

        INLINE_ALL
        M_OP(const qubit_t& qubit, const byte_t& type) : 
            qubit(qubit), type(type) { }

        inline
        void print_host(const bool& nonl = false) const {
            if (type < NR_GATETYPES) {
                PRINT(" %5s(", H_G2S[type]);
            }
            else {
                PRINT("  Unknown(");
            }
            PRINT("%6d", qubit);
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
            LOGGPU("%6d", qubit);
            LOGGPU(")%s", nonl ? "" : "\n");
        }
    };

}