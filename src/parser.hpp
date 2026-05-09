#pragma once

#include "definitions.hpp"
#include "statistics.hpp"
#include "gatetypes.hpp"
#include "datatypes.hpp"
#include "circuit.hpp"
#include "vector.hpp"

using std::to_string;

namespace QuaSARQ {

    enum CircuitMode { RANDOM_CIRCUIT, PARSED_CIRCUIT };

    #define WRITE_STATS(GATE) \
        stream += comment + "\t\t" + string(#GATE) + ": " + to_string(stats.circuit.gate_stats.types[GATE]) + " \t"; \
        stream += "(%" + to_string((uint64)percent((double)stats.circuit.gate_stats.types[GATE], stats.circuit.num_gates)) + ")\n"; \

    static const char *G2S_STIM[] = {
        FOREACH_GATE(GATE2STR)
    };

    struct ParsedGate {
        qubit_t c, t;
        float   p;      // depolarizing probability (0 for non-noise gates)
        byte_t  type;

        ParsedGate(const qubit_t& c, const qubit_t& t, const byte_t& type, const float& p = 0.0f) :
            c(c), t(t), type(type), p(p) {}
    };

    class CircuitQueue : public Vec<ParsedGate, size_t> {

        size_t head;

    public:

        CircuitQueue() : head(0) { }

        bool empty() const {
            return head == size();
        }

        const ParsedGate& front() {
            assert(head < size());
            return operator[](head);
        }

        void pop_front() { ++head; }

        void clear(const bool& free = false) {
            Vec<ParsedGate, size_t>::clear(free);
            head = 0;
        }
    };

    struct ObservableData {

        Vec<uint32, uint32>  record_refs; // measurement-history indices.
        Vec<uint32, uint32>  ref_starts;
        Vec<uint32, uint32>  ref_counts;
        Vec<uint32, uint32>  ids;         // the observable id (the k in OBSERVABLE_INCLUDE(k))

        ObservableData() {}

        void init() {
            record_refs.reserve(32);
            ref_starts.reserve(4);
            ref_counts.reserve(4);
            ids.reserve(4);
        }

        void destroy() {
            record_refs.clear(true);
            ref_starts.clear(true);
            ref_counts.clear(true);
            ids.clear(true);
        }

        uint32 num_observables() const { return ids.size(); }

        bool empty() const { return ids.empty(); }
    };

    struct CircuitIO {

        #define DELIM '\n'
        #define UNIX_DELIM '\r'
        #define MAX_GATENAME_LEN 64

        void* buffer;
        char* eof;
        uint32 measures_count;
        size_t size, max_qubits;
        Gate_stats gate_stats;
        CircuitQueue circuit_queue;
        ObservableData observables;
        bool measuring;

#if defined(__linux__) || defined(__CYGWIN__)
        int file;
#else
        ifstream file;
#endif

        CircuitIO() :
            buffer(nullptr)
            , eof(nullptr)
            , measures_count(0)
            , size(0)
            , max_qubits(0)
            , measuring(false)
        {
            init();
        }

        ~CircuitIO() { destroy(); }

        void init() {
            circuit_queue.reserve(MB);
            gate_stats.alloc();
            observables.init();
        }

        void destroy(const bool& free = false) {
            circuit_queue.clear(true);
            if (free) {
                observables.destroy();
                measures_count = 0;
            }
            gate_stats.destroy();
            if (buffer != nullptr) {
#if defined(__linux__) || defined(__CYGWIN__)
                if (munmap(buffer, size) != 0)
                    LOGERROR("cannot clean file mapping.");
#else
                std::free(buffer);
#endif
            }
            buffer = nullptr;
            eof = nullptr;
            size = 0;
            max_qubits = 0;
            measuring = false;
        }

        void write_circuit(string& stream, const int& format, const size_t& num_qubits_in_circuit, const Circuit& circuit);

        void write(const Circuit& circuit, const size_t& num_qubits_in_circuit, const int& format, const Statistics& stats);

        char* read(const char* circuit_path);

        int translate_gate(char* in, const int& gatelen);

        void read_gate_into(char*& str, CircuitQueue& target, Gate_stats& gstats);

        void read_gate(char*& str) {
            read_gate_into(str, circuit_queue, gate_stats);
        }

    };

}