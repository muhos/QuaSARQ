#pragma once

#include "definitions.hpp"
#include "statistics.hpp"
#include "gatetypes.hpp"
#include "datatypes.hpp"
#include "circuit.hpp"
#include "vector.hpp"
#include "record.cuh"

using std::to_string;

namespace QuaSARQ {

    #define WRITE_STATS(GATE) \
        stream += comment + "\t\t" + string(#GATE) + ": " + to_string(stats.circuit.gate_stats.types[GATE]) + " \t"; \
        stream += "(%" + to_string((uint64)percent((double)stats.circuit.gate_stats.types[GATE], stats.circuit.num_gates)) + ")\n"; \

    static const char *G2S_STIM[] = {
        FOREACH_GATE(GATE2STR)
    };

    struct ParsedGate {
        qubit_t c, t;
        float probs[15];
        byte_t type;
        byte_t expanded_from;

        ParsedGate(const qubit_t& c, const qubit_t& t, const byte_t& type) :
            c(c), t(t), type(type), expanded_from(0) { probs[0] = 0.0f; }
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

    struct ParsedBlock {
        CircuitQueue            gates;
        Gate_stats              gstats;
        RecordRefs              det;
        Vec<uint32, uint32>     det_mc;    // measures count within body just before each detector
        ObservableData          obs;
        Vec<uint32, uint32>     obs_mc;    // measures count within body just before each observable
        uint32                  measures;  // total M/MR gates in one REPEAT iteration

        ParsedBlock() : measures(0) {
            gates.reserve(256);
            gstats.alloc();
            det.init();
            det_mc.reserve(8);
            obs.init();
            obs_mc.reserve(4);
        }

        ~ParsedBlock() {
            gates.clear(true);
            gstats.destroy();
            det.destroy();
            det_mc.clear(true);
            obs.destroy();
            obs_mc.clear(true);
        }
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
        DetectorData detectors;
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
            measures_count = 0;
            observables.destroy();
            observables.init();
            detectors.destroy();
            detectors.init();
        }

        void destroy() {
            circuit_queue.clear(true);
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

        void parse_rec_refs(char*& str, RecordRefs& dest, const uint32& mc, const bool& deferred, const char* label);

        void read_gate_into(char*& str, CircuitQueue& target, Gate_stats& gstats, ParsedBlock* pb = nullptr);

        void read_gate(char*& str) {
            read_gate_into(str, circuit_queue, gate_stats);
            detectors.pinned.num_instructions = detectors.starts.size();
            observables.pinned.num_observables = observables.ids.size();
            observables.records.pinned.num_instructions = observables.records.starts.size();
        }

    };

}