#pragma once

#include "definitions.hpp"
#include "statistics.hpp"
#include "gatetypes.hpp"
#include "datatypes.hpp"
#include "circuit.hpp"
#include "vector.hpp"
#include "memory.cuh"

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

    struct RecordRefs {

        Vec<uint32, uint32>  refs; // measurement-history indices per instruction.
        Vec<uint32, uint32>  starts;  // the start index in record_refs for each instruction.
        Vec<uint32, uint32>  counts;  // the number of record_refs for each instruction.

        struct RawArrays {
            uint32* refs;
            uint32* starts;
            uint32* counts;

            size_t num_instructions;
            size_t num_counts;
            size_t num_refs;

            RawArrays() : 
                refs(nullptr), starts(nullptr), counts(nullptr),
                num_instructions(0), num_counts(0), num_refs(0) { }

            bool is_allocated() const {
                return !(refs == nullptr || starts == nullptr || counts == nullptr);
            }
        };

        RawArrays pinned, device;

        bool moved_to_pinned;

        RecordRefs() : pinned(), device(), moved_to_pinned(false) {}

        void init() {
            moved_to_pinned = false;
            refs.reserve(64);
            starts.reserve(16);
            counts.reserve(16);
        }

        void destroy() {
            refs.clear(true);
            starts.clear(true);
            counts.clear(true);
        }

        size_t bytes() const {
            return refs.size() * sizeof(uint32) + 
                   starts.size() * sizeof(uint32) + 
                   counts.size() * sizeof(uint32);
        }

        void alloc_pinned(DeviceAllocator& allocator);
        void alloc_device(DeviceAllocator& allocator);

        void move_to_pinned();
        void copy_to_device(const cudaStream_t& stream);

        bool empty() const { return !pinned.num_instructions; }
    };

    typedef RecordRefs DetectorData;

    struct ObservableData {

        RecordRefs records;
        Vec<uint32, uint32>  ids; // the observable id (the k in OBSERVABLE_INCLUDE(k))

        struct RawIds {
            uint32* ids;
            size_t num_observables;

            RawIds() : ids(nullptr), num_observables(0) {}
        };

        RawIds pinned, device;

        bool moved_to_pinned;

        ObservableData() : records(), pinned(), device(), moved_to_pinned(false) {}

        void init() {
            records.init();
            ids.reserve(4);
        }

        void destroy() {
            records.destroy();
            ids.clear(true);
        }

        size_t bytes() const {
            return records.bytes() +
                   ids.size() * sizeof(uint32);
        }

        void alloc_pinned(DeviceAllocator& allocator);
        void alloc_device(DeviceAllocator& allocator);
        void move_to_pinned();
        void copy_to_device(const cudaStream_t& stream);

        bool empty() const { return !pinned.num_observables; }
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

        void read_gate_into(char*& str, CircuitQueue& target, Gate_stats& gstats);

        void read_gate(char*& str) {
            read_gate_into(str, circuit_queue, gate_stats);
            detectors.pinned.num_instructions = detectors.starts.size();
            observables.pinned.num_observables = observables.ids.size();
            observables.records.pinned.num_instructions = observables.records.starts.size();
        }

    };

}