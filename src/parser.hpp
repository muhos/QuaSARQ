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

        uint32* pin_refs;
        uint32* pin_starts;
        uint32* pin_counts;

        size_t num_instructions;
        size_t num_counts;
        size_t num_refs;

        RecordRefs() :
                pin_refs(nullptr)
                , pin_starts(nullptr)
                , pin_counts(nullptr)
                , num_instructions(0)
                , num_counts(0)
                , num_refs(0) {}

        void init() {
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

        void alloc_pinned(DeviceAllocator& allocator) {
            if (pin_refs != nullptr || pin_starts != nullptr || pin_counts != nullptr) {
                LOGERROR("pinned memory already allocated for detector data");
            }
            pin_refs = allocator.allocate_pinned<uint32>(refs.size());
            pin_starts = allocator.allocate_pinned<uint32>(starts.size());
            pin_counts = allocator.allocate_pinned<uint32>(counts.size());
        }

        void move_to_pinned() {
            if (pin_refs == nullptr || pin_starts == nullptr || pin_counts == nullptr) {
                LOGERROR("pinned memory not allocated for detector data");
            }
            std::memcpy(pin_refs, refs.data(), refs.size() * sizeof(uint32));
            std::memcpy(pin_starts, starts.data(), starts.size() * sizeof(uint32));
            std::memcpy(pin_counts, counts.data(), counts.size() * sizeof(uint32));
            num_instructions = starts.size();
            num_counts = counts.size();
            num_refs = refs.size();
            destroy();
        }

        bool empty() const { return !num_instructions; }
    };

    typedef RecordRefs DetectorData;

    struct ObservableData {

        RecordRefs records;
        Vec<uint32, uint32>  ids; // the observable id (the k in OBSERVABLE_INCLUDE(k))

        uint32* pin_ids;
        size_t num_observables;


        ObservableData() :
              pin_ids(nullptr)
            , num_observables(0) {}

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

        void alloc_pinned(DeviceAllocator& allocator) {
            records.alloc_pinned(allocator);
            if (pin_ids != nullptr) {
                LOGERROR("pinned memory already allocated for observable data");
            }
            pin_ids = allocator.allocate_pinned<uint32>(ids.size());
        }

        void move_to_pinned() {
            records.move_to_pinned();
            if (pin_ids == nullptr) {
                LOGERROR("pinned memory not allocated for observable data");
            }
            std::memcpy(pin_ids, ids.data(), ids.size() * sizeof(uint32));
            num_observables = ids.size();
            destroy();
        }

        bool empty() const { return !num_observables; }
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
            detectors.num_instructions = detectors.starts.size();
            observables.num_observables = observables.ids.size();
            observables.records.num_instructions = observables.records.starts.size();
        }

    };

}