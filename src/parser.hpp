
#ifndef __PARSER_H
#define __PARSER_H

#include "definitions.hpp"
#include "statistics.hpp"
#include "gatetypes.hpp"

using std::to_string;

namespace QuaSARQ {

    enum CircuitMode { RANDOM_CIRCUIT, PARSED_CIRCUIT };

    struct Parsed_gate {
        qubit_t c, t;
        byte_t type;

        Parsed_gate(const qubit_t& c, const qubit_t& t, const byte_t& type) :
            c(c), t(t), type(type) {}
    };

    class Circuit_queue : public Vec<Parsed_gate, size_t> {

        size_t head;

    public:

        Circuit_queue() : head(0) { }

        bool empty() const {
            return head == size();
        }

        const Parsed_gate& front() {
            assert(head < size());
            return operator[](head);
        }

        void pop_front() { ++head; }

        void clear(const bool& free = false) {
            Vec<Parsed_gate, size_t>::clear(free);
            head = 0;
        }
    };

    struct CircuitIO {

        #define DELIM '\n'
        #define UNIX_DELIM '\r'
        #define MAX_GATENAME_LEN 16

        const char* GATE_STIM[NR_GATETYPES] = {
            "I",
            "X",
            "Y",
            "Z",
            "H",
            "S",
            "S_DAG",
            "CX",
            "CY",
            "CZ",
            "SWAP",
            "ISWAP"
        };

        const Gatetypes GATESTR_TO_BYTE[NR_GATETYPES] =
        {
            I,
            X,
            Y,
            Z,
            H,
            S,
            Sdg,
            CX,
            CY,
            CZ,
            Swap,
            iSwap
        };

        void* buffer;
        char* eof;
        size_t size;
        Gate_stats gate_stats;
        Circuit_queue circuit_queue;

#if defined(__linux__) || defined(__CYGWIN__)
        int file;
#else
        ifstream file;
#endif

        CircuitIO() :
            buffer(nullptr)
            , eof(nullptr)
            , size(0)
        { 
            init();
        }

        ~CircuitIO() { destroy(); }

        void init() {
            circuit_queue.reserve(MB);
            gate_stats.alloc();
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
        }

        void write(const Circuit& circuit, const size_t& num_qubits_in_circuit) {
            size_t max_qubits = num_qubits_in_circuit;
            size_t max_depth = circuit.depth();
            string path = "q" + to_string(max_qubits) + "_d" + to_string(max_depth) + ".stim";
            FILE* benchfile = nullptr;
            if (benchfile == nullptr) {
                LOGN2(1, "Opening \"%s%s%s\" benchmark file for writing..", CREPORTVAL, path.c_str(), CNORMAL);
                benchfile = fopen(path.c_str(), "w");
                if (benchfile == nullptr) { LOG2(1, "does not exist."); }
                LOGDONE(1, 3);
            }
            LOGN2(1, "Writing circuit with %s%zd qubits%s and %s%zd depth%s..", CREPORTVAL, max_qubits, CNORMAL, CREPORTVAL, max_depth, CNORMAL);
            string stream = "#" + to_string(max_qubits) + "\n";
	        for (depth_t d = 0; d < max_depth; d++) {
                const Window& window = circuit[d];
                for (qubit_t i = 0; i < window.size(); i++) {
                    gate_ref_t r = window[i];
                    const Gate& gate = circuit.gate(r);
                    if (gate.type == I) continue;
                    stream += string(GATE_STIM[gate.type]) + " " + to_string(gate.wires[0]);
                    if (gate.size > 1)
                        stream += " " + to_string(gate.wires[1]);
                    stream += "\n";
                }
            }
            fwrite(stream.c_str(), 1, stream.size(), benchfile);
            LOGDONE(1, 3);
            if (benchfile != nullptr) {
                fclose(benchfile);
                benchfile = nullptr;
            }
        }

        inline Gatetypes translate_gate(char* in, const int& gatelen) {
            for (int i = 0; i < NR_GATETYPES; i++) {
                const char* ref = GATE_STIM[i];
                int c = 0;
                while (ref[c]) {
                    if (ref[c] != in[c])
                        break;
                    c++;
                }
                if (gatelen == c)
                    return GATESTR_TO_BYTE[i];
            }
            LOGERROR("unknown gate %s.", in);
        }

        char* read(const char* circuit_path) {
            if (circuit_path == nullptr)
                LOGERROR("circuit path is empty.");        
            struct stat st;
            if (!canAccess(circuit_path, st))
                LOGERROR("circuit file is inaccessible.");
            size = st.st_size;
            LOG2(1, "Parsing circuit file \"%s%s%s\" (size: %s%zd%s MB)..", CREPORTVAL, circuit_path, CNORMAL, CREPORTVAL, ratio(size, MB), CNORMAL);
            char* stream = NULL;
#if defined(__linux__) || defined(__CYGWIN__)
            file = open(circuit_path, O_RDONLY, 0);
            if (file == -1) LOGERROR("cannot open input file");
            buffer = mmap(NULL, size, PROT_READ, MAP_PRIVATE, file, 0);
            stream = static_cast<char*>(buffer);
            close(file);
#else
            file.open(circuit_path, ifstream::in);
            if (!file.is_open()) LOGERROR("cannot open input file.");
            stream = calloc<char>(size + 1);
            buffer = static_cast<void*>(stream);
            file.read(stream, size);
            stream[size] = '\0';
            file.close();
#endif
            eof = stream + size;
            return stream;
        }

        void read_gate(char*& str) {
            eatWS(str);
            char gatestr[MAX_GATENAME_LEN];
            int gatename_len = 0;
            while ((isalpha(str[gatename_len]) || str[gatename_len] == '_') && gatename_len < MAX_GATENAME_LEN) {
                gatestr[gatename_len] = str[gatename_len];
                gatename_len++;
            }
            if (gatename_len == MAX_GATENAME_LEN)
                LOGERROR("gate name is too long.");
            gatestr[gatename_len] = '\0';           
            Gatetypes type = translate_gate(gatestr, gatename_len);
            str += gatename_len;    
            while (*str != DELIM && str < eof) {   
                if (*str == UNIX_DELIM) { 
                    str++;
                    continue;
                }
				const qubit_t c = toInteger(str);
                qubit_t t = c;
				if (gatename_len > 1 && type != Sdg) {
                    t = toInteger(str);
                }
                circuit_queue.push(Parsed_gate(c, t, type));
                gate_stats.types[type]++;
            }
        }

    };

}

#endif
