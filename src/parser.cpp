
#include "parser.hpp"

namespace QuaSARQ {

    void CircuitIO::read_gate_into(char*& str, CircuitQueue& target, Gate_stats& gstats) {
        eatWS(str);
        if (str >= eof || *str == '\0') return;
        // Parse gate / directive name: alpha, underscore, or digit.
        char gatestr[MAX_GATENAME_LEN];
        int gatename_len = 0;
        while (gatename_len < MAX_GATENAME_LEN &&
                (isalpha(str[gatename_len]) || str[gatename_len] == '_' || isDigit(str[gatename_len]))) {
            gatestr[gatename_len] = str[gatename_len];
            gatename_len++;
        }
        if (gatename_len == MAX_GATENAME_LEN)
            LOGERROR("gate name is too long.");
        if (gatename_len == 0) { eatLine(str); return; }
        gatestr[gatename_len] = '\0';
        str += gatename_len;

        // Drop: TICK, QUBIT_COORDS, SHIFT_COORDS (not needed for QuaSARQ).
        if (strcmp(gatestr, "TICK")         == 0 ||
            strcmp(gatestr, "QUBIT_COORDS") == 0 ||
            strcmp(gatestr, "SHIFT_COORDS") == 0) {
            eatLine(str);
            return;
        }

        // REPEAT block.
        if (strcmp(gatestr, "REPEAT") == 0) {
            eatWS(str);
            uint32 count = toInteger(str);
            eatWS(str);
            if (str >= eof || *str != '{')
                LOGERROR("expected '{' after REPEAT %u.", count);
            str++; // consume '{'

            CircuitQueue block;
            block.reserve(256);
            Gate_stats bstats;
            bstats.alloc();

            while (str < eof) {
                eatWS(str);
                if (*str == '}') { str++; break; }
                if (*str == '\0') break;
                if (*str == '#') { eatLine(str); continue; }
                read_gate_into(str, block, bstats);
            }

            for (uint32 i = 0; i < count; i++) {
                for (size_t j = 0; j < block.size(); j++) {
                    const ParsedGate& g = block[j];
                    target.push(g);
                    gstats.types[g.type]++;
                    if ((g.type == M || g.type == MR) && &target == &circuit_queue)
                        measures_count++;
                }
            }
            block.clear(true);
            bstats.destroy();
            return;
        }

        // DETECTOR(x, y, t, ...) rec[-1] rec[-2] ...
        if (strcmp(gatestr, "DETECTOR") == 0) {
            // Skip optional coordinates.
            if (str < eof && *str == '(') {
                while (str < eof && *str != ')' && *str != DELIM) str++;
                if (str < eof && *str == ')') str++;
            }
            detectors.ref_starts.push(detectors.record_refs.size());
            uint32 ref_count = 0;
            while (str < eof && *str != DELIM) {
                if (*str == UNIX_DELIM) { str++; continue; }
                char* peek = str;
                eatWS(peek);
                if (peek >= eof || *peek == DELIM || *peek == '\0') break;
                if (peek + 4 < eof &&
                    peek[0]=='r' && peek[1]=='e' && peek[2]=='c' &&
                    peek[3]=='[' && peek[4]=='-') {
                    str = peek + 5;
                    uint32 n = toInteger(str);
                    if (str < eof && *str == ']') str++;
                    if (n == 0 || n > measures_count)
                        LOGERROR("DETECTOR: rec[-%u] out of range (measures so far: %u).", n, measures_count);
                    detectors.record_refs.push(measures_count - n);
                    ref_count++;
                } else { eatLine(str); break; }
            }
            detectors.ref_counts.push(ref_count);
            return;
        }

        // OBSERVABLE_INCLUDE(k) rec[-1] rec[-2] ...
        if (strcmp(gatestr, "OBSERVABLE_INCLUDE") == 0) {
            uint32 obs_id = 0;
            if (str < eof && *str == '(') {
                str++;
                obs_id = toInteger(str);
                eatWS(str);
                if (str < eof && *str == ')') str++;
            }
            observables.ids.push(obs_id);
            observables.ref_starts.push((uint32)observables.record_refs.size());
            uint32 ref_count = 0;
            while (str < eof && *str != DELIM) {
                if (*str == UNIX_DELIM) { str++; continue; }
                char* peek = str;
                eatWS(peek);
                if (peek >= eof || *peek == DELIM || *peek == '\0') break;
                // Expect rec[-n]
                if (peek + 4 < eof &&
                    peek[0] == 'r' && peek[1] == 'e' && peek[2] == 'c' &&
                    peek[3] == '[' && peek[4] == '-') {
                    str = peek + 5; // skip "rec[-"
                    uint32 n = toInteger(str);
                    if (str < eof && *str == ']') str++;
                    if (n == 0 || n > measures_count)
                        LOGERROR("OBSERVABLE_INCLUDE: rec[-%u] out of range (measures so far: %zu)", n, measures_count);
                    observables.record_refs.push(measures_count - n);
                    ref_count++;
                } else {
                    eatLine(str);
                    break;
                }
            }
            observables.ref_counts.push(ref_count);
            return;
        }

        float gate_prob = 0.0f;
        if (*str == '(') {
            str++;
            gate_prob = toFloat(str);
            eatWS(str);
            if (str < eof && *str == ')') str++;
        }
        int gateindex = translate_gate(gatestr, gatename_len);
        if (gateindex < 0) {
            eatLine(str);
            return;
        }

        const Gatetypes type = Gatetypes(gateindex);

        if (type == M || type == MR) measuring = true;

        const bool is_2qubit = isGate2(int(type));
        while (str < eof && *str != DELIM) {
            if (*str == UNIX_DELIM) { str++; continue; }
            char* peek = str;
            eatWS(peek);
            if (!isDigit(*peek)) { eatLine(str); return; }
            const qubit_t c = toInteger(str);
            max_qubits = MAX(max_qubits, (size_t)(c) + 1);
            qubit_t t = c;
            if (is_2qubit) {
                t = toInteger(str);
                max_qubits = MAX(max_qubits, (size_t)(t) + 1);
            }
            target.push(ParsedGate(c, t, type, gate_prob));
            gstats.types[type]++;
            if ((type == M || type == MR) && &target == &circuit_queue)
                measures_count++;
        }
    }

    void CircuitIO::write_circuit(string& stream, const int& format, const size_t& num_qubits_in_circuit, const Circuit& circuit) {
        size_t max_depth = circuit.depth();
        for (depth_t d = 0; d < max_depth; d++) {
            const Window& gate_refs = circuit[d];
            const size_t max_refs = gate_refs.size();
            for (size_t i = 0; i < max_refs; i++) {
                gate_ref_t gate_ref = NO_REF;
                const Gate* gate = nullptr;
                gate_ref = gate_refs[i];
                gate = circuit.gateptr(gate_ref);
                if (gate->type == byte_t(I)) continue;
                string gatestr = string(G2S_STIM[gate->type]);
                if (format == 2) {
                    if (gatestr == "CX") 
                        gatestr = "C";
                    else if (gatestr == "S")
                        gatestr = "P";
                }
                stream += gatestr + " " + to_string(gate->wires[0]);
                if (gate->size > 1)
                    stream += " " + to_string(gate->wires[1]);
                stream += "\n";        
            }
        }
    }

    void CircuitIO::write(const Circuit& circuit, const size_t& num_qubits_in_circuit, const int& format, const Statistics& stats) {
        size_t max_qubits = num_qubits_in_circuit;
        size_t max_depth = circuit.depth();
        string path = "q" + to_string(max_qubits) + "_d" + to_string(max_depth);
        if (format == 1) path += ".stim";
        else if (format == 2) path += ".chp";
        FILE* benchfile = nullptr;
        if (benchfile == nullptr) {
            LOGN2(1, "Opening \"%s%s%s\" circuit file for writing..", CREPORTVAL, path.c_str(), CNORMAL);
            benchfile = fopen(path.c_str(), "w");
            if (benchfile == nullptr) { LOG2(1, "does not exist."); }
            LOGDONE(1, 4);
        }
        LOGN2(1, "Writing circuit with %s%zd qubits%s and %s%zd depth%s..", CREPORTVAL, max_qubits, CNORMAL, CREPORTVAL, max_depth, CNORMAL);
        string comment;
        if (format == 1) comment = "#";
        else if (format == 2) comment = "";
        string stream = comment + "This circuit is generated by QuaSARQ for benchmarking purposes.\n";
        stream += comment + "q" + to_string(max_qubits) + "\n";
        stream += comment + "d" + to_string(max_depth) + "\n";
        stream += comment + "Gates distribution:\n";
        FOREACH_GATE(WRITE_STATS);
        if (format == 2) stream += "#\n";
        write_circuit(stream, format, num_qubits_in_circuit, circuit);
        fwrite(stream.c_str(), 1, stream.size(), benchfile);
        LOGDONE(1, 4);
        if (benchfile != nullptr) {
            fclose(benchfile);
            benchfile = nullptr;
        }
    }

    int CircuitIO::translate_gate(char* in, const int& gatelen) {
        for (int i = 0; i < NR_GATETYPES; i++) {
            const char* ref = G2S_STIM[i];
            int c = 0;
            while (ref[c] && ref[c] == in[c]) c++;
            if (gatelen == c)
                return i;
        }
        return -1;
    }

    char* CircuitIO::read(const char* circuit_path) {
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

}