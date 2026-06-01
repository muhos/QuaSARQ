
#include "parser.hpp"

namespace QuaSARQ {

    void CircuitIO::parse_rec_refs(char*& str, RecordRefs& dest, const uint32& mc, const bool& deferred, const char* label) {
        dest.starts.push(dest.refs.size());
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
                if (n == 0 || n > mc)
                    LOGERROR("%s: rec[-%u] out of range (measures so far: %u).", label, n, mc);
                dest.refs.push(deferred ? n : mc - n);
                ref_count++;
            } 
            else { eatLine(str); break; }
        }
        dest.counts.push(ref_count);
    }

    void CircuitIO::parse_detector(char*& str, ParsedBlock* pb) {
        // Skip optional coordinate arguments.
        if (str < eof && *str == '(') {
            while (str < eof && *str != ')' && *str != DELIM) str++;
            if (str < eof && *str == ')') str++;
        }
        const uint32 mc = measures_count + (pb ? pb->measures : 0);
        RecordRefs& dest = pb ? pb->det : detectors;
        if (pb) pb->det_mc.push(pb->measures);
        parse_rec_refs(str, dest, mc, pb != nullptr, "DETECTOR");
    }

    void CircuitIO::parse_observable(char*& str, ParsedBlock* pb) {
        uint32 obs_id = 0;
        if (str < eof && *str == '(') {
            str++;
            obs_id = toInteger(str);
            eatWS(str);
            if (str < eof && *str == ')') str++;
        }
        const uint32 mc = measures_count + (pb ? pb->measures : 0);
        ObservableData& dest_obs = pb ? pb->obs : observables;
        RecordRefs& dest_rec     = pb ? pb->obs.records : observables.records;
        dest_obs.ids.push(obs_id);
        if (pb) pb->obs_mc.push(pb->measures);
        parse_rec_refs(str, dest_rec, mc, pb != nullptr, "OBSERVABLE_INCLUDE");
    }

    void CircuitIO::parse_repeat(char*& str, CircuitQueue& target, Gate_stats& gstats) {
        eatWS(str);
        uint32 count = toInteger(str);
        eatWS(str);
        if (str >= eof || *str != '{')
            LOGERROR("expected '{' after REPEAT %u.", count);
        str++; // consume '{'

        ParsedBlock block;
        while (str < eof) {
            eatWS(str);
            if (*str == '}') { str++; break; }
            if (*str == '\0') break;
            if (*str == '#') { eatLine(str); continue; }
            read_gate_into(str, block.gates, block.gstats, &block);
        }

        // Unroll count times.
        const uint32 measures_before = measures_count;
        for (uint32 i = 0; i < count; i++) {
            // Gates.
            for (size_t j = 0; j < block.gates.size(); j++) {
                const ParsedGate& g = block.gates[j];
                target.push(g);
                if (g.expanded_from == 0) {
                    gstats.types[g.type]++;
                } 
                else if (g.type == M || g.type == MR) {
                    gstats.types[g.expanded_from]++;
                }
                if ((g.type == M || g.type == MR) && &target == &circuit_queue)
                    measures_count++;
            }
            // Detectors.
            for (uint32 j = 0; j < block.det.starts.size(); j++) {
                const uint32 mc = measures_before + i * block.measures + block.det_mc[j];
                detectors.starts.push(detectors.refs.size());
                const uint32 s = block.det.starts[j];
                const uint32 c = block.det.counts[j];
                for (uint32 k = s; k < s + c; k++)
                    detectors.refs.push(mc - block.det.refs[k]);
                detectors.counts.push(c);
            }
            // Observables.
            for (uint32 j = 0; j < block.obs.ids.size(); j++) {
                const uint32 mc = measures_before + i * block.measures + block.obs_mc[j];
                observables.ids.push(block.obs.ids[j]);
                observables.records.starts.push(observables.records.refs.size());
                const uint32 s = block.obs.records.starts[j];
                const uint32 c = block.obs.records.counts[j];
                for (uint32 k = s; k < s + c; k++)
                    observables.records.refs.push(mc - block.obs.records.refs[k]);
                observables.records.counts.push(c);
            }
        }
    }

    void CircuitIO::read_gate_into(char*& str, CircuitQueue& target, Gate_stats& gstats, ParsedBlock* pb) {
        eatWS(str);
        if (str >= eof || *str == '\0') return;

        // Parse gate / directive name: alpha, underscore, or digit.
        char gatestr[MAX_GATENAME_LEN];
        int gatelen = 0;
        while (gatelen < MAX_GATENAME_LEN &&
               (isalpha(str[gatelen]) || str[gatelen] == '_' || isDigit(str[gatelen]))) {
            gatestr[gatelen] = str[gatelen];
            gatelen++;
        }
        if (gatelen == MAX_GATENAME_LEN) LOGERROR("gate name is too long.");
        if (gatelen == 0) { eatLine(str); return; }
        gatestr[gatelen] = '\0';
        str += gatelen;

        // Drop directives.
        if (strcmp(gatestr, "TICK")         == 0 ||
            strcmp(gatestr, "QUBIT_COORDS") == 0 ||
            strcmp(gatestr, "SHIFT_COORDS") == 0) {
            eatLine(str); return;
        }

        if (strcmp(gatestr, "REPEAT")             == 0) { parse_repeat(str, target, gstats); return; }
        if (strcmp(gatestr, "DETECTOR")           == 0) { parse_detector(str, pb);           return; }
        if (strcmp(gatestr, "OBSERVABLE_INCLUDE") == 0) { parse_observable(str, pb);         return; }
        if (try_expand_m_variants(str, gatestr, gatelen, target, gstats, pb))                return;
        if (try_expand_clifford  (str, gatestr, gatelen, target, gstats))                    return;

        // Parse optional probability argument(s).
        float gate_probs[15] = {};
        uint8 gate_nprobs = 0;
        if (str < eof && *str == '(') {
            str++;
            while (gate_nprobs < 15 && str < eof && *str != ')' && *str != DELIM) {
                eatWS(str);
                gate_probs[gate_nprobs++] = toFloat(str);
                eatWS(str);
                if (str < eof && *str == ',') str++;
            }
            eatWS(str);
            if (str < eof && *str == ')') str++;
        }

        const int gateindex = translate_gate(gatestr, gatelen);
        if (gateindex < 0) { eatLine(str); return; }

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
            ParsedGate pg(c, t, type);
            memcpy(pg.probs, gate_probs, gate_nprobs * sizeof(float));
            target.push(pg);
            gstats.types[type]++;
            if (type == M || type == MR) {
                if (&target == &circuit_queue) measures_count++;
                else if (pb != nullptr)        pb->measures++;
            }
        }
    }

    int CircuitIO::translate_gate(char* in, const int& gatelen) {
        // Stim gate aliases.
        struct Alias { const char* name; Gatetypes type; };
        static constexpr Alias aliases[] = {
            { "CNOT",       CX    },
            { "ZCX",        CX    },
            { "ZCY",        CY    },
            { "ZCZ",        CZ    },
            { "H_XZ",       H     },
            { "SQRT_Z",     S     },
            { "SQRT_Z_DAG", S_DAG },
            { "MZ",         M     },
            { "MRZ",        MR    },
            { "RZ",         R     },
        };
        for (const auto& a : aliases) {
            int c = 0;
            while (a.name[c] && a.name[c] == in[c]) c++;
            if (c == gatelen && !a.name[c]) return int(a.type);
        }
        // Canonical names.
        for (int i = 0; i < NR_GATETYPES; i++) {
            const char* ref = G2S_STIM[i];
            int c = 0;
            while (ref[c] && ref[c] == in[c]) c++;
            if (gatelen == c) return i;
        }
        return -1;
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
                    if (gatestr == "CX") gatestr = "C";
                    else if (gatestr == "S") gatestr = "P";
                }
                stream += gatestr + " " + to_string(gate->wires[0]);
                if (gate->size > 1)
                    stream += " " + to_string(gate->wires[1]);
                stream += "\n";
            }
        }
    }

    void CircuitIO::write(const Circuit& circuit, const size_t& num_qubits_in_circuit,
                          const int& format, const Statistics& stats) {
        size_t max_qubits = num_qubits_in_circuit;
        size_t max_depth  = circuit.depth();
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

    char* CircuitIO::read(const char* circuit_path) {
        if (circuit_path == nullptr)
            LOGERROR("circuit path is empty.");
        struct stat st;
        if (!canAccess(circuit_path, st))
            LOGERROR("circuit file is inaccessible.");
        size = st.st_size;
        LOG2(1, "Parsing circuit file \"%s%s%s\" (size: %s%zd%s MB)..",
             CREPORTVAL, circuit_path, CNORMAL, CREPORTVAL, ratio(size, MB), CNORMAL);
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
