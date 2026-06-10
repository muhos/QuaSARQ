
#include "parser.hpp"
#include "expansion.hpp"

namespace QuaSARQ {

    bool CircuitIO::try_expand_m_variants(char*& str, const char* gatestr, const int& gatelen,
                                          CircuitQueue& target, Gate_stats& gstats,
                                          ParsedBlock* pb) {
        const bool is_MX  = (gatelen == 2 && gatestr[0]=='M' && gatestr[1]=='X');
        const bool is_MY  = (gatelen == 2 && gatestr[0]=='M' && gatestr[1]=='Y');
        const bool is_MRX = (gatelen == 3 && gatestr[0]=='M' && gatestr[1]=='R' && gatestr[2]=='X');
        const bool is_MRY = (gatelen == 3 && gatestr[0]=='M' && gatestr[1]=='R' && gatestr[2]=='Y');
        if (!is_MX && !is_MY && !is_MRX && !is_MRY) return false;

        const Gatetypes mtype = (is_MX || is_MY) ? M : MR;
        const Gatetypes orig  = is_MX ? MX : (is_MY ? MY : (is_MRX ? MRX : MRY));
        const bool is_y = (is_MY || is_MRY);
        measuring = true;

        if (str < eof && *str == '(') {
            str++;
            while (str < eof && *str != ')' && *str != DELIM) str++;
            if (str < eof && *str == ')') str++;
        }

        // Collect all qubits on this line.
        Vec<qubit_t, size_t> qubits;
        while (str < eof && *str != DELIM) {
            if (*str == UNIX_DELIM) { str++; continue; }
            char* peek = str;
            eatWS(peek);
            if (!isDigit(*peek)) { eatLine(str); return true; }
            const qubit_t c = toInteger(str);
            max_qubits = MAX(max_qubits, (size_t)(c) + 1);
            qubits.push(c);
        }

        auto push_op = [&](const qubit_t& c, const Gatetypes& t) {
            ParsedGate pg(c, c, byte_t(t));
            pg.expanded_from = byte_t(orig);
            target.push(pg);
        };
        auto batch = [&](const Gatetypes& t) {
            for (size_t k = 0; k < qubits.size(); k++) 
                push_op(qubits[k], t);
        };

        // Phase 1: pre-measurement basis change.
        if (is_y) batch(S_DAG);
        batch(H);
        // Phase 2: measurement.
        for (size_t k = 0; k < qubits.size(); k++) {
            push_op(qubits[k], mtype);
            if (pb == nullptr) gstats.types[orig]++;
            if (&target == &circuit_queue) measures_count++;
            else if (pb != nullptr)        pb->measures++;
        }
        // Phase 3: post-measurement basis change.
        batch(H);
        if (is_y) batch(S);

        qubits.clear(true);
        return true;
    }

    bool CircuitIO::try_expand_clifford(char*& str, const char* gatestr, const int& gatelen, CircuitQueue& target, Gate_stats& gstats) {
        const CliffordDecomp* found = nullptr;
        for (const auto& cd : TABLE) {
            int c = 0;
            while (cd.name[c] && cd.name[c] == gatestr[c]) c++;
            if (c == gatelen && !cd.name[c]) { found = &cd; break; }
        }
        if (!found) return false;

        // Collect all qubit.
        Vec<qubit_t, size_t> qs_c, qs_t;
        while (str < eof && *str != DELIM) {
            if (*str == UNIX_DELIM) { str++; continue; }
            char* peek = str;
            eatWS(peek);
            if (!isDigit(*peek)) { eatLine(str); break; }
            const qubit_t c = toInteger(str);
            max_qubits = MAX(max_qubits, (size_t)(c) + 1);
            qs_c.push(c);
            if (found->is_2q) {
                const qubit_t t = toInteger(str);
                max_qubits = MAX(max_qubits, (size_t)(t) + 1);
                qs_t.push(t);
            } else {
                qs_t.push(c);  // 1-qubit: t == c
            }
        }

        auto push_op = [&](const qubit_t& c, const qubit_t& t, const PhaseOp& op) {
            qubit_t q0, q1;
            switch (op.sel) {
                case QC:  q0 = c; q1 = c; break;
                case QT:  q0 = t; q1 = t; break;
                case QCT: q0 = c; q1 = t; break;
                default:  q0 = t; q1 = c; break;  // QTC
            }
            ParsedGate pg(q0, q1, byte_t(op.type));
            target.push(pg);
            gstats.types[op.type]++;
        };
        
        // Emit the expansion for all qubits in batches.
        for (int phase = 0; phase < found->nops; phase++)
            for (size_t k = 0; k < qs_c.size(); k++)
                push_op(qs_c[k], qs_t[k], found->ops[phase]);

        qs_c.clear(true);
        qs_t.clear(true);
        return true;
    }

}
