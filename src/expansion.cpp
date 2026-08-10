
#include "parser.hpp"
#include "expansion.hpp"

namespace QuaSARQ {

    // Returns false if the line held a non-numeric token (line is eaten).
    inline bool collect_line_qubits(char*& str, const char* eof, size_t& max_qubits, Vec<qubit_t, size_t>& qubits) {
        while (str < eof && *str != DELIM) {
            if (*str == UNIX_DELIM) { str++; continue; }
            qubit_t c;
            if (!next_qubit_or_eat_line(str, max_qubits, c)) return false;
            qubits.push(c);
        }
        return true;
    }

    inline qubit_t max_qubit_in_line(const Vec<qubit_t, size_t>& qubits) {
        qubit_t line_max = 0;
        for (size_t k = 0; k < qubits.size(); k++)
            line_max = MAX(line_max, qubits[k]);
        return line_max;
    }

    // Marks the longest run of distinct qubits starting at 'begin' and returns its end.
    inline size_t mark_next_run(const Vec<qubit_t, size_t>& qubits, Vec<byte_t, size_t>& in_run, const size_t& begin) {
        size_t end = begin;
        while (end < qubits.size() && !in_run[qubits[end]]) {
            in_run[qubits[end]] = 1;
            end++;
        }
        return end;
    }

    inline void clear_run(const Vec<qubit_t, size_t>& qubits, Vec<byte_t, size_t>& in_run, const size_t& begin, const size_t& end) {
        for (size_t k = begin; k < end; k++)
            in_run[qubits[k]] = 0;
    }

    inline void push_expanded_op(CircuitQueue& target, const qubit_t& c, const Gatetypes& t, const Gatetypes& orig) {
        ParsedGate pg(c, c, byte_t(t));
        pg.expanded_from = byte_t(orig);
        target.push(pg);
    }

    inline void push_expanded_batch(CircuitQueue& target, const Vec<qubit_t, size_t>& qubits,
                                          const size_t& begin, const size_t& end,
                                          const Gatetypes& t, const Gatetypes& orig) {
        for (size_t k = begin; k < end; k++)
            push_expanded_op(target, qubits[k], t, orig);
    }

    inline void push_clifford_op(CircuitQueue& target, Gate_stats& gstats,
                                       const qubit_t& c, const qubit_t& t, const PhaseOp& op) {
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
    }

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

        skip_gate_args(str, eof);

        // Collect all qubits on this line.
        Vec<qubit_t, size_t> qubits;
        if (!collect_line_qubits(str, eof, max_qubits, qubits)) {
            qubits.clear(true);
            return true;
        }

        // Repeated qubits like MRX 0 0.. should be treated as
        // (H MR H)(H MR H), as if they were separate qubits.
        Vec<byte_t, size_t> in_run(size_t(max_qubit_in_line(qubits)) + 1, 0);

        size_t begin = 0;
        while (begin < qubits.size()) {
            const size_t end = mark_next_run(qubits, in_run, begin);
            // Phase 1: pre-measurement basis change.
            if (is_y) push_expanded_batch(target, qubits, begin, end, S_DAG, orig);
            push_expanded_batch(target, qubits, begin, end, H, orig);
            // Phase 2: measurement.
            for (size_t k = begin; k < end; k++) {
                push_expanded_op(target, qubits[k], mtype, orig);
                if (pb == nullptr) gstats.types[orig]++;
                if (&target == &circuit_queue) measures_count++;
                else if (pb != nullptr)        pb->measures++;
            }
            // Phase 3: post-measurement basis change.
            push_expanded_batch(target, qubits, begin, end, H, orig);
            if (is_y) push_expanded_batch(target, qubits, begin, end, S, orig);

            clear_run(qubits, in_run, begin, end);
            begin = end;
        }

        in_run.clear(true);
        qubits.clear(true);
        return true;
    }

    bool CircuitIO::try_expand_r_variants(char*& str, const char* gatestr, const int& gatelen,
                                          CircuitQueue& target, Gate_stats& gstats) {
        const bool is_RX = (gatelen == 2 && gatestr[0] == 'R' && gatestr[1] == 'X');
        const bool is_RY = (gatelen == 2 && gatestr[0] == 'R' && gatestr[1] == 'Y');
        if (!is_RX && !is_RY) return false;

        const Gatetypes orig = is_RX ? RX : RY;

        skip_gate_args(str, eof);

        Vec<qubit_t, size_t> qubits;
        if (!collect_line_qubits(str, eof, max_qubits, qubits)) {
            qubits.clear(true);
            return true;
        }

        // Repeated qubits like RX 0 0 must each be expanded independently.
        Vec<byte_t, size_t> in_run(size_t(max_qubit_in_line(qubits)) + 1, 0);

        size_t begin = 0;
        while (begin < qubits.size()) {
            const size_t end = mark_next_run(qubits, in_run, begin);
            for (size_t k = begin; k < end; k++) {
                push_expanded_op(target, qubits[k], R, orig);
                gstats.types[orig]++;
            }
            push_expanded_batch(target, qubits, begin, end, H, orig);
            if (is_RY) push_expanded_batch(target, qubits, begin, end, S, orig);

            clear_run(qubits, in_run, begin, end);
            begin = end;
        }

        in_run.clear(true);
        qubits.clear(true);
        return true;
    }

    bool CircuitIO::try_expand_clifford(char*& str, const char* gatestr, const int& gatelen, CircuitQueue& target, Gate_stats& gstats) {
        const CliffordDecomp* found = nullptr;
        for (const auto& cd : TABLE) {
            if (match_gate_name(cd.name, gatestr, gatelen)) { found = &cd; break; }
        }
        if (!found) return false;

        // Collect all qubit.
        Vec<qubit_t, size_t> qs_c, qs_t;
        while (str < eof && *str != DELIM) {
            if (*str == UNIX_DELIM) { str++; continue; }
            qubit_t c;
            if (!next_qubit_or_eat_line(str, max_qubits, c)) break;
            qs_c.push(c);
            if (found->is_2q) {
                qs_t.push(next_qubit(str, max_qubits));
            } else {
                qs_t.push(c);  // 1-qubit: t == c
            }
        }

        // Emit the expansion for all qubits in batches.
        for (int phase = 0; phase < found->nops; phase++)
            for (size_t k = 0; k < qs_c.size(); k++)
                push_clifford_op(target, gstats, qs_c[k], qs_t[k], found->ops[phase]);

        qs_c.clear(true);
        qs_t.clear(true);
        return true;
    }

}
