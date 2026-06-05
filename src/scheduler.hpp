#pragma once

#include "parser.hpp"
#include "circuit.hpp"
#include "statistics.hpp"

namespace QuaSARQ {

    struct ScheduleResult {
        size_t depth;
        size_t measuring_depth;
        size_t measuring_count;
        bool   measuring;
    };

    ScheduleResult schedule_circuit(CircuitIO& circuit_io, Circuit& circuit, WindowInfo& winfo, const size_t& num_qubits);

    ScheduleResult schedule_gates(
        CircuitIO&      circuit_io,
        Circuit&        circuit,
        WindowInfo&     winfo,
        const size_t&   num_qubits,
        Statistics&     stats,
        const bool&     sort = true
    );

}
