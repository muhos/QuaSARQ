
#include "simulate.hpp"
#include "print.cuh"
#include "equivalence.hpp"

namespace QuaSARQ {

    bool check_equivalence(std::string& circuit, std::string& other) {
        binding_clear_error();
        // The check compares the Clifford operation each circuit realises. A measurement is not
        // part of that operation, so a circuit containing one would be compared as though the
        // measurement were absent -- a wrong answer rather than a limitation.
        const size_t measured = scan_circuit(circuit).measurements;
        const size_t other_measured = scan_circuit(other).measurements;
        if (measured || other_measured) {
            LOGERROR("equivalence compares the Clifford operation a circuit realises, and cannot "
                     "account for measurements (%zd in the first circuit, %zd in the second).",
                     measured, other_measured);
        }
        OptionsGuard guard;
        apply_library_defaults();
        options.num_shots = 0;
        options.equivalence_en = true;
        Equivalence equivalence(circuit.data(), circuit.size(), other.data(), other.size());
        equivalence.check();
        return equivalence.is_equivalent();
    }

    Simulation::Simulation(const std::string& circuit) :
        circuit_text(circuit)
        , num_measurements(0)
        , num_qubits(0)
        , has_run(false)
    {
        const CircuitScan scan = scan_circuit(circuit_text);
        num_measurements = scan.measurements;
        num_qubits = scan.qubits;
    }

    // One row per generator: its sign, then one Pauli letter per qubit.
    static void collect_paulis(
        const Table&                xs,
        const Table&                zs,
        const Signs&                ss,
        const size_t&               num_qubits,
        const size_t&               num_words_major,
        const size_t&               offset,
              std::vector<std::string>& out)
    {
        for (size_t g = 0; g < num_qubits; g++) {
            std::string row;
            row.reserve(num_qubits + 1);
            row += (ss[offset + WORD_OFFSET(g)] & sign_t(BITMASK_GLOBAL(g))) ? '-' : '+';
            const size_t base = g * num_words_major + offset;
            for (size_t q = 0; q < num_qubits; q++) {
                const size_t idx = base + WORD_OFFSET(q);
                const word_t bit = BITMASK_GLOBAL(q);
                const bool x = bool(xs[idx] & bit);
                const bool z = bool(zs[idx] & bit);
                row += x ? (z ? 'Y' : 'X') : (z ? 'Z' : 'I');
            }
            out.push_back(row);
        }
    }

    void Simulation::run() {
        if (has_run) return;
        OptionsGuard guard;
        apply_library_defaults();
        options.num_shots = 0;
        // A split tableau holds one partition at a time and the next overwrites it, so the state
        // could not be returned whole. Sizing the pool for the whole tableau keeps it in one.
        if (auto_device_memory)
            options.max_gpu_memory = binding_auto_device_memory(num_qubits, num_measurements, 0);
        binding_clear_error();

        Simulating sim(circuit_text.data(), circuit_text.size(), false);
        sim.simulate();

        if (sim.partitions() > 1) {
            LOGERROR("the tableau needed %zd partitions, so the final state cannot be reported "
                     "whole; raise the device memory cap.", sim.partitions());
        }

        if (num_measurements && sim.records_measurements()) {
            MeasurementRecorder& recorder = sim.record_of();
            recorder.copy();
            const Vec<bool>& host = recorder.host_record();
            record.assign(host.data(), host.data() + MIN(size_t(host.size()), num_measurements));
        }
        record.resize(num_measurements, false);

        const Tableau& tab = sim.final_tableau();
        Table h_xs, h_zs; Signs h_ss;
        tab.copy_to_host(&h_xs, &h_zs, &h_ss);
        const size_t rows = sim.qubit_count();
        paulis.clear();
        paulis.reserve(tab.is_extended() ? 2 * rows : rows);
        collect_paulis(h_xs, h_zs, h_ss, rows, tab.num_words_major(), 0, paulis);
        if (tab.is_extended())
            collect_paulis(h_xs, h_zs, h_ss, rows, tab.num_words_major(), tab.num_words_minor(), paulis);
        h_xs.destroy();
        h_zs.destroy();
        h_ss.destroy();

        has_run = true;
    }

    const std::vector<bool>& Simulation::measurement_record() {
        run();
        return record;
    }

    const std::vector<std::string>& Simulation::pauli_strings() {
        run();
        return paulis;
    }

}
