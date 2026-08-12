#pragma once

#include <string>
#include <vector>
#include <memory>

#include "simulator.hpp"
#include "sampler.hpp"

namespace QuaSARQ {

    // Reaches the tableau and the recorder, which Simulator keeps protected.
    class Simulating : public Simulator {

    public:

        using Simulator::Simulator;

        const Tableau& final_tableau() const { return tableau; }
        MeasurementRecorder& record_of() { return recorder; }
        size_t partitions() const { return num_partitions; }
        size_t qubit_count() const { return num_qubits; }
        bool records_measurements() const { return measuring; }
    };

    // True when two circuits realise the same Clifford operation. Runs both on the device and
    // compares the tableaux they produce, so it answers for the circuits as a whole rather than
    // gate by gate.
    bool check_equivalence(std::string& circuit, std::string& other);

    // One deterministic run of a circuit: the measurement outcomes it produces, and the state it
    // ends in as Pauli strings. This is the tableau path, not the frame sampler, so there is a
    // single shot and no randomness beyond what the circuit's own error channels introduce.
    //
    // The state is the *inverse* tableau, which is what the simulator evolves and what
    // measurement readout needs. Rows are generators; an extended tableau gives the
    // destabilizers first and then the stabilizers.
    class Simulation {

        std::string circuit_text;
        size_t      num_measurements;
        size_t      num_qubits;
        bool        has_run;

        std::vector<bool>        record;
        std::vector<std::string> paulis;

    public:

        Simulation(const std::string& circuit);

        Simulation            (const Simulation&) = delete;
        Simulation& operator= (const Simulation&) = delete;

        size_t measurements() const { return num_measurements; }
        size_t qubits() const { return num_qubits; }

        // Runs the circuit if it has not run yet. Both outputs are cached, since the run is
        // deterministic and repeating it would only cost time.
        void run();

        const std::vector<bool>& measurement_record();
        const std::vector<std::string>& pauli_strings();
    };

}
