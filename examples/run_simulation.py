"""Runs circuits once on the tableau path and reports what they measured and where they ended up.

A single deterministic pass over the circuit. The state comes back as
Pauli strings, one per generator of the inverse tableau, with the destabilizers 
listed first and then the stabilizers. That is the same convention stim's 
current_inverse_tableau() uses, which is what the check at the end compares against.
"""

import time

import stim

import quasarq

CIRCUITS = {
    "bit flip":     "R 0 1\nX 0\nM 0 1",
    "measure reset": "R 0\nX 0\nMR 0\nM 0",
    "bell pair":    "R 0 1\nH 0\nCX 0 1\nM 0 1",
    "x basis":      "RX 0\nMX 0",
}

STATES = {
    "hadamard":  ("H 0", 1),
    "phase":     ("S 0", 1),
    "bell":      ("H 0\nCX 0 1", 2),
    "entangled": ("X 0\nH 1\nCX 1 2\nS 2", 3),
}


def stim_inverse_rows(text, qubits):
    # stim writes identity as '_' where quasarq writes 'I'.
    sim = stim.TableauSimulator()
    sim.set_num_qubits(qubits)
    sim.do_circuit(stim.Circuit(text))
    inverse = sim.current_inverse_tableau()
    rows = [str(inverse.x_output(q)) for q in range(qubits)]
    rows += [str(inverse.z_output(q)) for q in range(qubits)]
    return [row.replace("_", "I") for row in rows]


def run_measurements():
    print("Measurement outcomes\n")

    for name, text in CIRCUITS.items():
        sim = quasarq.simulate(text)
        outcomes = "".join("1" if bit else "0" for bit in sim.measurements())
        print(f"  {name:<14} {sim.num_qubits} qubits, {sim.num_measurements} measurements -> {outcomes}")

    print()


def run_states():
    print("Final state, as Pauli strings\n")

    agreed = True

    for name, (text, qubits) in STATES.items():
        rows = quasarq.simulate(text).paulis()
        want = stim_inverse_rows(text, qubits)
        agreed &= rows == want
        print(f"  {name:<10} {text!r}")
        print(f"    destabilizers {rows[:qubits]}")
        print(f"    stabilizers   {rows[qubits:]}")

    print()
    assert agreed, "the state must match stim's inverse tableau"


def run_scale(distance=25):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=distance,
        after_clifford_depolarization=0.001)

    start = time.perf_counter()
    sim = quasarq.simulate(circuit)
    generators = len(sim.paulis())
    elapsed = time.perf_counter() - start

    print(f"Surface code d={distance}: {sim.num_qubits} qubits, {sim.num_measurements} "
          f"measurements, {generators} generators in {elapsed:.2f}s\n")


if __name__ == "__main__":
    run_measurements()
    run_states()
    run_scale()
    print("All tests passed.")
