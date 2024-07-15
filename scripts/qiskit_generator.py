from timeit import default_timer as timer
import os, sys
import numpy as np
from qiskit import qasm2
from qiskit.circuit import ClassicalRegister, QuantumCircuit, CircuitInstruction
from qiskit.circuit import Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
import argparse
import matplotlib

matplotlib.use('TkAgg',force=True)

def random_circuit(num_qubits, depth, max_operands=2, measure=False, conditional=False, reset=False, seed=524287):
    if num_qubits == 0:
        return QuantumCircuit()
    if max_operands < 1 or max_operands > 4:
        raise CircuitError("max_operands must be between 1 and 4")
    max_operands = max_operands if num_qubits > max_operands else num_qubits

    gates_1q = [
        # (Gate class, number of qubits, number of parameters)
        #(standard_gates.IGate, 1, 0),
        #(standard_gates.SXGate, 1, 0),
        (standard_gates.XGate, 1, 0),
        #(standard_gates.RZGate, 1, 1),
        #(standard_gates.RGate, 1, 2),
        (standard_gates.HGate, 1, 0),
        #(standard_gates.PhaseGate, 1, 1),
        #(standard_gates.RXGate, 1, 1),
        #(standard_gates.RYGate, 1, 1),
        (standard_gates.SGate, 1, 0),
        (standard_gates.SdgGate, 1, 0),
        #(standard_gates.SXdgGate, 1, 0),
        #(standard_gates.TGate, 1, 0),
        #(standard_gates.TdgGate, 1, 0),
        #(standard_gates.UGate, 1, 3),
        #(standard_gates.U1Gate, 1, 1),
        #(standard_gates.U2Gate, 1, 2),
        #(standard_gates.U3Gate, 1, 3),
        (standard_gates.YGate, 1, 0),
        (standard_gates.ZGate, 1, 0),
    ]
    if reset:
        gates_1q.append((Reset, 1, 0))
    gates_2q = [
        (standard_gates.CXGate, 2, 0),
        #(standard_gates.DCXGate, 2, 0),
        #(standard_gates.CHGate, 2, 0),
        #(standard_gates.CPhaseGate, 2, 1),
        #(standard_gates.CRXGate, 2, 1),
        #(standard_gates.CRYGate, 2, 1),
        #(standard_gates.CRZGate, 2, 1),
        #(standard_gates.CSXGate, 2, 0),
        #(standard_gates.CUGate, 2, 4),
        #(standard_gates.CU1Gate, 2, 1),
        #(standard_gates.CU3Gate, 2, 3),
        (standard_gates.CYGate, 2, 0),
        (standard_gates.CZGate, 2, 0),
        #(standard_gates.RXXGate, 2, 1),
        #(standard_gates.RYYGate, 2, 1),
        #(standard_gates.RZZGate, 2, 1),
        #(standard_gates.RZXGate, 2, 1),
        #(standard_gates.XXMinusYYGate, 2, 2),
        #(standard_gates.XXPlusYYGate, 2, 2),
        #(standard_gates.ECRGate, 2, 0),
        #(standard_gates.CSGate, 2, 0),
        #(standard_gates.CSdgGate, 2, 0),
        (standard_gates.SwapGate, 2, 0),
        (standard_gates.iSwapGate, 2, 0),
    ]
    gates_3q = [
        (standard_gates.CCXGate, 3, 0),
        (standard_gates.CSwapGate, 3, 0),
        (standard_gates.CCZGate, 3, 0),
        (standard_gates.RCCXGate, 3, 0),
    ]
    gates_4q = [
        (standard_gates.C3SXGate, 4, 0),
        (standard_gates.RC3XGate, 4, 0),
    ]

    gates = gates_1q.copy()
    if max_operands >= 2:
        gates.extend(gates_2q)
    if max_operands >= 3:
        gates.extend(gates_3q)
    if max_operands >= 4:
        gates.extend(gates_4q)
    gates = np.array(
        gates, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
    )
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
        gate_specs = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)
        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack:
            gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

        # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomises those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        rng.shuffle(qubits)

        # We'veAttributeError: 'QuantumCircuit'  now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        if conditional and layer_number != 0:
            is_conditional = rng.random(size=len(gate_specs)) < 0.1
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0
            for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
                is_conditional,
            ):
                operation = gate(*parameters[p_start:p_end])
                if is_cond:
                    qc.measure(qc.qubits, cr)
                    # The condition values are required to be bigints, not Numpy's fixed-width type.
                    operation = operation.c_if(cr, int(condition_values[c_ptr]))
                    c_ptr += 1
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))
        else:
            for gate, q_start, q_end, p_start, p_end in zip(
                gate_specs["class"], q_indices[:-1], q_indices[1:], p_indices[:-1], p_indices[1:]
            ):
                operation = gate(*parameters[p_start:p_end])
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))

    if measure:
        qc.measure(qc.qubits, cr)

    return qc

print("\t\t Qiskit Random Circuit Generator\n")

parser = argparse.ArgumentParser("qiskit_generator")
parser.add_argument('-q', '--qubits', help="Number of qubits.", default=1000)
parser.add_argument('-d', '--depth', help="Circuit depth.", default=100)
parser.add_argument('-i', '--inc', help="Qubits increase.", default=1000)
parser.add_argument('-m', '--max', help="Maximum generatable qubits.", default=1000)
parser.add_argument('-p', '--path', help="Directory to save generated benchmarks.", default='clifford-benchmarks/qasm')
parser.add_argument('-v', '--visual', help="Visualize qiskit ciruit in a figure.", default=False, action='store_true')
args = parser.parse_args()

# Benchmarks directory.
bench_dir = args.path
os.makedirs(bench_dir, exist_ok=True)

qmin = int(args.qubits)
depth = int(args.depth)
inc = int(args.inc)
qmax = int(args.max)

if (qmin > qmax):
    print("Maximum qubits cannot be less than minimum qubits to generate.")

def generate():
    for q in range(qmin, qmax + inc, inc):
        path = bench_dir + '/q' + str(q) + '_d' + str(depth)
        print("Generating Qiskit random circuit for " + str(q) + " qubits and depth " + str(depth) + "...", end=''), sys.stdout.flush()
        start = timer()
        c = random_circuit(q, depth, measure=False)
        if (args.visual):
            c.draw(output='mpl',filename=path + '.jpg', style="clifford")
        stop = timer()
        time = stop - start
        print(" done in %.3f seconds" %(time)), sys.stdout.flush()

        print(" Converting QASM circuit to string...", end=''), sys.stdout.flush()
        start = timer()
        c_str = qasm2.dumps(c)
        circuit_size = len(c_str)
        stop = timer()
        time = stop - start
        print(" done in %.3f seconds" %(time)), sys.stdout.flush()

        with open(path + '.qasm', 'w') as f:
            print(" Writing qiskit circuit to file using QASM format...", end=''), sys.stdout.flush()
            start = timer()
            f.write(c_str)
            stop = timer()
            time = stop - start
            print(" done in %.3f seconds" %(time)), sys.stdout.flush()

generate()