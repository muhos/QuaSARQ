from timeit import default_timer as timer
from typing import Union, List
import os, sys
import stim

print("\t\t Stim Random Circuit Generator\n")

# Number of qubits.
q = 100 # defualt

q = int(input("Please enter number of qubits: "))

# Benchmarks directory.
bench_dir = 'stim-benchmarks'
os.makedirs(bench_dir, exist_ok=True)

# Ref: https://quantumcomputing.stackexchange.com/questions/27326/how-to-go-from-matrix-to-tableau-to-circuit-in-qiskit-or-stim
def tableau_to_circuit_optimized(tableau: stim.Tableau) -> stim.Circuit:
    remaining = tableau.inverse()
    recorded_circuit = stim.Circuit()
    def do(gate: str, targets: List[int]):
        recorded_circuit.append(gate, targets)
        remaining.append(stim.Tableau.from_named_gate(gate), targets)

    n = len(remaining)
    for col in range(n):
        # Find a cell with an anti-commuting pair of Paulis.
        for pivot_row in range(col, n):
            px = remaining.x_output_pauli(col, pivot_row)
            pz = remaining.z_output_pauli(col, pivot_row)
            if px and pz and px != pz:
                break
        else:
            raise NotImplementedError("Failed to find a pivot cell")

        # Move the pivot to the diagonal.
        if pivot_row != col:
            # Equivalent to 3 CX.
            do("SWAP", [pivot_row, col])

        # Transform the pivot to XZ.
        px = remaining.x_output_pauli(col, col)
        if px == 3:
            do("H", [col])
        elif px == 2:
            do("H_XY", [col])
        pz = remaining.z_output_pauli(col, col)
        if pz == 2:
            do("H_YZ", [col])

        # Use the pivot to remove all other terms in the X observable.
        for row in range(col + 1, n):
            px = remaining.x_output_pauli(col, row)
            if px:
                do("C" + "_XYZ"[px], [col, row])

        # Use the pivot to remove all other terms in the Z observable.
        for row in range(col + 1, n):
            pz = remaining.z_output_pauli(col, row)
            if pz:
                do("XC" + "_XYZ"[pz], [col, row])

        # Fix pauli signs.
        if remaining.z_output(col).sign == -1:
            do("X", [col])
        if remaining.x_output(col).sign == -1:
            do("Z", [col])

    return recorded_circuit

print("Generating Stim random tableau for " + str(q) + " qubits...", end=''), sys.stdout.flush()
start = timer()
tableau = stim.Tableau.random(q)
stop = timer()
time = stop - start
print(" done in %.3f seconds" %(time)), sys.stdout.flush()
#print(repr(tableau))

print('')

print("Default: Converting tableau to circuit {H, S, CX}...", end=''), sys.stdout.flush()
start = timer()
c = tableau.to_circuit(method="elimination")
print('')
print(c.diagram())
stop = timer()
time = stop - start
print(" done in %.3f seconds" %(time)), sys.stdout.flush()

print("Default: Verifying circuit {H, S, CX} is equivalent to tableau...", end=''), sys.stdout.flush()
start = timer()
assert stim.Tableau.from_circuit(c) == tableau
stop = timer()
time = stop - start
print(" VERIFIED in %.3f seconds" %(time)), sys.stdout.flush()

path = bench_dir + '/' + str(q) + '.stim'
with open(path, 'w') as f:
    print("Default: Writing circuit to file using Stim format...", end=''), sys.stdout.flush()
    f.write(str(q) + '\n')
    start = timer()
    c.to_file(f)
    #qasm = c.to_qasm(open_qasm_version=3)
    #f.write(qasm)
    stop = timer()
    time = stop - start
    print(" done in %.3f seconds" %(time)), sys.stdout.flush()
    
print('')

print("Compact: Converting tableau to circuit {H, H_XY, H_YZ, S, Swap, CX, CY, CZ, X, Y, Z, XCX, XCZ, XCY}...", end=''), sys.stdout.flush()
start = timer()
c_compact = tableau_to_circuit_optimized(tableau)
stop = timer()
time = stop - start
print(" done in %.3f seconds" %(time)), sys.stdout.flush()
#c = tableau_to_circuit_optimized(tableau)
#print(c)

print("Compact: Verifying circuit {H, H_XY, H_YZ, S, Swap, CX, CY, CZ, X, Y, Z, XCX, XCZ, XCY} is equivalent to tableau...", end=''), sys.stdout.flush()
start = timer()
assert stim.Tableau.from_circuit(c_compact) == tableau
stop = timer()
time = stop - start
print(" VERIFIED in %.3f seconds" %(time)), sys.stdout.flush()

path_compact = bench_dir + '/' + str(q) + '_compact.stim'
with open(path_compact, 'w') as f:
    print("Compact: Writing compact circuit to file using Stim format...", end=''), sys.stdout.flush()
    f.write(str(q) + '\n')
    start = timer()
    c_compact.to_file(f)
    #qasm = c.to_qasm(open_qasm_version=3)
    #f.write(qasm)
    stop = timer()
    time = stop - start
    print(" done in %.3f seconds" %(time)), sys.stdout.flush()
