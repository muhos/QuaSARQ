"""Asks whether two circuits realise the same Clifford operation.

This is a whole-circuit question, not a gate-by-gate comparison: the two circuits may differ in
depth, in gate count, even in which gates they use. Rewriting a circuit and then confirming that
nothing changed is the usual reason to need that check.

Measurements are not part of the operation a circuit realises, so a circuit containing one is
refused rather than compared as though the measurement were absent.
"""

import time

import stim

import quasarq

IDENTITIES = (
    ("H 0\nH 0",                        "I 0",                "H is its own inverse"),
    ("S 0\nS 0",                        "Z 0",                "S squared is Z"),
    ("S 0\nS 0\nS 0\nS 0",              "I 0",                "S has order four"),
    ("H 0\nS 0\nH 0\nS 0\nH 0\nS 0",    "I 0",                "(HS) has order three"),
    ("CX 0 1",                          "H 1\nCZ 0 1\nH 1",   "CX is CZ conjugated by H"),
    ("H 0\nH 1\nCX 0 1\nH 0\nH 1",      "CX 1 0",             "conjugating CX by H reverses it"),
)

DIFFERENCES = (
    ("X 0",      "Z 0",       "X is not Z"),
    ("S 0",      "Z 0",       "S is not Z"),
    ("CX 0 1",   "CX 1 0",    "CX is directed"),
    ("CX 0 1",   "CZ 0 1",    "CX is not CZ"),
)


def run_identities():
    print("Clifford identities\n")

    for a, b, why in IDENTITIES:
        verdict = quasarq.equivalent(a, b)
        print(f"  {'equivalent' if verdict else 'DIFFERENT ':<12} {why}")
        assert verdict, why

    print()

    for a, b, why in DIFFERENCES:
        verdict = quasarq.equivalent(a, b)
        print(f"  {'EQUIVALENT' if verdict else 'different ':<12} {why}")
        assert not verdict, why

    print()


def run_rewrite(qubits=12, layers=40):

    original, rewritten = [], []

    for layer in range(layers):
        for q in range(layer % 2, qubits - 1, 2):
            original.append(f"CX {q} {q + 1}")
            rewritten += [f"H {q + 1}", f"CZ {q} {q + 1}", f"H {q + 1}"]
        for q in range(qubits):
            gate = "H" if (layer + q) % 3 else "S"
            original.append(f"{gate} {q}")
            rewritten.append(f"{gate} {q}")

    original, rewritten = "\n".join(original), "\n".join(rewritten)
    broken = "\n".join(rewritten.split("\n")[:-1])

    print(f"Rewriting a {qubits}-qubit circuit of {len(original.splitlines())} gates "
          f"into {len(rewritten.splitlines())}\n")

    start = time.perf_counter()
    same = quasarq.equivalent(original, rewritten)
    print(f"  rewrite preserves the circuit : {same}   ({time.perf_counter() - start:.2f}s)")

    start = time.perf_counter()
    damaged = quasarq.equivalent(original, broken)
    print(f"  dropping one gate breaks it   : {not damaged}   ({time.perf_counter() - start:.2f}s)")

    assert same and not damaged, "the rewrite must hold and the damaged circuit must not"
    print()


def run_inputs():
    # A circuit can be given as text, a quasarq.Circuit, or a stim.Circuit.
    target = "H 1\nCZ 0 1\nH 1"
    forms = (
        ("text",            "CX 0 1"),
        ("quasarq.Circuit", quasarq.Circuit("CX 0 1")),
        ("stim.Circuit",    stim.Circuit("CX 0 1")),
    )

    print("Accepted inputs\n")
    for name, form in forms:
        verdict = quasarq.equivalent(form, target)
        print(f"  {name:<16} {verdict}")
        assert verdict

    # Measurements cannot be compared this way, so they are refused.
    try:
        quasarq.equivalent("H 0\nM 0", "H 0")
        raise AssertionError("a circuit with measurements should be refused")
    except RuntimeError as e:
        print(f"\n  circuits with measurements are refused:\n    {e}")

    print()


if __name__ == "__main__":
    run_identities()
    run_rewrite()
    run_inputs()
    print("All tests passed.")
