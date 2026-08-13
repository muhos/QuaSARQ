# Changelog

## 1.7.0 [2026-08-13]

**QuaSARQ is now on PyPI.**

```bash
pip install quasarq
```

- Sample detection events, observable flips, or raw measurement outcomes directly from a
  `stim.Circuit` (or circuit text), returned as numpy arrays.
- `quasarq.sinter` plugs the GPU sampler into `sinter`, decoding with pymatching.
- Simulate a circuit's final state, or check two circuits for equivalence, from Python.
- Several processes can share one GPU, so `sinter` can run concurrent workers on a single card.
- Faster random number generation and retuned kernel launch settings.

The package needs Linux x86_64, CPython 3.10–3.13, and an NVIDIA driver 525 or newer with a
Pascal or newer GPU. It needs no CUDA toolkit and no compiler. Building the command-line tool
from source is unchanged.

Earlier releases are listed under [tags](https://github.com/muhos/QuaSARQ/tags).
