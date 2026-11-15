import os

# Before PySCF/numpy load: degenerate orbitals rotate between runs under threaded BLAS,
# which makes integrals (not energies) irreproducible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root():
    return ROOT


def molecule(*atoms, charge=0, spin=0):
    return {
        "atoms": [{"element": element, "xyz": list(xyz)} for element, xyz in atoms],
        "charge": charge,
        "spin": spin,
    }


H2 = molecule(("H", (0, 0, 0)), ("H", (0, 0, 0.735)))
HEH_CATION = molecule(("He", (0, 0, 0)), ("H", (0, 0, 0.772)), charge=1)
LIH = molecule(("Li", (0, 0, 0)), ("H", (0, 0, 1.595)))
WATER = molecule(("O", (0, 0, 0.1173)), ("H", (0, 0.7572, -0.4692)), ("H", (0, -0.7572, -0.4692)))
