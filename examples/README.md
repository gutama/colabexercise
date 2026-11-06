# Examples

Every file here is validated by `tests/test_workflow.py::test_every_example_validates`. The energies are the
anchors the tests reproduce (STO-3G, PySCF 2.15, `OMP_NUM_THREADS=1`).

| Config | What it shows | SCF (Ha) | FCI / CASCI (Ha) |
|---|---|---|---|
| `h2-pyscf.json` | Minimal HF + integral export (0.74 Å) | −1.1167593074 | −1.1372838345 |
| `heh-cation-qubit.json` | JW qubit Hamiltonian on an **asymmetric** system, which exposes qubit-ordering bugs that H₂ hides | −2.8413824898 | −2.8510240300 |
| `lih-bk.json` | Bravyi–Kitaev (12 qubits) with FCI, CCSD and CCSD(T) references | −7.8620238601 | −7.8824019323 |
| `h2o-cas44.json` | Active space CAS(4e,4o): frozen core, 8 qubits | −74.9630231385 | −74.9704543855 |
| `h2o-dft-gpu.json` | B3LYP/def2-SVP on GPU4PySCF. Needs `--gpu`; not validated on GPU hardware here | — | — |
| `h2-qe.json` | Quantum ESPRESSO gamma-point SCF; needs `pw.x` and the pseudopotential (see README) | — | — |

```bash
quantum-workflow examples/h2o-cas44.json --check                 # validate, show defaults
quantum-workflow examples/h2o-cas44.json --output results/h2o
quantum-workflow-adapt results/h2o
```
