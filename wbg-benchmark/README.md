# Wide bandgap semiconductor screening

This repository implements the first four stages of a GaN-class and 4H-SiC-class
bulk screening benchmark. ASE writes periodic Quantum ESPRESSO decks. pymatgen
builds and inspects crystal structures and detects repeated primitive cells.
The workflow records a checkpoint per completed QE task and evaluates explicit
gates as `pass`, `fail`, or `pending`.

The structure of the benchmark follows the attached handoff and
[Material Discovery Bench](https://discoveredmaterials.com/benchmark/).
Its downloadable candidate data serve as a test of structural deduplication.
The MDB thermal, dielectric and mechanical estimates stay separate from the
semiconductor evidence in this project.

| Stage | Automated here | Evidence to supply for scoring |
| --- | --- | --- |
| 1. Triage | `pw.x` PBE `vc-relax` → fresh PBE SCF on the relaxed cell | Hull distance from a compatible phase diagram using competing phases, for example from Materials Project |
| 2. Electronic | `pw.x` HSE06 SCF and sampled band edges from QE XML | Verified fundamental gap from dense sampling or validated interpolation. Optional electron and hole masses |
| 3. Lattice and dielectric | PBE `ph.x` → `q2r.x` → `matdyn.x` before HSE06, then scalar intrinsic breakdown from the verified hybrid gap and phonon cutoff | Static permittivity with the ionic response. Optional direction-specific breakdown evidence |
| 4. Mobility and FOM | AMSET triage, then an EPW IBTE handoff for candidates that pass. BFOM and JFOM calculation | Transport results at the configured conditions and the track's matched reference |

Stage 3's stability check runs before Stage 2. The breakdown gate waits for the
HSE06 fundamental gap, and Stage 4 waits for all three preceding stages to pass.
Stage 4 prepares transport handoffs and validates the returned evidence.

## Quick start

Python 3.11+ on Linux or macOS.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

wbg plan configs/gan.json --workdir runs/gan-plan
wbg plan configs/sic.json --workdir runs/sic-plan
pytest -q
```

The plans contain a seed CIF, a dependency graph and preview QE decks. The
previews use the initial geometry. Execution generates the SCF/HSE decks from
the saved relaxed geometry. The geometry tests verify four atoms for wurtzite
GaN and eight atoms for 4H-SiC. Both nominal seeds have space group 186. The
4H-SiC seed uses ideal ABAC stacking with approximate lattice parameters.
Relax both structures before using them as references.

To execute, provide `pw.x`, `ph.x`, `q2r.x`, and `matdyn.x` on PATH or set
`qe.commands` to the corresponding executable/MPI argument lists. Copy the
case config beside the supplied configs, point it at vetted scalar-relativistic
PBE ONCV/NC UPF v2 pseudopotentials, and fill in each file's `source` and
`sha256` (for example, via `sha256sum`). For Ga, use a potential that includes
the 3d semicore states and examine sensitivity to the pseudopotential choice.
Adjust the band count and converge cutoffs, meshes and properties on both
calibration cases. The default 80 Ry, density cutoff ratio of four, and k/q
meshes are starting values. The PBE grids use `pbe_kmesh` as an axis-by-axis
minimum and round up to multiples of the DFPT grid so that each k count
times its real-space lattice-vector length is at least 35 Å by default. The
relaxation deck uses the seed cell, and the PBE SCF uses the relaxed cell.

```bash
wbg run configs/gan-local.json --workdir runs/gan --through 1
wbg evidence-schema --output runs/evidence.schema.json
wbg run configs/gan-local.json --workdir runs/gan --through 2 --evidence runs/gan-hull.json
wbg run configs/gan-local.json --workdir runs/gan --through 4 --evidence runs/gan-evidence.json
wbg evaluate configs/gan-local.json --workdir runs/gan --evidence runs/gan-evidence.json
```

Use the same pattern with `configs/sic.json`. The calibration flag lets the
reference compounds exercise the QE stages even when a screening gate is
pending or fails. It preserves their gate scores. Candidate cases need the
Stage 1 hull and PBE gap before DFPT, and a passing phonon minimum before HSE06.
`--through 2` includes the PBE phonon prerequisite and runs the same QE tasks as
`--through 3`. Once the gap and breakdown gates pass, run through Stage 4 with
the latest evidence to refresh the AMSET or EPW handoff. Setting
`transport` to `amset` stops at triage. Full Stage 4 promotion requires EPW.
Each failed task keeps its attempt directory. Reruns reuse verified completed
outputs. Use a fresh work directory for schema-v1 checkpoints or changes to
the structure, config, or UPF files.

The reference build is stock QE input syntax. A compatible build of
[Discovered Materials’ dm_qe](https://github.com/discovered-materials/dm_qe)
can be selected with the same command configuration. The scaffold creates no
Colab sessions and installs no QE engines or pseudopotentials.

## Screening criteria

| Gate | GaN-class | 4H-SiC-class |
| --- | --- | --- |
| PBE mesh gap and hull | ≥ 1.8 eV and ≤ 50 meV/atom | Same |
| HSE06 fundamental gap | ≥ 3.40 eV | ≥ 3.26 eV |
| Minimum sampled signed phonon frequency | ≥ −1 THz | Same |
| Breakdown field | ≥ 3.3 MV/cm | ≥ 2.8 MV/cm |
| Bulk electron mobility in AMSET and then EPW | ≥ 1,000 cm²/V·s | ≥ 500 cm²/V·s |
| Device figures of merit | BFOM above matched GaN baseline | BFOM above matched 4H-SiC baseline |

The bulk comparison defaults to 300 K and 10¹⁶ cm⁻³. The GaN template uses the
basal direction and the SiC template uses the c direction. These choices are
configurable. JFOM is reported when independent saturation velocity is
available. A 2DEG requires its own comparison protocol. See
[the evidence contract](docs/evidence.md) for units, provenance and transport
requirements.

## Later stages

| Stage | Planned calculation and gate |
| --- | --- |
| 5. Thermal conductivity | Phono3py displaced supercells with PBE `pw.x` forces, or Pheasy force-constant fitting. Converge supercell size and phonon BTE q mesh, including three-phonon and isotope scattering. The lattice thermal-conductivity floors are 130 W/(m·K) for GaN and 300 W/(m·K) for 4H-SiC. Record temperature and tensor component. |
| 6. Dopability | HSE06 supercells for native defects and likely dopants across charge states. Start with 64–128 atoms and test size convergence. Use finite-size corrections such as FNV or Kumagai–Oba and calculate charge transition levels and self-compensation. Screen for n-type ionization below 50–100 meV for GaN or bipolar ionization below 150–200 meV for 4H-SiC, with explicit checks for Fermi-level pinning. Follow with expert synthesis review. |

Use `wbg dedupe path/a.cif path/b.cif` on candidate structures and again on
relaxed structures before scoring a cohort. The matcher compares reduced
primitive geometries within composition groups. Space-group and Wyckoff
summaries support review. Structural similarity does not establish literature
novelty or the viability of a synthesis route.

```bash
wbg audit-mdb /path/to/mdb-materials.zip --limit 30 --output runs/mdb-audit.json
```

Validation covers QE deck generation, both seed structures, output parsers,
provenance checks, and the Stage 1–4 flow with explicitly synthetic QE fixtures.
Physical calibration energies, transport and convergence studies require
separate runs with vetted inputs.
