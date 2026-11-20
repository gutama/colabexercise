# Evidence contract

Generate the machine-readable schema with `wbg evidence-schema`. Evidence JSON
may be partial. Gates with missing inputs stay pending. Paths in this file are
relative to the evidence JSON directory. Each numerical result carries a
`source` with the artifact path, its SHA-256 digest, and the method or protocol.
Use the saved `relaxed_structure_sha256` in `transport-request.json` or hash
`relax/attempt-*/relaxed.json` for the completed relaxation. The evaluator
checks that the evidence refers to that candidate and relaxed structure and
that each cited local file matches its hash. Review the scientific provenance
and data extraction independently before comparing candidates.

| JSON field | Value and method |
| --- | --- |
| `hull_mev_atom` | meV/atom from a phase hull. Use compatible PBE reference phases and identical pseudopotential lineage. Include elemental references, cutoffs, and energy conventions. Avoid mixing raw QE energies with corrected database entries. |
| `hse06_gap_ev` | eV from the global VBM and CBM. Check off-grid extrema, especially the indirect minimum of 4H-SiC. Cite the dense mesh or Wannier interpolation and compare the interpolated bands with HSE06. |
| `electron_mass_m0`, `hole_mass_m0` | Optional positive masses in units of the electron mass. Document band, valley, direction and fit window. |
| `relative_static_epsilon` | Dimensionless dielectric component along the configured axis, including the ionic response. Keep the coordinate frame consistent across the candidate and reference. Γ DFPT provides electronic permittivity and Born charges for this step. |
| `breakdown_mv_cm` | Optional alternative in MV/cm for the configured direction. Cite the directional model and inputs, or the measurement protocol. The default screening field uses the scalar Kim–Pilania–Ramprasad model. |
| `amset` | Tier 1 transport record with `backend: "amset"`, `electron_cm2_vs`, the four `scattering` entries `ADP`, `POP`, `IMP` and `PIE`, and `free_carrier_screening: true`. Match case conditions. |
| `transport` | Final phonon-limited EPW result with `backend: "epw"`, `solver: "ibte"`, `electron_cm2_vs` and optional `hole_cm2_vs`, both in cm²/(V·s). Match the case temperature, carrier density, axis and bulk regime. |
| `transport.saturation_cm_s` | Optional independent high-field velocity in cm/s for JFOM. Use a high-field transport model or measurement. |
| `baseline` | Track reference ID (`gan-wurtzite` or `sic-4h`), static ε and field with their joint source, and a matched EPW IBTE transport record. Set `breakdown_method` to `kim_pilania_ramprasad` for a reference calculated with the default model or `directional` for the corresponding directional protocol. |

Scalar evidence has this shape:

```json
{
  "value": 0.0,
  "source": {
    "artifact": "phase-hull.json",
    "sha256": "<64 lowercase hex characters>",
    "method": "PBE, converged competing phase and elemental references"
  }
}
```

The displayed value illustrates the schema. The source file must contain the
actual computation and inputs. Calibration compounds are separated in reports
and provide reference geometries and calibration workflows. BFOM promotion
requires a strict improvement over the reference.

The scalar intrinsic field is
`0.24442 × exp(0.315 × sqrt(Eg_eV × νmax_THz))` in MV/cm. The workflow uses the
validated HSE06 fundamental gap and the maximum sampled phonon frequency from
`matdyn.x`. The frequency is converted from cm⁻¹ to ordinary THz with the
factor 0.0299792458. Each evaluation records the inputs and provenance. Treat
the result as an intrinsic screening descriptor with no directional resolution.
[Kim et al., Eq. (3)](https://doi.org/10.1021/acs.chemmater.5b04109) ·
[Unit conversion and examples, Supplement §SV](https://arxiv.org/html/2610.31777v1)

BFOM uses absolute vacuum permittivity in F/cm multiplied by the relative static
dielectric constant, mobility in cm²/(V·s) and the cube of the field in V/cm.
The reported unit is W/cm². JFOM is (E × v / 2π)², with E in V/cm and saturation
velocity in cm/s. Match the dielectric component and transport conditions, and
use the same field protocol for the candidate and reference.

## External stage inputs

- **Hull**. A phase diagram needs competing phases outside the candidate run.
  Use Materials Project to identify competing phases and recompute candidate,
  elemental and competing-phase energies with consistent QE settings. Assemble
  mutually compatible energy entries, then use pymatgen's
  `PhaseDiagram.get_e_above_hull` and convert eV/atom to meV/atom. The initial
  scaffold consumes this result and its recorded phase set.
- **Wannier**. Generate `pw2wannier90.x` and Wannier90 inputs after selecting
  material-specific projections and disentanglement windows. Validate the
  interpolation near the fundamental band edges and record the mesh used for
  effective mass fits.
- **Phonons**. PBE `matdyn.x` reads `q2r.x` force constants and samples a uniform
  reciprocal mesh. A finite q mesh can miss off-grid soft modes. Check mesh
  convergence, acoustic sum rules and nonanalytic LO–TO behavior before
  reporting physical stability.
- **AMSET**. Use a validated QE-to-AMSET representation of the bands/wavefunctions
  and provide the elastic, deformation potential, piezoelectric and dielectric
  tensors and the effective polar optical frequency. The effective polar optical
  frequency is a separate AMSET input from the spectral cutoff used for the field.
  Record the settings, scattering channels and mobility in `amset`, then rerun
  through Stage 4 to release EPW once it passes.
- **EPW**. Build a matched coarse QE NSCF mesh, validate Wannier interpolation,
  and collect `dvscf`, dynamical matrices and `phsave`. Set `iterative_bte=.true.`
  and converge on commensurate fine k/q grids, for example by testing 60³ through
  100³ where justified. Preserve the interpolation and convergence artifacts.
