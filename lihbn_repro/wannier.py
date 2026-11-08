"""Provenance-friendly Wannier90 parsing and quadratic-fermion validation.

This does NOT implement interacting Li-hBN or periodic HF/ACE. All matrix
energies are in eV (the usual Wannier90 *_hr.dat convention); no automatic
Hartree/Rydberg conversion is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping
import hashlib
import math

import numpy as np

UP_WANNIER_LABELS = (19, 33, 55, 56, 57, 67, 68, 69, 70, 71, 72)
DOWN_WANNIER_LABELS = (19, 31, 55, 56, 57, 67, 68, 69, 70, 71, 72)


@dataclass(frozen=True)
class WannierHR:
    path: str
    sha256: str
    num_wann: int
    nrpts: int
    R: tuple[tuple[int, int, int], ...]
    degeneracies: tuple[int, ...]
    blocks: Mapping[tuple[int, int, int], np.ndarray]
    max_hr_hermiticity: float

    def h_at_k(self, k_fractional=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Wannier-gauge H(k) with R-vector degeneracies, k fractional reciprocal.

        No comparison with QE band energies or gauge-matched U(k) is implied.
        """
        k = np.asarray(k_fractional, dtype=float)
        if k.shape != (3,) or not np.all(np.isfinite(k)):
            raise ValueError("k must contain three finite fractional reciprocal coordinates")
        H = np.zeros((self.num_wann, self.num_wann), dtype=np.complex128)
        for R, d in zip(self.R, self.degeneracies):
            H += self.blocks[R] * (np.exp(2j * np.pi * np.dot(k, R)) / d)
        return H

    def onsite(self) -> np.ndarray:
        """Unweighted R=(0,0,0) H(R) block; NOT the full Gamma H(k)."""
        r0 = (0, 0, 0)
        if r0 not in self.blocks:
            raise ValueError("R=0 block not found")
        return self.blocks[r0].copy()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hr(path: str | Path, *, expected_wann: int | None = None,
             expected_nrpts: int | None = None, herm_tol: float = 1e-7,
             max_entries: int = 3_000_000) -> WannierHR:
    """Read and validate a Wannier90 *_hr.dat file without guessing missing data.

    Exactly one (R,m,n) entry is required per R and matrix location; tests
    H(R)=H(-R)^dagger and equality of partner degeneracies.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            raise ValueError("Empty hr.dat")
        nw = int(f.readline().strip())
        nr = int(f.readline().strip())
        if nw < 1 or nr < 1 or nr * nw * nw > max_entries:
            raise ValueError("Invalid or unadmitted dimensions in hr.dat")
        if expected_wann is not None and nw != expected_wann:
            raise ValueError(f"Wannier count mismatch: {nw}, expected {expected_wann}")
        if expected_nrpts is not None and nr != expected_nrpts:
            raise ValueError(f"R count mismatch: {nr}, expected {expected_nrpts}")
        deg = []
        while len(deg) < nr:
            line = f.readline()
            if not line:
                raise ValueError("Truncated degeneracy header")
            try:
                deg.extend(int(x) for x in line.split())
            except ValueError as exc:
                raise ValueError("Malformed degeneracy header") from exc
        if len(deg) != nr or min(deg) <= 0:
            raise ValueError("Invalid degeneracy count or value")
        order = []
        matrices: dict[tuple[int,int,int],np.ndarray] = {}
        covered: dict[tuple[int,int,int],np.ndarray] = {}
        records = 0
        for line in f:
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 7:
                raise ValueError(f"Malformed hr record at row {records+1}")
            r = tuple(int(x) for x in p[:3])
            i, j = int(p[3]) - 1, int(p[4]) - 1
            if not 0 <= i < nw or not 0 <= j < nw:
                raise ValueError(f"Bad matrix index at row {records+1}")
            if r not in matrices:
                if len(order) >= nr:
                    raise ValueError("More R vectors than declared")
                matrices[r] = np.zeros((nw,nw),dtype=complex)
                covered[r] = np.zeros((nw,nw),dtype=bool)
                order.append(r)
            if covered[r][i,j]:
                raise ValueError(f"Duplicate (R,m,n)={r,i+1,j+1}")
            covered[r][i,j] = True
            val = complex(float(p[5].replace("D","E")), float(p[6].replace("D","E")))
            if not np.isfinite(val):
                raise ValueError("Non-finite matrix element")
            matrices[r][i,j] = val
            records += 1
        if len(order) != nr or records != nr * nw * nw:
            raise ValueError("Unexpected R or matrix-record count")
        if not all(np.all(mask) for mask in covered.values()):
            raise ValueError("Missing (R,m,n) records")
        dmap = dict(zip(order, deg))
        max_defect = 0.0
        for r in order:
            negative = tuple(-x for x in r)
            if negative not in matrices:
                raise ValueError(f"Missing -R partner for {r}")
            if dmap[r] != dmap[negative]:
                raise ValueError(f"Unequal +R/-R degeneracies for {r}")
            defect = float(np.max(np.abs(matrices[r] - matrices[negative].conj().T)))
            max_defect = max(max_defect, defect)
        if max_defect > herm_tol:
            raise ValueError(f"H(R) Hermiticity defect {max_defect:g} > {herm_tol:g}")
    return WannierHR(str(path.resolve()), sha256(path), nw, nr,
                     tuple(order), tuple(deg), matrices, max_defect)


def spin_blocks(up: WannierHR, down: WannierHR, *,
                mode: str = "onsite", up_labels=UP_WANNIER_LABELS,
                down_labels=DOWN_WANNIER_LABELS) -> tuple[np.ndarray, np.ndarray]:
    """Return selected spin-resolved 11x11 blocks in note's 1-based ordering.

    mode='onsite' uses H(R=0) as in the note. mode='gamma' includes all R.
    """
    if mode not in ("onsite", "gamma"):
        raise ValueError("mode must be onsite or gamma")
    Hup = up.onsite() if mode == "onsite" else up.h_at_k()
    Hdn = down.onsite() if mode == "onsite" else down.h_at_k()
    def select(mat, labels):
        ints = [int(x) for x in labels]
        if len(ints) != len(set(ints)) or any(x < 1 or x > mat.shape[0] for x in ints):
            raise ValueError("Bad one-based Wannier labels")
        i = np.array(ints) - 1
        return mat[np.ix_(i, i)].copy()
    return select(Hup, up_labels), select(Hdn, down_labels)



def spin_blocked_to_interleaved(h_up: np.ndarray, h_down: np.ndarray) -> np.ndarray:
    """Permute diag(h_up,h_down) to (up0,down0,up1,down1,...) coordinates.

    Only a one-particle change of basis ordering; not a correlated two-body
    model and NOT a valid spin-independent ``effective_hamiltonian`` payload.
    """
    up, down = np.asarray(h_up, dtype=complex), np.asarray(h_down, dtype=complex)
    if up.ndim != 2 or up.shape[0] == 0 or up.shape != down.shape or up.shape[0] != up.shape[1]:
        raise ValueError("Both spin blocks must be nonempty equally sized square matrices")
    if any(not np.all(np.isfinite(h)) or np.max(np.abs(h - h.conj().T)) > 1e-9
           for h in (up, down)):
        raise ValueError("Both spin blocks must be finite and Hermitian")
    m = up.shape[0]
    result = np.zeros((2*m, 2*m), dtype=complex)
    result[0::2, 0::2] = up
    result[1::2, 1::2] = down
    return result

def one_particle_eigenpairs(h: np.ndarray, n_electrons: int | None = None) -> dict:
    """Orbital eigenvectors (columns) and optional Slater state projector.

    Input h is a single-spin one-particle Hamiltonian in eV.
    """
    h = np.asarray(h, dtype=complex)
    if h.ndim != 2 or not h.shape[0] or h.shape[0] != h.shape[1]:
        raise ValueError("Square nonempty matrix required")
    if not np.all(np.isfinite(h)) or np.max(np.abs(h-h.conj().T)) > 1e-9:
        raise ValueError("Finite Hermitian matrix required")
    energies, orbitals = np.linalg.eigh(h)
    # Fix column phase; degenerate subspaces still have an arbitrary gauge.
    for j in range(orbitals.shape[1]):
        pivot = orbitals[np.argmax(np.abs(orbitals[:,j])),j]
        orbitals[:,j] *= np.conj(pivot) / abs(pivot)
    residual = np.max(np.linalg.norm(h@orbitals - orbitals*energies[np.newaxis,:],axis=0))
    ans = {"energies_ev":energies,"orbitals":orbitals,
           "max_residual_ev":float(residual),
           "orthogonality_defect":float(np.linalg.norm(orbitals.conj().T@orbitals - np.eye(len(energies))))}
    if n_electrons is not None:
        if not isinstance(n_electrons,int) or not 0 <= n_electrons <= len(energies):
            raise ValueError("Invalid n_electrons for spin block")
        occ = orbitals[:,:n_electrons]
        rho = occ@occ.conj().T
        ans.update(n_electrons=n_electrons,
                   slater_energy_ev=float(np.sum(energies[:n_electrons])),
                   one_rdm=rho,
                   idempotency_defect=float(np.linalg.norm(rho@rho-rho)))
    return ans


def _basis_bitmasks(m: int, n: int) -> list[int]:
    if not 0 <= n <= m:
        raise ValueError("Bad sector occupation")
    return [sum(1<<i for i in subset) for subset in combinations(range(m),n)]


def _slater_coefficients(orbitals: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.ones(1, dtype=complex)
    return np.asarray([np.linalg.det(orbitals[np.ix_(sub,range(n))])
                       for sub in combinations(range(orbitals.shape[0]),n)],dtype=complex)


def sector_crosscheck(h_up: np.ndarray, h_down: np.ndarray,
                      n_up: int, n_down: int, max_dimension: int = 500) -> dict:
    """Independent fixed-(Nup,Ndown) second-quantized exact solve for small D.

    Uses a direct creation/annihilation matrix construction and compares the
    resulting ground energy and state with a Slater determinant. No interactions.
    """
    a = one_particle_eigenpairs(h_up,n_up); b = one_particle_eigenpairs(h_down,n_down)
    m1,m2 = h_up.shape[0],h_down.shape[0]
    d = math.comb(m1,n_up)*math.comb(m2,n_down)
    if d > max_dimension:
        return {"skipped":True, "reason": "sector dimension exceeds admission limit", "dimension":d}
    up_masks = _basis_bitmasks(m1,n_up)
    down_masks = _basis_bitmasks(m2,n_down)
    basis = [u | (v<<m1) for u in up_masks for v in down_masks]
    index = {bit:i for i,bit in enumerate(basis)}
    H1 = np.zeros((m1+m2,m1+m2),dtype=complex)
    H1[:m1,:m1] = h_up
    H1[m1:,m1:] = h_down
    H = np.zeros((d,d),dtype=complex)
    terms = [(p,q,H1[p,q]) for p in range(m1+m2)
             for q in range(m1+m2) if abs(H1[p,q])>1e-14]
    for col,occ in enumerate(basis):
        for p,q,coeff in terms:
            if not occ & (1<<q):
                continue
            unoccupied_q = occ ^ (1<<q)
            if unoccupied_q & (1<<p):
                continue
            sign_q = -1 if (occ & ((1<<q)-1)).bit_count()%2 else 1
            sign_p = -1 if (unoccupied_q & ((1<<p)-1)).bit_count()%2 else 1
            row_bitmask = unoccupied_q | (1<<p)
            row = index.get(row_bitmask)
            if row is None:
                raise AssertionError("Hamiltonian leaked out of fixed spin sector")
            H[row,col] += coeff * sign_p * sign_q
    vals, vecs = np.linalg.eigh(H)
    cu = _slater_coefficients(a["orbitals"],n_up)
    cd = _slater_coefficients(b["orbitals"],n_down)
    trial = np.kron(cu,cd)
    expected = a["slater_energy_ev"]+b["slater_energy_ev"]
    residual = np.linalg.norm(H@trial - expected*trial)
    gap = float(vals[1]-vals[0]) if len(vals)>1 else float("inf")
    return {"skipped":False,"dimension":d,"analytic_ground_energy_ev":expected,
            "sector_ground_energy_ev":float(vals[0]),
            "energy_error_ev":float(abs(vals[0]-expected)),
            "slater_residual_ev":float(residual),
            "slater_norm":float(np.linalg.norm(trial)),
            "eigenstate_fidelity":float(abs(np.vdot(vecs[:,0],trial))**2) if gap>1e-8 else None,
            "gap_ev":gap}


def make_demo_hr(path: str | Path, *, nw: int = 11, seed: int = 1) -> Path:
    """Generate an unmistakably synthetic, H(R)-Hermitian miniature hr.dat fixture.

    Only for software smoke tests, NEVER a purported Li-hBN material result.
    """
    path = Path(path)
    if nw < 1:
        raise ValueError("nw must be positive")
    rng = np.random.default_rng(seed)
    mat0 = rng.normal(size=(nw,nw)) + 1j*rng.normal(size=(nw,nw))
    mat0 = (mat0 + mat0.conj().T)/2
    off = (rng.normal(size=(nw,nw)) + 1j*rng.normal(size=(nw,nw)))*0.02
    Rs = ((0,0,0),(1,0,0),(-1,0,0))
    blocks = (mat0,off,off.conj().T)
    with path.open('w',encoding='utf-8') as f:
        f.write('synthetic reproduction test; NOT Li-hBN data\n')
        f.write(f'{nw}\n3\n')
        f.write('1 1 1\n')
        for R,block in zip(Rs,blocks):
            for i in range(nw):
                for j in range(nw):
                    val=block[i,j]
                    f.write(f'{R[0]} {R[1]} {R[2]} {i+1} {j+1} {val.real:.16e} {val.imag:.16e}\n')
    return path
