"""Check Wannier90 .eig against spin-resolved QE NSCF CSV; independent data only."""
from pathlib import Path
import numpy as np


def compare_eig_csv(csv_path: str | Path, up_path: str | Path,
                    down_path: str | Path, *, expected_kpoints=144,
                    expected_bands=200) -> dict:
    """Compare explicit (k_index,band,spin) records in eV, without reordering .eig."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    necessary = {"k_index","spin","band","energy_eV"}
    if not necessary.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(necessary - set(df.columns))}")
    spin = df["spin"].astype(str).str.strip().str.lower()
    mapping = {"1":"up", "2":"down", "up":"up", "down":"down"}
    if not spin.isin(mapping).all():
        raise ValueError("Unrecognized spin tags; expected 1/2 or up/down")
    df = df.copy()
    df["spin_name"] = spin.map(mapping)
    if len(df)!=2*expected_kpoints*expected_bands:
        raise ValueError("Wrong total NSCF row count")
    report = {}
    for name,path in (("up",up_path),("down",down_path)):
        table = df[df.spin_name==name].sort_values(["k_index","band"])
        if len(table)!=expected_kpoints*expected_bands:
            raise ValueError(f"{name}: wrong NSCF channel size")
        if table.duplicated(["k_index","band"]).any():
            raise ValueError(f"{name}: duplicate k/band pairs")
        expected_k = np.repeat(np.arange(1, expected_kpoints + 1), expected_bands)
        expected_band = np.tile(np.arange(1, expected_bands + 1), expected_kpoints)
        if not np.array_equal(table.k_index.to_numpy(), expected_k) or not np.array_equal(
            table.band.to_numpy(), expected_band
        ):
            raise ValueError(f"{name}: missing or unexpected k-point/band indices")
        eig = np.loadtxt(path,ndmin=2)
        if eig.shape != (len(table),3):
            raise ValueError(f"{name}: expected {(len(table),3)} .eig shape, got {eig.shape}")
        if not np.array_equal(eig[:,0],table.band.to_numpy()) or not np.array_equal(eig[:,1],table.k_index.to_numpy()):
            raise ValueError(f"{name}: .eig band/k indexing differs from sorted NSCF records")
        delta=eig[:,2]-table.energy_eV.to_numpy(dtype=float)
        if not np.isfinite(delta).all():
            raise ValueError("Non-finite eigenvalue comparison")
        report[name]={"records":len(delta),"rms_ev":float(np.sqrt(np.mean(delta**2))),
                      "max_abs_ev":float(np.max(np.abs(delta))),
                      "count_above_1e-6_ev":int(np.count_nonzero(np.abs(delta)>1e-6))}
    return report
