"""Molecular electronic-structure workflows: PySCF/QE runs, Hamiltonian and qubit export."""
import importlib.metadata as _md

try:
    __version__ = _md.version("colab-quantum-chemistry")
except _md.PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0+unknown"


def distribution_name() -> str:
    """Installed distribution that provides this package (survives template renames)."""
    names = _md.packages_distributions().get(__name__, [])
    return names[0] if names else "colab-quantum-chemistry"
