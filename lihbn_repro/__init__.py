"""Reproduction utilities for Li-hBN, with explicit evidence scopes.

Only one-particle Wannier calculations and read-only QE log audits are included.
"""
from .wannier import (UP_WANNIER_LABELS, DOWN_WANNIER_LABELS, parse_hr, make_demo_hr,
                      spin_blocks, one_particle_eigenpairs, spin_blocked_to_interleaved, sector_crosscheck)

__all__ = ["UP_WANNIER_LABELS", "DOWN_WANNIER_LABELS", "parse_hr",
           "make_demo_hr", "spin_blocks", "one_particle_eigenpairs", "spin_blocked_to_interleaved",
           "sector_crosscheck"]
