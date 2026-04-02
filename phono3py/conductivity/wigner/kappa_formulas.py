"""Shared constants and helpers for the Wigner transport equation."""

from __future__ import annotations

from phonopy.physical_units import get_physical_units

# Threshold in THz below which two modes are considered degenerate and treated
# as a population (diagonal) term instead of a coherence (off-diagonal) term.
DEGENERATE_FREQUENCY_THRESHOLD_THZ = 1e-4


def get_conversion_factor_WTE(volume: float) -> float:
    """Return unit conversion factor for Wigner transport equation kappa.

    Parameters
    ----------
    volume : float
        Primitive-cell volume in Angstrom^3.

    Returns
    -------
    float
        Conversion factor in W/(m*K).

    """
    u = get_physical_units()
    return (
        (u.THz * u.Angstrom) ** 2  # group velocity squared
        * u.EV  # specific heat in eV/K
        * u.Hbar  # Lorentzian eV^-1 to s
        / (volume * u.Angstrom**3)  # unit cell volume
    )
