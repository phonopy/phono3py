"""Tests for the fc4 first-order frequency shift (phono3py.phonon4.frequency_shift).

At Gamma (primitive = supercell, mesh = [1, 1, 1]) the one-loop first-order shift
returned by ``FrequencyShift`` is identical, mode by mode, to first-order
perturbation theory on the temperature-renormalized harmonic force constants

    Phi2_eff = Phi2 + (1/2) Phi4 : <u u>,

with ``<u u>`` the quantum harmonic displacement covariance. This identity holds
for arbitrary fc2 / fc4, so it is checked here on a small random system. The
covariance uses phonopy's thermal-displacement unit convention, independent of
the fc4 code; see ``tools/freqshift_validation_report`` for the derivation. This
covers the contraction magnitude, the mass weighting, and the unit conversion,
but not the finite-q phase convention (at Gamma the eigenvectors are real).

"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, Supercell

from phono3py.phonon4.fc4 import set_permutation_symmetry_fc4
from phono3py.phonon4.frequency_shift import FrequencyShift, _bose_einstein

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CUTOFF = 1e-4
TEMPERATURES = [300.0, 1000.0, 3000.0]


def _system(seed: int = 0) -> tuple[Supercell, Primitive]:
    """Return (supercell, primitive) for a 3-atom random cell, primitive = cell."""
    rng = np.random.default_rng(seed)
    lattice = np.eye(3) * 4.0 + rng.normal(scale=0.1, size=(3, 3))
    lattice = (lattice + lattice.T) / 2
    cell = PhonopyAtoms(
        cell=lattice,
        symbols=["Si", "O", "C"],  # distinct masses -> nontrivial mass weighting
        scaled_positions=rng.random((3, 3)),
    )
    ph = Phonopy(
        cell, supercell_matrix=np.eye(3, dtype=int), primitive_matrix=np.eye(3)
    )
    return ph.supercell, ph.primitive


def _random_fc2(natom: int, seed: int) -> NDArray[np.double]:
    """Random symmetric, positive-definite fc2 (all modes nu > 0)."""
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(3 * natom, 3 * natom))
    mat = b @ b.T + 5.0 * np.eye(3 * natom)
    fc2 = mat.reshape(natom, 3, natom, 3).transpose(0, 2, 1, 3).copy()
    return 0.5 * (fc2 + fc2.transpose(1, 0, 3, 2))


def _random_fc4(natom: int, seed: int) -> NDArray[np.double]:
    """Random permutation-symmetric fc4 in the full layout."""
    rng = np.random.default_rng(seed)
    fc4 = rng.normal(scale=20.0, size=(natom, natom, natom, natom, 3, 3, 3, 3))
    set_permutation_symmetry_fc4(fc4)
    return fc4


def _renormalized_shift(
    fc4: NDArray[np.double],
    masses: NDArray[np.double],
    freqs: NDArray[np.double],
    eigvecs: NDArray[np.cdouble],
    temperature: float,
    factor: float,
) -> NDArray[np.double]:
    """Independent first-order shift d(nu) from Phi2_eff = Phi2 + 1/2 Phi4:<uu>."""
    units = get_physical_units()
    # Covariance <u u> in Angstrom^2: mass weight 1/sqrt(M_i M_j) with M in kg
    # = (AMU number) * AMU, so the 1/AMU lives here.
    inv_sqrt_m_kg = 1.0 / np.sqrt(np.repeat(masses, 3) * units.AMU)
    # dD = M^-1/2 dPhi2 M^-1/2 uses the AMU-number masses, as the dynamical matrix.
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses, 3))
    cuu = np.zeros((len(inv_sqrt_m), len(inv_sqrt_m)), dtype="complex128")
    for nu, e in zip(freqs, eigvecs.T, strict=True):
        if nu <= CUTOFF:
            continue
        n = _bose_einstein(np.array([nu]), temperature)[0]
        q2 = (
            units.Hbar
            * units.EV
            / units.Angstrom**2
            * (2 * n + 1)
            / (2 * nu * 1e12 * 2 * np.pi)
        )
        cuu += q2 * np.outer(e, e.conj())
    cuu *= np.outer(inv_sqrt_m_kg, inv_sqrt_m_kg)

    natom = len(masses)
    cuu4 = cuu.reshape(natom, 3, natom, 3)
    dphi2 = 0.5 * np.einsum("ijklabgd,kgld->iajb", fc4, cuu4, optimize=True)
    dd = dphi2.reshape(3 * natom, 3 * natom) * np.outer(inv_sqrt_m, inv_sqrt_m)
    shifts = np.zeros(3 * natom)
    for s, (nu, e) in enumerate(zip(freqs, eigvecs.T, strict=True)):
        if nu <= CUTOFF:
            continue
        shifts[s] = factor**2 * (e.conj() @ dd @ e).real / (2 * nu)
    return shifts


@pytest.mark.parametrize("lang", ["C", "Rust"])
def test_frequency_shift_matches_renormalization(lang: Literal["C", "Rust"]) -> None:
    """FrequencyShift equals the real-space renormalization shift at Gamma."""
    if lang == "Rust":
        pytest.importorskip("phonors")
    factor = get_physical_units().DefaultToTHz
    supercell, primitive = _system(0)
    natom = len(primitive)
    fc2 = _random_fc2(natom, 1)
    fc4 = _random_fc4(natom, 2)
    masses = np.array(primitive.masses, dtype="double")

    fs = FrequencyShift(
        fc4,
        fc2,
        supercell,
        primitive,
        mesh=np.array([1, 1, 1], dtype="int64"),
        temperatures=np.array(TEMPERATURES),
        cutoff_frequency=CUTOFF,
        lang=lang,
    )
    shifts = fs.run(grid_point=0)  # (n_temperatures, n_band)
    freqs = fs.frequencies[0]
    eigvecs = fs._eigenvectors[0]

    # Sanity: the random system has all modes above the cutoff and a nonzero shift.
    assert (freqs > CUTOFF).all()
    assert np.abs(shifts).max() > 1e-6

    for i_t, temperature in enumerate(TEMPERATURES):
        ref = _renormalized_shift(fc4, masses, freqs, eigvecs, temperature, factor)
        np.testing.assert_allclose(shifts[i_t], ref, rtol=1e-9, atol=1e-12)
