"""First-order (loop) phonon frequency shift from fc4 (experimental).

Port of the 2015 phono4py ``FrequencyShift`` (Python / no-symmetry path) onto
modern phonopy. The first-order correction to the phonon frequency from the
quartic force constants is

    Delta(q, s) = C * sum_{q', s'} Phi4_normal(q,s; q',s') * (2 n_{q',s'} + 1)

where ``Phi4_normal`` is the fc4 contracted with the eigenvectors at ``q`` and
``q'`` (and divided by the two frequencies), ``n`` is the Bose-Einstein
occupation, and ``C`` collects the physical-unit conversion. Only the
no-symmetry path (sum over the full mesh, weight 1) is implemented.

The fc4 Fourier transform uses :class:`phono3py.phonon4.real_to_reciprocal.
RealToReciprocalFc4`, and the eigenvector contraction here keeps the matching
2015 phase convention.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive

from phono3py._lang import resolve_lang
from phono3py.phonon4.real_to_reciprocal import RealToReciprocalFc4


def _bose_einstein(frequencies: NDArray[np.double], t: float) -> NDArray[np.double]:
    """Return Bose-Einstein occupations for frequencies (THz) at temperature t (K).

    Frequencies at or below zero get zero occupation.
    """
    units = get_physical_units()
    occ = np.zeros_like(frequencies)
    if t <= 0:
        return occ
    mask = frequencies > 0
    x = units.THzToEv * frequencies[mask] / (units.KB * t)
    occ[mask] = 1.0 / (np.exp(x) - 1.0)
    return occ


class ReciprocalToNormalFc4:
    """Contract a reciprocal-space fc4 with eigenvectors to normal coordinates."""

    def __init__(
        self,
        primitive: Primitive,
        frequencies: NDArray[np.double],
        eigenvectors: NDArray[np.complex128],
        cutoff_frequency: float = 1e-4,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method."""
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._cutoff_frequency = cutoff_frequency
        self._inv_sqrt_masses = np.array(
            1.0 / np.sqrt(primitive.masses), dtype="double"
        )
        self._num_atom = len(primitive)
        self._lang = resolve_lang(lang)

    def run(
        self,
        fc4_reciprocal: NDArray[np.complex128],
        gp: int,
        band_index: int,
        gp1: int,
    ) -> NDArray[np.complex128]:
        """Return the fc4 normal-coordinate elements for band ``band_index`` at ``gp``.

        Returns an array indexed by the band index at ``gp1`` (shape (num_band,),
        complex); entries with frequency below the cutoff are left zero.
        """
        num_band = self._num_atom * 3
        fc4_normal = np.zeros(num_band, dtype="complex128")
        f1 = self._frequencies[gp][band_index]

        if self._lang == "Rust":
            import phonors  # type: ignore[import-untyped]

            phonors.reciprocal_to_normal_fc4(
                fc4_normal,
                np.ascontiguousarray(fc4_reciprocal),
                np.ascontiguousarray(self._eigenvectors[gp][:, band_index]),
                np.ascontiguousarray(self._eigenvectors[gp1]),
                float(f1),
                np.ascontiguousarray(self._frequencies[gp1]),
                self._inv_sqrt_masses,
                self._cutoff_frequency,
            )
            return fc4_normal

        if f1 <= self._cutoff_frequency:
            return fc4_normal
        e1 = self._eigenvectors[gp][:, band_index]
        f2 = self._frequencies[gp1]
        for i in range(num_band):
            if f2[i] > self._cutoff_frequency:
                e2 = self._eigenvectors[gp1][:, i]
                fc4_normal[i] = self._sum_in_atoms(fc4_reciprocal, e1, e2) / f1 / f2[i]
        return fc4_normal

    def _sum_in_atoms(
        self,
        fc4_reciprocal: NDArray[np.complex128],
        e1: NDArray[np.complex128],
        e2: NDArray[np.complex128],
    ) -> complex:
        n = self._num_atom
        w = self._inv_sqrt_masses[:, None]
        ec1 = e1.reshape(n, 3)
        ec2 = e2.reshape(n, 3)
        a = ec1.conj() * w  # leg 1: e1*, /sqrt(m)
        b = ec1 * w  # leg 2: e1
        c = ec2 * w  # leg 3: e2
        d = ec2.conj() * w  # leg 4: e2*
        return complex(
            np.einsum(
                "im,jn,kp,lq,ijklmnpq->", a, b, c, d, fc4_reciprocal, optimize=True
            )
        )


class FrequencyShift:
    """fc4 first-order phonon frequency shift (experimental, no-symmetry)."""

    def __init__(
        self,
        fc4: NDArray[np.double],
        fc2: NDArray[np.double],
        supercell: PhonopyAtoms,
        primitive: Primitive,
        mesh: NDArray[np.int64],
        temperatures: NDArray[np.double] | None = None,
        band_indices: NDArray[np.int64] | None = None,
        is_compact_fc4: bool = False,
        frequency_factor_to_THz: float | None = None,
        cutoff_frequency: float = 1e-4,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Parameters
        ----------
        fc4 : ndarray
            fc4 (full or compact; set ``is_compact_fc4`` accordingly).
        fc2 : ndarray
            Harmonic force constants for the dynamical matrix.
        supercell, primitive : PhonopyAtoms, Primitive
            Cells.
        mesh : array_like
            Reciprocal sampling mesh, shape ``(3,)``.
        temperatures : array_like, optional
            Temperatures in K. Default is ``[0.0]``.
        band_indices : array_like, optional
            Band indices to compute. Default is all bands.
        is_compact_fc4 : bool, optional
            Whether ``fc4`` is compact. Default is False.
        frequency_factor_to_THz : float, optional
            Frequency unit factor. Default is phonopy's ``DefaultToTHz``.
        cutoff_frequency : float, optional
            Frequencies at or below this (THz) are skipped. Default 1e-4.

        """
        units = get_physical_units()
        self._primitive = primitive
        self._mesh = np.array(mesh, dtype="int64")
        self._temperatures = (
            np.array([0.0])
            if temperatures is None
            else np.array(temperatures, dtype="double")
        )
        num_band = len(primitive) * 3
        self._band_indices = (
            np.arange(num_band) if band_indices is None else np.array(band_indices)
        )
        if frequency_factor_to_THz is None:
            frequency_factor_to_THz = units.DefaultToTHz
        self._factor = frequency_factor_to_THz
        self._cutoff_frequency = cutoff_frequency
        self._lang = resolve_lang(lang)

        self._dm = get_dynamical_matrix(fc2, supercell, primitive)
        self._r2r = RealToReciprocalFc4(
            fc4, primitive, self._mesh, is_compact_fc4, lang=self._lang
        )

        self._grid_address = np.array(
            list(np.ndindex(*self._mesh.tolist())), dtype="int64"
        )
        self._frequencies, self._eigenvectors = self._solve_phonons()
        self._r2n = ReciprocalToNormalFc4(
            primitive,
            self._frequencies,
            self._eigenvectors,
            cutoff_frequency,
            lang=self._lang,
        )

        # Delta unit conversion (THz), matching the 2015 phono4py.
        self._unit_conversion = (
            units.EV
            / units.Angstrom**4
            / units.AMU**2
            / (2 * np.pi * units.THz) ** 2
            * units.Hbar
            * units.EV
            / (2 * np.pi * units.THz)
            / 8
            / np.prod(self._mesh)
        )

    def _solve_phonons(self) -> tuple[NDArray[np.double], NDArray[np.complex128]]:
        num_grid = len(self._grid_address)
        num_band = len(self._primitive) * 3
        frequencies = np.zeros((num_grid, num_band), dtype="double")
        eigenvectors = np.zeros((num_grid, num_band, num_band), dtype="complex128")
        for gi, address in enumerate(self._grid_address):
            q = address / self._mesh
            self._dm.run(q)
            eigvals, eigenvectors[gi] = np.linalg.eigh(self._dm.dynamical_matrix)
            frequencies[gi] = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * self._factor
        return frequencies, eigenvectors

    @property
    def grid_address(self) -> NDArray[np.int64]:
        """Return the (no-symmetry) grid addresses."""
        return self._grid_address

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies on the grid (THz)."""
        return self._frequencies

    def run(self, grid_point: int) -> NDArray[np.double]:
        """Return frequency shifts at a grid point, shape (n_temperatures, n_bands)."""
        address0 = self._grid_address[grid_point]
        num_grid = len(self._grid_address)
        num_band = len(self._primitive) * 3

        # fc4_normal[gp1, band_j, band'] for the requested bands.
        fc4_normal = np.zeros(
            (num_grid, len(self._band_indices), num_band), dtype="complex128"
        )
        for gi, address1 in enumerate(self._grid_address):
            quartet = np.array([-address0, address0, address1, -address1])
            fc4_reciprocal = self._r2r.run(quartet)
            for j, band_index in enumerate(self._band_indices):
                fc4_normal[gi, j] = self._r2n.run(
                    fc4_reciprocal, grid_point, int(band_index), gi
                )

        shifts = np.zeros((len(self._temperatures), len(self._band_indices)))
        for i_t, temperature in enumerate(self._temperatures):
            for j in range(len(self._band_indices)):
                total = 0.0 + 0.0j
                for gi in range(num_grid):
                    occ = _bose_einstein(self._frequencies[gi], temperature)
                    total += (
                        fc4_normal[gi, j] * self._unit_conversion * (2 * occ + 1)
                    ).sum()
                shifts[i_t, j] = total.real
        return shifts
