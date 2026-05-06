"""Isotope scattering calculation."""

# Copyright (C) 2015 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, get_dynamical_matrix
from phonopy.phonon.grid import BZGrid
from phonopy.phonon.tetrahedron_mesh import get_tetrahedra_frequencies
from phonopy.phonon.tetrahedron_method import (
    TetrahedronMethod,
    get_integration_weights,
)
from phonopy.structure.atomic_data import get_atomic_data
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phono3py.phonon.func import gaussian
from phono3py.phonon.solver import run_phonon_solver_c, run_phonon_solver_py


def get_unique_grid_points(
    grid_points: NDArray[np.int64],
    bz_grid: BZGrid,
    lang: Literal["C", "Rust"] = "C",
) -> NDArray[np.int64]:
    """Collect grid points on tetrahedron vertices around input grid points.

    Find grid points of 24 tetrahedra around each grid point and
    collect those grid points that are unique.

    Parameters
    ----------
    grid_points : array_like
        Grid point indices.
    bz_grid : BZGrid
        Grid information in reciprocal space.

    Returns
    -------
    ndarray
        Unique grid points on tetrahedron vertices around input grid points.
        shape=(unique_grid_points, ), dtype='int64'.

    """
    _grid_points = np.ascontiguousarray(grid_points, dtype="int64")
    thm = TetrahedronMethod(bz_grid.microzone_lattice)
    unique_vertices = np.array(
        np.dot(thm.get_unique_tetrahedra_vertices(), bz_grid.P.T),
        dtype="int64",
        order="C",
    )
    neighboring_grid_points = np.zeros(
        len(unique_vertices) * len(_grid_points), dtype="int64"
    )
    args = (
        neighboring_grid_points,
        _grid_points,
        unique_vertices,
        bz_grid.D_diag,
        bz_grid.addresses,
        bz_grid.gp_map,
        bz_grid.store_dense_gp_map * 1 + 1,
    )
    if lang == "Rust":
        import phonors  # type: ignore[import-untyped]

        phonors.neighboring_grid_points(*args)
    else:
        import phono3py._phono3py as phono3c  # type: ignore

        phono3c.neighboring_grid_points(*args)

    return np.array(np.unique(neighboring_grid_points), dtype="int64")


def get_mass_variances(
    primitive: PhonopyAtoms | None = None,
    symbols: Sequence[str] | None = None,
    isotope_data: dict | None = None,
) -> NDArray[np.double]:
    """Calculate mass variances."""
    _symbols: Sequence[str]
    if primitive is not None:
        _symbols = primitive.symbols
    elif symbols is not None:
        _symbols = symbols
    else:
        raise RuntimeError("primitive or symbols have to be given.")

    _isotope_data = {}
    phonopy_isotope_data = get_atomic_data().isotope_data
    for s in _symbols:
        if isotope_data is not None and s in isotope_data:
            _isotope_data[s] = isotope_data[s]
        else:
            _isotope_data[s] = phonopy_isotope_data[s]

    mass_variances = []
    for s in _symbols:
        masses = np.array([x[1] for x in _isotope_data[s]])
        fractions = np.array([x[2] for x in _isotope_data[s]])
        m_ave = np.dot(masses, fractions)
        g = np.dot(fractions, (1 - masses / m_ave) ** 2)
        mass_variances.append(g)

    return np.array(mass_variances, dtype="double")


class Isotope:
    """Isotope scattering calculation class."""

    def __init__(
        self,
        mesh: float | NDArray[np.int64] | Sequence[int] | Sequence[Sequence[int]],
        primitive: Primitive,
        mass_variances: Sequence[float]
        | NDArray[np.double]
        | None = None,  # length of list is num_atom.
        isotope_data: dict | None = None,
        band_indices: Sequence[int] | NDArray[np.int64] | None = None,
        sigma: float | None = None,
        bz_grid: BZGrid | None = None,
        frequency_factor_to_THz: float | None = None,
        use_grg: bool = False,
        symprec: float = 1e-5,
        cutoff_frequency: float | None = None,
        lapack_zheev_uplo: Literal["L", "U"] = "L",
        lang: Literal["C", "Python", "Rust"] = "C",
    ):
        """Init method."""
        self._mesh = mesh
        if mass_variances is None:
            self._mass_variances = get_mass_variances(
                primitive, isotope_data=isotope_data
            )
        else:
            self._mass_variances = np.array(mass_variances, dtype="double")
        self._primitive = primitive
        self._sigma = sigma
        self._symprec = symprec
        if cutoff_frequency is None:
            self._cutoff_frequency = 0.0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._lapack_zheev_uplo: Literal["L", "U"] = lapack_zheev_uplo
        self._lang: Literal["C", "Python", "Rust"] = lang
        from phono3py._lang import log_dispatch

        log_dispatch(lang, "Isotope.__init__")
        self._nac_q_direction: NDArray[np.double] | None = None

        self._grid_points: NDArray[np.int64] | None = None
        self._frequencies: NDArray[np.double] | None = None
        self._eigenvectors: NDArray[np.cdouble] | None = None
        self._phonon_done: NDArray[np.byte] | None = None
        self._dm: DynamicalMatrix | None = None
        self._gamma: NDArray[np.double] | None = None
        self._integration_weights: NDArray[np.double] | None = None

        num_band = len(self._primitive) * 3
        self._band_indices: NDArray[np.int64]
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype="int64")
        else:
            self._band_indices = np.array(band_indices, dtype="int64")

        if bz_grid is None:
            primitive_symmetry = Symmetry(self._primitive, self._symprec)
            self._bz_grid = BZGrid(
                self._mesh,
                lattice=self._primitive.cell,
                symmetry_dataset=primitive_symmetry.dataset,
                use_grg=use_grg,
                lang="Rust" if self._lang == "Rust" else "C",
            )
        else:
            self._bz_grid = bz_grid

    def set_grid_point(self, grid_point: int) -> None:
        """Initialize grid points."""
        self._grid_point = grid_point
        self._grid_points = np.arange(len(self._bz_grid.addresses), dtype="int64")  # type: ignore[assignment]

        if self._phonon_done is None:
            self._allocate_phonon()

    def run(self) -> None:
        """Run isotope scattering calculation.

        The backend is selected by ``self._lang`` set at construction.

        """
        if self._lang == "C":
            self._run_c()
        elif self._lang == "Rust":
            self._run_rust()
        else:
            self._run_py()

    @property
    def sigma(self) -> float | None:
        """Setter and getter of smearing width."""
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float | None) -> None:
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

    @property
    def dynamical_matrix(self) -> DynamicalMatrix | None:
        """Return DynamicalMatrix* class instance."""
        return self._dm

    @property
    def band_indices(self) -> NDArray[np.int64]:
        """Return specified band indices."""
        return self._band_indices

    @property
    def gamma(self) -> NDArray[np.double] | None:
        """Return scattering strength."""
        return self._gamma

    @property
    def bz_grid(self) -> BZGrid:
        """Return BZgrid class instance."""
        return self._bz_grid

    @property
    def mass_variances(self) -> NDArray[np.double]:
        """Return mass variances."""
        return self._mass_variances

    def get_phonons(
        self,
    ) -> tuple[
        NDArray[np.double] | None, NDArray[np.cdouble] | None, NDArray[np.byte] | None
    ]:
        """Return phonons on grid."""
        return self._frequencies, self._eigenvectors, self._phonon_done

    def set_phonons(
        self,
        frequencies: NDArray[np.double],
        eigenvectors: NDArray[np.cdouble],
        phonon_done: NDArray[np.byte],
        dm: DynamicalMatrix | None = None,
    ) -> None:
        """Set phonons on grid."""
        self._frequencies = frequencies  # type: ignore[assignment]
        self._eigenvectors = eigenvectors  # type: ignore[assignment]
        self._phonon_done = phonon_done  # type: ignore[assignment]
        if dm is not None:
            self._dm = dm

    def init_dynamical_matrix(
        self,
        fc2: NDArray[np.double],
        supercell: PhonopyAtoms,
        primitive: Primitive,
        nac_params: dict | None = None,
        frequency_scale_factor: float | None = None,
        decimals: int | None = None,
    ) -> None:
        """Initialize dynamical matrix."""
        self._primitive = primitive
        self._dm = get_dynamical_matrix(  # type: ignore[assignment]
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
        )

    def set_nac__qdirection(
        self, nac_q_direction: Sequence[float] | NDArray[np.double] | None = None
    ) -> None:
        """Set q-direction at q->0 used for NAC."""
        self._nac_q_direction = (
            np.array(nac_q_direction, dtype="double")
            if nac_q_direction is not None
            else None
        )

    def _run_c(self) -> None:
        assert self._grid_points is not None

        self._run_phonon_solver_c(self._grid_points)
        import phono3py._phono3py as phono3c  # type: ignore

        gamma = np.zeros(len(self._band_indices), dtype="double")
        weights_in_bzgp = np.ones(len(self._grid_points), dtype="double")
        if self._sigma is None:
            self._set_integration_weights(lang=self._lang)
            phono3c.thm_isotope_strength(
                gamma,
                self._grid_point,
                self._bz_grid.grg2bzg,
                weights_in_bzgp,
                self._mass_variances,
                self._frequencies,
                self._eigenvectors,
                self._band_indices,
                self._integration_weights,
                self._cutoff_frequency,
            )
        else:
            phono3c.isotope_strength(
                gamma,
                self._grid_point,
                self._bz_grid.grg2bzg,
                weights_in_bzgp,
                self._mass_variances,
                self._frequencies,
                self._eigenvectors,
                self._band_indices,
                self._sigma,
                self._cutoff_frequency,
            )

        self._gamma = gamma / np.prod(self._bz_grid.D_diag)

    def _run_rust(self) -> None:
        """Run isotope scattering via the Rust backend.

        Mirrors ``_run_c`` but dispatches to ``phonors``.

        """
        assert self._grid_points is not None

        self._run_phonon_solver_c(self._grid_points)
        import phonors  # type: ignore

        gamma = np.zeros(len(self._band_indices), dtype="double")
        weights_in_bzgp = np.ones(len(self._grid_points), dtype="double")
        if self._sigma is None:
            self._set_integration_weights(lang=self._lang)
            phonors.thm_isotope_strength(
                gamma,
                self._grid_point,
                self._bz_grid.grg2bzg,
                weights_in_bzgp,
                self._mass_variances,
                self._frequencies,
                self._eigenvectors,
                self._band_indices,
                self._integration_weights,
                self._cutoff_frequency,
            )
        else:
            phonors.isotope_strength(
                gamma,
                self._grid_point,
                self._bz_grid.grg2bzg,
                weights_in_bzgp,
                self._mass_variances,
                self._frequencies,
                self._eigenvectors,
                self._band_indices,
                self._sigma,
                self._cutoff_frequency,
            )

        self._gamma = gamma / np.prod(self._bz_grid.D_diag)

    def _set_integration_weights(
        self, lang: Literal["C", "Python", "Rust"] = "C"
    ) -> None:
        if lang == "Python":
            self._set_integration_weights_py()
        else:
            self._set_integration_weights_c(lang=lang)

    def _set_integration_weights_c(self, lang: Literal["C", "Rust"] = "C") -> None:
        """Set tetrahedron method integration weights.

        self._frequencies are those on all BZ-grid. So all those grid points in
        BZ-grid, i.e., self._grid_points, are passed to get_integration_weights.

        """
        assert self._frequencies is not None
        assert self._grid_points is not None

        unique_grid_points = get_unique_grid_points(
            self._grid_points, self._bz_grid, lang=lang
        )
        self._run_phonon_solver_c(unique_grid_points)
        freq_points = np.array(
            self._frequencies[self._grid_point, self._band_indices],
            dtype="double",
            order="C",
        )
        self._integration_weights = get_integration_weights(
            freq_points,
            self._frequencies,
            self._bz_grid,
            grid_points=self._grid_points,
            lang=lang,
        )

    def _set_integration_weights_py(self) -> None:
        """Set tetrahedron method integration weights.

        Python implementation corresponding to _set_integration_weights_c.

        """
        assert self._grid_points is not None
        assert self._frequencies is not None

        thm = TetrahedronMethod(self._bz_grid.microzone_lattice)
        assert thm.tetrahedra is not None

        num_grid_points = len(self._grid_points)
        num_band = len(self._primitive) * 3
        self._integration_weights = np.zeros(
            (num_grid_points, len(self._band_indices), num_band), dtype="double"
        )

        for i, gp in enumerate(self._grid_points):
            tfreqs = get_tetrahedra_frequencies(
                gp,  # In BZ-grid used only to access self._bz_grid.addresses.
                self._bz_grid.D_diag,
                self._bz_grid.addresses,
                np.array(
                    np.dot(thm.tetrahedra, self._bz_grid.P.T),
                    dtype="int64",
                    order="C",
                ),
                self._bz_grid.grg2bzg,
                self._frequencies,
                grid_order=[
                    1,
                    self._bz_grid.D_diag[0],
                    self._bz_grid.D_diag[0] * self._bz_grid.D_diag[1],
                ],
                lang="Python",
            )

            for bi, frequencies in enumerate(tfreqs):
                thm.set_tetrahedra_omegas(frequencies)
                thm.run(self._frequencies[self._grid_point, self._band_indices])
                iw = thm.get_integration_weight()
                self._integration_weights[i, :, bi] = iw

    def _run_py(self) -> None:
        assert self._grid_points is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        for gp in self._grid_points:
            self._run_phonon_solver_py(gp)

        if self._sigma is None:
            self._set_integration_weights(lang=self._lang)

        t_inv = []
        for bi in self._band_indices:
            vec0 = self._eigenvectors[self._grid_point][:, bi].conj()
            f0 = self._frequencies[self._grid_point][bi]
            ti_sum = 0.0
            for gp in self._bz_grid.grg2bzg:
                for j, (f, vec) in enumerate(
                    zip(self._frequencies[gp], self._eigenvectors[gp].T, strict=True)
                ):
                    if f < self._cutoff_frequency:
                        continue
                    ti_sum_band = np.sum(
                        np.abs((vec * vec0).reshape(-1, 3).sum(axis=1)) ** 2
                        * self._mass_variances
                    )
                    if self._sigma is None:
                        assert self._integration_weights is not None
                        ti_sum += ti_sum_band * self._integration_weights[gp, bi, j]
                    else:
                        ti_sum += ti_sum_band * gaussian(f0 - f, self._sigma)
            t_inv.append(np.pi / 2 / np.prod(self._bz_grid.D_diag) * f0**2 * ti_sum)

        self._gamma = np.array(t_inv, dtype="double") / 2

    def _run_phonon_solver_c(self, grid_points: NDArray[np.int64]) -> None:
        assert self._dm is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._phonon_done is not None
        run_phonon_solver_c(
            self._dm,
            self._frequencies,
            self._eigenvectors,
            self._phonon_done,
            grid_points,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            self._frequency_factor_to_THz,
            self._nac_q_direction,
            self._lapack_zheev_uplo,
        )

    def _run_phonon_solver_py(self, grid_point: int) -> None:
        assert self._phonon_done is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._dm is not None
        run_phonon_solver_py(
            grid_point,
            self._phonon_done,
            self._frequencies,
            self._eigenvectors,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            self._dm,
            self._frequency_factor_to_THz,
            self._lapack_zheev_uplo,
        )

    def _allocate_phonon(self) -> None:
        num_band = len(self._primitive) * 3
        num_grid = len(self._bz_grid.addresses)
        self._phonon_done = np.zeros(num_grid, dtype="byte")
        self._frequencies = np.zeros((num_grid, num_band), dtype="double", order="C")
        self._eigenvectors = np.zeros(
            (num_grid, num_band, num_band), dtype="cdouble", order="C"
        )
