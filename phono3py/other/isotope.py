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

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, get_dynamical_matrix
from phonopy.phonon.degeneracy import get_degenerate_ids
from phonopy.phonon.grid import BZGrid
from phonopy.phonon.tetrahedron_method import (
    TetrahedronMethod,
    get_integration_weights,
    get_tetrahedra_frequencies,
    get_tetrahedra_relative_gr_grid_address,
)
from phonopy.structure.atomic_data import get_atomic_data
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phono3py._lang import log_dispatch, resolve_lang
from phono3py.phonon.degeneracy import average_over_degenerate_sets
from phono3py.phonon.func import gaussian
from phono3py.phonon.solver import (
    PhononData,
    run_phonon_solver_c,
    run_phonon_solver_py,
    run_phonon_solver_rust,
    zero_gamma_acoustic_frequencies,
)


def get_unique_grid_points(
    grid_points: NDArray[np.int64],
    bz_grid: BZGrid,
    symmetrize_tetrahedra: bool = False,
    lang: Literal["C", "Rust"] = "Rust",
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
    symmetrize_tetrahedra : bool, optional, default=False
        Use the vertices of the 24 tetrahedra rotated by all the point-group
        operations.

    Returns
    -------
    ndarray
        Unique grid points on tetrahedron vertices around input grid points.
        shape=(unique_grid_points, ), dtype='int64'.

    """
    lang = resolve_lang(lang)
    _grid_points = np.ascontiguousarray(grid_points, dtype="int64")
    relative_grid_address = get_tetrahedra_relative_gr_grid_address(
        bz_grid, symmetrize_tetrahedra=symmetrize_tetrahedra
    )
    unique_vertices = np.array(
        np.unique(relative_grid_address.reshape(-1, 3), axis=0),
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
        symmetrize_tetrahedra: bool = False,
        average_degenerate_weights: bool = False,
        exclude_gamma_acoustic: bool = False,
        lang: Literal["C", "Python", "Rust"] = "Rust",
    ):
        """Init method.

        Parameters
        ----------
        mesh : float or array_like
            Sampling mesh, given as a length, three mesh numbers, or a grid
            matrix. shape=(3,) or (3, 3). It is used only when ``bz_grid`` is
            None.
        primitive : Primitive
            Primitive cell.
        mass_variances : array_like, optional
            Mass variance of each atom in the primitive cell,
            sum_i f_i (1 - m_i / m_ave)^2 over the isotopes i with fraction f_i
            and mass m_i. If None, computed by ``get_mass_variances`` from
            ``isotope_data``. shape=(atoms,), dtype='double'
        isotope_data : dict, optional
            Isotopes of each element, overriding phonopy's data, e.g.,
            ``{"Si": [(28, 27.977, 0.922), (29, 28.976, 0.047),
            (30, 29.974, 0.031)]}`` with (mass number, mass, fraction). Used
            only when ``mass_variances`` is None.
        band_indices : array_like, optional
            Bands at which the scattering rate is calculated. If None, all
            bands. shape=(bands,), dtype='int64'
        sigma : float, optional
            Width of the Gaussian smearing in THz. If None, the tetrahedron
            method is used.
        bz_grid : BZGrid, optional
            Grid in reciprocal space. If None, built from ``mesh`` and the
            symmetry of ``primitive``.
        frequency_factor_to_THz : float, optional
            Factor that converts the phonon frequencies to THz. If None,
            ``get_physical_units().DefaultToTHz``.
        use_grg : bool, optional, default=False
            Use a generalized regular grid when ``bz_grid`` is built here.
        symprec : float, optional, default=1e-5
            Tolerance of the symmetry search when ``bz_grid`` is built here.
        cutoff_frequency : float, optional
            Phonon modes with frequency below this value in THz are left out.
            If None, 0.
        lapack_zheev_uplo : str, optional, default='L'
            'L' or 'U' passed to the LAPACK zheev phonon solver.
        symmetrize_tetrahedra : bool, optional, default=False
            When True, the integration weights of the tetrahedron method are
            averaged over the 24 tetrahedra rotated by all the point-group
            operations. The 24 tetrahedra are cut along one main diagonal, so
            the weights can differ between symmetrically equivalent q-points.
            Averaging removes the difference. Not available with
            ``lang='C'``.
        average_degenerate_weights : bool, optional, default=False
            When True, the integration weights of the tetrahedron method are
            averaged over each set of degenerate bands at the given grid point
            and over each set of degenerate bands at every q'. The tetrahedron
            method lifts the degeneracy of the bands on the tetrahedron
            vertices, so the weights differ among degenerate bands and the
            result depends on the choice of eigenvectors in the degenerate
            subspaces. The scattering rates are also averaged over each set of
            degenerate bands at the given grid point. All bands are calculated
            internally, then ``band_indices`` are selected. Used only with
            the tetrahedron method (``sigma=None``). Since the modes at Gamma
            with frequencies below ``cutoff_frequency`` are left out, use it
            with ``exclude_gamma_acoustic=True`` or a positive
            ``cutoff_frequency`` to remove the dependence on the eigenvectors
            of the acoustic modes at Gamma.
        exclude_gamma_acoustic : bool, optional, default=False
            When True, the frequencies of the three modes at Gamma with the
            smallest absolute values are set to zero after the phonons are
            solved or set. The acoustic modes at Gamma are then zero on every
            platform, instead of small nonzero values from rounding.
        lang : str, optional, default='Rust'
            Backend, 'C', 'Python' or 'Rust'.

        """
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
        self._symmetrize_tetrahedra = symmetrize_tetrahedra
        self._average_degenerate_weights = average_degenerate_weights
        self._exclude_gamma_acoustic = exclude_gamma_acoustic
        if lang in ("C", "Rust"):
            lang = resolve_lang(lang)
        self._lang: Literal["C", "Python", "Rust"] = lang
        log_dispatch(lang, "Isotope.__init__")
        self._nac_q_direction: NDArray[np.double] | None = None

        self._grid_points: NDArray[np.int64] | None = None
        self._phonons: PhononData | None = None
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

        if self._phonons is None:
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
    def symmetrize_tetrahedra(self) -> bool:
        """Return whether tetrahedron weights are averaged over the point group."""
        return self._symmetrize_tetrahedra

    @property
    def average_degenerate_weights(self) -> bool:
        """Return whether tetrahedron weights are averaged over degenerate bands."""
        return self._average_degenerate_weights

    @property
    def exclude_gamma_acoustic(self) -> bool:
        """Return whether the acoustic frequencies at Gamma are set to zero."""
        return self._exclude_gamma_acoustic

    @property
    def mass_variances(self) -> NDArray[np.double]:
        """Return mass variances."""
        return self._mass_variances

    @property
    def phonons(self) -> PhononData | None:
        """Return phonons on grid.

        None before phonons are allocated or set. The arrays in the returned
        PhononData are those used in this instance, not copies.

        """
        return self._phonons

    def get_phonons(
        self,
    ) -> tuple[
        NDArray[np.double] | None, NDArray[np.cdouble] | None, NDArray[np.byte] | None
    ]:
        """Return frequencies, eigenvectors and phonon_done on grid.

        This method is deprecated and will be removed in v5.0. Use the
        ``phonons`` property.

        """
        warnings.warn(
            "get_phonons() is deprecated and will be removed in v5.0. "
            "Use the phonons property.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._phonons is None:
            return None, None, None
        return (
            self._phonons.frequencies,
            self._phonons.eigenvectors,
            self._phonons.phonon_done,
        )

    def set_phonons(
        self,
        phonons: PhononData | NDArray[np.double],
        eigenvectors: NDArray[np.cdouble] | None = None,
        phonon_done: NDArray[np.byte] | None = None,
        dm: DynamicalMatrix | None = None,
    ) -> None:
        """Set phonons on grid.

        The arrays in ``phonons`` are used as they are, not copied.

        Passing frequencies, eigenvectors and phonon_done as separate arrays
        is deprecated and will be removed in v5.0. Pass a PhononData instance.

        Parameters
        ----------
        phonons : PhononData
            Phonons on the BZ grid of this instance. Frequencies when the
            deprecated form is used.
        eigenvectors : ndarray, optional
            Deprecated. Eigenvectors on the BZ grid.
        phonon_done : ndarray, optional
            Deprecated. 1 where phonons are calculated, otherwise 0.
        dm : DynamicalMatrix, optional
            Dynamical matrix used when phonons are solved later. Pass it by
            keyword.

        """
        if not isinstance(phonons, PhononData):
            if eigenvectors is None or phonon_done is None:
                raise TypeError(
                    "set_phonons() takes a PhononData instance, or frequencies, "
                    "eigenvectors and phonon_done."
                )
            warnings.warn(
                "set_phonons(frequencies, eigenvectors, phonon_done) is deprecated "
                "and will be removed in v5.0. Pass a PhononData instance.",
                DeprecationWarning,
                stacklevel=2,
            )
            phonons = PhononData(
                frequencies=phonons,
                eigenvectors=eigenvectors,
                phonon_done=phonon_done,
                degenerate_ids=get_degenerate_ids(phonons),
            )
        self._phonons = phonons
        if dm is not None:
            self._dm = dm
        if self._exclude_gamma_acoustic:
            gp_Gamma = self._bz_grid.gp_Gamma
            zero_gamma_acoustic_frequencies(
                phonons.frequencies, phonons.phonon_done, gp_Gamma
            )
            if gp_Gamma is not None:
                phonons.degenerate_ids[gp_Gamma] = get_degenerate_ids(
                    phonons.frequencies[[gp_Gamma]]
                )[0]

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
            lang="Rust" if self._lang == "Rust" else "C",
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

        self._run_phonon_solver_on_grid(self._grid_points)
        assert self._phonons is not None
        import phono3py._phono3py as phono3c  # type: ignore

        calc_band_indices = self._get_calc_band_indices()
        gamma = np.zeros(len(calc_band_indices), dtype="double")
        weights_in_bzgp = np.ones(len(self._grid_points), dtype="double")
        if self._sigma is None:
            self._set_integration_weights(lang=self._lang)
            phono3c.thm_isotope_strength(
                gamma,
                self._grid_point,
                self._bz_grid.grg2bzg,
                weights_in_bzgp,
                self._mass_variances,
                self._phonons.frequencies,
                self._phonons.eigenvectors,
                calc_band_indices,
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
                self._phonons.frequencies,
                self._phonons.eigenvectors,
                calc_band_indices,
                self._sigma,
                self._cutoff_frequency,
            )

        self._gamma = self._finalize_gamma(gamma / np.prod(self._bz_grid.D_diag))

    def _run_rust(self) -> None:
        """Run isotope scattering via the Rust backend.

        Mirrors ``_run_c`` but dispatches to ``phonors``.

        """
        assert self._grid_points is not None

        self._run_phonon_solver_on_grid(self._grid_points)
        assert self._phonons is not None
        import phonors  # type: ignore

        calc_band_indices = self._get_calc_band_indices()
        gamma = np.zeros(len(calc_band_indices), dtype="double")
        weights_in_bzgp = np.ones(len(self._grid_points), dtype="double")
        if self._sigma is None:
            self._set_integration_weights(lang=self._lang)
            phonors.thm_isotope_strength(
                gamma,
                self._grid_point,
                self._bz_grid.grg2bzg,
                weights_in_bzgp,
                self._mass_variances,
                self._phonons.frequencies,
                self._phonons.eigenvectors,
                calc_band_indices,
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
                self._phonons.frequencies,
                self._phonons.eigenvectors,
                calc_band_indices,
                self._sigma,
                self._cutoff_frequency,
            )

        self._gamma = self._finalize_gamma(gamma / np.prod(self._bz_grid.D_diag))

    def _set_integration_weights(
        self, lang: Literal["C", "Python", "Rust"] = "Rust"
    ) -> None:
        if lang == "Python":
            self._set_integration_weights_py()
        else:
            self._set_integration_weights_native(lang=lang)

    def _set_integration_weights_native(
        self, lang: Literal["C", "Rust"] = "Rust"
    ) -> None:
        """Set tetrahedron method integration weights.

        The frequencies are those on all BZ-grid. So all those grid points in
        BZ-grid, i.e., self._grid_points, are passed to get_integration_weights.

        """
        assert self._phonons is not None
        assert self._grid_points is not None

        unique_grid_points = get_unique_grid_points(
            self._grid_points,
            self._bz_grid,
            symmetrize_tetrahedra=self._symmetrize_tetrahedra,
            lang=lang,
        )
        self._run_phonon_solver_on_grid(unique_grid_points)
        freq_points = np.array(
            self._phonons.frequencies[self._grid_point, self._get_calc_band_indices()],
            dtype="double",
            order="C",
        )
        self._integration_weights = self._finalize_integration_weights(
            get_integration_weights(
                freq_points,
                self._phonons.frequencies,
                self._bz_grid,
                grid_points=self._grid_points,
                lang=lang,
                symmetrize_tetrahedra=self._symmetrize_tetrahedra,
            )
        )

    def _set_integration_weights_py(self) -> None:
        """Set tetrahedron method integration weights.

        Python implementation corresponding to _set_integration_weights_native.

        """
        assert self._grid_points is not None
        assert self._phonons is not None

        relative_grid_address = get_tetrahedra_relative_gr_grid_address(
            self._bz_grid, symmetrize_tetrahedra=self._symmetrize_tetrahedra
        )
        thm = TetrahedronMethod(None, relative_grid_address=relative_grid_address)

        num_grid_points = len(self._grid_points)
        num_band = len(self._primitive) * 3
        calc_band_indices = self._get_calc_band_indices()
        integration_weights = np.zeros(
            (num_grid_points, len(calc_band_indices), num_band), dtype="double"
        )

        for i, gp in enumerate(self._grid_points):
            tfreqs = get_tetrahedra_frequencies(
                gp, self._bz_grid, relative_grid_address, self._phonons.frequencies
            )

            for bi, frequencies in enumerate(tfreqs):
                thm.set_tetrahedra_omegas(frequencies)
                thm.run(self._phonons.frequencies[self._grid_point, calc_band_indices])
                iw = thm.get_integration_weight()
                integration_weights[i, :, bi] = iw

        self._integration_weights = self._finalize_integration_weights(
            integration_weights
        )

    def _get_calc_band_indices(self) -> NDArray[np.int64]:
        """Return bands at which gamma and integration weights are calculated.

        When band_indices include degenerate bands at the given grid point,
        all bands are needed to average over their degenerate sets. The
        results are then reduced to band_indices in ``_finalize_gamma``.

        """
        if self._average_over_initial_bands():
            return np.arange(len(self._primitive) * 3, dtype="int64")
        return self._band_indices

    def _average_over_initial_bands(self) -> bool:
        """Return whether band_indices include degenerate bands to average."""
        if not self._averaging:
            return False
        assert self._phonons is not None
        degenerate_ids = self._phonons.degenerate_ids[self._grid_point]
        # bincount gives the size of each degenerate set at its smallest band
        # index, which is the value of degenerate_ids.
        set_sizes = np.bincount(degenerate_ids)[degenerate_ids[self._band_indices]]
        return bool((set_sizes > 1).any())

    @property
    def _averaging(self) -> bool:
        """Return whether degenerate bands are averaged."""
        return self._average_degenerate_weights and self._sigma is None

    def _finalize_integration_weights(
        self, integration_weights: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Average weights over degenerate bands.

        Parameters
        ----------
        integration_weights : ndarray
            Weights at the bands of ``_get_calc_band_indices``.
            shape=(grid_points, bands0, bands), dtype='double'

        """
        if not self._averaging:
            return integration_weights

        assert self._phonons is not None
        assert self._grid_points is not None
        degenerate_ids = self._phonons.degenerate_ids

        # Degenerate sets of the given grid point.
        if self._average_over_initial_bands():
            integration_weights = average_over_degenerate_sets(
                integration_weights, degenerate_ids[self._grid_point], 1
            )
        # Degenerate sets at every q'.
        for i, gp in enumerate(self._grid_points):
            integration_weights[i] = average_over_degenerate_sets(
                integration_weights[i], degenerate_ids[gp], 1
            )
        return integration_weights

    def _finalize_gamma(self, gamma: NDArray[np.double]) -> NDArray[np.double]:
        """Average gamma over degenerate bands and select band_indices.

        Nothing is done when band_indices have no degenerate bands.

        With the weights equal among the degenerate bands, the sum of gamma
        over a degenerate set does not depend on the choice of eigenvectors,
        but gamma of each band does. Gamma is therefore averaged over each
        set.

        """
        if not self._average_over_initial_bands():
            return gamma

        assert self._phonons is not None
        gamma = average_over_degenerate_sets(
            gamma, self._phonons.degenerate_ids[self._grid_point], 0
        )
        return np.array(gamma[self._band_indices])

    def _run_py(self) -> None:
        assert self._grid_points is not None
        assert self._phonons is not None

        for gp in self._grid_points:
            self._run_phonon_solver_py(gp)

        if self._sigma is None:
            self._set_integration_weights(lang=self._lang)

        t_inv = []
        for ib, bi in enumerate(self._get_calc_band_indices()):
            vec0 = self._phonons.eigenvectors[self._grid_point][:, bi].conj()
            f0 = self._phonons.frequencies[self._grid_point][bi]
            ti_sum = 0.0
            for gp in self._bz_grid.grg2bzg:
                for j, (f, vec) in enumerate(
                    zip(
                        self._phonons.frequencies[gp],
                        self._phonons.eigenvectors[gp].T,
                        strict=True,
                    )
                ):
                    if f < self._cutoff_frequency:
                        continue
                    ti_sum_band = np.sum(
                        np.abs((vec * vec0).reshape(-1, 3).sum(axis=1)) ** 2
                        * self._mass_variances
                    )
                    if self._sigma is None:
                        assert self._integration_weights is not None
                        ti_sum += ti_sum_band * self._integration_weights[gp, ib, j]
                    else:
                        ti_sum += ti_sum_band * gaussian(f0 - f, self._sigma)
            t_inv.append(np.pi / 2 / np.prod(self._bz_grid.D_diag) * f0**2 * ti_sum)

        self._gamma = self._finalize_gamma(np.array(t_inv, dtype="double") / 2)

    def _run_phonon_solver_on_grid(self, grid_points: NDArray[np.int64]) -> None:
        assert self._dm is not None
        assert self._phonons is not None
        solver = run_phonon_solver_rust if self._lang == "Rust" else run_phonon_solver_c
        solver(
            self._dm,
            self._phonons.frequencies,
            self._phonons.eigenvectors,
            self._phonons.phonon_done,
            grid_points,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            self._frequency_factor_to_THz,
            self._nac_q_direction,
            self._lapack_zheev_uplo,
            exclude_gamma_acoustic=self._exclude_gamma_acoustic,
            degenerate_ids=self._phonons.degenerate_ids,
        )

    def _run_phonon_solver_py(self, grid_point: int) -> None:
        assert self._phonons is not None
        assert self._dm is not None
        run_phonon_solver_py(
            grid_point,
            self._phonons.phonon_done,
            self._phonons.frequencies,
            self._phonons.eigenvectors,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            self._dm,
            self._frequency_factor_to_THz,
            self._lapack_zheev_uplo,
            exclude_gamma_acoustic=self._exclude_gamma_acoustic,
            degenerate_ids=self._phonons.degenerate_ids,
        )

    def _allocate_phonon(self) -> None:
        self._phonons = PhononData.allocate(
            len(self._bz_grid.addresses), len(self._primitive) * 3
        )
