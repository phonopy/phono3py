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

from typing import Optional, Union

import numpy as np
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.phonon.tetrahedron_mesh import get_tetrahedra_frequencies
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.atoms import isotope_data as phonopy_isotope_data
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.tetrahedron_method import TetrahedronMethod
from phonopy.units import VaspToTHz

from phono3py.other.tetrahedron_method import (
    get_integration_weights,
    get_unique_grid_points,
)
from phono3py.phonon.func import gaussian
from phono3py.phonon.grid import BZGrid
from phono3py.phonon.solver import run_phonon_solver_c, run_phonon_solver_py


def get_mass_variances(
    primitive: Optional[PhonopyAtoms] = None,
    symbols: Optional[Union[list[str], tuple[str]]] = None,
    isotope_data: Optional[dict] = None,
):
    """Calculate mass variances."""
    if primitive is not None:
        _symbols = primitive.symbols
    elif symbols is not None:
        _symbols = symbols
    else:
        raise RuntimeError("primitive or symbols have to be given.")

    _isotope_data = {}
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
        mesh,
        primitive,
        mass_variances=None,  # length of list is num_atom.
        isotope_data=None,
        band_indices=None,
        sigma=None,
        bz_grid=None,
        frequency_factor_to_THz=VaspToTHz,
        use_grg=False,
        symprec=1e-5,
        cutoff_frequency=None,
        lapack_zheev_uplo="L",
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
        self._bz_grid = bz_grid
        self._symprec = symprec
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._nac_q_direction = None

        self._grid_points = None
        self._frequencies = None
        self._eigenvectors = None
        self._phonon_done = None
        self._dm = None
        self._gamma = None
        self._tetrahedron_method = None

        num_band = len(self._primitive) * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype="int_")
        else:
            self._band_indices = np.array(band_indices, dtype="int_")

        if self._bz_grid is None:
            primitive_symmetry = Symmetry(self._primitive, self._symprec)
            self._bz_grid = BZGrid(
                self._mesh,
                lattice=self._primitive.cell,
                symmetry_dataset=primitive_symmetry.dataset,
                use_grg=use_grg,
                store_dense_gp_map=True,
            )

    def set_grid_point(self, grid_point):
        """Initialize grid points."""
        self._grid_point = grid_point
        self._grid_points = np.arange(len(self._bz_grid.addresses), dtype="int_")

        if self._phonon_done is None:
            self._allocate_phonon()

    def run(self, lang="C"):
        """Run isotope scattering calculation."""
        if lang == "C":
            self._run_c()
        else:
            self._run_py()

    @property
    def sigma(self):
        """Setter and getter of smearing width."""
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

    @property
    def dynamical_matrix(self):
        """Return DynamicalMatrix* class instance."""
        return self._dm

    @property
    def band_indices(self):
        """Return specified band indices."""
        return self._band_indices

    @property
    def gamma(self):
        """Return scattering strength."""
        return self._gamma

    @property
    def bz_grid(self):
        """Return BZgrid class instance."""
        return self._bz_grid

    @property
    def mass_variances(self):
        """Return mass variances."""
        return self._mass_variances

    def get_phonons(self):
        """Return phonons on grid."""
        return self._frequencies, self._eigenvectors, self._phonon_done

    def set_phonons(self, frequencies, eigenvectors, phonon_done, dm=None):
        """Set phonons on grid."""
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._phonon_done = phonon_done
        if dm is not None:
            self._dm = dm

    def init_dynamical_matrix(
        self,
        fc2,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        nac_params=None,
        frequency_scale_factor=None,
        decimals=None,
    ):
        """Initialize dynamical matrix."""
        self._primitive = primitive
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
        )

    def set_nac_q_direction(self, nac_q_direction=None):
        """Set q-direction at q->0 used for NAC."""
        if nac_q_direction is not None:
            self._nac_q_direction = np.array(nac_q_direction, dtype="double")

    def _run_c(self):
        self._run_phonon_solver_c(self._grid_points)
        import phono3py._phono3py as phono3c

        gamma = np.zeros(len(self._band_indices), dtype="double")
        weights_in_bzgp = np.ones(len(self._grid_points), dtype="double")
        if self._sigma is None:
            self._set_integration_weights()
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

    def _set_integration_weights(self, lang="C"):
        if lang == "C":
            self._set_integration_weights_c()
        else:
            self._set_integration_weights_py()

    def _set_integration_weights_c(self):
        """Set tetrahedron method integration weights.

        self._frequencies are those on all BZ-grid. So all those grid points in
        BZ-grid, i.e., self._grid_points, are passed to get_integration_weights.

        """
        unique_grid_points = get_unique_grid_points(self._grid_points, self._bz_grid)
        self._run_phonon_solver_c(unique_grid_points)
        freq_points = np.array(
            self._frequencies[self._grid_point, self._band_indices],
            dtype="double",
            order="C",
        )
        self._integration_weights = get_integration_weights(
            freq_points, self._frequencies, self._bz_grid, grid_points=self._grid_points
        )

    def _set_integration_weights_py(self):
        """Set tetrahedron method integration weights.

        Python implementation corresponding to _set_integration_weights_c.

        """
        thm = TetrahedronMethod(self._bz_grid.microzone_lattice)
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
                    dtype="int_",
                    order="C",
                ),
                self._bz_grid.grg2bzg,
                self._frequencies,
                grid_order=[
                    1,
                    self._bz_grid.D_diag[0],
                    self._bz_grid.D_diag[0] * self._bz_grid.D_diag[1],
                ],
                lang="Py",
            )

            for bi, frequencies in enumerate(tfreqs):
                thm.set_tetrahedra_omegas(frequencies)
                thm.run(self._frequencies[self._grid_point, self._band_indices])
                iw = thm.get_integration_weight()
                self._integration_weights[i, :, bi] = iw

    def _run_py(self):
        for gp in self._grid_points:
            self._run_phonon_solver_py(gp)

        if self._sigma is None:
            self._set_integration_weights(lang="Py")

        t_inv = []
        for bi in self._band_indices:
            vec0 = self._eigenvectors[self._grid_point][:, bi].conj()
            f0 = self._frequencies[self._grid_point][bi]
            ti_sum = 0.0
            for gp in self._bz_grid.grg2bzg:
                for j, (f, vec) in enumerate(
                    zip(self._frequencies[gp], self._eigenvectors[gp].T)
                ):
                    if f < self._cutoff_frequency:
                        continue
                    ti_sum_band = np.sum(
                        np.abs((vec * vec0).reshape(-1, 3).sum(axis=1)) ** 2
                        * self._mass_variances
                    )
                    if self._sigma is None:
                        ti_sum += ti_sum_band * self._integration_weights[gp, bi, j]
                    else:
                        ti_sum += ti_sum_band * gaussian(f0 - f, self._sigma)
            t_inv.append(np.pi / 2 / np.prod(self._bz_grid.D_diag) * f0**2 * ti_sum)

        self._gamma = np.array(t_inv, dtype="double") / 2

    def _run_phonon_solver_c(self, grid_points):
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

    def _run_phonon_solver_py(self, grid_point):
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

    def _allocate_phonon(self):
        num_band = len(self._primitive) * 3
        num_grid = len(self._bz_grid.addresses)
        self._phonon_done = np.zeros(num_grid, dtype="byte")
        self._frequencies = np.zeros((num_grid, num_band), dtype="double")
        itemsize = self._frequencies.itemsize
        self._eigenvectors = np.zeros(
            (num_grid, num_band, num_band), dtype=("c%d" % (itemsize * 2))
        )
