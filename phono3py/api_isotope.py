"""API for isotope scattering."""

# Copyright (C) 2019 Atsushi Togo
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
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive

from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import BZGrid


class Phono3pyIsotope:
    """Class to calculate isotope scattering."""

    def __init__(
        self,
        mesh: Sequence[int] | NDArray[np.int64],
        primitive: Primitive,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        band_indices: Sequence[int] | NDArray[np.int64] | None = None,
        sigmas: Sequence[float | None] | None = None,
        frequency_factor_to_THz: float | None = None,
        use_grg: bool = False,
        symprec: float = 1e-5,
        cutoff_frequency: float | None = None,
        lapack_zheev_uplo: Literal["L", "U"] = "L",
    ) -> None:
        """Init method."""
        self._sigmas: Sequence[float | None]
        if sigmas is None:
            self._sigmas = [None]
        else:
            self._sigmas = sigmas

        self._iso = Isotope(
            mesh,
            primitive,
            mass_variances=mass_variances,
            band_indices=band_indices,
            frequency_factor_to_THz=frequency_factor_to_THz,
            use_grg=use_grg,
            symprec=symprec,
            cutoff_frequency=cutoff_frequency,
            lapack_zheev_uplo=lapack_zheev_uplo,
        )

    @property
    def dynamical_matrix(self) -> DynamicalMatrix | None:
        """Return dynamical matrix class instance."""
        return self._iso.dynamical_matrix

    @property
    def grid(self) -> BZGrid:
        """Return BZGrid class instance."""
        return self._iso.bz_grid

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return calculated isotope scattering."""
        return self._gamma

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies at grid points."""
        return self._iso.get_phonons()[0][self._grid_points]  # type: ignore[index]

    @property
    def grid_points(self) -> NDArray[np.int64]:
        """Return grid points in BZ-grid."""
        return self._grid_points

    def run(
        self,
        grid_points: Sequence[int] | NDArray[np.int64],
        lang: Literal["C", "Python"] = "C",
    ) -> None:
        """Calculate isotope scattering."""
        gamma = np.zeros(
            (len(self._sigmas), len(grid_points), len(self._iso.band_indices)),
            dtype="double",
        )
        self._grid_points = np.array(grid_points, dtype="int64")

        for j, gp in enumerate(grid_points):
            self._iso.set_grid_point(gp)

            print("--------------- Isotope scattering ---------------")
            print("Grid point: %d" % gp)
            adrs = self._iso.bz_grid.addresses[gp]
            bz_grid = self._iso.bz_grid
            print(bz_grid.D_diag)
            q = np.dot(adrs.astype("double") / bz_grid.D_diag, bz_grid.Q.T)
            print("q-point: %s" % q)

            if self._sigmas:
                for i, sigma in enumerate(self._sigmas):
                    if sigma is None:
                        print("Tetrahedron method")
                    else:
                        print("Sigma: %s" % sigma)
                    self._iso.sigma = sigma
                    self._iso.run(lang=lang)
                    gamma[i, j] = self._iso.gamma
                    frequencies = self._iso.get_phonons()[0]
                    print("")
                    print("Phonon-isotope scattering rate in THz (1/4pi-tau)")
                    print(" Frequency     Rate")
                    for g, f in zip(self._iso.gamma, frequencies[gp], strict=True):  # type: ignore[index, arg-type]
                        print("%8.3f     %5.3e" % (f, g))
            else:
                print("sigma or tetrahedron method has to be set.")
        self._gamma = gamma

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
        self._iso.init_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
        )

    def set_sigma(self, sigma: float | None) -> None:
        """Set sigma. None means tetrahedron method."""
        self._iso.sigma = sigma
