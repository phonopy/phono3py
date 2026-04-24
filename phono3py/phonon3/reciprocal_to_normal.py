"""Transform fc3 in reciprocal space to phonon space."""

# Copyright (C) 2020 Atsushi Togo
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

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms


def run_reciprocal_to_normal_squared_rust(
    fc3_reciprocal: NDArray[np.cdouble],
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    masses: NDArray[np.double],
    band_indices: NDArray[np.int64],
    g_pos: NDArray[np.int64],
    cutoff_frequency: float,
    n_out: int,
) -> NDArray[np.double]:
    """Compute ``|fc3_normal|^2 / (f0*f1*f2)`` at a triplet using the Rust backend.

    ``fc3_reciprocal`` uses the atom-first layout
    ``(num_patom, num_patom, num_patom, 3, 3, 3)`` produced by
    ``run_real_to_reciprocal_rust``.  ``frequencies`` has shape
    ``(3, num_band)`` and ``eigenvectors`` has shape
    ``(3, num_band, num_band)``; each row selects the quantity at one
    vertex of the triplet.  ``eigenvectors`` is indexed as
    ``[component, band]`` (un-scaled; mass scaling is applied inside).

    ``g_pos`` has shape ``(num_g_pos, 4)`` with columns
    ``(i0, j, k, dest)`` where ``i0`` indexes ``band_indices`` and
    ``dest`` is the flat offset in the output.  Output length is
    ``n_out``; entries not touched by ``g_pos`` are zero.

    """
    import phono3py_rs

    fc3_normal_squared = np.zeros(n_out, dtype="double")
    f0, f1, f2 = np.ascontiguousarray(frequencies, dtype="double")
    e0, e1, e2 = (np.ascontiguousarray(e, dtype="complex128") for e in eigenvectors)

    phono3py_rs.reciprocal_to_normal_squared(
        fc3_normal_squared,
        np.ascontiguousarray(g_pos, dtype="int64"),
        np.ascontiguousarray(fc3_reciprocal, dtype="complex128"),
        f0,
        f1,
        f2,
        e0,
        e1,
        e2,
        np.ascontiguousarray(masses, dtype="double"),
        np.ascontiguousarray(band_indices, dtype="int64"),
        float(cutoff_frequency),
    )
    return fc3_normal_squared


class ReciprocalToNormal:
    """Class to transform fc3 in reciprocal space to phonon space.

    This is an implementation in python for prototyping and the test.
    Equivalent routine is implemented in C, and this is what usually
    we use.

    """

    def __init__(
        self,
        primitive: PhonopyAtoms,
        frequencies: NDArray[np.double],
        eigenvectors: NDArray[np.cdouble],
        band_indices: Sequence[int] | NDArray[np.int64],
        cutoff_frequency: float = 0,
    ) -> None:
        """Init method."""
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._band_indices = band_indices
        self._cutoff_frequency = cutoff_frequency

        self._fc3_normal: NDArray[np.cdouble]
        self._fc3_reciprocal: NDArray[np.cdouble]

    def run(
        self,
        fc3_reciprocal: NDArray[np.cdouble],
        grid_triplet: NDArray[np.int64],
    ) -> None:
        """Calculate fc3 in phonon coordinates."""
        num_band = len(self._primitive) * 3
        self._fc3_reciprocal = fc3_reciprocal
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._fc3_normal = np.zeros(
            (len(self._band_indices), num_band, num_band), dtype=dtype
        )
        self._reciprocal_to_normal(grid_triplet)

    def get_reciprocal_to_normal(self) -> NDArray[np.cdouble] | None:
        """Return fc3 in phonon coordinates."""
        return self._fc3_normal

    def _reciprocal_to_normal(self, grid_triplet: NDArray[np.int64]) -> None:
        e1, e2, e3 = self._eigenvectors[grid_triplet]
        f1, f2, f3 = self._frequencies[grid_triplet]
        num_band = len(f1)
        cutoff = self._cutoff_frequency
        for i, j, k in list(np.ndindex(len(self._band_indices), num_band, num_band)):
            bi = int(self._band_indices[i])
            if f1[bi] > cutoff and f2[j] > cutoff and f3[k] > cutoff:
                fc3_elem = self._sum_in_atoms((bi, j, k), (e1, e2, e3))
                fff = np.sqrt(f1[bi] * f2[j] * f3[k])
                self._fc3_normal[i, j, k] = fc3_elem / fff

    def _sum_in_atoms(
        self,
        band_indices: tuple[int, int, int],
        eigvecs: tuple[NDArray[np.cdouble], NDArray[np.cdouble], NDArray[np.cdouble]],
    ) -> complex:
        num_atom = len(self._primitive)
        (e1, e2, e3) = eigvecs
        (b1, b2, b3) = band_indices

        sum_fc3 = 0j
        for i, j, k in list(np.ndindex((num_atom,) * 3)):
            sum_fc3_cart = 0
            for ll, m, n in list(np.ndindex((3, 3, 3))):
                sum_fc3_cart += (
                    e1[i * 3 + ll, b1]
                    * e2[j * 3 + m, b2]
                    * e3[k * 3 + n, b3]
                    * self._fc3_reciprocal[i, j, k, ll, m, n]
                )
            mass_sqrt = np.sqrt(np.prod(self._primitive.masses[[i, j, k]]))
            sum_fc3 += sum_fc3_cart / mass_sqrt

        return sum_fc3
