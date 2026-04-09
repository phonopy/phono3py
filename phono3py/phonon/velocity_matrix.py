"""Calculate group velocity matrix."""

# Copyright (C) 2021 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
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
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.phonon.group_velocity import GroupVelocity

LITTLE_GROUP_TOLERANCE = 1e-5


class VelocityMatrix(GroupVelocity):
    """Class to calculate group velocities matrices of phonons.

     v_qjj' = 1/(2*sqrt(omega_qj*omega_qj')) * <e(q,j)|dD/dq|e(q,j')>

    Attributes
    ----------
    velocity_matrices : ndarray

    """

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        q_length: float | None = None,
        rotations_cartesian: NDArray[np.double] | None = None,
        reciprocal_operations: NDArray[np.int64] | None = None,
        frequency_factor_to_THz: float | None = None,
        cutoff_frequency: float = 1e-4,
    ) -> None:
        """Init method.

        See details of parameters at phonopy `GroupVelocity` class.

        ``rotations_cartesian`` is generated from ``reciprocal_operations`` and
        their orders are corresponding.

        """
        self._dynmat: DynamicalMatrix
        self._reciprocal_lattice: NDArray[np.double]
        self._q_length: float | None = None
        self._ddm: DerivativeOfDynamicalMatrix | None
        self._factor: float
        self._cutoff_frequency: float
        self._directions: NDArray[np.double]
        self._perturbation: NDArray[np.double] | None = None

        GroupVelocity.__init__(
            self,
            dynamical_matrix,
            q_length=q_length,
            frequency_factor_to_THz=frequency_factor_to_THz,
            cutoff_frequency=cutoff_frequency,
        )

        self._reciprocal_operations = reciprocal_operations
        self._rotations_cartesian = rotations_cartesian

        self._velocity_matrices = None

    def run(
        self,
        q_points: Sequence[Sequence[float]]
        | Sequence[NDArray[np.double]]
        | NDArray[np.double],
        perturbation: Sequence[float] | NDArray[np.double] | None = None,
    ) -> None:
        """Run group velocity matrix calculate at q-points.

        Calculated group velocities are stored in
        self._velocity_matrices.

        Parameters
        ----------
        q_points : array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : array-like
            Direction in fractional coordinates of reciprocal space.

        """
        self._perturbation = perturbation  # type: ignore[assignment]
        if perturbation is None:
            # Give an random direction to break symmetry
            self._directions[0] = np.array([1, 2, 3])
        else:
            self._directions[0] = np.dot(self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])
        vm = [
            self._calculate_velocity_matrix_at_q(q)
            for q in np.asarray(q_points, dtype="double")
        ]
        self._velocity_matrices = np.array(vm, dtype="cdouble", order="C")

    @property
    def velocity_matrices(self) -> NDArray[np.cdouble] | None:
        """Return group velocity matrices.

        Returns
        -------
        velocity_matrices : ndarray
            shape=(q-points, 3, num_band, num_band), order='C'
            dtype=complex that is "c%d" % (np.dtype('double').itemsize * 2)

        """
        return self._velocity_matrices

    @property
    def group_velocities(
        self,
    ) -> NDArray[np.double] | None:
        """Return group velocities."""
        if self._velocity_matrices is None:
            return None
        shape = self._velocity_matrices.shape
        gv = np.zeros((shape[0], shape[2], 3), dtype="double", order="C")
        for i_q, vm in enumerate(self._velocity_matrices):
            for i, vm_a in enumerate(vm):
                gv[i_q, :, i] = np.diag(vm_a.real)

        return gv

    def _calculate_velocity_matrix_at_q(
        self, q: NDArray[np.double]
    ) -> NDArray[np.cdouble]:

        self._dynmat.run(q)
        dm = self._dynmat.dynamical_matrix
        assert dm is not None
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real  # type: ignore
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
        deg_sets = degenerate_sets(freqs)
        ddms = self._get_dD(q)
        rot_eigvecs = np.zeros_like(eigvecs)

        for deg in deg_sets:
            rot_eigvecs[:, deg] = self._rot_eigsets(ddms, eigvecs[:, deg])
        condition = freqs > self._cutoff_frequency
        freqs = np.where(condition, freqs, 1)
        rot_eigvecs = rot_eigvecs * np.where(condition, 1 / np.sqrt(2 * freqs), 0)

        vm = np.zeros((3,) + eigvecs.shape, dtype="cdouble")
        projector = self._get_projector(q)
        for i, ddm in enumerate(np.einsum("ij,jkl->ikl", projector, ddms[1:])):
            vm[i] += rot_eigvecs.T.conj() @ ddm @ rot_eigvecs

        return self._hermitian_velocity_matrix(vm) * (self._factor**2)

    def _hermitian_velocity_matrix(
        self, vm: NDArray[np.cdouble]
    ) -> NDArray[np.cdouble]:
        vm = (vm + vm.transpose(0, 2, 1).conj()) / 2
        return vm

    def _rot_eigsets(
        self, ddms: NDArray[np.cdouble], eigsets: NDArray[np.cdouble]
    ) -> NDArray[np.cdouble]:
        """Treat degeneracy.

        Eigenvectors of degenerates bands in eigsets are rotated to make
        the velocity analytical in a specified direction (self._directions[0]).

        Parameters
        ----------
        ddms : list of ndarray
            List of delta (derivative or finite difference) of dynamical
            matrices along several q-directions for perturbation.
            shape=(len(self._directions), num_band, num_band), dtype=complex
        eigsets : ndarray
            List of phonon eigenvectors of degenerate bands.
            shape=(num_band, num_degenerates), dtype=complex

        Returns
        -------
        rot_eigvecs : ndarray
            Rotated eigenvectors.
            shape=(num_band, num_degenerates), dtype=complex

        """
        _, eigvecs = np.linalg.eigh(np.dot(eigsets.T.conj(), np.dot(ddms[0], eigsets)))
        rot_eigsets = np.dot(eigsets, eigvecs)

        return rot_eigsets

    def _get_projector(self, q: NDArray[np.double]) -> NDArray[np.double]:
        """Return little group rotations at q."""
        assert (
            self._reciprocal_operations is not None
            and self._rotations_cartesian is not None
        )
        projector = np.zeros((3, 3), dtype="double", order="C")
        order = 0
        for r, r_cart in zip(
            self._reciprocal_operations, self._rotations_cartesian, strict=True
        ):
            q_in_BZ = q - np.rint(q)
            diff = q_in_BZ - np.dot(r, q_in_BZ)
            if (np.abs(diff) < LITTLE_GROUP_TOLERANCE).all():
                projector += r_cart
                order += 1
        return projector / order

    def _get_little_group(self, q: NDArray[np.double]) -> list[NDArray[np.double]]:
        """Return little group rotations at q."""
        assert (
            self._reciprocal_operations is not None
            and self._rotations_cartesian is not None
        )
        rotations = []
        for r, r_cart in zip(
            self._reciprocal_operations, self._rotations_cartesian, strict=True
        ):
            q_in_BZ = q - np.rint(q)
            diff = q_in_BZ - np.dot(r, q_in_BZ)
            if (np.abs(diff) < LITTLE_GROUP_TOLERANCE).all():
                rotations.append(r_cart)
        return rotations
