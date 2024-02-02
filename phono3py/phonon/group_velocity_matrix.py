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

import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.units import VaspToTHz
from phonopy.utils import similarity_transformation


class GroupVelocityMatrix(GroupVelocity):
    """Class to calculate group velocities matricies of phonons.

     v_qjj' = 1/(2*sqrt(omega_qj*omega_qj')) * <e(q,j)|dD/dq|e(q,j')>

    Attributes
    ----------
    group_velocity_matrices : ndarray

    """

    def __init__(
        self,
        dynamical_matrix,
        q_length=None,
        symmetry=None,
        frequency_factor_to_THz=VaspToTHz,
        cutoff_frequency=1e-4,
    ):
        """Init method.

        See details of parameters at phonopy `GroupVelocity` class.

        """
        self._dynmat = None
        self._reciprocal_lattice = None
        self._q_length = None
        self._ddm = None
        self._symmetry = None
        self._factor = None
        self._cutoff_frequency = None
        self._directions = None
        self._q_points = None
        self._perturbation = None

        GroupVelocity.__init__(
            self,
            dynamical_matrix,
            q_length=q_length,
            symmetry=symmetry,
            frequency_factor_to_THz=frequency_factor_to_THz,
            cutoff_frequency=cutoff_frequency,
        )

        self._group_velocity_matrices = None
        self._complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)

    def run(self, q_points, perturbation=None):
        """Run group velocity matrix calculate at q-points.

        Calculated group velocities are stored in
        self._group_velocity_matrices.

        Parameters
        ----------
        q_points : array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : array-like
            Direction in fractional coordinates of reciprocal space.

        """
        self._q_points = q_points
        self._perturbation = perturbation
        if perturbation is None:
            # Give an random direction to break symmetry
            self._directions[0] = np.array([1, 2, 3])
        else:
            self._directions[0] = np.dot(self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])

        gvm = [self._calculate_group_velocity_matrix_at_q(q) for q in self._q_points]
        self._group_velocity_matrices = np.array(
            gvm, dtype=self._complex_dtype, order="C"
        )

    @property
    def group_velocity_matrices(self):
        """Return group velocity matrices.

        Returns
        -------
        group_velocity_matrices : ndarray
            shape=(q-points, 3, num_band, num_band), order='C'
            dtype=complex that is "c%d" % (np.dtype('double').itemsize * 2)

        """
        return self._group_velocity_matrices

    def _calculate_group_velocity_matrix_at_q(self, q):
        self._dynmat.run(q)
        dm = self._dynmat.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
        deg_sets = degenerate_sets(freqs)
        ddms = self._get_dD(np.array(q))
        rot_eigvecs = np.zeros_like(eigvecs)

        for deg in deg_sets:
            rot_eigvecs[:, deg] = self._rot_eigsets(ddms, eigvecs[:, deg])
        condition = freqs > self._cutoff_frequency
        freqs = np.where(condition, freqs, 1)
        rot_eigvecs = rot_eigvecs * np.where(condition, 1 / np.sqrt(2 * freqs), 0)

        gvm = np.zeros((3,) + eigvecs.shape, dtype=self._complex_dtype)
        for i, ddm in enumerate(ddms[1:]):
            ddm = ddm * (self._factor**2)
            gvm[i] = np.dot(rot_eigvecs.T.conj(), np.dot(ddm, rot_eigvecs))

        if self._perturbation is None:
            if self._symmetry is None:
                return gvm
            else:
                return self._symmetrize_group_velocity_matrix(gvm, q)
        else:
            return gvm

    def _symmetrize_group_velocity_matrix(self, gvm, q):
        """Symmetrize obtained group velocity matrices.

        The following symmetries are applied:
            1. site symmetries
            2. band hermicity

        """
        # site symmetries
        rotations = []
        for r in self._symmetry.reciprocal_operations:
            q_in_BZ = q - np.rint(q)
            diff = q_in_BZ - np.dot(r, q_in_BZ)
            if (np.abs(diff) < self._symmetry.tolerance).all():
                rotations.append(r)

        gvm_sym = np.zeros_like(gvm)
        for r in rotations:
            r_cart = similarity_transformation(self._reciprocal_lattice, r)
            gvm_sym += np.einsum("ij,jkl->ikl", r_cart, gvm)
        gvm_sym = gvm_sym / len(rotations)

        # band hermicity
        gvm_sym = (gvm_sym + gvm_sym.transpose(0, 2, 1).conj()) / 2

        return gvm_sym

    def _rot_eigsets(self, ddms, eigsets):
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
