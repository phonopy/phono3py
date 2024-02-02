"""Velocity operator of Simoncelli, Marzari, and Mauri."""

# Copyright (C) 2013 Atsushi Togo
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
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.units import VaspToTHz


class VelocityOperator(GroupVelocity):
    """Class to calculate velocity operator of phonons."""

    def __init__(
        self,
        dynamical_matrix,
        q_length=None,
        symmetry=None,
        frequency_factor_to_THz=VaspToTHz,
        cutoff_frequency=1e-4,
    ):
        """Init method.

        dynamical_matrix : DynamicalMatrix or DynamicalMatrixNAC
            Dynamical matrix class instance.
        q_length : float
            This is used such as D(q + q_length) - D(q - q_length) for
            calculating finite difference of dynamical matrix.
            Default is None, which gives 1e-5.
        symmetry : Symmetry
            This is used to symmetrize group velocity at each q-points.
            Default is None, which means no symmetrization.
        frequency_factor_to_THz : float
            Unit conversion factor to convert to THz. Default is VaspToTHz.
        cutoff_frequency : float
            Group velocity is set zero if phonon frequency is below this value.

        """
        self._dynmat = dynamical_matrix
        primitive = dynamical_matrix.primitive
        self._reciprocal_lattice_inv = primitive.cell
        self._reciprocal_lattice = np.linalg.inv(self._reciprocal_lattice_inv)
        self._q_length = q_length
        if self._q_length is None:
            self._q_length = 5e-6
        self._symmetry = symmetry
        self._factor = frequency_factor_to_THz
        self._cutoff_frequency = cutoff_frequency

        self._directions = np.array(
            [
                [
                    1,
                    2,
                    3,
                ],
                # this is a random direction, not used and left here for historical
                # reasons.
                [1, 0, 0],  # x
                [0, 1, 0],  # y
                [0, 0, 1],
            ],
            dtype="double",
        )  # z
        self._directions[0] /= np.linalg.norm(
            self._directions[0]
        )  # normalize the random direction

        self._q_points = None
        self._velocity_operators = None
        self._perturbation = None

    def run(self, q_points, perturbation=None):
        """Velocity operators are computed at q-points.

        q_points : Array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : Array-like
            Direction in fractional coordinates of reciprocal space.

        """
        self._q_points = q_points
        self._perturbation = perturbation
        if perturbation is None:
            # Give a random direction to break symmetry
            self._directions[0] = np.array([1, 0, 0])  # normalized later
        else:
            self._directions[0] = np.dot(self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])

        gv_operator = [
            self._calculate_velocity_operator_at_q(q) for q in self._q_points
        ]
        self._velocity_operators = np.array(gv_operator, dtype="complex", order="C")

    @property
    def velocity_operators(self):
        """Return velocity operators."""
        return self._velocity_operators

    def _calculate_velocity_operator_at_q(self, q):
        self._dynmat.run(q)
        dm = self._dynmat.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
        nat3 = len(freqs)
        gv_operator = np.zeros((nat3, nat3, 3), dtype="complex", order="C")
        #
        # compute the finite differences along several directions
        # ddms is a list of FD derivatives of the dynamical matrix
        # computed along several directions
        ddms = self._get_dsqrtD_FD(np.array(q))
        #
        # ddms[0] cointains the FD derivative in the direction in which the velocity
        # operator is diagonalized
        for id_dir in range(0, 3):
            gv_operator[:, :, id_dir] = (
                np.matmul(eigvecs.T.conj(), np.matmul(ddms[id_dir + 1], eigvecs))
                * self._factor
            )
            #
            # enforce the velocity operator to be Hermitian
            gv_operator[:, :, id_dir] = (
                gv_operator[:, :, id_dir].T.conj() + gv_operator[:, :, id_dir]
            ) / 2.0

        return gv_operator

    def _get_dsqrtD_FD(self, q):
        """Compute finite difference of sqrt of dynamical matrices."""
        #
        ddm = []
        # _q_length is a float specifying the modulus of the q-vector displacement
        # used to compute the finite differences
        for dqc in self._directions * self._q_length:
            dq = np.dot(self._reciprocal_lattice_inv, dqc)
            ddm.append(
                self._delta_sqrt_dynamical_matrix(q, dq, self._dynmat)
                / self._q_length
                / 2
            )
        return np.array(ddm)

    def _sqrt_dynamical_matrix(self, flag_gamma, dm):
        # returns the sqrt of the dynamical matrix in the cartesian basis
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        # will give NaN in case of negative frequencies
        freqs = np.sign(eigvals) * np.sqrt(abs(eigvals))
        # print('raw=',freqs)
        #
        # enforce eigenvalues and eigenvectors to zero for acoustic modes at Gamma point
        if flag_gamma:
            freqs[0] = 0.0
            eigvecs[:, 0] = 0.0
            freqs[1] = 0.0
            eigvecs[:, 1] = 0.0
            freqs[2] = 0.0
            eigvecs[:, 2] = 0.0
        #
        if any(f < 0.0 for f in freqs):
            print("ERROR: negative frequency=", freqs)
        # this is a diagonal matrix containing the frequencies on the diagonal
        # we take the element-wise sqrt of a diagonal matrix, in eigenmodes basis
        omega_matrix = np.sqrt(np.matmul(eigvecs.T.conj(), np.matmul(dm, eigvecs)))
        omega_matrix = np.diag(np.diag(omega_matrix))
        # now we go back to the Cartesian basis
        sqrt_dm = np.matmul(eigvecs, np.matmul(omega_matrix, eigvecs.T.conj()))
        #
        return sqrt_dm

    def _delta_sqrt_dynamical_matrix(self, q, delta_q, dynmat):
        #
        flag_gamma = False
        if np.linalg.norm(q) < np.linalg.norm(delta_q):
            flag_gamma = True

        if (
            (self._dynmat.is_nac())
            and (self._dynmat.nac_method == "gonze")
            and flag_gamma
        ):
            dynmat.run(
                q - delta_q, q_direction=(q - delta_q) / np.linalg.norm(q - delta_q)
            )
            dm1 = dynmat.dynamical_matrix
            sqrt_dm1 = self._sqrt_dynamical_matrix(flag_gamma, dm1)
            dynmat.run(
                q + delta_q, q_direction=(q + delta_q) / np.linalg.norm(q + delta_q)
            )
            dm2 = dynmat.dynamical_matrix
            sqrt_dm2 = self._sqrt_dynamical_matrix(flag_gamma, dm2)
        else:
            dynmat.run(q - delta_q)
            dm1 = dynmat.dynamical_matrix
            sqrt_dm1 = self._sqrt_dynamical_matrix(flag_gamma, dm1)
            dynmat.run(q + delta_q)
            dm2 = dynmat.dynamical_matrix
            sqrt_dm2 = self._sqrt_dynamical_matrix(flag_gamma, dm2)
        #
        #
        return sqrt_dm2 - sqrt_dm1
