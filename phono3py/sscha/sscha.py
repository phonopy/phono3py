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

"""SSCHA calculation.

Formulae implemented are based on these papers:

    Ref. 1: Bianco et al. https://doi.org/10.1103/PhysRevB.96.014111
    Ref. 2: Aseginolaza et al. https://doi.org/10.1103/PhysRevB.100.214307

"""

import numpy as np
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.units import VaspToTHz

from phono3py.phonon.func import sigma_squared


def get_sscha_matrices(supercell, force_constants, cutoff_frequency=None):
    """Return instance of DispCorrMatrix.

    This can be used to compute probability distribution of supercell displacements
    as follows. Suppose `disp` is sets of displacements of supercells and the shape is
    `disp.shape == (n_snapshots, n_satom, 3)`.

    ```python
    uu = get_sscha_matrices(supercell, force_constants)
    uu.run(temperature)
    dmat = disp.reshape(n_snapshots, 3 * n_satom)
    vals = -(dmat * np.dot(dmat, uu.upsilon_matrix)).sum(axis=1) / 2
    prob = uu.prefactor * np.exp(vals)
    ```

    """
    sc_ph = SupercellPhonon(supercell, force_constants)
    uu = DispCorrMatrix(sc_ph, cutoff_frequency=cutoff_frequency)
    return uu


class SupercellPhonon:
    """Supercell phonon class.

    Dynamical matrix is created for supercell atoms and solved in real.
    All phonons at commensurate points are folded to those at Gamma point.
    Phonon eigenvectors are represented in real type.

    Attributes
    ----------
    eigenvalues : ndarray
        Phonon eigenvalues of supercell dynamical matrix.
        shape=(3 * num_satom, ), dtype='double', order='C'
    eigenvectors : ndarray
        Phonon eigenvectors of supercell dynamical matrix.
        shape=(3 * num_satom, 3 * num_satom), dtype='double', order='C'
    frequencies : ndarray
        Phonon frequencies of supercell dynamical matrix. Frequency conversion
        factor to THz is multiplied.
        shape=(3 * num_satom, ), dtype='double', order='C'
    force_constants : ndarray
        Supercell force constants. The array shape is different from
        the input force constants.
        shape=(3 * num_satom, 3 * num_satom), dtype='double', order='C'
    supercell : PhonopyAtoms or its derived class
        Supercell.

    """

    def __init__(self, supercell, force_constants, frequency_factor_to_THz=VaspToTHz):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.
        force_constants : array_like
            Second order force constants.
            shape=(num_satom, num_satom, 3, 3), dtype='double', order='C'
        frequency_factor_to_THz : float
            Frequency conversion factor to THz.

        """
        self._supercell = supercell
        N = len(supercell)
        _fc2 = np.array(
            np.transpose(force_constants, axes=[0, 2, 1, 3]), dtype="double", order="C"
        )
        _fc2 = _fc2.reshape((3 * N, 3 * N))
        _fc2 = np.array(_fc2, dtype="double", order="C")
        inv_sqrt_masses = 1.0 / np.repeat(np.sqrt(supercell.masses), 3)
        dynmat = np.array(
            inv_sqrt_masses * (inv_sqrt_masses * _fc2).T, dtype="double", order="C"
        )
        eigvals, eigvecs = np.linalg.eigh(dynmat)
        freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
        freqs *= frequency_factor_to_THz
        self._eigenvalues = np.array(eigvals, dtype="double", order="C")
        self._eigenvectors = np.array(eigvecs, dtype="double", order="C")
        self._frequencies = np.array(freqs, dtype="double", order="C")
        self._force_constants = _fc2

    @property
    def eigenvalues(self):
        """Return eigenvalues."""
        return self._eigenvalues

    @property
    def eigenvectors(self):
        """Return eigenvectors."""
        return self._eigenvectors

    @property
    def frequencies(self):
        """Return frequencies."""
        return self._frequencies

    @property
    def force_constants(self):
        """Return harmonic force cosntants."""
        return self._force_constants

    @property
    def supercell(self):
        """Return supercell."""
        return self._supercell


class DispCorrMatrix:
    """Calculate displacement correlation matrix from supercell phonon.

    Attributes
    ----------
    psi_matrix : ndarray
        Displacement-displacement correlation matrix at temperature.
        Physical unit is [Angstrom^2].
        shape=(3 * num_satom, 3 * num_satom), dtype='double', order='C'
    upsilon_matrix : ndarray
        Inverse displacement-displacement correlation matrix at temperature.
        Physical unit is [1/Angstrom^2].
        shape=(3 * num_satom, 3 * num_satom), dtype='double', order='C'
    supercell_phonon : SupercellPhonon
        Supercell phonon object. Phonons at Gamma point, where
        eigenvectors are not complex type but real type.

    """

    def __init__(self, supercell_phonon, cutoff_frequency=None):
        """Init method.

        Parameters
        ----------
        supercell_phonon : SupercellPhonon
            Supercell phonon object. Phonons at Gamma point, where
            eigenvectors are not complex type but real type.
        cutoff_frequency : float, optional
            Phonons are ignored if they have frequencies less than this value.
            None sets 1e-5.

        """
        self._supercell_phonon = supercell_phonon
        if cutoff_frequency is None:
            self._cutoff_frequency = 1e-5
        else:
            self._cutoff_frequency = cutoff_frequency
        self._psi_matrix = None
        self._upsilon_matrix = None
        self._determinant = None

    def run(self, T):
        """Calculate displacement correlation matrix from supercell phonon.

        N doesn't appear in the computation explicitly because N=1, i.e.,
        the factor is included in supercell eigenvectors.

        """
        freqs = self._supercell_phonon.frequencies
        eigvecs = self._supercell_phonon.eigenvectors
        sqrt_masses = np.repeat(np.sqrt(self._supercell_phonon.supercell.masses), 3)
        inv_sqrt_masses = np.repeat(
            1.0 / np.sqrt(self._supercell_phonon.supercell.masses), 3
        )

        # ignore zero and imaginary frequency modes
        condition = freqs > self._cutoff_frequency
        _freqs = np.where(condition, freqs, 1)
        _a2 = sigma_squared(_freqs, T)
        a2 = np.where(condition, _a2, 0)
        a2_inv = np.where(condition, 1 / _a2, 0)

        matrix = np.dot(a2 * eigvecs, eigvecs.T)
        self._psi_matrix = np.array(
            inv_sqrt_masses * (inv_sqrt_masses * matrix).T, dtype="double", order="C"
        )

        matrix = np.dot(a2_inv * eigvecs, eigvecs.T)
        self._upsilon_matrix = np.array(
            sqrt_masses * (sqrt_masses * matrix).T, dtype="double", order="C"
        )

        self._prefactor = np.sqrt(1 / np.prod(2 * np.pi * np.extract(condition, a2)))

    @property
    def upsilon_matrix(self):
        """Return Upsilon matrix."""
        return self._upsilon_matrix

    @property
    def psi_matrix(self):
        """Return Psi matrix."""
        return self._psi_matrix

    @property
    def supercell_phonon(self):
        """Return SupercellPhonon class instance."""
        return self._supercell_phonon

    @property
    def prefactor(self):
        """Return prefactor of probability distribution."""
        return self._prefactor


class DispCorrMatrixMesh:
    """Calculate upsilon and psi matrix from normal phonon.

    This calculation is similar to the transformation from
    dynamical matrices to force constants. Instead of creating
    dynamcial matrices from eigenvalues and eigenvectors,
    1/a**2 or a**2 and eigenvectors are used, where a is mode length.

    psi_matrix : ndarray
        Displacement-displacement correlation matrix at temperature.
        Physical unit is [Angstrom^2].
        shape=(3 * num_satom, 3 * num_satom), dtype='double', order='C'
    upsilon_matrix : ndarray
        Inverse displacement-displacement correlation matrix at temperature.
        Physical unit is [1/Angstrom^2].
        shape=(3 * num_satom, 3 * num_satom), dtype='double', order='C'
    commensurate_points : ndarray
        Commensurate q-points of transformation matrix from primitive cell
        to supercell.
        shape=(det(transformation_matrix), 3), dtype='double', order='C'.

    """

    def __init__(self, primitive, supercell, cutoff_frequency=1e-5):
        """Init method."""
        self._d2f = DynmatToForceConstants(primitive, supercell, is_full_fc=True)
        self._masses = supercell.masses
        self._cutoff_frequency = cutoff_frequency

        self._psi_matrix = None
        self._upsilon_matrix = None

    @property
    def commensurate_points(self):
        """Return commensurate points."""
        return self._d2f.commensurate_points

    def run(self, frequencies, eigenvectors, T):
        """Calculate displacement correlation matrix from normal phonon results.

        Parameters
        ----------
        frequencies : ndarray
            Supercell phonon frequencies in THz (without 2pi).
            shape=(grid_point, band), dtype='double', order='C'
        eigenvectors : ndarray
            Supercell phonon eigenvectors.
            shape=(grid_point, band, band), dtype='double', order='C'

        """
        condition = frequencies > self._cutoff_frequency
        _freqs = np.where(condition, frequencies, 1)
        _a2 = sigma_squared(_freqs, T)
        a2 = np.where(condition, _a2, 0)
        a2_inv = np.where(condition, 1 / _a2, 0)
        N = len(self._masses)
        shape = (N * 3, N * 3)

        self._d2f.create_dynamical_matrices(a2_inv, eigenvectors)
        self._d2f.run()
        matrix = self._d2f.force_constants
        matrix = np.transpose(matrix, axes=[0, 2, 1, 3]).reshape(shape)
        self._upsilon_matrix = np.array(matrix, dtype="double", order="C")

        self._d2f.create_dynamical_matrices(a2, eigenvectors)
        self._d2f.run()
        matrix = self._d2f.force_constants
        for i, m_i in enumerate(self._masses):
            for j, m_j in enumerate(self._masses):
                matrix[i, j] /= m_i * m_j
        matrix = np.transpose(matrix, axes=[0, 2, 1, 3]).reshape(shape)
        self._psi_matrix = np.array(matrix, dtype="double", order="C")

    @property
    def upsilon_matrix(self):
        """Return Upsilon matrix."""
        return self._upsilon_matrix

    @property
    def psi_matrix(self):
        """Return Psi matrix."""
        return self._psi_matrix


class SecondOrderFC:
    r"""SSCHA second order force constants by ensemble average.

    This class is made just for the test of the ensemble average in
    Ref. 1, and will not be used for usual fc2 calculation.

    \Phi_{ab} = - \sum_{p} \Upsilon_{ap} \left< u^p f_b
                \right>_{tilde{\rho}_{\mathfrak{R},\Phi}}

    Attributes
    ----------
    displacements : ndarray
        shape=(snap_shots, 3 * num_satoms), dtype='double', order='C'
    forces : ndarray
        shape=(snap_shots, 3 * num_satoms), dtype='double', order='C'
    fc2 : ndarray
        shape=(num_satom, num_satom, 3, 3)

    """

    def __init__(
        self,
        displacements,
        forces,
        supercell_phonon,
        cutoff_frequency=1e-5,
        log_level=0,
    ):
        """Init method.

        Parameters
        ----------
        displacements : ndarray
            shape=(snap_shots, num_satom, 3), dtype='double', order='C'
        forces : ndarray
            shape=(snap_shots, num_satom, 3), dtype='double', order='C'
        supercell_phonon : SupercellPhonon
            Supercell phonon object. Phonons at Gamma point, where
            eigenvectors are not complex type but real type.
        cutoff_frequency : float
            Phonons are ignored if they have frequencies less than this value.

        """
        assert displacements.shape == forces.shape
        shape = displacements.shape
        u = np.array(displacements.reshape(shape[0], -1), dtype="double", order="C")
        f = np.array(forces.reshape(shape[0], -1), dtype="double", order="C")
        self._displacements = u
        self._forces = f
        self._uu = DispCorrMatrix(supercell_phonon, cutoff_frequency=cutoff_frequency)
        self._force_constants = supercell_phonon.force_constants
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level

    @property
    def displacements(self):
        """Return input displacements."""
        return self._displacements

    @property
    def forces(self):
        """Return input forces."""
        return self._forces

    @property
    def fc2(self):
        """Return fc2 calculated stochastically."""
        return self._fc2

    def run(self, T=300.0):
        """Calculate fc2 stochastically.

        As displacement correlation matrix <u_a u_b>^-1, two choices exist.

        1. Upsilon matrix
            self._uu.run(T)
            Y = self._uu.upsilon_matrix
        2. From displacements used for the force calculations
            Y = np.linalg.pinv(np.dot(u.T, u) / u.shape[0])

        They should give close results, otherwise self-consistency
        is not reached well. The later seems better agreement with
        least square fit to fc2 under not very good self-consistency
        condition.

        """
        u = self._displacements
        f = self._forces
        Y = np.linalg.pinv(np.dot(u.T, u) / u.shape[0])
        u_inv = np.dot(u, Y)

        if self._log_level:
            nelems = np.prod(u.shape)
            # print("sum u_inv:", u_inv.sum(axis=0) / u.shape[0])
            print("sum all u_inv:", u_inv.sum() / nelems)
            print("rms u_inv:", np.sqrt((u_inv**2).sum() / nelems))
            print("rms u:", np.sqrt((u**2).sum() / nelems))
            print("rms forces:", np.sqrt((self._forces**2).sum() / nelems))
            # print("drift forces:",
            #       self._forces.sum(axis=0) / self._forces.shape[0])

        fc2 = -np.dot(u_inv.T, f) / f.shape[0]
        N = Y.shape[0] // 3
        self._fc2 = np.array(
            np.transpose(fc2.reshape(N, 3, N, 3), axes=[0, 2, 1, 3]),
            dtype="double",
            order="C",
        )


class ThirdOrderFC:
    r"""SSCHA third order force constants.

    Eq. 45a in Ref.1 (See top docstring of this file)

    \Phi_{abc} = - \sum_{pq} \Upsilon_{ap} \Upsilon_{bq}
    \left< u^p u^q \mathfrak{f}_c \right>_{tilde{\rho}_{\mathfrak{R},\Phi}}

    \mathfrak{f}_i = f_i - \left[
    \left< f_i \right>_{\tilde{\rho}_{\mathfrak{R},\Phi}}
    - \sum_j \Phi_{ij}u^j \right]

    Attributes
    ----------
    displacements : ndarray
        shape=(snap_shots, 3 * num_satoms), dtype='double', order='C'
    forces : ndarray
        shape=(snap_shots, 3 * num_satoms), dtype='double', order='C'
    fc3 : ndarray
        shape=(num_satom, num_satom, num_satom, 3, 3, 3)

    """

    def __init__(
        self,
        displacements,
        forces,
        supercell_phonon,
        cutoff_frequency=1e-5,
        log_level=0,
    ):
        """Init method.

        Parameters
        ----------
        displacements : ndarray
            shape=(snap_shots, num_satom, 3), dtype='double', order='C'
        forces : ndarray
            shape=(snap_shots, num_satom, 3), dtype='double', order='C'
        upsilon_matrix : DispCorrMatrix
            Displacement-displacement correlation matrix class instance.
        cutoff_frequency : float
            Phonons are ignored if they have frequencies less than this value.

        """
        assert displacements.shape == forces.shape
        shape = displacements.shape
        u = np.array(displacements.reshape(shape[0], -1), dtype="double", order="C")
        f = np.array(forces.reshape(shape[0], -1), dtype="double", order="C")
        self._displacements = u
        self._forces = f
        self._uu = DispCorrMatrix(supercell_phonon, cutoff_frequency=cutoff_frequency)
        self._force_constants = supercell_phonon.force_constants
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level

        self._drift = u.sum(axis=0) / u.shape[0]
        self._fmat = None
        self._fc3 = None

    @property
    def displacements(self):
        """Return input displacements."""
        return self._displacements

    @property
    def forces(self):
        """Return input forces."""
        return self._forces

    @property
    def fc3(self):
        """Return fc3 calculated stochastically."""
        return self._fc3

    @property
    def ff(self):
        """Return force matrix."""
        return self._fmat

    def run(self, T=300.0):
        """Calculate fc3 stochastically."""
        if self._fmat is None:
            self._fmat = self._run_fmat()

        fc3 = self._run_fc3_ave(T)
        N = fc3.shape[0] // 3
        fc3 = fc3.reshape((N, 3, N, 3, N, 3))
        self._fc3 = np.array(
            np.transpose(fc3, axes=[0, 2, 4, 1, 3, 5]), dtype="double", order="C"
        )

    def _run_fmat(self):
        f = self._forces
        u = self._displacements
        fc2 = self._force_constants
        return f - f.sum(axis=0) / f.shape[0] + np.dot(u, fc2)

    def _run_fc3_ave(self, T):
        # self._uu.run(T)
        # Y = self._uu.upsilon_matrix
        f = self._fmat
        u = self._displacements
        Y = np.linalg.pinv(np.dot(u.T, u) / u.shape[0])

        # This is faster than multiplying Y after ansemble average at least
        # in python implementation.
        u_inv = np.dot(u, Y)

        if self._log_level:
            N = np.prod(u.shape)
            print("rms u_inv:", np.sqrt((u_inv**2).sum() / N))
            print("rms u:", np.sqrt((u**2).sum() / N))
            print("rms forces:", np.sqrt((self._forces**2).sum() / N))
            print("rms f:", np.sqrt((f**2).sum() / N))

        return -np.einsum("li,lj,lk->ijk", u_inv, u_inv, f) / f.shape[0]
