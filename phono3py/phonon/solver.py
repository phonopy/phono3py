"""Create dynamical matrix and solve harmonic phonons on grid."""

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

import numpy as np
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    DynamicalMatrixGL,
    DynamicalMatrixNAC,
)
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import sparse_to_dense_svecs


def run_phonon_solver_c(
    dm,
    frequencies,
    eigenvectors,
    phonon_done,
    grid_points,
    grid_address,
    QDinv,
    frequency_conversion_factor=None,
    nac_q_direction=None,  # in reduced coordinates
    lapack_zheev_uplo="L",
):
    """Build and solve dynamical matrices on grid in C-API.

    Note
    ----
    When LAPACKE is linked in C, `phononcalc.phonons_at_gridpoints` constructs
    and solves dynamical matrices on grid points. Otherwise, it only constructs
    dynamical matrices and solves them in python.

    Parameters
    ----------
    dm : DynamicalMatrix
        DynamicalMatrix instance.
    frequencies, eigenvectors, phonon_done :
        See Interaction.get_phonons().
    grid_points : ndarray
        Grid point indices.
        shape=(grid_points, ), dtype='int64'
    grid_address : ndarray
        See BZGrid.addresses.
    QDinv : ndarray
        See BZGrid.QDinv.
    frequency_conversion_factor : float, optional
        Frequency conversion factor that is multiplied with sqrt or eigenvalue
        of dynamical matrix. Default is VaspToTHz.
    nac_q_direction : array_like, optional
        See Interaction.nac_q_direction. Default is None.
    lapack_zheev_uplo : str, optional
        'U' or 'L' for lapack zheev solver. Default is 'L'.

    """
    import phono3py._phono3py as phono3c  # type: ignore[import-untyped]
    import phono3py._phononcalc as phononcalc  # type: ignore[import-untyped]

    if frequency_conversion_factor is None:
        _frequency_conversion_factor = get_physical_units().DefaultToTHz
    else:
        _frequency_conversion_factor = frequency_conversion_factor

    (
        svecs,
        multi,
        masses,
        rec_lattice,  # column vectors
        positions,
        born,
        nac_factor,
        dielectric,
    ) = _extract_params(dm)

    if isinstance(dm, DynamicalMatrixGL):
        gonze_nac_dataset = dm.Gonze_nac_dataset
        if gonze_nac_dataset[0] is None:
            dm.make_Gonze_nac_dataset()
            gonze_nac_dataset = dm.Gonze_nac_dataset
        (
            gonze_fc,  # fc where the dipole-diple contribution is removed.
            dd_q0,  # second term of dipole-dipole expression.
            G_cutoff,  # Cutoff radius in reciprocal space. This will not be used.
            G_list,  # List of G points where d-d interactions are integrated.
            Lambda,
        ) = gonze_nac_dataset  # Convergence parameter
        assert Lambda is not None
        fc = gonze_fc
        use_GL_NAC = True
    else:
        use_GL_NAC = False
        positions = np.zeros(3)  # dummy variable
        dd_q0 = np.zeros(2)  # dummy variable
        G_list = np.zeros(3)  # dummy variable
        Lambda = 0  # dummy variable
        if not isinstance(dm, DynamicalMatrixNAC):
            born = np.zeros((3, 3))  # dummy variable
            dielectric = np.zeros(3)  # dummy variable
        fc = dm.force_constants

    if nac_q_direction is None:
        is_nac_q_zero = False
        _nac_q_direction = np.zeros(3)
    else:
        is_nac_q_zero = True
        _nac_q_direction = np.array(nac_q_direction, dtype="double")

    assert grid_points.dtype == np.dtype("int64")
    assert grid_points.flags.c_contiguous
    assert QDinv.dtype == np.dtype("double")
    assert QDinv.flags.c_contiguous
    assert lapack_zheev_uplo in ("L", "U")

    if not phono3c.include_lapacke():
        # phonon_done is set even with phono3c.include_lapacke() == 0 for which
        # dynamical matrices are not diagonalized in
        # phononcalc.phonons_at_gridpoints.
        phonon_undone = np.where(phonon_done == 0)[0]

    fc_p2s, fc_s2p = _get_fc_elements_mapping(dm, fc)
    phononcalc.phonons_at_gridpoints(
        frequencies,
        eigenvectors,
        phonon_done,
        grid_points,
        grid_address,
        QDinv,
        fc,
        svecs,
        multi,
        positions,
        masses,
        fc_p2s,
        fc_s2p,
        _frequency_conversion_factor,
        born,
        dielectric,
        rec_lattice,
        _nac_q_direction,
        float(nac_factor),
        dd_q0,
        G_list,
        float(Lambda),
        isinstance(dm, DynamicalMatrixNAC) * 1,
        is_nac_q_zero * 1,
        use_GL_NAC * 1,
        lapack_zheev_uplo,
    )

    if not phono3c.include_lapacke():
        # The variable `eigenvectors` contains dynamical matrices.
        # They are diagonalized in python as follows.
        for gp in phonon_undone:
            frequencies[gp], eigenvectors[gp] = np.linalg.eigh(
                eigenvectors[gp], UPLO=lapack_zheev_uplo
            )
        frequencies[phonon_undone] = (
            np.sign(frequencies[phonon_undone])
            * np.sqrt(np.abs(frequencies[phonon_undone]))
            * _frequency_conversion_factor
        )


def run_phonon_solver_py(
    grid_point,
    phonon_done,
    frequencies,
    eigenvectors,
    grid_address,
    QDinv,
    dynamical_matrix,
    frequency_conversion_factor,
    lapack_zheev_uplo,
):
    """Build and solve dynamical matrices on grid in python."""
    gp = grid_point
    if phonon_done[gp] == 0:
        phonon_done[gp] = 1
        q = np.dot(grid_address[gp], QDinv.T)
        dynamical_matrix.run(q)
        dm = dynamical_matrix.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm, UPLO=lapack_zheev_uplo)
        eigvals = eigvals.real  # type: ignore[no-untyped-call]
        frequencies[gp] = (
            np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * frequency_conversion_factor
        )
        eigenvectors[gp] = eigvecs


def _extract_params(dm: DynamicalMatrix | DynamicalMatrixNAC):
    svecs, multi = dm.primitive.get_smallest_vectors()
    if dm.primitive.store_dense_svecs:
        _svecs = svecs
        _multi = multi
    else:
        _svecs, _multi = sparse_to_dense_svecs(svecs, multi)

    masses = np.array(dm.primitive.masses, dtype="double")
    rec_lattice = np.array(np.linalg.inv(dm.primitive.cell), dtype="double", order="C")
    positions = np.array(dm.primitive.positions, dtype="double", order="C")
    if isinstance(dm, DynamicalMatrixNAC):
        born = dm.born
        nac_factor = dm.nac_factor
        dielectric = dm.dielectric_constant
    else:
        born = None
        nac_factor = 0
        dielectric = None

    return (
        _svecs,
        _multi,
        masses,
        rec_lattice,
        positions,
        born,
        nac_factor,
        dielectric,
    )


def _get_fc_elements_mapping(dm, fc):
    p2s_map = dm.primitive.p2s_map
    s2p_map = dm.primitive.s2p_map
    if fc.shape[0] == fc.shape[1]:  # full fc
        fc_p2s = p2s_map
        fc_s2p = s2p_map
    else:  # compact fc
        primitive = dm.primitive
        p2p_map = primitive.p2p_map
        s2pp_map = np.array(
            [p2p_map[s2p_map[i]] for i in range(len(s2p_map))], dtype="intc"
        )
        fc_p2s = np.arange(len(p2s_map), dtype="intc")
        fc_s2p = s2pp_map

    return np.array(fc_p2s, dtype="int64"), np.array(fc_s2p, dtype="int64")
