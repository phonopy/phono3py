"""Utilities to handle q-point triplets."""

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

from typing import TYPE_CHECKING, Union

import numpy as np
from phonopy.structure.tetrahedron_method import TetrahedronMethod

from phono3py.other.tetrahedron_method import get_tetrahedra_relative_grid_address
from phono3py.phonon.func import gaussian
from phono3py.phonon.grid import BZGrid, get_grid_point_from_address_py

if TYPE_CHECKING:
    from phono3py.phonon3.interaction import Interaction
    from phono3py.phonon3.joint_dos import JointDos


def get_triplets_at_q(
    grid_point,
    bz_grid: BZGrid,
    reciprocal_rotations=None,
    is_time_reversal=True,
    swappable=True,
):
    """Generate q-point triplets.

    Parameters
    ----------
    grid_point : int
        A grid point in the grid type chosen by `store_dense_gp_map`.
    bz_grid : BZGrid
        Data structure to represent BZ grid.
    reciprocal_rotations : array_like or None, optional
        Rotation matrices {R} with respect to reciprocal basis vectors.
        Defined by q'=Rq.
        dtype='int_', shape=(rotations, 3, 3)
    is_time_reversal : bool, optional
        Inversion symemtry is added if it doesn't exist. Default is True.
    swappable : bool, optional
        q1 and q2 among (q0, q1, q2) can be swapped. Deafult is True.

    Returns
    -------
    triplets_at_q : ndarray
        Symmetry reduced number of triplets are stored as grid point
        integer numbers in BZGrid system.
        shape=(n_triplets, 3), dtype='int_'
    weights : ndarray
        Weights of triplets in Brillouin zone
        shape=(n_triplets,), dtype='int_'
    map_triplets : ndarray or None
        Mapping table of all triplets to symmetrically independent
        tripelts in generalized regular grid system. More precisely,
        this gives a list of index mapping from all q-points to
        independent q' of q+q'+q''=G. Considering q' is enough because
        q is fixed and q''=G-q-q' where G is automatically determined
        to choose smallest |G|.
        shape=(prod(mesh),), dtype='int_'
    map_q : ndarray
        Irreducible q-points stabilized by q-point of specified grid_point
        in generalized regular grid system.
        shape=(prod(mesh),), dtype='int_'

    """
    if reciprocal_rotations is None:
        rotations = bz_grid.rotations
    else:
        rotations = reciprocal_rotations

    map_triplets, map_q = _get_triplets_reciprocal_mesh_at_q(
        bz_grid.bzg2grg[grid_point],
        bz_grid.D_diag,
        rotations,
        is_time_reversal=is_time_reversal,
        swappable=swappable,
    )
    triplets_at_q, weights = _get_BZ_triplets_at_q(grid_point, bz_grid, map_triplets)

    assert np.prod(bz_grid.D_diag) == weights.sum(), (
        "Num grid points %d, sum of weight %d"
        % (np.prod(bz_grid.D_diag), weights.sum())
    )

    return triplets_at_q, weights, map_triplets, map_q


def get_all_triplets(grid_point, bz_grid):
    """Return all triplets of a grid point.

    Almost equivalent to ``get_nosym_triplets_at_q``.
    Symmetry reduced set of triplets is obtained by ``get_triplets_at_q``.

    """
    triplets_at_q, _ = _get_BZ_triplets_at_q(
        grid_point, bz_grid, np.arange(np.prod(bz_grid.D_diag), dtype="int_")
    )

    return triplets_at_q


def get_nosym_triplets_at_q(grid_point, bz_grid: BZGrid):
    """Return triplets information without imposing mesh symmetry.

    See the docstring of get_triplets_at_q.

    """
    map_triplets = np.arange(np.prod(bz_grid.D_diag), dtype="int_")
    map_q = np.arange(np.prod(bz_grid.D_diag), dtype="int_")
    triplets_at_q, weights = _get_BZ_triplets_at_q(grid_point, bz_grid, map_triplets)

    return triplets_at_q, weights, map_triplets, map_q


def get_triplets_integration_weights(
    interaction: Union["Interaction", "JointDos"],
    frequency_points,
    sigma,
    sigma_cutoff=None,
    is_collision_matrix=False,
    lang="C",
):
    """Calculate triplets integration weights.

    Returns
    -------
    g : ndarray
        Triplets integration weights.
        shape=(2 or 3, triplets, freq_points, bands, bands), dtype='double'.
    g_zero : ndarray
        Location of strictly zero elements.
        shape=(triplets, freq_points, bands, bands), dtype='byte'

    """
    triplets = interaction.get_triplets_at_q()[0]
    frequencies = interaction.get_phonons()[0]
    num_band = frequencies.shape[1]
    g_zero = None

    if is_collision_matrix:
        g = np.empty(
            (3, len(triplets), len(frequency_points), num_band, num_band),
            dtype="double",
            order="C",
        )
    else:
        g = np.empty(
            (2, len(triplets), len(frequency_points), num_band, num_band),
            dtype="double",
            order="C",
        )
    g[:] = 0

    if sigma:
        if lang == "C":
            import phono3py._phono3py as phono3c

            g_zero = np.zeros(g.shape[1:], dtype="byte", order="C")
            if sigma_cutoff is None:
                cutoff = -1
            else:
                cutoff = float(sigma_cutoff)
            # cutoff < 0 disables g_zero feature.
            phono3c.triplets_integration_weights_with_sigma(
                g, g_zero, frequency_points, triplets, frequencies, sigma, cutoff
            )
        else:
            for i, tp in enumerate(triplets):
                f1s = frequencies[tp[1]]
                f2s = frequencies[tp[2]]
                for j, k in list(np.ndindex((num_band, num_band))):
                    f1 = f1s[j]
                    f2 = f2s[k]
                    g0 = gaussian(frequency_points - f1 - f2, sigma)
                    g[0, i, :, j, k] = g0
                    g1 = gaussian(frequency_points + f1 - f2, sigma)
                    g2 = gaussian(frequency_points - f1 + f2, sigma)
                    g[1, i, :, j, k] = g1 - g2
                    if len(g) == 3:
                        g[2, i, :, j, k] = g0 + g1 + g2
    else:
        if lang == "C":
            g_zero = np.zeros(g.shape[1:], dtype="byte", order="C")
            _set_triplets_integration_weights_c(
                g,
                g_zero,
                interaction,
                frequency_points,
            )
        else:
            _set_triplets_integration_weights_py(g, interaction, frequency_points)

    return g, g_zero


def _get_triplets_reciprocal_mesh_at_q(
    fixed_grid_number, D_diag, rec_rotations, is_time_reversal=True, swappable=True
):
    """Search symmetry reduced triplets fixing one q-point.

    Triplets of (q0, q1, q2) are searched. This method doesn't consider
    translationally equivalent points on BZ surface.

    Parameters
    ----------
    fixed_grid_number : int
        Grid point of q0
    D_diag : array_like
        Diagonal part of the diagonal matrix by SNF. shape=(3,), dtype='int_'
    rec_rotations : array_like
        Rotation matrices in reciprocal space, where the rotation matrix R is
        defined like q'=Rq. Time reversal symmetry is taken care of by
        is_time_reversal. shape=(n_rot, 3, 3), dtype='int_'
    is_time_reversal : bool
        Inversion symemtry is added if it doesn't exist.
    swappable : bool
        q1 and q2 can be swapped. By this number of triplets decreases.

    Returns
    -------
    map_triplets : ndarray or None
        Mapping table of all triplets to symmetrically independent tripelts.
        More precisely, this gives a list of index mapping from all q-points to
        independent q' of q+q'+q''=G. Considering q' is enough because q is
        fixed and q''=G-q-q' where G is automatically determined to choose
        smallest |G| without considering BZ surface (see docstring of
        _get_BZ_triplets_at_q.) shape=(prod(mesh),), dtype='int_'
    map_q : ndarray
        Irreducible q-points stabilized by q-point of specified grid_point.
        shape=(prod(mesh),), dtype='int_'

    """
    import phono3py._phono3py as phono3c

    map_triplets = np.zeros(np.prod(D_diag), dtype="int_")
    map_q = np.zeros(np.prod(D_diag), dtype="int_")

    num_triplets = phono3c.triplets_reciprocal_mesh_at_q(
        map_triplets,
        map_q,
        fixed_grid_number,
        np.array(D_diag, dtype="int_"),
        is_time_reversal * 1,
        np.array(rec_rotations, dtype="int_", order="C"),
        swappable * 1,
    )
    assert num_triplets == len(np.unique(map_triplets))

    return map_triplets, map_q


def _get_BZ_triplets_at_q(bz_grid_index, bz_grid: BZGrid, map_triplets):
    """Grid point triplets are searched considering BZ surface.

    Looking for q+q'+q''=G with smallest |G|. In this condition,
    a pair in (q, q', q'') can be translationally equivalent points.
    This is treated on BZ-grid.

    Note
    ----
    Symmetry information of triplets is encoded in ``map_triplets``.

    Parameters
    ----------
    bz_grid_index : int
        Grid point of q0 as defined in BZ-grid.
    bz_grid : BZGrid
        Data structure to represent BZ grid.
    map_triplets : ndarray or None
        Mapping table of all triplets to symmetrically
        independent tripelts. More precisely, this gives a list of
        index mapping from all q-points to independent q' of
        q+q'+q''=G. Considering q' is enough because q is fixed and
        q''=G-q-q' where G is automatically determined to choose
        smallest |G| without considering BZ surface (see docstring of
        _get_BZ_triplets_at_q.)
        shape=(prod(mesh),), dtype='int_'

    Returns
    -------
    triplets : ndarray
        Symmetry reduced number of triplets are stored as grid point
        integer numbers.
        shape=(n_triplets, 3), dtype='int_'
    ir_weights : ndarray
        Weights of triplets at a fixed q0.
        shape=(n_triplets,), dtype='int_'

    """
    import phono3py._phono3py as phono3c

    weights = np.zeros(len(map_triplets), dtype="int_")
    for g in map_triplets:
        weights[g] += 1
    ir_weights = np.extract(weights > 0, weights)
    triplets = -np.ones((len(ir_weights), 3), dtype="int_")
    num_ir_ret = phono3c.BZ_triplets_at_q(
        triplets,
        bz_grid_index,
        bz_grid.addresses,
        bz_grid.gp_map,
        map_triplets,
        np.array(bz_grid.D_diag, dtype="int_"),
        bz_grid.Q,
        bz_grid.store_dense_gp_map * 1 + 1,
    )

    assert num_ir_ret == len(ir_weights)

    return triplets, np.array(ir_weights, dtype="int_")


def _set_triplets_integration_weights_c(
    g: np.ndarray,
    g_zero: np.ndarray,
    pp: Union["Interaction", "JointDos"],
    frequency_points,
):
    import phono3py._phono3py as phono3c

    tetrahedra = get_tetrahedra_relative_grid_address(pp.bz_grid.microzone_lattice)
    triplets_at_q = pp.get_triplets_at_q()[0]
    frequencies = pp.get_phonons()[0]
    phono3c.triplets_integration_weights(
        g,
        g_zero,
        frequency_points,  # f0
        np.array(np.dot(tetrahedra, pp.bz_grid.P.T), dtype="int_", order="C"),
        pp.bz_grid.D_diag,
        triplets_at_q,
        frequencies,  # f1
        frequencies,  # f2
        pp.bz_grid.addresses,
        pp.bz_grid.gp_map,
        pp.bz_grid.store_dense_gp_map * 1 + 1,
        g.shape[0],
    )


def _set_triplets_integration_weights_py(
    g, pp: Union["Interaction", "JointDos"], frequency_points
):
    """Python version of _set_triplets_integration_weights_c.

    Tetrahedron method engine is that implemented in phonopy written mainly in C.

    """
    thm = TetrahedronMethod(pp.bz_grid.microzone_lattice)
    triplets_at_q = pp.get_triplets_at_q()[0]
    tetrahedra_vertices = _get_tetrahedra_vertices(
        np.array(np.dot(thm.tetrahedra, pp.bz_grid.P.T), dtype="int_", order="C"),
        triplets_at_q,
        pp.bz_grid,
    )
    pp.run_phonon_solver()
    frequencies = pp.get_phonons()[0]
    num_band = frequencies.shape[1]
    for i, vertices in enumerate(tetrahedra_vertices):
        for j, k in list(np.ndindex((num_band, num_band))):
            f1_v = frequencies[vertices[0], j]
            f2_v = frequencies[vertices[1], k]
            thm.set_tetrahedra_omegas(f1_v + f2_v)
            thm.run(frequency_points)
            g0 = thm.get_integration_weight()
            g[0, i, :, j, k] = g0
            thm.set_tetrahedra_omegas(-f1_v + f2_v)
            thm.run(frequency_points)
            g1 = thm.get_integration_weight()
            thm.set_tetrahedra_omegas(f1_v - f2_v)
            thm.run(frequency_points)
            g2 = thm.get_integration_weight()
            g[1, i, :, j, k] = g1 - g2
            if len(g) == 3:
                g[2, i, :, j, k] = g0 + g1 + g2


def _get_tetrahedra_vertices(relative_address, triplets_at_q, bz_grid: BZGrid):
    """Return vertices of tetrahedra used for tetrahedron method.

    Equivalent function is implemented in C and this python version exists
    only for the test and assumes q1+q2+q3=G.

    """
    num_triplets = len(triplets_at_q)
    vertices = np.zeros((num_triplets, 2, 24, 4), dtype="int_")
    for i, tp in enumerate(triplets_at_q):
        for j, adrs_shift in enumerate((relative_address, -relative_address)):
            adrs = bz_grid.addresses[tp[j + 1]] + adrs_shift
            gps = get_grid_point_from_address_py(adrs, bz_grid.D_diag)
            vertices[i, j] = bz_grid.grg2bzg[gps]
    return vertices
