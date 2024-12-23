"""Tetrahedron method python wrapper."""

# Copyright (C) 2021 Atsushi Togo
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

import numpy as np
from phonopy.structure.tetrahedron_method import TetrahedronMethod

from phono3py.phonon.grid import BZGrid


def get_unique_grid_points(grid_points, bz_grid: BZGrid):
    """Collect grid points on tetrahedron vertices around input grid points.

    Find grid points of 24 tetrahedra around each grid point and
    collect those grid points that are unique.

    Parameters
    ----------
    grid_points : array_like
        Grid point indices.
    bz_grid : BZGrid
        Grid information in reciprocal space.

    Returns
    -------
    ndarray
        Unique grid points on tetrahedron vertices around input grid points.
        shape=(unique_grid_points, ), dtype='long'.

    """
    import phono3py._phono3py as phono3c

    if _check_ndarray_state(grid_points, "long"):
        _grid_points = grid_points
    else:
        _grid_points = np.array(grid_points, dtype="long")
    thm = TetrahedronMethod(bz_grid.microzone_lattice)
    unique_vertices = np.array(
        np.dot(thm.get_unique_tetrahedra_vertices(), bz_grid.P.T),
        dtype="long",
        order="C",
    )
    neighboring_grid_points = np.zeros(
        len(unique_vertices) * len(_grid_points), dtype="long"
    )
    phono3c.neighboring_grid_points(
        neighboring_grid_points,
        _grid_points,
        unique_vertices,
        bz_grid.D_diag,
        bz_grid.addresses,
        bz_grid.gp_map,
        bz_grid.store_dense_gp_map * 1 + 1,
    )
    unique_grid_points = np.array(np.unique(neighboring_grid_points), dtype="long")
    return unique_grid_points


def get_integration_weights(
    sampling_points,
    grid_values,
    bz_grid: BZGrid,
    grid_points=None,
    bzgp2irgp_map=None,
    function="I",
):
    """Return tetrahedron method integration weights.

    Parameters
    ----------
    sampling_points : array_like
        Values at which the integration weights are computed.
        shape=(sampling_points, ), dtype='double'
    grid_values : array_like
        Values of tetrahedron vertices. Usually they are phonon frequencies, but
        the same shape array can be used instead of frequencies.
        shape=(regular_grid_points, num_band), dtype='double'
    bz_grid : BZGrid
        Grid information in reciprocal space.
    grid_points : array_like, optional, default=None
        Grid point indices in BZ-grid. If None, all regular grid points in
        BZ-grid. shape=(grid_points, ), dtype='long'
    bzgp2irgp_map : array_like, optional, default=None
        Grid point index mapping from bz_grid to index of the first dimension of
        `grid_values` array, i.e., usually irreducible grid point count.
    function : str, 'I' or 'J', optional, default='I'
        'J' is for intetration and 'I' is for its derivative.

    Returns
    -------
    integration_weights : ndarray
        shape=(grid_points, sampling_points, num_band), dtype='double',
        order='C'

    """
    import phono3py._phono3py as phono3c

    relative_grid_addresses = np.array(
        np.dot(
            get_tetrahedra_relative_grid_address(bz_grid.microzone_lattice), bz_grid.P.T
        ),
        dtype="long",
        order="C",
    )
    if grid_points is None:
        _grid_points = bz_grid.grg2bzg
    elif _check_ndarray_state(grid_points, "long"):
        _grid_points = grid_points
    else:
        _grid_points = np.array(grid_points, dtype="long")
    if _check_ndarray_state(grid_values, "double"):
        _grid_values = grid_values
    else:
        _grid_values = np.array(grid_values, dtype="double", order="C")
    if _check_ndarray_state(sampling_points, "double"):
        _sampling_points = sampling_points
    else:
        _sampling_points = np.array(sampling_points, dtype="double")
    if bzgp2irgp_map is None:
        _bzgp2irgp_map = np.arange(len(grid_values), dtype="long")
    elif _check_ndarray_state(bzgp2irgp_map, "long"):
        _bzgp2irgp_map = bzgp2irgp_map
    else:
        _bzgp2irgp_map = np.array(bzgp2irgp_map, dtype="long")

    num_grid_points = len(_grid_points)
    num_band = _grid_values.shape[1]
    integration_weights = np.zeros(
        (num_grid_points, len(_sampling_points), num_band), dtype="double", order="C"
    )
    phono3c.integration_weights_at_grid_points(
        integration_weights,
        _sampling_points,
        relative_grid_addresses,
        bz_grid.D_diag,
        _grid_points,
        _grid_values,
        bz_grid.addresses,
        bz_grid.gp_map,
        _bzgp2irgp_map,
        bz_grid.store_dense_gp_map * 1 + 1,
        function,
    )
    return integration_weights


def get_tetrahedra_relative_grid_address(microzone_lattice):
    """Return relative (differences of) grid addresses from the central.

    Parameter
    ---------
    microzone_lattice : ndarray or list of list
        column vectors of parallel piped microzone lattice, i.e.,
        microzone_lattice = np.linalg.inv(cell.get_cell()) / mesh

    """
    import phono3py._phono3py as phono3c

    relative_grid_address = np.zeros((24, 4, 3), dtype="long", order="C")
    phono3c.tetrahedra_relative_grid_address(
        relative_grid_address, np.array(microzone_lattice, dtype="double", order="C")
    )

    return relative_grid_address


def _check_ndarray_state(array, dtype):
    """Check contiguousness and dtype."""
    return array.dtype == dtype and array.flags.c_contiguous
