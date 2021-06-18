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


def get_unique_grid_points(grid_points, bz_grid):
    """Collect grid points that are on tetrahedron vertices."""
    import phono3py._phono3py as phono3c
    thm = TetrahedronMethod(bz_grid.microzone_lattice)
    unique_vertices = np.array(
        np.dot(thm.get_unique_tetrahedra_vertices(), bz_grid.P.T),
        dtype='int_', order='C')
    neighboring_grid_points = np.zeros(
        len(unique_vertices) * len(grid_points), dtype='int_')
    phono3c.neighboring_grid_points(
        neighboring_grid_points,
        grid_points,
        unique_vertices,
        bz_grid.D_diag,
        bz_grid.addresses,
        bz_grid.gp_map,
        bz_grid.is_dense_gp_map * 1 + 1)
    unique_grid_points = np.array(np.unique(neighboring_grid_points),
                                  dtype='int_')
    return unique_grid_points


def get_integration_weights(sampling_points,
                            grid_values,
                            bz_grid,
                            grid_points=None,
                            function='I'):
    """Return tetrahedron method integration weights.

    Parameters
    ----------
    sampling_points : array_like
        Values at which the integration weights are computed.
        shape=(sampling_points, ), dtype='double'
    grid_values : array_like
        Values of tetrahedron vertices. Usually they are phonon frequencies,
        but the same shape array can be used instead of frequencies.
        shape=(regular_grid_points, num_band), dtype='double'
    bz_grid : BZGrid
        Grid information in reciprocal space.
    grid_points : array_like, optional, default=None
        Grid point indices. If None, all regular grid points.
        shape=(grid_points, ), dtype='int_'
    function : str, 'I' or 'J', optional, default='I'
        'J' is for intetration and 'I' is for its derivative.

    """
    import phono3py._phono3py as phono3c
    thm = TetrahedronMethod(bz_grid.microzone_lattice)

    if grid_points is None:
        _grid_points = bz_grid.grg2bzg
    elif _check_ndarray_state(grid_points, 'int_'):
        _grid_points = grid_points
    else:
        _grid_points = np.array(grid_points, dtype='int_')
    if _check_ndarray_state(grid_values, 'double'):
        _grid_values = grid_values
    else:
        _grid_values = np.array(grid_values, dtype='double', order='C')
    if _check_ndarray_state(sampling_points, 'double'):
        _sampling_points = sampling_points
    else:
        _sampling_points = np.array(sampling_points, dtype='double')

    num_grid_points = len(_grid_points)
    num_band = _grid_values.shape[1]
    integration_weights = np.zeros(
        (num_grid_points, len(_sampling_points), num_band), dtype='double')
    phono3c.integration_weights_at_grid_points(
        integration_weights,
        _sampling_points,
        np.array(np.dot(thm.get_tetrahedra(), bz_grid.P.T),
                 dtype='int_', order='C'),
        bz_grid.D_diag,
        _grid_points,
        _grid_values,
        bz_grid.addresses,
        bz_grid.gp_map,
        bz_grid.is_dense_gp_map * 1 + 1,
        function)
    return integration_weights


def _check_ndarray_state(array, dtype):
    """Check contiguousness and dtype."""
    return (array.dtype == dtype and
            array.flags.c_contiguous)
