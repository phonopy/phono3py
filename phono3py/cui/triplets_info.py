"""Show and write triplets information."""

# Copyright (C) 2015 Atsushi Togo
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

from phono3py.file_IO import write_grid_address_to_hdf5, write_ir_grid_points
from phono3py.phonon.grid import get_ir_grid_points
from phono3py.phonon3.triplets import get_triplets_at_q


def write_grid_points(
    primitive,
    bz_grid,
    band_indices=None,
    sigmas=None,
    temperatures=None,
    is_kappa_star=True,
    is_lbte=False,
    compression="gzip",
    filename=None,
):
    """Write grid points into files."""
    ir_grid_points, ir_grid_weights = _get_ir_grid_points(
        bz_grid, is_kappa_star=is_kappa_star
    )
    write_ir_grid_points(
        bz_grid, ir_grid_points, ir_grid_weights, np.linalg.inv(primitive.cell)
    )
    gadrs_hdf5_fname = write_grid_address_to_hdf5(
        bz_grid.addresses,
        bz_grid.D_diag,
        bz_grid.gp_map,
        compression=compression,
        filename=filename,
    )

    print('Ir-grid points are written into "ir_grid_points.yaml".')
    print('Grid addresses are written into "%s".' % gadrs_hdf5_fname)

    if is_lbte and temperatures is not None:
        num_temp = len(temperatures)
        num_sigma = len(sigmas)
        num_ir_gp = len(ir_grid_points)
        num_band = len(primitive) * 3
        num_gp = len(bz_grid.addresses)
        if band_indices is None:
            num_band0 = num_band
        else:
            num_band0 = len(band_indices)
        print("Memory requirements:")
        size = (num_band0 * 3 * num_ir_gp * num_band * 3) * 8 / 1.0e9
        print(
            "- Piece of collision matrix at each grid point, temp and "
            "sigma: %.2f Gb" % size
        )
        size = (num_ir_gp * num_band * 3) ** 2 * 8 / 1.0e9
        print("- Full collision matrix at each temp and sigma: %.2f Gb" % size)
        size = num_gp * (num_band**2 * 16 + num_band * 8 + 1) / 1.0e9
        print("- Phonons: %.2f Gb" % size)
        size = num_gp * 5 * 4 / 1.0e9
        print("- Grid point information: %.2f Gb" % size)
        size = (
            num_ir_gp
            * num_band0
            * (3 + 6 + num_temp * 2 + num_sigma * num_temp * 15 + 2)
            * 8
            / 1.0e9
        )
        print("- Phonon properties: %.2f Gb" % size)


def show_num_triplets(
    primitive, bz_grid, band_indices=None, grid_points=None, is_kappa_star=True
):
    """Show numbers of triplets at grid points."""
    tp_nums = _TripletsNumbers(bz_grid, is_kappa_star=is_kappa_star)
    num_band = len(primitive) * 3
    if band_indices is None:
        num_band0 = num_band
    else:
        num_band0 = len(band_indices)

    if grid_points:
        _grid_points = grid_points
    else:
        _grid_points = tp_nums.ir_grid_points

    print("-" * 76)
    print("Grid point        q-point        No. of triplets     Approx. Mem.")
    for gp in _grid_points:
        num_triplets = tp_nums.get_number_of_triplets(gp)
        q = np.dot(bz_grid.addresses[gp], bz_grid.QDinv.T)
        size = num_triplets * num_band0 * num_band**2 * 8 / 1e6
        print(
            "  %5d     (%5.2f %5.2f %5.2f)  %8d              %d Mb"
            % (gp, q[0], q[1], q[2], num_triplets, size)
        )


class _TripletsNumbers:
    def __init__(self, bz_grid, is_kappa_star=True):
        self._bz_grid = bz_grid
        (
            self.ir_grid_points,
            _,
        ) = _get_ir_grid_points(self._bz_grid, is_kappa_star=is_kappa_star)

    def get_number_of_triplets(self, gp):
        num_triplets = _get_number_of_triplets(self._bz_grid, gp, swappable=True)
        return num_triplets


def _get_ir_grid_points(bz_grid, is_kappa_star=True):
    if is_kappa_star:
        ir_grid_points, ir_grid_weights, _ = get_ir_grid_points(bz_grid)
        ir_grid_points = np.array(bz_grid.grg2bzg[ir_grid_points], dtype="int_")
    else:
        ir_grid_points = np.array(
            bz_grid.grg2bzg[: np.prod(bz_grid.D_diag)], dtype="int_", order="C"
        )
        ir_grid_weights = np.ones(len(ir_grid_points), dtype="int_")

    return ir_grid_points, ir_grid_weights


def _get_number_of_triplets(bz_grid, grid_point, swappable=True):
    triplets_at_q = get_triplets_at_q(grid_point, bz_grid, swappable=swappable)[0]
    return len(triplets_at_q)
