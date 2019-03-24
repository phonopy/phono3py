# Copyright (C) 2015 Atsushi Togo
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
from phono3py.file_IO import (write_ir_grid_points,
                              write_grid_address_to_hdf5)
from phono3py.phonon3.triplets import (get_coarse_ir_grid_points,
                                       get_number_of_triplets)


def write_grid_points(primitive,
                      mesh,
                      mesh_divs=None,
                      band_indices=None,
                      sigmas=None,
                      temperatures=None,
                      coarse_mesh_shifts=None,
                      is_kappa_star=True,
                      is_lbte=False,
                      compression=None,
                      symprec=1e-5,
                      filename=None):
    print("-" * 76)
    if mesh is None:
        print("To write grid points, mesh numbers have to be specified.")
    else:
        (ir_grid_points,
         grid_weights,
         bz_grid_address,
         grid_mapping_table) = get_coarse_ir_grid_points(
             primitive,
             mesh,
             mesh_divs,
             coarse_mesh_shifts,
             is_kappa_star=is_kappa_star,
             symprec=symprec)
        write_ir_grid_points(mesh,
                             mesh_divs,
                             ir_grid_points,
                             grid_weights,
                             bz_grid_address,
                             np.linalg.inv(primitive.get_cell()))
        gadrs_hdf5_fname = write_grid_address_to_hdf5(bz_grid_address,
                                                      mesh,
                                                      grid_mapping_table,
                                                      compression=compression,
                                                      filename=filename)

        print("Ir-grid points are written into \"ir_grid_points.yaml\".")
        print("Grid addresses are written into \"%s\"." % gadrs_hdf5_fname)

        if is_lbte and temperatures is not None:
            num_temp = len(temperatures)
            num_sigma = len(sigmas)
            num_ir_gp = len(ir_grid_points)
            num_band = primitive.get_number_of_atoms() * 3
            num_gp = len(bz_grid_address)
            if band_indices is None:
                num_band0 = num_band
            else:
                num_band0 = len(band_indices)
            print("Memory requirements:")
            size = (num_band0 * 3 * num_ir_gp * num_band * 3) * 8 / 1.0e9
            print("- Piece of collision matrix at each grid point, temp and "
                  "sigma: %.2f Gb" % size)
            size = (num_ir_gp * num_band * 3) ** 2 * 8 / 1.0e9
            print("- Full collision matrix at each temp and sigma: %.2f Gb"
                  % size)
            size = num_gp * (num_band ** 2 * 16 + num_band * 8 + 1) / 1.0e9
            print("- Phonons: %.2f Gb" % size)
            size = num_gp * 5 * 4 / 1.0e9
            print("- Grid point information: %.2f Gb" % size)
            size = (num_ir_gp * num_band0 *
                    (3 + 6 + num_temp * 2 + num_sigma * num_temp * 15 + 2) *
                    8 / 1.0e9)
            print("- Phonon properties: %.2f Gb" % size)


def show_num_triplets(primitive,
                      mesh,
                      mesh_divs=None,
                      band_indices=None,
                      grid_points=None,
                      coarse_mesh_shifts=None,
                      is_kappa_star=True,
                      symprec=1e-5):
    print("-" * 76)

    ir_grid_points, _, grid_address, _ = get_coarse_ir_grid_points(
        primitive,
        mesh,
        mesh_divs,
        coarse_mesh_shifts,
        is_kappa_star=is_kappa_star,
        symprec=symprec)

    if grid_points:
        _grid_points = grid_points
    else:
        _grid_points = ir_grid_points

    num_band = primitive.get_number_of_atoms() * 3
    if band_indices is None:
        num_band0 = num_band
    else:
        num_band0 = len(band_indices)

    print("Grid point        q-point        No. of triplets     Memory size")
    for gp in _grid_points:
        num_triplets = get_number_of_triplets(primitive,
                                              mesh,
                                              gp,
                                              swappable=True,
                                              symprec=symprec)
        q = grid_address[gp] / np.array(mesh, dtype='double')
        size = num_triplets * num_band0 * num_band ** 2 * 8 / 1e6
        print("  %5d     (%5.2f %5.2f %5.2f)  %8d              %d Mb" %
              (gp, q[0], q[1], q[2], num_triplets, size))
