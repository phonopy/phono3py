"""API for joint-density-of-states calculation."""

# Copyright (C) 2019 Atsushi Togo
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
from phonopy.structure.cells import Primitive, Supercell
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz

from phono3py.file_IO import write_joint_dos
from phono3py.phonon.grid import BZGrid
from phono3py.phonon3.imag_self_energy import (
    get_freq_points_batches,
    get_frequency_points,
)
from phono3py.phonon3.joint_dos import JointDos


class Phono3pyJointDos:
    """Class to calculate joint-density-of-states."""

    def __init__(
        self,
        supercell: Supercell,
        primitive: Primitive,
        fc2,
        mesh=None,
        nac_params=None,
        nac_q_direction=None,
        sigmas=None,
        cutoff_frequency=1e-4,
        frequency_step=None,
        num_frequency_points=None,
        num_points_in_batch=None,
        temperatures=None,
        frequency_factor_to_THz=VaspToTHz,
        frequency_scale_factor=None,
        use_grg=False,
        SNF_coordinates="reciprocal",
        is_mesh_symmetry=True,
        is_symmetry=True,
        symprec=1e-5,
        output_filename=None,
        log_level=0,
    ):
        """Init method."""
        self._primitive = primitive
        self._supercell = supercell
        self._fc2 = fc2
        self._temperatures = temperatures
        self._nac_params = nac_params
        self._nac_q_direction = nac_q_direction
        if sigmas is None:
            self._sigmas = [None]
        else:
            self._sigmas = sigmas
        self._cutoff_frequency = cutoff_frequency
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor
        self._is_mesh_symmetry = is_mesh_symmetry
        self._is_symmetry = is_symmetry

        self._use_grg = use_grg
        self._SNF_coordinates = SNF_coordinates
        self._symprec = symprec
        self._filename = output_filename
        self._log_level = log_level

        self._bz_grid = None
        self._joint_dos = None
        self._num_frequency_points_in_batch = num_points_in_batch
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points

        self._primitive_symmetry = Symmetry(
            self._primitive, self._symprec, self._is_symmetry
        )

        if mesh is not None:
            self.mesh_numbers = mesh
            self.initialize(mesh)

    @property
    def grid(self):
        """Return BZGrid class instance."""
        return self._bz_grid

    @property
    def nac_params(self):
        """Setter and getter of parameters for non-analytical term correction."""
        return self._nac_params

    @property
    def num_frequency_points_in_batch(self):
        """Getter and setter of num_frequency_points_in_batch.

        Number of sampling frequency points per batch.
        Larger value gives better concurrency in tetrahedron method,
        but requires more memory.

        """
        return self._num_frequency_points_in_batch

    @num_frequency_points_in_batch.setter
    def num_frequency_points_in_batch(self, nelems_in_batch):
        self._num_frequency_points_in_batch = nelems_in_batch

    @property
    def mesh_numbers(self):
        """Setter and getter of sampling mesh numbers in reciprocal space."""
        if self._bz_grid is None:
            return None
        else:
            return self._bz_grid.D_diag

    @mesh_numbers.setter
    def mesh_numbers(self, mesh_numbers):
        self._bz_grid = BZGrid(
            mesh_numbers,
            lattice=self._primitive.cell,
            symmetry_dataset=self._primitive_symmetry.dataset,
            is_time_reversal=self._is_symmetry,
            use_grg=self._use_grg,
            force_SNF=False,
            SNF_coordinates=self._SNF_coordinates,
            store_dense_gp_map=True,
        )

    def initialize(self, mesh_numbers):
        """Initialize JointDos."""
        self._jdos = JointDos(
            self._primitive,
            self._supercell,
            self._bz_grid,
            self._fc2,
            nac_params=self._nac_params,
            cutoff_frequency=self._cutoff_frequency,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            frequency_scale_factor=self._frequency_scale_factor,
            is_mesh_symmetry=self._is_mesh_symmetry,
            symprec=self._symprec,
            filename=self._filename,
            log_level=self._log_level,
        )
        if self._log_level:
            print("Generating grid system ... ", end="", flush=True)

        self.mesh_numbers = mesh_numbers

        if self._log_level:
            if self._bz_grid.grid_matrix is None:
                print("[ %d %d %d ]" % tuple(self._bz_grid.D_diag))
            else:
                print("")
                print(
                    "Generalized regular grid: [ %d %d %d ]"
                    % tuple(self._bz_grid.D_diag)
                )
                print("Grid generation matrix:")
                print("  [ %d %d %d ]" % tuple(self._bz_grid.grid_matrix[0]))
                print("  [ %d %d %d ]" % tuple(self._bz_grid.grid_matrix[1]))
                print("  [ %d %d %d ]" % tuple(self._bz_grid.grid_matrix[2]))

    def run(self, grid_points, write_jdos=False):
        """Calculate joint-density-of-states."""
        if self._log_level:
            print(
                "--------------------------------- Joint DOS "
                "---------------------------------"
            )
            print("Running harmonic phonon calculations...", flush=True)

        self._jdos.run_phonon_solver()
        frequencies, _, _ = self._jdos.get_phonons()
        self._jdos.run_phonon_solver_at_gamma()
        max_phonon_freq = np.max(frequencies)
        self._jdos.run_phonon_solver_at_gamma(is_nac=True)

        self._frequency_points = get_frequency_points(
            max_phonon_freq=max_phonon_freq,
            sigmas=self._sigmas,
            frequency_points=None,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points,
        )
        batches = get_freq_points_batches(
            len(self._frequency_points), nelems=self._num_frequency_points_in_batch
        )
        if self._temperatures is None:
            temperatures = [None]
        else:
            temperatures = self._temperatures
        self._joint_dos = np.zeros(
            (
                len(self._sigmas),
                len(temperatures),
                len(self._frequency_points),
                2,
            ),
            dtype="double",
            order="C",
        )

        for i, gp in enumerate(grid_points):
            if (self._bz_grid.addresses[gp] == 0).all():
                self._jdos.nac_q_direction = self._nac_q_direction
            else:
                self._jdos.nac_q_direction = None
            self._jdos.set_grid_point(gp)

            if self._log_level:
                weights = self._jdos.get_triplets_at_q()[1]
                print(
                    "======================= "
                    "Grid point %d (%d/%d) "
                    "=======================" % (gp, i + 1, len(grid_points))
                )
                adrs = self._jdos.bz_grid.addresses[gp]
                q = np.dot(adrs, self._bz_grid.QDinv.T)
                print("q-point: (%5.2f %5.2f %5.2f)" % tuple(q))
                print("Number of triplets: %d" % len(weights))
                print("Frequency")
                for f in self._jdos.get_phonons()[0][gp]:
                    print("%8.3f" % f)

            if not self._sigmas:
                raise RuntimeError("sigma or tetrahedron method has to be set.")

            for i_s, sigma in enumerate(self._sigmas):
                self._jdos.sigma = sigma
                if self._log_level:
                    if sigma is None:
                        print("Tetrahedron method is used.")
                    else:
                        print("Smearing method with sigma=%s is used." % sigma)
                    print(
                        f"Calculations at {len(self._frequency_points)} "
                        f"frequency points are devided into {len(batches)} batches."
                    )
                for i_t, temperature in enumerate(temperatures):
                    self._jdos.temperature = temperature

                    for ib, freq_indices in enumerate(batches):
                        if self._log_level:
                            print(
                                f"{ib + 1}/{len(batches)}: {freq_indices + 1}",
                                flush=True,
                            )
                        self._jdos.frequency_points = self._frequency_points[
                            freq_indices
                        ]
                        self._jdos.run()
                        self._joint_dos[i_s, i_t, freq_indices] = self._jdos.joint_dos

                    if write_jdos:
                        filename = self._write(gp, i_sigma=i_s)
                        if self._log_level:
                            print('JDOS is written into "%s".' % filename)

    @property
    def dynamical_matrix(self):
        """Return DynamicalMatrix class instance."""
        return self._jdos.dynamical_matrix

    @property
    def frequency_points(self):
        """Return frequency points."""
        return self._frequency_points

    @property
    def joint_dos(self):
        """Return calculated joint-density-of-states."""
        return self._joint_dos

    def _write(self, gp, i_sigma=0):
        return write_joint_dos(
            gp,
            self._bz_grid.D_diag,
            self._frequency_points,
            self._joint_dos[i_sigma],
            sigma=self._sigmas[i_sigma],
            temperatures=self._temperatures,
            filename=self._filename,
            is_mesh_symmetry=self._is_mesh_symmetry,
        )
