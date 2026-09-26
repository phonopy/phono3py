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

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.phonon.grid import BZGrid
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import Primitive, Supercell
from phonopy.structure.symmetry import Symmetry

from phono3py._lang import resolve_lang
from phono3py.file_IO import write_joint_dos_at_t
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
        fc2: NDArray[np.double],
        mesh: float
        | Sequence[int]
        | Sequence[Sequence[int]]
        | NDArray[np.int64]
        | None = None,
        nac_params: dict | None = None,
        nac_q_direction: NDArray[np.double] | None = None,
        sigmas: Sequence[float | None] | None = None,
        cutoff_frequency: float = 1e-4,
        frequency_step: float | None = None,
        num_frequency_points: int | None = None,
        num_points_in_batch: int | None = None,
        temperatures: Sequence[float | None] | None = None,
        frequency_factor_to_THz: float | None = None,
        frequency_scale_factor: float | None = None,
        use_grg: bool = False,
        SNF_coordinates: Literal["reciprocal", "direct"] = "reciprocal",
        is_mesh_symmetry: bool = True,
        is_symmetry: bool = True,
        symprec: float = 1e-5,
        output_filename: str | os.PathLike | None = None,
        log_level: int = 0,
        symmetrize_tetrahedra: bool = False,
        lang: Literal["C", "Python", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Parameters
        ----------
        supercell : Supercell
            Supercell of fc2.
        primitive : Primitive
            Primitive cell.
        fc2 : ndarray
            Second-order force constants.
            shape=(atoms in supercell, atoms in supercell, 3, 3) or
            (atoms in primitive, atoms in supercell, 3, 3), dtype='double'
        mesh : float or array_like, optional
            Sampling mesh, given as a length, three mesh numbers, or a grid
            matrix. shape=(3,) or (3, 3). If None, ``initialize`` has to be
            called with it before ``run``.
        nac_params : dict, optional
            Parameters of the non-analytical term correction.
        nac_q_direction : array_like, optional
            Direction of q from Gamma in fractional coordinates, used for the
            non-analytical term correction at the Gamma point. shape=(3,)
        sigmas : list of float or None, optional
            Widths of the Gaussian smearing in THz, one calculation for each.
            None in the list means the tetrahedron method. If None, [None].
        cutoff_frequency : float, optional, default=1e-4
            Phonon modes with frequency below this value in THz are left out.
        frequency_step : float, optional
            Spacing of the frequency points in THz. Used when
            ``num_frequency_points`` is None.
        num_frequency_points : int, optional
            Number of frequency points. The frequency points run from 0 to
            twice the maximum phonon frequency, plus four times the largest
            sigma. If neither this nor ``frequency_step`` is given, 201.
        num_points_in_batch : int, optional
            Number of frequency points computed at once. A larger value gives
            better concurrency but uses more memory.
        temperatures : list of float or None, optional
            Temperatures in K. For None in the list, the JDOS is computed
            without phonon occupations; otherwise it is weighted by them. If
            None, [None].
        frequency_factor_to_THz : float, optional
            Deprecated. Passing a non-None value emits a
            ``DeprecationWarning``. If None,
            ``get_physical_units().DefaultToTHz``.
        frequency_scale_factor : float, optional
            Factor multiplied to all phonon frequencies. If None, not scaled.
        use_grg : bool, optional, default=False
            Use a generalized regular grid.
        SNF_coordinates : str, optional, default='reciprocal'
            'reciprocal' or 'direct', the space in which the grid matrix is
            brought to the Smith normal form.
        is_mesh_symmetry : bool, optional, default=True
            Sum over the triplets irreducible by the symmetry of the grid
            point, weighted by their multiplicities, instead of all triplets.
        is_symmetry : bool, optional, default=True
            Use the crystal symmetry and time reversal for the grid.
        symprec : float, optional, default=1e-5
            Tolerance of the symmetry search.
        output_filename : str or os.PathLike, optional
            Inserted into the names of the files written by ``run`` with
            ``write_jdos=True``.
        log_level : int, optional, default=0
            Verbosity of the standard output.
        symmetrize_tetrahedra : bool, optional, default=False
            When True, the integration weights of the tetrahedron method are
            averaged over the 24 tetrahedra rotated by all the point-group
            operations. The 24 tetrahedra are cut along one main diagonal, so
            the weights can differ between symmetrically equivalent q-points.
            Averaging removes the difference. Not available with
            ``lang='C'``.
        lang : str, optional, default='Rust'
            Backend, 'C', 'Python' or 'Rust'.

        """
        self._primitive = primitive
        self._supercell = supercell
        self._fc2 = fc2
        if temperatures is None:
            self._temperatures = [None]
        else:
            self._temperatures = temperatures
        self._nac_params = nac_params
        self._nac_q_direction = nac_q_direction
        if sigmas is None:
            self._sigmas: list[float | None] = [None]
        else:
            self._sigmas = list(sigmas)
        self._cutoff_frequency = cutoff_frequency
        if frequency_factor_to_THz is None:
            self._frequency_factor_to_THz: float = get_physical_units().DefaultToTHz
        else:
            warnings.warn(
                "frequency_factor_to_THz parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor
        self._is_mesh_symmetry = is_mesh_symmetry
        self._is_symmetry = is_symmetry

        self._use_grg = use_grg
        self._SNF_coordinates: Literal["reciprocal", "direct"] = SNF_coordinates
        self._symprec = symprec
        self._filename = output_filename
        self._log_level = log_level
        self._symmetrize_tetrahedra = symmetrize_tetrahedra
        if lang in ("C", "Rust"):
            lang = resolve_lang(lang)
        self._lang: Literal["C", "Python", "Rust"] = lang

        self._bz_grid: BZGrid | None = None
        self._jdos: JointDos | None = None
        self._joint_dos: NDArray[np.double] | None = None
        self._frequency_points: NDArray[np.double] | None = None
        self._num_frequency_points_in_batch = num_points_in_batch
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points

        self._primitive_symmetry = Symmetry(
            self._primitive, self._symprec, self._is_symmetry
        )

        if mesh is not None:
            self.initialize(mesh)

    @property
    def grid(self) -> BZGrid | None:
        """Return BZGrid class instance."""
        return self._bz_grid

    @property
    def nac_params(self) -> dict | None:
        """Setter and getter of parameters for non-analytical term correction."""
        return self._nac_params

    @property
    def num_frequency_points_in_batch(self) -> int | None:
        """Getter and setter of num_frequency_points_in_batch.

        Number of sampling frequency points per batch.
        Larger value gives better concurrency in tetrahedron method,
        but requires more memory.

        """
        return self._num_frequency_points_in_batch

    @num_frequency_points_in_batch.setter
    def num_frequency_points_in_batch(self, nelems_in_batch: int | None) -> None:
        self._num_frequency_points_in_batch = nelems_in_batch

    @property
    def mesh_numbers(self) -> NDArray[np.int64] | None:
        """Setter and getter of sampling mesh numbers in reciprocal space."""
        if self._bz_grid is None:
            return None
        else:
            return self._bz_grid.D_diag

    @mesh_numbers.setter
    def mesh_numbers(
        self,
        mesh_numbers: float
        | Sequence[int]
        | Sequence[Sequence[int]]
        | NDArray[np.int64],
    ) -> None:
        self._bz_grid = BZGrid(
            mesh_numbers,
            lattice=self._primitive.cell,
            symmetry_dataset=self._primitive_symmetry.dataset,
            is_time_reversal=self._is_symmetry,
            use_grg=self._use_grg,
            force_SNF=False,
            SNF_coordinates=self._SNF_coordinates,
            store_dense_gp_map=True,
            lang="Rust" if self._lang == "Rust" else "C",
        )

    def initialize(
        self, mesh: float | Sequence[int] | Sequence[Sequence[int]] | NDArray[np.int64]
    ) -> None:
        """Initialize JointDos."""
        self.mesh_numbers = mesh
        assert self._bz_grid is not None
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
            log_level=self._log_level,
            symmetrize_tetrahedra=self._symmetrize_tetrahedra,
            lang=self._lang,
        )
        if self._log_level:
            print("Generating grid system ... ", end="", flush=True)

        if self._log_level:
            assert self._bz_grid is not None
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

    def run(
        self,
        grid_points: Sequence[int] | NDArray[np.int64],
        write_jdos: bool = False,
    ) -> None:
        """Calculate joint-density-of-states."""
        assert self._jdos is not None
        assert self._bz_grid is not None

        if self._log_level:
            print(
                "--------------------------------- Joint DOS "
                "---------------------------------"
            )
            print("Running harmonic phonon calculations...", flush=True)

        self._jdos.run_phonon_solver()
        frequencies, _, _ = self._jdos.get_phonons()
        assert frequencies is not None
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
        self._joint_dos = np.zeros(  # type: ignore[call-overload]
            (
                len(self._sigmas),
                len(self._temperatures),
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
                assert weights is not None
                print("Number of triplets: %d" % len(weights))
                print("Frequency")
                _freqs = self._jdos.get_phonons()[0]
                assert _freqs is not None
                for f in _freqs[gp]:
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
                        f"frequency points are divided into {len(batches)} batches."
                    )
                for i_t, temperature in enumerate(self._temperatures):
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
                        filename = write_joint_dos_at_t(
                            gp,
                            self._bz_grid.D_diag,
                            self._frequency_points,
                            self._joint_dos[i_s, i_t],
                            sigma=self._sigmas[i_s],
                            temperature=self._temperatures[i_t],
                            filename=self._filename,
                            is_mesh_symmetry=self._is_mesh_symmetry,
                        )
                        if self._log_level:
                            print('JDOS is written into "%s".' % filename)

    @property
    def dynamical_matrix(self) -> DynamicalMatrix:
        """Return DynamicalMatrix class instance."""
        assert self._jdos is not None
        return self._jdos.dynamical_matrix

    @property
    def frequency_points(self) -> NDArray[np.double] | None:
        """Return frequency points."""
        return self._frequency_points

    @property
    def joint_dos(self) -> NDArray[np.double] | None:
        """Return calculated joint-density-of-states."""
        return self._joint_dos
