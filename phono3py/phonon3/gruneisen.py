"""Calculation of mode Grueneisen parameters from fc3."""

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

import sys
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrixGL,
    DynamicalMatrixNAC,
    get_dynamical_matrix,
)
from phonopy.phonon.grid import (
    BZGrid,
    get_ir_grid_points,
    get_qpoints_from_bz_grid_points,
)
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry


def run_gruneisen_parameters(
    fc2: NDArray[np.double],
    fc3: NDArray[np.double],
    supercell: PhonopyAtoms,
    primitive: Primitive,
    band_paths: list[NDArray[np.double]] | None,
    mesh: float | Sequence[int] | Sequence[Sequence[int]] | NDArray[np.int64] | None,
    primitive_symmetry: Symmetry,
    qpoints: Sequence[Sequence[float]] | NDArray[np.double] | None,
    nac_params: dict | None = None,
    nac_q_direction: NDArray[np.double] | None = None,
    ion_clamped: bool = False,
    factor: float | None = None,
    symprec: float = 1e-5,
    output_filename: str | None = None,
    log_level: int = 1,
) -> None:
    """Run mode Grueneisen parameter calculation.

    The results is written into files.

    """
    gruneisen = Gruneisen(
        fc2,
        fc3,
        supercell,
        primitive,
        nac_params=nac_params,
        nac_q_direction=nac_q_direction,
        ion_clamped=ion_clamped,
        factor=factor,
        symprec=symprec,
    )

    if log_level > 0:
        dm = gruneisen.dynamical_matrix
        if isinstance(dm, DynamicalMatrixGL):
            dm.show_nac_message()

    if mesh is not None:
        gruneisen.set_sampling_mesh(mesh, primitive_symmetry=primitive_symmetry)
        filename_ext = ".hdf5"
    elif band_paths is not None:
        gruneisen.set_band_structure(band_paths)
        filename_ext = ".yaml"
    elif qpoints is not None:
        gruneisen.set_qpoints(np.asarray(qpoints, dtype="double", order="C"))
        filename_ext = ".yaml"

    if log_level:
        print("-" * 23 + " Phonon Gruneisen parameter " + "-" * 23)
        if mesh is not None:
            print("Mesh sampling: [ %d %d %d ]" % tuple(gruneisen.mesh_numbers))  # type: ignore[arg-type]
        elif band_paths is not None:
            print("Paths in reciprocal reduced coordinates:")
            for path in band_paths:
                print(
                    "[%5.2f %5.2f %5.2f] --> [%5.2f %5.2f %5.2f]"
                    % (tuple(path[0]) + tuple(path[-1]))
                )
        if ion_clamped:
            print("To be calculated with ion clamped.")

        sys.stdout.flush()

    gruneisen.run()

    if output_filename is None:
        filename = "gruneisen"
    else:
        filename = "gruneisen." + output_filename
    gruneisen.write(filename=filename)

    if log_level:
        print("Gruneisen parameters are written in %s" % (filename + filename_ext))


class Gruneisen:
    """Calculate mode Grueneisen parameters from fc3."""

    def __init__(
        self,
        fc2: NDArray[np.double],
        fc3: NDArray[np.double],
        supercell: PhonopyAtoms,
        primitive: Primitive,
        nac_params: dict | None = None,
        nac_q_direction: NDArray[np.double] | None = None,
        ion_clamped: bool = False,
        factor: float | None = None,
        symprec: float = 1e-5,
    ) -> None:
        """Init method."""
        self._fc2 = fc2
        self._fc3 = fc3
        self._scell = supercell
        self._pcell = primitive
        self._ion_clamped = ion_clamped
        if factor is None:
            self._factor = get_physical_units().DefaultToTHz
        else:
            self._factor = factor
        self._symprec = symprec
        self._dm = get_dynamical_matrix(
            self._fc2,
            self._scell,
            self._pcell,
            nac_params=nac_params,
        )
        self._nac_q_direction = nac_q_direction

        self._svecs, self._multi = self._pcell.get_smallest_vectors()

        if self._ion_clamped:
            num_atom_prim = len(self._pcell)
            self._X = np.zeros((num_atom_prim, 3, 3, 3), dtype=float)
        else:
            self._X = self._get_X()
        self._dPhidu = self._get_dPhidu()

        self._gruneisen_parameters: (
            NDArray[np.double] | Sequence[NDArray[np.double]] | None
        ) = None
        self._frequencies: NDArray[np.double] | Sequence[NDArray[np.double]] | None = (
            None
        )
        self._qpoints: NDArray[np.double] | None = None
        self._mesh: NDArray[np.int64] | None = None
        self._band_paths: list[NDArray[np.double]] | None = None
        self._band_distances: list[list[float]] | None = None
        self._run_mode: str | None = None
        self._weights: NDArray[np.int64] | None = None

    def run(self) -> None:
        """Run mode Grueneisen parameter calculation."""
        if self._run_mode == "band":
            (
                self._gruneisen_parameters,
                self._frequencies,
            ) = self._calculate_band_paths()  # type: ignore[assignment]
        elif self._run_mode == "qpoints" or self._run_mode == "mesh":
            (
                self._gruneisen_parameters,
                self._frequencies,
            ) = self._calculate_at_qpoints(self._qpoints)  # type: ignore[arg-type]
        else:
            sys.stderr.write("Q-points are not specified.\n")

    @property
    def mesh_numbers(self) -> NDArray[np.int64] | None:
        """Return mesh numbers."""
        return self._mesh

    @property
    def dynamical_matrix(self) -> DynamicalMatrixGL | DynamicalMatrixNAC:
        """Return DynamicalMatrix instance."""
        return self._dm  # type: ignore[return-value]

    @property
    def frequencies(self) -> NDArray[np.double] | Sequence[NDArray[np.double]] | None:
        """Return frequencies."""
        return self._frequencies

    @property
    def gruneisen_parameters(
        self,
    ) -> NDArray[np.double] | Sequence[NDArray[np.double]] | None:
        """Return mode Grueneisen paraterms."""
        return self._gruneisen_parameters

    def get_gruneisen_parameters(
        self,
    ) -> NDArray[np.double] | Sequence[NDArray[np.double]] | None:
        """Return mode Grueneisen paraterms."""
        warnings.warn(
            "Use attribute, Gruneisen.gruneisen_parameters "
            "instead of Gruneisen.get_gruneisen_parameters().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.gruneisen_parameters

    def set_qpoints(self, qpoints: NDArray[np.double]) -> None:
        """Set q-points."""
        self._run_mode = "qpoints"
        self._qpoints = qpoints

    def set_sampling_mesh(
        self,
        mesh: float | Sequence[int] | Sequence[Sequence[int]] | NDArray[np.int64],
        primitive_symmetry: Symmetry | None = None,
        use_grg: bool = False,
    ) -> None:
        """Set sampling mesh."""
        self._run_mode = "mesh"
        dataset = primitive_symmetry.dataset if primitive_symmetry is not None else None
        bz_grid = BZGrid(
            mesh,
            lattice=self._pcell.cell,
            symmetry_dataset=dataset,
            use_grg=use_grg,
        )
        ir_grid_points, self._weights, _ = get_ir_grid_points(bz_grid)
        ir_grid_points = np.array(bz_grid.grg2bzg[ir_grid_points], dtype="int64")
        self._qpoints = get_qpoints_from_bz_grid_points(ir_grid_points, bz_grid)
        self._mesh = bz_grid.D_diag

    def set_band_structure(self, paths: list[NDArray[np.double]]) -> None:
        """Set band structure paths."""
        self._run_mode = "band"
        self._band_paths = paths
        rec_lattice = np.linalg.inv(self._pcell.cell)
        self._band_distances = []
        for path in paths:
            distances_at_path = [0.0]
            for i in range(len(path) - 1):
                distances_at_path.append(
                    float(np.linalg.norm(np.dot(rec_lattice, path[i + 1] - path[i])))
                    + distances_at_path[-1]
                )
            self._band_distances.append(distances_at_path)

    def write(self, filename: str = "gruneisen") -> None:
        """Write result in a file."""
        if self._gruneisen_parameters is not None:
            if self._run_mode == "band":
                self._write_band_yaml(filename + ".yaml")
            elif self._run_mode == "qpoints":
                self._write_mesh_yaml(filename + ".yaml")
            elif self._run_mode == "mesh":
                self._write_mesh_hdf5(filename + ".hdf5")

    def _write_mesh_yaml(self, filename: str) -> None:
        assert self._qpoints is not None
        assert self._gruneisen_parameters is not None
        assert self._frequencies is not None
        with open(filename, "w") as f:
            if self._run_mode == "mesh":
                f.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))  # type: ignore[arg-type]
            f.write("nqpoint: %d\n" % len(self._qpoints))
            f.write("phonon:\n")
            for i, (q, g_at_q, freqs_at_q) in enumerate(
                zip(
                    self._qpoints,
                    self._gruneisen_parameters,
                    self._frequencies,
                    strict=True,
                )
            ):
                f.write("- q-position: [ %10.7f, %10.7f, %10.7f ]\n" % tuple(q))
                if self._weights is not None:
                    f.write("  multiplicity: %d\n" % self._weights[i])
                f.write("  band:\n")
                for j, (g, freq) in enumerate(zip(g_at_q, freqs_at_q, strict=True)):
                    f.write("  - # %d\n" % (j + 1))
                    f.write("    frequency: %15.10f\n" % freq)
                    f.write("    gruneisen: %15.10f\n" % (g.trace() / 3))
                    f.write("    gruneisen_tensor:\n")
                    for g_xyz in g:
                        f.write("    - [ %10.7f, %10.7f, %10.7f ]\n" % tuple(g_xyz))

    def _write_band_yaml(self, filename: str) -> None:
        assert self._band_paths is not None
        assert self._band_distances is not None
        assert self._gruneisen_parameters is not None
        assert self._frequencies is not None
        with open(filename, "w") as f:
            f.write("path:\n\n")
            for path, distances, gs, fs in zip(
                self._band_paths,
                self._band_distances,
                self._gruneisen_parameters,
                self._frequencies,
                strict=True,
            ):
                f.write("- nqpoint: %d\n" % len(path))
                f.write("  phonon:\n")
                for q, d, g_at_q, freqs_at_q in zip(
                    path, distances, gs, fs, strict=True
                ):
                    f.write("  - q-position: [ %10.7f, %10.7f, %10.7f ]\n" % tuple(q))
                    f.write("    distance: %10.7f\n" % d)
                    f.write("    band:\n")
                    for j, (g, freq) in enumerate(zip(g_at_q, freqs_at_q, strict=True)):
                        f.write("    - # %d\n" % (j + 1))
                        f.write("      frequency: %15.10f\n" % freq)
                        f.write("      gruneisen: %15.10f\n" % (g.trace() / 3))
                        f.write("      gruneisen_tensor:\n")
                        for g_xyz in g:
                            f.write(
                                "      - [ %10.7f, %10.7f, %10.7f ]\n" % tuple(g_xyz)
                            )
                    f.write("\n")

    def _write_mesh_hdf5(self, filename: str = "gruneisen.hdf5") -> None:
        import h5py

        assert self._gruneisen_parameters is not None
        g = self._gruneisen_parameters
        assert isinstance(g, np.ndarray)
        gruneisen = np.array(
            (g[:, :, 0, 0] + g[:, :, 1, 1] + g[:, :, 2, 2]) / 3,
            dtype="double",
            order="C",
        )

        with h5py.File(filename, "w") as w:
            w.create_dataset("mesh", data=self._mesh)
            w.create_dataset("gruneisen", data=gruneisen)
            w.create_dataset("gruneisen_tensor", data=self._gruneisen_parameters)
            w.create_dataset("weight", data=self._weights)
            w.create_dataset("frequency", data=self._frequencies)
            w.create_dataset("qpoint", data=self._qpoints)

    def _calculate_at_qpoints(
        self, qpoints: NDArray[np.double]
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        gruneisen_parameters = []
        frequencies = []
        for i, q in enumerate(qpoints):
            if isinstance(self._dm, DynamicalMatrixNAC):
                if (np.abs(q) < 1e-5).all():  # If q is almost at Gamma
                    if self._run_mode == "band":
                        # Direction estimated from neighboring point
                        if i > 0:
                            q_dir = qpoints[i] - qpoints[i - 1]
                        elif i == 0 and len(qpoints) > 1:
                            q_dir = qpoints[i + 1] - qpoints[i]
                        else:
                            q_dir = None
                        g, omega2 = self._get_gruneisen_tensor(q, nac_q_direction=q_dir)
                    else:  # Specified q-vector
                        g, omega2 = self._get_gruneisen_tensor(
                            q, nac_q_direction=self._nac_q_direction
                        )
                else:  # If q is away from Gamma-point, then q-vector
                    g, omega2 = self._get_gruneisen_tensor(q, nac_q_direction=q)
            else:  # Without NAC
                g, omega2 = self._get_gruneisen_tensor(q)
            gruneisen_parameters.append(g)
            frequencies.append(np.sqrt(abs(omega2)) * np.sign(omega2) * self._factor)

        return (
            np.array(gruneisen_parameters, dtype="double", order="C"),
            np.array(frequencies, dtype="double", order="C"),
        )

    def _calculate_band_paths(
        self,
    ) -> tuple[list[NDArray[np.double]], list[NDArray[np.double]]]:
        assert self._band_paths is not None
        gruneisen_parameters = []
        frequencies = []
        for path in self._band_paths:
            (gruneisen_at_path, frequencies_at_path) = self._calculate_at_qpoints(path)
            gruneisen_parameters.append(gruneisen_at_path)
            frequencies.append(frequencies_at_path)

        return gruneisen_parameters, frequencies

    def _get_gruneisen_tensor(
        self,
        q: NDArray[np.double],
        nac_q_direction: NDArray[np.double] | None = None,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        if nac_q_direction is None:
            self._dm.run(q)
        else:
            self._dm.run(q, nac_q_direction)  # type: ignore[arg-type]
        assert self._dm.dynamical_matrix is not None
        omega2, w = np.linalg.eigh(self._dm.dynamical_matrix)
        g = np.zeros((len(omega2), 3, 3), dtype=float)
        num_atom_prim = len(self._pcell)
        dDdu = self._get_dDdu(q)

        for s in range(len(omega2)):
            if (np.abs(q) < 1e-5).all() and s < 3:
                continue
            for i in range(3):
                for j in range(3):
                    for nu in range(num_atom_prim):
                        for pi in range(num_atom_prim):
                            g[s] += (
                                w[nu * 3 + i, s].conjugate()
                                * dDdu[nu, pi, i, j]
                                * w[pi * 3 + j, s]
                            ).real

            g[s] *= -1.0 / 2 / omega2[s]

        return g, omega2

    def _get_dDdu(self, q: NDArray[np.double]) -> NDArray[np.cdouble]:
        num_atom_prim = len(self._pcell)
        p2s = self._pcell.p2s_map
        s2p = self._pcell.s2p_map
        m = self._pcell.masses
        dPhidu = self._dPhidu
        dDdu = np.zeros((num_atom_prim, num_atom_prim, 3, 3, 3, 3), dtype=complex)

        for nu in range(num_atom_prim):
            for pi, p in enumerate(p2s):
                for Ppi, s in enumerate(s2p):
                    if not s == p:
                        continue
                    adrs = self._multi[Ppi, nu][1]
                    multi = self._multi[Ppi, nu][0]
                    phase = (
                        np.exp(
                            2j
                            * np.pi
                            * np.dot(self._svecs[adrs : (adrs + multi), :], q)
                        ).sum()
                        / multi
                    )
                    dDdu[nu, pi] += phase * dPhidu[nu, Ppi]
                dDdu[nu, pi] /= np.sqrt(m[nu] * m[pi])

        return dDdu

    def _get_dPhidu(self) -> NDArray[np.double]:
        fc3 = self._fc3
        num_atom_prim = len(self._pcell)
        num_atom_super = len(self._scell)
        p2s = self._pcell.p2s_map
        dPhidu = np.zeros((num_atom_prim, num_atom_super, 3, 3, 3, 3), dtype=float)

        for nu in range(num_atom_prim):
            Y = self._get_Y(nu)
            for pi in range(num_atom_super):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for ll in range(3):
                                for _ in range(3):
                                    dPhidu[nu, pi, i, j, k, ll] = (
                                        fc3[p2s[nu], pi, :, i, j, :] * Y[:, :, k, ll]
                                    ).sum()
                                    # Symmetrization?
                                    # (Y[:,:,k,l] + Y[:,:,l,k]) / 2).sum()

        return dPhidu

    def _get_Y(self, nu: int) -> NDArray[np.double]:
        X = self._X
        lat = self._pcell.cell
        num_atom_super = len(self._scell)
        R = self._get_R(num_atom_super, nu, lat)
        s2p = self._pcell.s2p_map
        p2p = self._pcell.p2p_map

        Y = np.zeros((num_atom_super, 3, 3, 3), dtype=float)

        for Mmu in range(num_atom_super):
            for i in range(3):
                Y[Mmu, i, i, :] = R[Mmu, :]
            Y[Mmu] += X[p2p[s2p[Mmu]]]

        return Y

    def _get_X(self) -> NDArray[np.double]:
        num_atom_super = len(self._scell)
        num_atom_prim = len(self._pcell)
        p2s = self._pcell.p2s_map
        lat = self._pcell.cell
        X = np.zeros((num_atom_prim, 3, 3, 3), dtype=float)
        G = self._get_Gamma()
        P = self._fc2

        for mu in range(num_atom_prim):
            for nu in range(num_atom_prim):
                R = self._get_R(num_atom_super, nu, lat)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for ll in range(3):
                                X[mu, i, j, k] -= G[mu, nu, i, ll] * np.dot(
                                    P[p2s[nu], :, ll, j], R[:, k]
                                )

        return X

    def _get_R(
        self, num_atom_super: int, nu: int, lat: NDArray[np.double]
    ) -> NDArray[np.double]:
        R = []
        for Npi in range(num_atom_super):
            adrs = self._multi[Npi, nu][1]
            multi = self._multi[Npi, nu][0]
            R.append(
                np.dot(self._svecs[adrs : (adrs + multi), :].sum(axis=0) / multi, lat)
            )
        return np.array(R)

    def _get_Gamma(self) -> NDArray[np.double]:
        num_atom_prim = len(self._pcell)
        m = self._pcell.masses
        self._dm.run([0, 0, 0])
        assert self._dm.dynamical_matrix is not None
        vals, vecs = np.linalg.eigh(self._dm.dynamical_matrix.real)
        G = np.zeros((num_atom_prim, num_atom_prim, 3, 3), dtype=float)

        for pi in range(num_atom_prim):
            for mu in range(num_atom_prim):
                for k in range(3):
                    for i in range(3):
                        # Eigenvectors are real.
                        # 3: means optical modes
                        G[pi, mu, k, i] = (
                            1.0
                            / np.sqrt(m[pi] * m[mu])
                            * (
                                vecs[pi * 3 + k, 3:] * vecs[mu * 3 + i, 3:] / vals[3:]
                            ).sum()
                        )
        return G
