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

import sys

import numpy as np
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.grid_points import get_qpoints
from phonopy.units import VaspToTHz


def run_gruneisen_parameters(
    fc2,
    fc3,
    supercell,
    primitive,
    band_paths,
    mesh,
    rotations,
    qpoints,
    nac_params=None,
    nac_q_direction=None,
    ion_clamped=False,
    factor=None,
    symprec=1e-5,
    output_filename=None,
    log_level=1,
):
    """Run mode Grueneisen parameter calculation.

    The results is written into files.

    """
    if log_level:
        print("-" * 23 + " Phonon Gruneisen parameter " + "-" * 23)
        if mesh is not None:
            print("Mesh sampling: [ %d %d %d ]" % tuple(mesh))
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
        if dm.is_nac() and dm.nac_method == "gonze":
            dm.show_Gonze_nac_message()

    if mesh is not None:
        gruneisen.set_sampling_mesh(mesh, rotations=rotations, is_gamma_center=True)
        filename_ext = ".hdf5"
    elif band_paths is not None:
        gruneisen.set_band_structure(band_paths)
        filename_ext = ".yaml"
    elif qpoints is not None:
        gruneisen.set_qpoints(qpoints)
        filename_ext = ".yaml"

    gruneisen.run()

    if output_filename is None:
        filename = "gruneisen"
    else:
        filename = "gruneisen." + output_filename
    gruneisen.write(filename=filename)

    if log_level:
        print("Gruneisen parameters are written in %s" % (filename + filename_ext))


class Gruneisen:
    """Calculat mode Grueneisen parameters from fc3."""

    def __init__(
        self,
        fc2,
        fc3,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        nac_params=None,
        nac_q_direction=None,
        ion_clamped=False,
        factor=VaspToTHz,
        symprec=1e-5,
    ):
        """Init method."""
        self._fc2 = fc2
        self._fc3 = fc3
        self._scell = supercell
        self._pcell = primitive
        self._ion_clamped = ion_clamped
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

        self._gruneisen_parameters = None
        self._frequencies = None
        self._qpoints = None
        self._mesh = None
        self._band_paths = None
        self._band_distances = None
        self._run_mode = None
        self._weights = None

    def run(self):
        """Run mode Grueneisen parameter calculation."""
        if self._run_mode == "band":
            (
                self._gruneisen_parameters,
                self._frequencies,
            ) = self._calculate_band_paths()
        elif self._run_mode == "qpoints" or self._run_mode == "mesh":
            (
                self._gruneisen_parameters,
                self._frequencies,
            ) = self._calculate_at_qpoints(self._qpoints)
        else:
            sys.stderr.write("Q-points are not specified.\n")

    @property
    def dynamical_matrix(self):
        """Return DynamicalMatrix instance."""
        return self._dm

    def get_gruneisen_parameters(self):
        """Return mode Grueneisen paraterms."""
        return self._gruneisen_parameters

    def set_qpoints(self, qpoints):
        """Set q-points."""
        self._run_mode = "qpoints"
        self._qpoints = qpoints

    def set_sampling_mesh(
        self, mesh, rotations=None, shift=None, is_gamma_center=False
    ):
        """Set sampling mesh."""
        self._run_mode = "mesh"
        self._mesh = np.array(mesh, dtype="intc")
        self._qpoints, self._weights = get_qpoints(
            self._mesh,
            np.linalg.inv(self._pcell.cell),
            q_mesh_shift=shift,
            is_gamma_center=is_gamma_center,
            rotations=rotations,
        )

    def set_band_structure(self, paths):
        """Set band structure paths."""
        self._run_mode = "band"
        self._band_paths = paths
        rec_lattice = np.linalg.inv(self._pcell.cell)
        self._band_distances = []
        for path in paths:
            distances_at_path = [0.0]
            for i in range(len(path) - 1):
                distances_at_path.append(
                    np.linalg.norm(np.dot(rec_lattice, path[i + 1] - path[i]))
                    + distances_at_path[-1]
                )
            self._band_distances.append(distances_at_path)

    def write(self, filename="gruneisen"):
        """Write result in a file."""
        if self._gruneisen_parameters is not None:
            if self._run_mode == "band":
                self._write_band_yaml(filename + ".yaml")
            elif self._run_mode == "qpoints":
                self._write_mesh_yaml(filename + ".yaml")
            elif self._run_mode == "mesh":
                self._write_mesh_hdf5(filename + ".hdf5")

    def _write_mesh_yaml(self, filename):
        with open(filename, "w") as f:
            if self._run_mode == "mesh":
                f.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))
            f.write("nqpoint: %d\n" % len(self._qpoints))
            f.write("phonon:\n")
            for i, (q, g_at_q, freqs_at_q) in enumerate(
                zip(self._qpoints, self._gruneisen_parameters, self._frequencies)
            ):
                f.write("- q-position: [ %10.7f, %10.7f, %10.7f ]\n" % tuple(q))
                if self._weights is not None:
                    f.write("  multiplicity: %d\n" % self._weights[i])
                f.write("  band:\n")
                for j, (g, freq) in enumerate(zip(g_at_q, freqs_at_q)):
                    f.write("  - # %d\n" % (j + 1))
                    f.write("    frequency: %15.10f\n" % freq)
                    f.write("    gruneisen: %15.10f\n" % (g.trace() / 3))
                    f.write("    gruneisen_tensor:\n")
                    for g_xyz in g:
                        f.write("    - [ %10.7f, %10.7f, %10.7f ]\n" % tuple(g_xyz))

    def _write_band_yaml(self, filename):
        with open(filename, "w") as f:
            f.write("path:\n\n")
            for path, distances, gs, fs in zip(
                self._band_paths,
                self._band_distances,
                self._gruneisen_parameters,
                self._frequencies,
            ):
                f.write("- nqpoint: %d\n" % len(path))
                f.write("  phonon:\n")
                for q, d, g_at_q, freqs_at_q in zip(path, distances, gs, fs):
                    f.write("  - q-position: [ %10.7f, %10.7f, %10.7f ]\n" % tuple(q))
                    f.write("    distance: %10.7f\n" % d)
                    f.write("    band:\n")
                    for j, (g, freq) in enumerate(zip(g_at_q, freqs_at_q)):
                        f.write("    - # %d\n" % (j + 1))
                        f.write("      frequency: %15.10f\n" % freq)
                        f.write("      gruneisen: %15.10f\n" % (g.trace() / 3))
                        f.write("      gruneisen_tensor:\n")
                        for g_xyz in g:
                            f.write(
                                "      - [ %10.7f, %10.7f, %10.7f ]\n" % tuple(g_xyz)
                            )
                    f.write("\n")

    def _write_mesh_hdf5(self, filename="gruneisen.hdf5"):
        import h5py

        g = self._gruneisen_parameters
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

    def _calculate_at_qpoints(self, qpoints):
        gruneisen_parameters = []
        frequencies = []
        for i, q in enumerate(qpoints):
            if self._dm.is_nac():
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

    def _calculate_band_paths(self):
        gruneisen_parameters = []
        frequencies = []
        for path in self._band_paths:
            (gruneisen_at_path, frequencies_at_path) = self._calculate_at_qpoints(path)
            gruneisen_parameters.append(gruneisen_at_path)
            frequencies.append(frequencies_at_path)

        return gruneisen_parameters, frequencies

    def _get_gruneisen_tensor(self, q, nac_q_direction=None):
        if nac_q_direction is None:
            self._dm.run(q)
        else:
            self._dm.run(q, nac_q_direction)
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

    def _get_dDdu(self, q):
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

    def _get_dPhidu(self):
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

    def _get_Y(self, nu):
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

    def _get_X(self):
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

    def _get_R(self, num_atom_super, nu, lat):
        R = []
        for Npi in range(num_atom_super):
            adrs = self._multi[Npi, nu][1]
            multi = self._multi[Npi, nu][0]
            R.append(
                np.dot(self._svecs[adrs : (adrs + multi), :].sum(axis=0) / multi, lat)
            )
        return np.array(R)

    def _get_Gamma(self):
        num_atom_prim = len(self._pcell)
        m = self._pcell.masses
        self._dm.run([0, 0, 0])
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
