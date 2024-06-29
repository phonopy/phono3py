"""File I/O methods."""

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

import os
import warnings
from collections.abc import Sequence
from typing import Optional, Union

import h5py
import numpy as np
from phonopy.cui.load_helper import read_force_constants_from_hdf5

# This import is deactivated for a while.
# from phonopy.file_IO import write_force_constants_to_hdf5
from phonopy.file_IO import check_force_constants_indices, get_cell_from_disp_yaml

from phono3py.version import __version__


def write_disp_fc3_yaml(dataset, supercell, filename="disp_fc3.yaml"):
    """Write disp_fc3.yaml.

    This function should not be called from phono3py script from version 3.

    """
    warnings.warn(
        "write_disp_fc3_yaml() is deprecated.", DeprecationWarning, stacklevel=2
    )

    w = open(filename, "w")
    w.write("natom: %d\n" % dataset["natom"])

    num_first = len(dataset["first_atoms"])
    w.write("num_first_displacements: %d\n" % num_first)
    if "cutoff_distance" in dataset:
        w.write("cutoff_distance: %f\n" % dataset["cutoff_distance"])

    num_second = 0
    num_disp_files = 0
    for d1 in dataset["first_atoms"]:
        num_disp_files += 1
        num_second += len(d1["second_atoms"])
        for d2 in d1["second_atoms"]:
            if "included" in d2:
                if d2["included"]:
                    num_disp_files += 1
            else:
                num_disp_files += 1

    w.write("num_second_displacements: %d\n" % num_second)
    w.write("num_displacements_created: %d\n" % num_disp_files)

    if "duplicates" in dataset:
        w.write("duplicates:\n")
        for i, j in dataset["duplicates"]:
            w.write("- [ %d, %d ]\n" % (i, j))

    w.write("first_atoms:\n")
    count1 = 0
    count2 = len(dataset["first_atoms"])
    for disp1 in dataset["first_atoms"]:
        count1 += 1
        disp_cart1 = disp1["displacement"]
        w.write("- number: %5d\n" % (disp1["number"] + 1))
        w.write("  displacement:\n")
        w.write(
            "    [%20.16f,%20.16f,%20.16f ] # %05d\n"
            % (disp_cart1[0], disp_cart1[1], disp_cart1[2], count1)
        )
        w.write("  displacement_id: %d\n" % count1)
        w.write("  second_atoms:\n")

        included = None
        atom2_list = np.array(
            [disp2["number"] for disp2 in disp1["second_atoms"]], dtype=int
        )
        _, indices = np.unique(atom2_list, return_index=True)
        for atom2 in atom2_list[indices]:
            disp2_list = []
            for disp2 in disp1["second_atoms"]:
                if disp2["number"] == atom2:
                    disp2_list.append(disp2)

            disp2 = disp2_list[0]
            atom2 = disp2["number"]
            if "included" in disp2:
                included = disp2["included"]
            pair_distance = disp2["pair_distance"]
            w.write("  - number: %5d\n" % (atom2 + 1))
            w.write("    distance: %f\n" % pair_distance)
            if included is not None:
                if included:
                    w.write("    included: %s\n" % "true")
                else:
                    w.write("    included: %s\n" % "false")
            w.write("    displacements:\n")

            for disp2 in disp2_list:
                count2 += 1

                # Assert all disp2s belonging to same atom2 appear straight.
                assert disp2["id"] == count2

                disp_cart2 = disp2["displacement"]
                w.write(
                    "    - [%20.16f,%20.16f,%20.16f ] # %05d\n"
                    % (disp_cart2[0], disp_cart2[1], disp_cart2[2], count2)
                )

            ids = ["%d" % disp2["id"] for disp2 in disp2_list]
            w.write("    displacement_ids: [ %s ]\n" % ", ".join(ids))

    _write_cell_yaml(w, supercell)

    w.close()

    return num_first + num_second, num_disp_files


def write_disp_fc2_yaml(dataset, supercell, filename="disp_fc2.yaml"):
    """Write disp_fc2.yaml.

    This function should not be called from phono3py script from version 3.

    """
    warnings.warn(
        "write_disp_fc2_yaml() is deprecated.", DeprecationWarning, stacklevel=2
    )

    w = open(filename, "w")
    w.write("natom: %d\n" % dataset["natom"])

    num_first = len(dataset["first_atoms"])
    w.write("num_first_displacements: %d\n" % num_first)
    w.write("first_atoms:\n")
    for i, disp1 in enumerate(dataset["first_atoms"]):
        disp_cart1 = disp1["displacement"]
        w.write("- number: %5d\n" % (disp1["number"] + 1))
        w.write("  displacement:\n")
        w.write(
            "    [%20.16f,%20.16f,%20.16f ] # %05d\n"
            % (disp_cart1[0], disp_cart1[1], disp_cart1[2], i + 1)
        )

    if supercell is not None:
        _write_cell_yaml(w, supercell)

    w.close()

    return num_first


def write_FORCES_FC2(disp_dataset, forces_fc2=None, fp=None, filename="FORCES_FC2"):
    """Write FORCES_FC2.

    fp : IO object, optional, default=None
        When this is given, FORCES_FC2 content is written into this IO object.

    """
    if fp is None:
        w = open(filename, "w")
    else:
        w = fp

    for i, disp1 in enumerate(disp_dataset["first_atoms"]):
        w.write("# File: %-5d\n" % (i + 1))
        w.write("# %-5d " % (disp1["number"] + 1))
        w.write("%20.16f %20.16f %20.16f\n" % tuple(disp1["displacement"]))
        if "forces" in disp1 and forces_fc2 is None:
            force_set = disp1["forces"]
        else:
            force_set = forces_fc2[i]
        for forces in force_set:
            w.write("%15.10f %15.10f %15.10f\n" % tuple(forces))

    if fp is None:
        w.close()


def write_FORCES_FC3(disp_dataset, forces_fc3=None, fp=None, filename="FORCES_FC3"):
    """Write FORCES_FC3.

    fp : IO object, optional, default=None
        When this is given, FORCES_FC3 content is written into this IO object.

    """
    if fp is None:
        w = open(filename, "w")
    else:
        w = fp

    natom = disp_dataset["natom"]
    num_disp1 = len(disp_dataset["first_atoms"])
    count = num_disp1
    file_count = num_disp1

    write_FORCES_FC2(disp_dataset, forces_fc2=forces_fc3, fp=w)

    for disp1 in disp_dataset["first_atoms"]:
        atom1 = disp1["number"]
        for disp2 in disp1["second_atoms"]:
            atom2 = disp2["number"]
            w.write("# File: %-5d\n" % (count + 1))
            w.write("# %-5d " % (atom1 + 1))
            w.write("%20.16f %20.16f %20.16f\n" % tuple(disp1["displacement"]))
            w.write("# %-5d " % (atom2 + 1))
            w.write("%20.16f %20.16f %20.16f\n" % tuple(disp2["displacement"]))

            # For supercell calculation reduction
            included = True
            if "included" in disp2:
                included = disp2["included"]
            if included:
                if "forces" in disp2 and forces_fc3 is None:
                    force_set = disp2["forces"]
                else:
                    force_set = forces_fc3[file_count]
                for force in force_set:
                    w.write("%15.10f %15.10f %15.10f\n" % tuple(force))
                file_count += 1
            else:
                # for forces in forces_fc3[i]:
                #     w.write("%15.10f %15.10f %15.10f\n" % (tuple(forces)))
                for _ in range(natom):
                    w.write("%15.10f %15.10f %15.10f\n" % (0, 0, 0))
            count += 1

    if fp is None:
        w.close()


def write_fc3_to_hdf5(fc3, filename="fc3.hdf5", p2s_map=None, compression="gzip"):
    """Write fc3 in fc3.hdf5.

    Parameters
    ----------
    force_constants : ndarray
        Force constants
        shape=(n_satom, n_satom, n_satom, 3, 3, 3) or
        (n_patom, n_satom, n_satom,3,3,3), dtype=double
    filename : str
        Filename to be used.
    p2s_map : ndarray, optional
        Primitive atom indices in supercell index system
        shape=(n_patom,), dtype=intc
    compression : str or int, optional
        h5py's lossless compression filters (e.g., "gzip", "lzf"). None gives
        no compression. See the detail at docstring of
        h5py.Group.create_dataset. Default is "gzip".

    """
    with h5py.File(filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("fc3", data=fc3, compression=compression)
        if p2s_map is not None:
            w.create_dataset("p2s_map", data=p2s_map)


def read_fc3_from_hdf5(filename="fc3.hdf5", p2s_map=None):
    """Read fc3 from fc3.hdf5.

    fc3 can be in full or compact format. They are distinguished by
    the array shape.

    p2s_map is optionally stored in fc3.hdf5, which gives the mapping
    table from primitive cell atoms to corresponding supercell atoms by
    respective indices.

    """
    with h5py.File(filename, "r") as f:
        fc3 = f["fc3"][:]
        if "p2s_map" in f:
            p2s_map_in_file = f["p2s_map"][:]
            check_force_constants_indices(
                fc3.shape[:2], p2s_map_in_file, p2s_map, filename
            )
        if fc3.dtype == np.double and fc3.flags.c_contiguous:
            return fc3
        else:
            msg = (
                "%s has to be read by h5py as numpy ndarray of "
                "dtype='double' and c_contiguous." % filename
            )
            raise TypeError(msg)
    return None


def write_fc2_to_hdf5(
    force_constants,
    filename="fc2.hdf5",
    p2s_map=None,
    physical_unit=None,
    compression="gzip",
):
    """Write fc2 in fc2.hdf5.

    write_force_constants_to_hdf5 was copied from phonopy because
    it in phonopy doesn't support 'version' dataset.

    """

    def write_force_constants_to_hdf5(
        force_constants,
        filename="force_constants.hdf5",
        p2s_map=None,
        physical_unit=None,
        compression=None,
        version=None,
    ):
        try:
            import h5py
        except ImportError as exc:
            raise ModuleNotFoundError("You need to install python-h5py.") from exc

        with h5py.File(filename, "w") as w:
            w.create_dataset(
                "force_constants", data=force_constants, compression=compression
            )
            if p2s_map is not None:
                w.create_dataset("p2s_map", data=p2s_map)
            if physical_unit is not None:
                dset = w.create_dataset(
                    "physical_unit", (1,), dtype="S%d" % len(physical_unit)
                )
                dset[0] = np.bytes_(physical_unit)
            if version is not None:
                w.create_dataset("version", data=np.bytes_(version))

    write_force_constants_to_hdf5(
        force_constants,
        filename=filename,
        p2s_map=p2s_map,
        physical_unit=physical_unit,
        compression=compression,
        version=__version__,
    )


def read_fc2_from_hdf5(filename="fc2.hdf5", p2s_map=None):
    """Read fc2 from fc2.hdf5."""
    return read_force_constants_from_hdf5(
        filename=filename, p2s_map=p2s_map, calculator="vasp"
    )


def write_grid_address_to_hdf5(
    grid_address,
    mesh,
    grid_mapping_table,
    bz_grid=None,
    compression="gzip",
    filename=None,
):
    """Write grid addresses to grid_address.hdf5."""
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "grid_address" + suffix + ".hdf5"
    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("mesh", data=mesh)
        if bz_grid is not None and bz_grid.grid_matrix is not None:
            w.create_dataset("grid_matrix", data=bz_grid.grid_matrix)
            w.create_dataset("P_matrix", data=bz_grid.P)
            w.create_dataset("Q_matrix", data=bz_grid.Q)
        w.create_dataset("grid_address", data=grid_address, compression=compression)
        w.create_dataset(
            "grid_mapping_table", data=grid_mapping_table, compression=compression
        )
        return full_filename
    return None


def write_imag_self_energy_at_grid_point(
    gp,
    band_indices,
    mesh,
    frequencies,
    gammas,
    sigma=None,
    temperature=None,
    scattering_event_class=None,
    filename=None,
    is_mesh_symmetry=True,
):
    """Write imaginary part of self-energy spectrum in gamma-*.dat."""
    gammas_filename = "gammas"
    gammas_filename += "-m%d%d%d-g%d-" % (mesh[0], mesh[1], mesh[2], gp)
    if sigma is not None:
        gammas_filename += "s" + _del_zeros(sigma) + "-"

    if temperature is not None:
        gammas_filename += "t" + _del_zeros(temperature) + "-"

    for i in band_indices:
        gammas_filename += "b%d" % (i + 1)

    if scattering_event_class is not None:
        gammas_filename += "-c%d" % scattering_event_class

    if filename is not None:
        gammas_filename += ".%s" % filename
    elif not is_mesh_symmetry:
        gammas_filename += ".nosym"
    gammas_filename += ".dat"

    w = open(gammas_filename, "w")
    for freq, g in zip(frequencies, gammas):
        w.write("%15.7f %20.15e\n" % (freq, g))
    w.close()

    return gammas_filename


def write_joint_dos(
    gp,
    mesh,
    frequencies,
    jdos,
    sigma=None,
    temperatures=None,
    filename=None,
    is_mesh_symmetry=True,
):
    """Write joint-DOS spectrum in jdos-*.dat."""
    if temperatures is None:
        return _write_joint_dos_at_t(
            gp,
            mesh,
            frequencies,
            jdos[0],
            sigma=sigma,
            temperature=None,
            filename=filename,
            is_mesh_symmetry=is_mesh_symmetry,
        )
    else:
        for jdos_at_t, t in zip(jdos, temperatures):
            return _write_joint_dos_at_t(
                gp,
                mesh,
                frequencies,
                jdos_at_t,
                sigma=sigma,
                temperature=t,
                filename=filename,
                is_mesh_symmetry=is_mesh_symmetry,
            )


def _write_joint_dos_at_t(
    grid_point,
    mesh,
    frequencies,
    jdos,
    sigma=None,
    temperature=None,
    filename=None,
    is_mesh_symmetry=True,
):
    suffix = _get_filename_suffix(
        mesh, grid_point=grid_point, sigma=sigma, filename=filename
    )
    jdos_filename = "jdos%s" % suffix
    if temperature is not None:
        jdos_filename += "-t" + _del_zeros(temperature)
    if not is_mesh_symmetry:
        jdos_filename += ".nosym"
    if filename is not None:
        jdos_filename += ".%s" % filename
    jdos_filename += ".dat"

    with open(jdos_filename, "w") as w:
        for omega, vals in zip(frequencies, jdos):
            w.write("%15.7f" % omega)
            w.write((" %20.15e" * len(vals)) % tuple(vals))
            w.write("\n")
        return jdos_filename


def write_real_self_energy_at_grid_point(
    gp,
    band_indices,
    frequency_points,
    deltas,
    mesh,
    epsilon,
    temperature,
    filename=None,
    is_mesh_symmetry=True,
):
    """Write real part of self-energy spectrum in deltas-*.dat."""
    deltas_filename = "deltas"
    deltas_filename += _get_filename_suffix(mesh, grid_point=gp)
    if epsilon > 1e-5:
        deltas_filename += "-e" + _del_zeros(epsilon)
    else:
        deltas_filename += "-e%.3e" % epsilon
    if temperature is not None:
        deltas_filename += "-t" + _del_zeros(temperature) + "-"
    for i in band_indices:
        deltas_filename += "b%d" % (i + 1)
    if filename is not None:
        deltas_filename += ".%s" % filename
    elif not is_mesh_symmetry:
        deltas_filename += ".nosym"
    deltas_filename += ".dat"

    with open(deltas_filename, "w") as w:
        for freq, v in zip(frequency_points, deltas):
            w.write("%15.7f %20.15e\n" % (freq, v))

    return deltas_filename


def write_real_self_energy_to_hdf5(
    grid_point,
    band_indices,
    temperatures,
    deltas,
    mesh,
    epsilon,
    bz_grid=None,
    frequency_points=None,
    frequencies=None,
    filename=None,
):
    """Wirte real part of self energy (currently only bubble) in hdf5.

    deltas : ndarray
        Real part of self energy.

        With frequency_points:
            shape=(temperatures, band_indices, frequency_points),
            dtype='double', order='C'
        otherwise:
            shape=(temperatures, band_indices), dtype='double', order='C'

    """
    full_filename = "deltas"
    suffix = _get_filename_suffix(mesh, grid_point=grid_point)
    _band_indices = np.array(band_indices, dtype="intc")

    full_filename += suffix
    if epsilon > 1e-5:
        full_filename += "-e" + _del_zeros(epsilon)
    else:
        full_filename += "-e%.3e" % epsilon
    full_filename += ".hdf5"

    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("grid_point", data=grid_point)
        w.create_dataset("mesh", data=mesh)
        if bz_grid is not None and bz_grid.grid_matrix is not None:
            w.create_dataset("grid_matrix", data=bz_grid.grid_matrix)
            w.create_dataset("P_matrix", data=bz_grid.P)
            w.create_dataset("Q_matrix", data=bz_grid.Q)
        w.create_dataset("band_index", data=_band_indices)
        w.create_dataset("delta", data=deltas)
        w.create_dataset("temperature", data=temperatures)
        w.create_dataset("epsilon", data=epsilon)
        if frequency_points is not None:
            w.create_dataset("frequency_points", data=frequency_points)
        if frequencies is not None:
            w.create_dataset("frequency", data=frequencies)

    return full_filename


def write_spectral_function_at_grid_point(
    gp,
    band_indices,
    frequency_points,
    spectral_functions,
    mesh,
    temperature,
    sigma=None,
    filename=None,
    is_mesh_symmetry=True,
):
    """Write spectral function spectrum in spectral-*.dat."""
    spectral_filename = "spectral"
    spectral_filename += _get_filename_suffix(mesh, grid_point=gp, sigma=sigma)
    if temperature is not None:
        spectral_filename += "-t" + _del_zeros(temperature) + "-"
    for i in band_indices:
        spectral_filename += "b%d" % (i + 1)
    if filename is not None:
        spectral_filename += ".%s" % filename
    elif not is_mesh_symmetry:
        spectral_filename += ".nosym"
    spectral_filename += ".dat"

    with open(spectral_filename, "w") as w:
        for freq, v in zip(frequency_points, spectral_functions):
            w.write("%15.7f %20.15e\n" % (freq, v))

    return spectral_filename


def write_spectral_function_to_hdf5(
    grid_point,
    band_indices,
    temperatures,
    spectral_functions,
    shifts,
    half_linewidths,
    mesh,
    bz_grid=None,
    sigma=None,
    frequency_points=None,
    frequencies=None,
    all_band_exist=False,
    filename=None,
):
    """Wirte spectral functions (currently only bubble) in hdf5.

    spectral_functions : ndarray
        Spectral functions.
        shape=(temperature, band_index, frequency_points),
        dtype='double', order='C'

    """
    full_filename = "spectral"
    if all_band_exist:
        _band_indices = None
    else:
        _band_indices = np.hstack(band_indices).astype("int_")
    suffix = _get_filename_suffix(
        mesh, grid_point=grid_point, band_indices=_band_indices, sigma=sigma
    )
    _band_indices = np.array(band_indices, dtype="intc")

    full_filename += suffix
    if filename is not None:
        full_filename += f".{filename}"
    full_filename += ".hdf5"

    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("grid_point", data=grid_point)
        w.create_dataset("mesh", data=mesh)
        if bz_grid is not None and bz_grid.grid_matrix is not None:
            w.create_dataset("grid_matrix", data=bz_grid.grid_matrix)
            w.create_dataset("P_matrix", data=bz_grid.P)
            w.create_dataset("Q_matrix", data=bz_grid.Q)
        w.create_dataset("band_index", data=_band_indices)
        w.create_dataset("spectral_function", data=spectral_functions)
        w.create_dataset("shift", data=shifts)
        w.create_dataset("half_linewidth", data=half_linewidths)
        w.create_dataset("temperature", data=temperatures)
        if frequency_points is not None:
            w.create_dataset("frequency_point", data=frequency_points)
        if frequencies is not None:
            w.create_dataset("frequency", data=frequencies)

    return full_filename


def write_collision_to_hdf5(
    temperature,
    mesh,
    gamma=None,
    gamma_isotope=None,
    collision_matrix=None,
    grid_point=None,
    band_index=None,
    sigma=None,
    sigma_cutoff=None,
    filename=None,
):
    """Write collision matrix to collision-*.hdf5."""
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        band_indices=band_indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
    )
    full_filename = "collision" + suffix + ".hdf5"
    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("temperature", data=temperature)
        if gamma is not None:
            w.create_dataset("gamma", data=gamma)
        if gamma_isotope is not None:
            w.create_dataset("gamma_isotope", data=gamma_isotope)
        if collision_matrix is not None:
            w.create_dataset("collision_matrix", data=collision_matrix)
        if grid_point is not None:
            w.create_dataset("grid_point", data=grid_point)
        if band_index is not None:
            w.create_dataset("band_index", data=(band_index + 1))
        if sigma is not None:
            w.create_dataset("sigma", data=sigma)
        if sigma_cutoff is not None:
            w.create_dataset("sigma_cutoff_width", data=sigma_cutoff)

        text = "Collisions "
        if grid_point is not None:
            text += "at grid adress %d " % grid_point
        if sigma is not None:
            if grid_point is not None:
                text += "and "
            else:
                text += "at "
            text += "sigma %s " % _del_zeros(sigma)
        text += "were written into "
        if sigma is not None:
            text += "\n"
        text += '"%s".' % ("collision" + suffix + ".hdf5")
        print(text)

    return full_filename


def write_full_collision_matrix(collision_matrix, filename="fcm.hdf5"):
    """Write full (non-symmetrized) collision matrix to collision-*.hdf5."""
    with h5py.File(filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("collision_matrix", data=collision_matrix)


def write_unitary_matrix_to_hdf5(
    temperature,
    mesh,
    unitary_matrix=None,
    sigma=None,
    sigma_cutoff=None,
    solver=None,
    filename=None,
    verbose=False,
):
    """Write eigenvectors of collision matrices at temperatures.

    Depending on the choice of the solver, eigenvectors are sotred in
    either column-wise or row-wise.

    """
    suffix = _get_filename_suffix(
        mesh, sigma=sigma, sigma_cutoff=sigma_cutoff, filename=filename
    )
    hdf5_filename = "unitary" + suffix + ".hdf5"
    with h5py.File(hdf5_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("temperature", data=temperature)
        if unitary_matrix is not None:
            w.create_dataset("unitary_matrix", data=unitary_matrix)
        if solver is not None:
            w.create_dataset("solver", data=solver)

        if verbose:
            if len(temperature) > 1:
                text = "Unitary matrices "
            else:
                text = "Unitary matrix "
            if sigma is not None:
                text += "at sigma %s " % _del_zeros(sigma)
                if sigma_cutoff is not None:
                    text += "(%4.2f SD) " % sigma_cutoff
            if len(temperature) > 1:
                text += "were written into "
            else:
                text += "was written into "
            if sigma is not None:
                text += "\n"
            text += '"%s".' % hdf5_filename
            print(text)


def write_collision_eigenvalues_to_hdf5(
    temperatures,
    mesh,
    collision_eigenvalues,
    sigma=None,
    sigma_cutoff=None,
    filename=None,
    verbose=True,
):
    """Write eigenvalues of collision matrix to coleigs-*.hdf5."""
    suffix = _get_filename_suffix(
        mesh, sigma=sigma, sigma_cutoff=sigma_cutoff, filename=filename
    )
    with h5py.File("coleigs" + suffix + ".hdf5", "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("temperature", data=temperatures)
        w.create_dataset("collision_eigenvalues", data=collision_eigenvalues)
        w.close()

        if verbose:
            text = "Eigenvalues of collision matrix "
            if sigma is not None:
                text += "with sigma %s\n" % sigma
            text += "were written into "
            text += '"%s"' % ("coleigs" + suffix + ".hdf5")
            print(text)


def write_kappa_to_hdf5(
    temperature,
    mesh,
    boundary_mfp: float = None,
    bz_grid=None,
    frequency=None,
    group_velocity=None,
    gv_by_gv=None,
    velocity_operator=None,
    mean_free_path=None,
    heat_capacity=None,
    kappa=None,
    mode_kappa=None,
    kappa_P_exact=None,
    kappa_P_RTA=None,
    kappa_C=None,
    kappa_TOT_exact=None,
    kappa_TOT_RTA=None,
    mode_kappa_P_exact=None,  # k_P from the exact solution of the LBTE
    mode_kappa_P_RTA=None,  # k_P in the RTA calculated in the LBTE
    mode_kappa_C=None,
    kappa_RTA=None,  # RTA calculated in LBTE
    mode_kappa_RTA=None,  # RTA calculated in LBTE
    f_vector=None,
    gamma=None,
    gamma_isotope=None,
    gamma_N=None,
    gamma_U=None,
    averaged_pp_interaction=None,
    qpoint=None,
    weight=None,
    grid_point=None,
    band_index=None,
    sigma=None,
    sigma_cutoff=None,
    kappa_unit_conversion=None,
    compression="gzip",
    filename=None,
    verbose=True,
):
    """Write thermal conductivity related properties in kappa-*.hdf5."""
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        band_indices=band_indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
    )
    full_filename = "kappa" + suffix + ".hdf5"
    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("temperature", data=temperature)
        w.create_dataset("mesh", data=mesh)
        if bz_grid is not None and bz_grid.grid_matrix is not None:
            w.create_dataset("grid_matrix", data=bz_grid.grid_matrix)
            w.create_dataset("P_matrix", data=bz_grid.P)
            w.create_dataset("Q_matrix", data=bz_grid.Q)
        if frequency is not None:
            if isinstance(frequency, np.floating):
                w.create_dataset("frequency", data=frequency)
            else:
                w.create_dataset("frequency", data=frequency, compression=compression)
        if group_velocity is not None:
            w.create_dataset(
                "group_velocity", data=group_velocity, compression=compression
            )
        if gv_by_gv is not None:
            w.create_dataset("gv_by_gv", data=gv_by_gv)
        if velocity_operator is not None:
            w.create_dataset(
                "velocity_operator", data=velocity_operator, compression=compression
            )
        # if mean_free_path is not None:
        #     w.create_dataset('mean_free_path', data=mean_free_path,
        #                      compression=compression)
        if heat_capacity is not None:
            w.create_dataset(
                "heat_capacity", data=heat_capacity, compression=compression
            )
        if kappa is not None:
            w.create_dataset("kappa", data=kappa)
        if mode_kappa is not None:
            w.create_dataset("mode_kappa", data=mode_kappa, compression=compression)
        if kappa_RTA is not None:
            w.create_dataset("kappa_RTA", data=kappa_RTA)
        if mode_kappa_RTA is not None:
            w.create_dataset(
                "mode_kappa_RTA", data=mode_kappa_RTA, compression=compression
            )
        if kappa_P_exact is not None:
            w.create_dataset("kappa_P_exact", data=kappa_P_exact)
        if kappa_P_RTA is not None:
            w.create_dataset("kappa_P_RTA", data=kappa_P_RTA)
        if kappa_C is not None:
            w.create_dataset("kappa_C", data=kappa_C)
        if kappa_TOT_exact is not None:
            w.create_dataset("kappa_TOT_exact", data=kappa_TOT_exact)
        if kappa_TOT_RTA is not None:
            w.create_dataset("kappa_TOT_RTA", data=kappa_TOT_RTA)
        if mode_kappa_P_exact is not None:
            w.create_dataset(
                "mode_kappa_P_exact", data=mode_kappa_P_exact, compression=compression
            )
        if mode_kappa_P_RTA is not None:
            w.create_dataset(
                "mode_kappa_P_RTA", data=mode_kappa_P_RTA, compression=compression
            )
        if mode_kappa_C is not None:
            w.create_dataset("mode_kappa_C", data=mode_kappa_C, compression=compression)
        if f_vector is not None:
            w.create_dataset("f_vector", data=f_vector, compression=compression)
        if gamma is not None:
            w.create_dataset("gamma", data=gamma, compression=compression)
        if gamma_isotope is not None:
            w.create_dataset(
                "gamma_isotope", data=gamma_isotope, compression=compression
            )
        if gamma_N is not None:
            w.create_dataset("gamma_N", data=gamma_N, compression=compression)
        if gamma_U is not None:
            w.create_dataset("gamma_U", data=gamma_U, compression=compression)
        if averaged_pp_interaction is not None:
            w.create_dataset(
                "ave_pp", data=averaged_pp_interaction, compression=compression
            )
        if qpoint is not None:
            w.create_dataset("qpoint", data=qpoint, compression=compression)
        if weight is not None:
            w.create_dataset("weight", data=weight, compression=compression)
        if grid_point is not None:
            w.create_dataset("grid_point", data=grid_point)
        if band_index is not None:
            w.create_dataset("band_index", data=(band_index + 1))
        if sigma is not None:
            w.create_dataset("sigma", data=sigma)
        if sigma_cutoff is not None:
            w.create_dataset("sigma_cutoff_width", data=sigma_cutoff)
        if kappa_unit_conversion is not None:
            w.create_dataset("kappa_unit_conversion", data=kappa_unit_conversion)
        if boundary_mfp is not None:
            w.create_dataset("boundary_mfp", data=boundary_mfp)

        if verbose:
            text = "Thermal conductivity related properties "
            if grid_point is not None:
                try:
                    gp_text = "at gp-%d " % grid_point
                    text += gp_text
                except TypeError:
                    pass
                if band_index is not None:
                    text += "and band_index-%d\n" % (band_index + 1)
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % sigma
                if sigma_cutoff is None:
                    text += "\n"
                else:
                    text += "(%4.2f SD)\n" % sigma_cutoff
                text += "were written into "
            else:
                text += "were written into "
                if band_index is None:
                    text += "\n"
            text += '"%s".' % full_filename
            print(text)

        return full_filename


def read_gamma_from_hdf5(
    mesh,
    grid_point=None,
    band_index=None,
    sigma=None,
    sigma_cutoff=None,
    filename=None,
):
    """Read gamma from kappa-*.hdf5 file."""
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        band_indices=band_indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
    )
    full_filename = "kappa" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        return None, full_filename

    read_data = {}

    with h5py.File(full_filename, "r") as f:
        read_data["gamma"] = f["gamma"][:]
        for key in ("gamma_isotope", "ave_pp", "gamma_N", "gamma_U"):
            if key in f.keys():
                if len(f[key].shape) > 0:
                    read_data[key] = f[key][:]
                else:
                    read_data[key] = f[key][()]

    return read_data, full_filename


def read_collision_from_hdf5(
    mesh,
    indices=None,
    grid_point=None,
    band_index=None,
    sigma=None,
    sigma_cutoff=None,
    filename=None,
    only_temperatures=False,
    verbose=True,
):
    """Read colliction matrix.

    indices : array_like of int
        Indices of temperatures.

    """
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        band_indices=band_indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
    )
    full_filename = "collision" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        if verbose:
            print("%s not found." % full_filename)
        return None

    if only_temperatures:
        with h5py.File(full_filename, "r") as f:
            temperatures = np.array(f["temperature"][:], dtype="double")
            return None, None, temperatures

    with h5py.File(full_filename, "r") as f:
        if indices == "all":
            colmat_shape = (1,) + f["collision_matrix"].shape
            collision_matrix = np.zeros(colmat_shape, dtype="double", order="C")
            gamma = np.array(f["gamma"][:], dtype="double", order="C")
            collision_matrix[0] = f["collision_matrix"][:]
            temperatures = np.array(f["temperature"][:], dtype="double")
        else:
            colmat_shape = (1, len(indices)) + f["collision_matrix"].shape[1:]
            collision_matrix = np.zeros(colmat_shape, dtype="double")
            gamma = np.array(f["gamma"][indices], dtype="double", order="C")
            collision_matrix[0] = f["collision_matrix"][indices]
            temperatures = np.array(f["temperature"][indices], dtype="double")

        if verbose:
            text = "Collisions "
            if band_index is None:
                if grid_point is not None:
                    text += "at grid point %d " % grid_point
            else:
                if grid_point is not None:
                    text += "at (grid point %d, band index %d) " % (
                        grid_point,
                        band_index,
                    )
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % _del_zeros(sigma)
                if sigma_cutoff is not None:
                    text += "(%4.2f SD)" % sigma_cutoff
            if band_index is None and grid_point is not None:
                text += " were read from "
                text += "\n"
            else:
                text += "\n"
                text += "were read from "
            text += '"%s".' % full_filename
            print(text)

        return collision_matrix, gamma, temperatures


def write_pp_to_hdf5(
    mesh,
    pp=None,
    g_zero=None,
    grid_point=None,
    triplet=None,
    weight=None,
    triplet_map=None,
    triplet_all=None,
    sigma=None,
    sigma_cutoff=None,
    filename=None,
    verbose=True,
    check_consistency=False,
    compression="gzip",
):
    """Write ph-ph interaction strength in its hdf5 file."""
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
    )
    full_filename = "pp" + suffix + ".hdf5"

    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        if pp is not None:
            if g_zero is None:
                w.create_dataset("pp", data=pp, compression=compression)
                if triplet is not None:
                    w.create_dataset("triplet", data=triplet, compression=compression)
                if weight is not None:
                    w.create_dataset("weight", data=weight, compression=compression)
                if triplet_map is not None:
                    w.create_dataset(
                        "triplet_map", data=triplet_map, compression=compression
                    )
                if triplet_all is not None:
                    w.create_dataset(
                        "triplet_all", data=triplet_all, compression=compression
                    )
            else:
                x = g_zero.ravel()
                nonzero_pp = np.array(pp.ravel()[x == 0], dtype="double")
                bytelen = len(x) // 8
                remlen = len(x) % 8
                y = x[: bytelen * 8].reshape(-1, 8)
                z = np.packbits(y)
                if remlen != 0:
                    z_rem = np.packbits(x[bytelen * 8 :])

                # No compression for pp because already almost random.
                w.create_dataset("nonzero_pp", data=nonzero_pp, compression=None)
                w.create_dataset("pp_shape", data=pp.shape, compression=compression)
                w.create_dataset("g_zero_bits", data=z, compression=compression)
                if remlen != 0:
                    w.create_dataset("g_zero_bits_reminder", data=z_rem)

                # This is only for the test and coupled with read_pp_from_hdf5.
                if check_consistency:
                    w.create_dataset("pp", data=pp, compression=compression)
                    w.create_dataset("g_zero", data=g_zero, compression=compression)

        if verbose:
            text = ""
            text += "Ph-ph interaction strength "
            if grid_point is not None:
                text += "at gp-%d " % grid_point
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % sigma
                if sigma_cutoff is None:
                    text += "\n"
                else:
                    text += "(%4.2f SD)\n" % sigma_cutoff
                text += "were written into "
            else:
                text += "were written into "
                text += "\n"
            text += '"%s".' % full_filename
            print(text)

        return full_filename


def read_pp_from_hdf5(
    mesh,
    grid_point=None,
    sigma=None,
    sigma_cutoff=None,
    filename=None,
    verbose=True,
    check_consistency=False,
):
    """Read ph-ph interaction strength from its hdf5 file."""
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
    )
    full_filename = "pp" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        if verbose:
            print("%s not found." % full_filename)
        return None

    with h5py.File(full_filename, "r") as f:
        if "nonzero_pp" in f:
            nonzero_pp = f["nonzero_pp"][:]
            pp_shape = f["pp_shape"][:]
            z = f["g_zero_bits"][:]
            bytelen = np.prod(pp_shape) // 8
            remlen = 0
            if "g_zero_bits_reminder" in f:
                z_rem = f["g_zero_bits_reminder"][:]
                remlen = np.prod(pp_shape) - bytelen * 8

            bits = np.unpackbits(z)
            if not bits.flags["C_CONTIGUOUS"]:
                bits = np.array(bits, dtype="uint8")

            g_zero = np.zeros(pp_shape, dtype="byte", order="C")
            b = g_zero.ravel()
            b[: (bytelen * 8)] = bits
            if remlen != 0:
                b[-remlen:] = np.unpackbits(z_rem)[:remlen]

            pp = np.zeros(pp_shape, dtype="double", order="C")
            pp_ravel = pp.ravel()
            pp_ravel[g_zero.ravel() == 0] = nonzero_pp

            # check_consistency==True in write_pp_to_hdf5 required.
            if check_consistency and g_zero is not None:
                if verbose:
                    print("Checking consistency of ph-ph interanction " "strength.")
                assert (g_zero == f["g_zero"][:]).all()
                assert np.allclose(pp, f["pp"][:])
        else:
            pp = np.zeros(f["pp"].shape, dtype="double", order="C")
            pp[:] = f["pp"][:]
            g_zero = None

        if verbose:
            print('Ph-ph interaction strength was read from "%s".' % full_filename)

        return pp, g_zero

    return None


def write_gamma_detail_to_hdf5(
    temperature,
    mesh,
    bz_grid=None,
    gamma_detail=None,
    grid_point=None,
    triplet=None,
    weight=None,
    triplet_map=None,
    triplet_all=None,
    frequency_points=None,
    band_index=None,
    sigma=None,
    sigma_cutoff=None,
    compression="gzip",
    filename=None,
    verbose=True,
):
    """Write detailed gamma in its hdf5 file."""
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        band_indices=band_indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
    )
    full_filename = "gamma_detail" + suffix + ".hdf5"

    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("temperature", data=temperature)
        w.create_dataset("mesh", data=mesh)
        if bz_grid is not None and bz_grid.grid_matrix is not None:
            w.create_dataset("grid_matrix", data=bz_grid.grid_matrix)
            w.create_dataset("P_matrix", data=bz_grid.P)
            w.create_dataset("Q_matrix", data=bz_grid.Q)
        if gamma_detail is not None:
            w.create_dataset("gamma_detail", data=gamma_detail, compression=compression)
        if triplet is not None:
            w.create_dataset("triplet", data=triplet, compression=compression)
        if weight is not None:
            w.create_dataset("weight", data=weight, compression=compression)
        if triplet_map is not None:
            w.create_dataset("triplet_map", data=triplet_map, compression=compression)
        if triplet_all is not None:
            w.create_dataset("triplet_all", data=triplet_all, compression=compression)
        if grid_point is not None:
            w.create_dataset("grid_point", data=grid_point)
        if band_index is not None:
            w.create_dataset("band_index", data=(band_index + 1))
        if sigma is not None:
            w.create_dataset("sigma", data=sigma)
        if sigma_cutoff is not None:
            w.create_dataset("sigma_cutoff_width", data=sigma_cutoff)
        if frequency_points is not None:
            w.create_dataset("frequency_point", data=frequency_points)

        if verbose:
            text = ""
            text += "Phonon triplets contributions to Gamma "
            if grid_point is not None:
                text += "at gp-%d " % grid_point
                if band_index is not None:
                    text += "and band_index-%d\n" % (band_index + 1)
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % sigma
                if sigma_cutoff is None:
                    text += "\n"
                else:
                    text += "(%4.2f SD)\n" % sigma_cutoff
                text += "were written into "
            else:
                text += "were written into "
                if band_index is None:
                    text += "\n"
            text += '"%s".' % full_filename
            print(text)

        return full_filename

    return None


def write_phonon_to_hdf5(
    frequency,
    eigenvector,
    grid_address,
    mesh,
    bz_grid=None,
    ir_grid_points=None,
    ir_grid_weights=None,
    compression="gzip",
    filename=None,
):
    """Write phonon on grid in its hdf5 file."""
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "phonon" + suffix + ".hdf5"

    with h5py.File(full_filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("mesh", data=mesh)
        if bz_grid is not None and bz_grid.grid_matrix is not None:
            w.create_dataset("grid_matrix", data=bz_grid.grid_matrix)
            w.create_dataset("P_matrix", data=bz_grid.P)
            w.create_dataset("Q_matrix", data=bz_grid.Q)
        w.create_dataset("grid_address", data=grid_address, compression=compression)
        w.create_dataset("frequency", data=frequency, compression=compression)
        w.create_dataset("eigenvector", data=eigenvector, compression=compression)
        if ir_grid_points is not None:
            w.create_dataset(
                "ir_grid_points", data=ir_grid_points, compression=compression
            )
        if ir_grid_weights is not None:
            w.create_dataset(
                "ir_grid_weights", data=ir_grid_weights, compression=compression
            )
        return full_filename

    return None


def read_phonon_from_hdf5(mesh, filename=None, verbose=True):
    """Read phonon from its hdf5 file."""
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "phonon" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        if verbose:
            print("%s not found." % full_filename)
        return None

    with h5py.File(full_filename, "r") as f:
        frequencies = np.array(f["frequency"][:], dtype="double", order="C")
        itemsize = frequencies.itemsize
        eigenvectors = np.array(
            f["eigenvector"][:], dtype=("c%d" % (itemsize * 2)), order="C"
        )
        mesh_in_file = np.array(f["mesh"][:], dtype="intc")
        grid_address = np.array(f["grid_address"][:], dtype="intc", order="C")

        assert (mesh_in_file == mesh).all(), "Mesh numbers are inconsistent."

        if verbose:
            print('Phonons were read from "%s".' % full_filename, flush=True)

        return frequencies, eigenvectors, grid_address

    return None


def write_ir_grid_points(bz_grid, grid_points, grid_weights, primitive_lattice):
    """Write ir-grid-points in yaml."""
    lines = []
    lines.append("mesh: [ %d, %d, %d ]" % tuple(bz_grid.D_diag))
    lines.append("reciprocal_lattice:")
    for vec, axis in zip(primitive_lattice.T, ("a*", "b*", "c*")):
        lines.append("- [ %12.8f, %12.8f, %12.8f ] # %2s" % (tuple(vec) + (axis,)))
    lines.append("microzone_lattice:")
    for vec, axis in zip(bz_grid.microzone_lattice.T, ("a*", "b*", "c*")):
        lines.append("- [ %12.8f, %12.8f, %12.8f ] # %2s" % (tuple(vec) + (axis,)))
    lines.append("num_reduced_ir_grid_points: %d" % len(grid_points))
    lines.append("ir_grid_points:  # [address, weight]")

    for g, weight in zip(grid_points, grid_weights):
        lines.append("- grid_point: %d" % g)
        lines.append("  weight: %d" % weight)
        lines.append(
            "  grid_address: [ %12d, %12d, %12d ]" % tuple(bz_grid.addresses[g])
        )
        q = np.dot(bz_grid.addresses[g], bz_grid.QDinv.T)
        lines.append("  q-point:      [ %12.7f, %12.7f, %12.7f ]" % tuple(q))
    lines.append("")

    with open("ir_grid_points.yaml", "w") as w:
        w.write("\n".join(lines))


def parse_disp_fc2_yaml(filename="disp_fc2.yaml", return_cell=False):
    """Parse disp_fc2.yaml file.

    This function should not be called from phono3py script from version 3.

    """
    warnings.warn(
        "parse_disp_fc2_yaml() is deprecated.", DeprecationWarning, stacklevel=2
    )

    dataset = _parse_yaml(filename)
    natom = dataset["natom"]
    new_dataset = {}
    new_dataset["natom"] = natom
    new_first_atoms = []
    for first_atoms in dataset["first_atoms"]:
        first_atoms["number"] -= 1
        atom1 = first_atoms["number"]
        disp1 = first_atoms["displacement"]
        new_first_atoms.append({"number": atom1, "displacement": disp1})
    new_dataset["first_atoms"] = new_first_atoms

    if return_cell:
        cell = get_cell_from_disp_yaml(dataset)
        return new_dataset, cell
    else:
        return new_dataset


def parse_disp_fc3_yaml(filename="disp_fc3.yaml", return_cell=False):
    """Parse disp_fc3.yaml file.

    This function should not be called from phono3py script from version 3.

    """
    warnings.warn(
        "parse_disp_fc3_yaml() is deprecated.", DeprecationWarning, stacklevel=2
    )

    dataset = _parse_yaml(filename)
    natom = dataset["natom"]
    new_dataset = {}
    new_dataset["natom"] = natom
    if "cutoff_distance" in dataset:
        new_dataset["cutoff_distance"] = dataset["cutoff_distance"]
    new_first_atoms = []
    for first_atoms in dataset["first_atoms"]:
        atom1 = first_atoms["number"] - 1
        disp1 = first_atoms["displacement"]
        new_second_atoms = []
        for second_atom in first_atoms["second_atoms"]:
            disp2_dataset = {"number": second_atom["number"] - 1}
            if "included" in second_atom:
                disp2_dataset.update({"included": second_atom["included"]})
            if "distance" in second_atom:
                disp2_dataset.update({"pair_distance": second_atom["distance"]})
            for disp2 in second_atom["displacements"]:
                disp2_dataset.update({"displacement": disp2})
                new_second_atoms.append(disp2_dataset.copy())
        new_first_atoms.append(
            {"number": atom1, "displacement": disp1, "second_atoms": new_second_atoms}
        )
    new_dataset["first_atoms"] = new_first_atoms

    if return_cell:
        cell = get_cell_from_disp_yaml(dataset)
        return new_dataset, cell
    else:
        return new_dataset


def parse_FORCES_FC2(disp_dataset, filename="FORCES_FC2", unit_conversion_factor=None):
    """Parse type1 FORCES_FC2 file and store forces in disp_dataset."""
    num_atom = disp_dataset["natom"]
    num_disp = len(disp_dataset["first_atoms"])
    forces_fc2 = []
    with open(filename, "r") as f2:
        for _ in range(num_disp):
            forces = _parse_force_lines(f2, num_atom)
            if forces is None:
                return []
            else:
                forces_fc2.append(forces)

    for i, disp1 in enumerate(disp_dataset["first_atoms"]):
        if unit_conversion_factor is not None:
            disp1["forces"] = forces_fc2[i] * unit_conversion_factor
        else:
            disp1["forces"] = forces_fc2[i]


def parse_FORCES_FC3(
    disp_dataset, filename="FORCES_FC3", use_loadtxt=False, unit_conversion_factor=None
):
    """Parse type1 FORCES_FC3 and store forces in disp_dataset."""
    num_atom = disp_dataset["natom"]
    num_disp = len(disp_dataset["first_atoms"])
    for disp1 in disp_dataset["first_atoms"]:
        num_disp += len(disp1["second_atoms"])

    if use_loadtxt:
        forces_fc3 = np.loadtxt(filename).reshape((num_disp, -1, 3))
    else:
        forces_fc3 = np.zeros((num_disp, num_atom, 3), dtype="double", order="C")
        with open(filename, "r") as f3:
            for i in range(num_disp):
                forces = _parse_force_lines(f3, num_atom)
                if forces is None:
                    raise RuntimeError("Failed to parse %s." % filename)
                else:
                    forces_fc3[i] = forces

    if unit_conversion_factor is not None:
        forces_fc3 *= unit_conversion_factor

    i = 0
    for disp1 in disp_dataset["first_atoms"]:
        disp1["forces"] = forces_fc3[i]
        i += 1

    for disp1 in disp_dataset["first_atoms"]:
        for disp2 in disp1["second_atoms"]:
            disp2["forces"] = forces_fc3[i]
            i += 1


def get_filename_suffix(
    mesh,
    grid_point=None,
    band_indices=None,
    sigma=None,
    sigma_cutoff=None,
    temperature=None,
    filename=None,
):
    """Return filename suffix corresponding to parameters."""
    return _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        band_indices=band_indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        temperature=temperature,
        filename=filename,
    )


def get_length_of_first_line(f):
    """Return length of first line of text file.

    This is used to distinguish the data format of the text file.
    Empty lines and lines starting with # are ignored.

    """
    for line in f:
        if line.strip() == "":
            continue
        elif line.strip()[0] == "#":
            continue
        else:
            f.seek(0)
            return len(line.split())

    raise RuntimeError("File doesn't contain relevant infomration.")


def _get_filename_suffix(
    mesh,
    grid_point: Optional[int] = None,
    band_indices: Optional[Union[np.ndarray, Sequence]] = None,
    sigma: Optional[float] = None,
    sigma_cutoff: Optional[float] = None,
    temperature: Optional[float] = None,
    filename: Optional[str] = None,
):
    """Return filename suffix corresponding to parameters."""
    suffix = "-m%d%d%d" % tuple(mesh)
    try:
        gp_suffix = "-g%d" % grid_point
        suffix += gp_suffix
    except TypeError:
        pass
    if band_indices is not None:
        suffix += "-"
        for bi in band_indices:
            suffix += "b%d" % (bi + 1)
    if sigma is not None:
        suffix += "-s" + _del_zeros(sigma)
        if sigma_cutoff is not None:
            sigma_cutoff_str = _del_zeros(sigma_cutoff)
            suffix += "-sd" + sigma_cutoff_str
    if temperature is not None:
        suffix += "-t" + _del_zeros(temperature)
    if filename is not None:
        suffix += "." + filename

    return suffix


def _del_zeros(val):
    """Remove trailing zeros after decimal point."""
    return ("%f" % val).rstrip("0").rstrip(r"\.")


def _parse_yaml(file_yaml):
    """Open yaml file and return the dictionary.

    Used only from parse_disp_fc3_yaml and parse_disp_fc2_yaml.
    So this is obsolete at v2 and later versions.

    """
    warnings.warn("_parse_yaml() is deprecated.", DeprecationWarning, stacklevel=2)

    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(file_yaml) as f:
        string = f.read()
    data = yaml.load(string, Loader=Loader)
    return data


def _parse_force_lines(forcefile, num_atom):
    forces = []
    for line in forcefile:
        if line.strip() == "":
            continue
        if line.strip()[0] == "#":
            continue
        forces.append([float(x) for x in line.strip().split()])
        if len(forces) == num_atom:
            break

    if not len(forces) == num_atom:
        return None
    else:
        return np.array(forces)


def _write_cell_yaml(w, supercell):
    """Write cell info.

    This is only used from write_disp_fc3_yaml and write_disp_fc2_yaml.
    These methods are also deprecated.

    """
    warnings.warn("write_cell_yaml() is deprecated.", DeprecationWarning, stacklevel=2)

    w.write("lattice:\n")
    for axis in supercell.get_cell():
        w.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_scaled_positions()
    w.write("atoms:\n")
    for i, (s, v) in enumerate(zip(symbols, positions)):
        w.write("- symbol: %-2s # %d\n" % (s, i + 1))
        w.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % tuple(v))
