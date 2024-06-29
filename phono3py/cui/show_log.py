"""Utilities to show various logs for main CUI script."""

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

import sys

import numpy as np
from phonopy.structure.cells import print_cell

from phono3py import Phono3py
from phono3py.cui.settings import Phono3pySettings


def show_general_settings(
    settings, run_mode, phono3py, cell_filename, input_filename, output_filename
):
    """Show general setting information."""
    is_primitive_axes_auto = (
        isinstance(phono3py.primitive_matrix, str)
        and phono3py.primitive_matrix == "auto"
    )
    primitive_matrix = phono3py.primitive_matrix
    supercell_matrix = phono3py.supercell_matrix
    phonon_supercell_matrix = phono3py.phonon_supercell_matrix

    print("-" * 29 + " General settings " + "-" * 29)
    print("Run mode: %s" % run_mode)
    if output_filename:
        print("Output filename is modified by %s." % output_filename)
    if input_filename:
        print("Input filename is modified by %s." % input_filename)
    if settings.hdf5_compression:
        print("HDF5 data compression filter: %s" % settings.hdf5_compression)

    if phono3py.calculator:
        print("Calculator interface: %s" % phono3py.calculator)
    print('Crystal structure was read from "%s".' % cell_filename)

    if (np.diag(np.diag(supercell_matrix)) - supercell_matrix).any():
        print("Supercell matrix (dim):")
        for v in supercell_matrix:
            print("  %s" % v)
    else:
        print("Supercell (dim): %s" % np.diag(supercell_matrix))
    if phonon_supercell_matrix is not None:
        if (np.diag(np.diag(phonon_supercell_matrix)) - phonon_supercell_matrix).any():
            print("Phonon supercell matrix (dim-fc2):")
            for v in phonon_supercell_matrix:
                print("  %s" % v)
        else:
            print("Phonon supercell (dim-fc2): %s" % np.diag(phonon_supercell_matrix))
    if is_primitive_axes_auto:
        print("Primitive matrix (Auto):")
        for v in primitive_matrix:
            print("  %s" % v)
    elif primitive_matrix is not None:
        print("Primitive matrix:")
        for v in primitive_matrix:
            print("  %s" % v)


def show_phono3py_cells(phono3py: Phono3py):
    """Show crystal structures."""
    primitive = phono3py.primitive
    supercell = phono3py.supercell
    phonon_primitive = phono3py.phonon_primitive
    phonon_supercell = phono3py.phonon_supercell

    print("-" * 30 + " primitive cell " + "-" * 30)
    print_cell(primitive)
    print("-" * 32 + " supercell " + "-" * 33)
    print_cell(supercell, mapping=primitive.s2p_map)
    if phono3py.phonon_supercell_matrix is not None:
        print("-" * 19 + " primitive cell for harmonic phonon " + "-" * 20)
        print_cell(phonon_primitive)
        print("-" * 21 + " supercell for harmonic phonon " + "-" * 22)
        print_cell(phonon_supercell, mapping=phonon_primitive.s2p_map)
    print("-" * 76, flush=True)


def show_phono3py_force_constants_settings(settings: Phono3pySettings):
    """Show force constants settings."""
    read_fc3 = settings.read_fc3
    read_fc2 = settings.read_fc2
    symmetrize_fc3r = settings.is_symmetrize_fc3_r or settings.fc_symmetry
    symmetrize_fc2 = settings.is_symmetrize_fc2 or settings.fc_symmetry

    print("-" * 29 + " Force constants " + "-" * 30)
    if settings.fc_calculator == "alm" and not read_fc2:
        print("Use ALM for getting fc2 (simultaneous fit to fc2 and fc3)")
    else:
        print(
            "Imposing translational and index exchange symmetry to fc2: %s"
            % symmetrize_fc2
        )

    if settings.is_isotope or settings.is_joint_dos:
        pass
    elif settings.fc_calculator == "alm" and not read_fc3:
        print("Use ALM for getting fc3")
    else:
        print(
            "Imposing translational and index exchange symmetry to fc3: "
            "%s" % symmetrize_fc3r
        )

    if settings.cutoff_fc3_distance is not None:
        print("FC3 cutoff distance: %s" % settings.cutoff_fc3_distance)


def show_phono3py_settings(phono3py, settings, updated_settings, log_level):
    """Show general calculation settings."""
    sigmas = updated_settings["sigmas"]
    temperatures = updated_settings["temperatures"]
    temperature_points = updated_settings["temperature_points"]
    cutoff_frequency = updated_settings["cutoff_frequency"]
    frequency_factor_to_THz = updated_settings["frequency_factor_to_THz"]
    frequency_scale_factor = updated_settings["frequency_scale_factor"]
    frequency_step = updated_settings["frequency_step"]
    num_frequency_points = updated_settings["num_frequency_points"]

    print("-" * 27 + " Calculation settings " + "-" * 27)
    if settings.is_nac:
        print("Non-analytical term correction (NAC): %s" % settings.is_nac)
        if phono3py.nac_params:
            print("NAC unit conversion factor: %9.5f" % phono3py.nac_params["factor"])
        if settings.nac_q_direction is not None:
            print("NAC q-direction: %s" % settings.nac_q_direction)
    if settings.band_indices is not None and not settings.is_bterta:
        print(
            ("Band indices: [" + " %s" * len(settings.band_indices) + " ]")
            % tuple([np.array(bi) + 1 for bi in settings.band_indices])
        )
    if sigmas:
        text = "BZ integration: "
        for i, sigma in enumerate(sigmas):
            if sigma:
                text += "Smearing=%s" % sigma
                if settings.sigma_cutoff_width is not None:
                    text += "(%4.2f SD)" % settings.sigma_cutoff_width
            else:
                text += "Tetrahedron-method"
            if i < len(sigmas) - 1:
                text += ", "
        print(text)

    if settings.is_lbte and settings.read_collision is not None:
        pass
    elif settings.is_joint_dos:
        pass
    elif settings.is_bterta:
        if len(temperatures) > 5:
            text = (" %.1f " * 5 + "...") % tuple(temperatures[:5])
            text += " %.1f" % temperatures[-1]
        else:
            text = (" %.1f " * len(temperatures)) % tuple(temperatures)
        print("Temperature: " + text)
    elif temperature_points is not None:
        print(
            ("Temperatures:" + " %.1f " * len(temperature_points))
            % tuple(temperature_points)
        )
        if settings.scattering_event_class is not None:
            print("Scattering event class: %s" % settings.scattering_event_class)

    if cutoff_frequency:
        print("Cutoff frequency: %s" % cutoff_frequency)

    if settings.use_ave_pp and (settings.is_bterta or settings.is_lbte):
        print("Use averaged ph-ph interaction")

    const_ave_pp = settings.constant_averaged_pp_interaction
    if const_ave_pp is not None and (settings.is_bterta or settings.is_lbte):
        print("Constant ph-ph interaction: %6.3e" % const_ave_pp)

    print("Frequency conversion factor to THz: %9.5f" % frequency_factor_to_THz)
    if frequency_scale_factor is not None:
        print("Frequency scale factor: %8.5f" % frequency_scale_factor)

    if (
        settings.is_joint_dos
        or settings.is_imag_self_energy
        or settings.is_real_self_energy
        or settings.is_spectral_function
    ):
        if frequency_step is not None:
            print("Frequency step for spectrum: %s" % frequency_step)
        if num_frequency_points is not None:
            print("Number of frequency sampling points: %d" % num_frequency_points)

    if settings.mesh_numbers is not None:
        try:
            mesh_length = float(settings.mesh_numbers)
            print(f"Length for sampling mesh generation: {mesh_length:.2f}")
        except TypeError:
            mesh_numbers = tuple(np.ravel(settings.mesh_numbers))
            nums = (("%d " * len(mesh_numbers)).strip()) % mesh_numbers
            print(f"Mesh sampling: [ {nums} ]")

    sys.stdout.flush()


def show_grid_points(grid_points):
    """Show grid point list."""
    text = "Grid point to be calculated: "
    if len(grid_points) > 8:
        for i, gp in enumerate(grid_points):
            if i % 10 == 0:
                text += "\n"
                text += " "
            text += "%d " % gp
    else:
        for gp in grid_points:
            text += "%d " % gp
    print(text)
