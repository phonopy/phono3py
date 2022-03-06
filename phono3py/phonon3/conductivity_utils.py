"""Utilities for lattice thermal conductivity calculation."""
# Copyright (C) 2022 Atsushi Togo
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


from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from phono3py.file_IO import (
    read_gamma_from_hdf5,
    write_gamma_detail_to_hdf5,
    write_kappa_to_hdf5,
    write_pp_to_hdf5,
)
from phono3py.phonon3.conductivity import unit_to_WmK
from phono3py.phonon3.triplets import get_all_triplets

if TYPE_CHECKING:
    from phono3py.phonon3.conductivity_RTA import ConductivityRTA
    from phono3py.phonon3.conductivity import ConductivityBase
    from phono3py.phonon3.conductivity_Wigner import ConductivityWignerRTA

from phono3py.phonon3.interaction import Interaction


class ConductivityRTAWriter:
    """Collection of result writers."""

    @staticmethod
    def write_gamma(
        br: Union["ConductivityRTA", "ConductivityWignerRTA"],
        interaction: Interaction,
        i: int,
        compression: str = "gzip",
        filename: Optional[str] = None,
        verbose: bool = True,
    ):
        """Write mode kappa related properties into a hdf5 file."""
        grid_points = br.grid_points
        group_velocities = br.group_velocities
        try:
            gv_by_gv = br.gv_by_gv
        except AttributeError:
            gv_by_gv = None
        mode_heat_capacities = br.mode_heat_capacities
        ave_pp = br.averaged_pp_interaction
        mesh = br.mesh_numbers
        bz_grid = br.bz_grid
        temperatures = br.temperatures
        gamma = br.gamma
        gamma_isotope = br.gamma_isotope
        sigmas = br.sigmas
        sigma_cutoff = br.sigma_cutoff_width
        volume = interaction.primitive.volume
        gamma_N, gamma_U = br.get_gamma_N_U()

        gp = grid_points[i]
        if all_bands_exist(interaction):
            if ave_pp is None:
                ave_pp_i = None
            else:
                ave_pp_i = ave_pp[i]
            frequencies = interaction.get_phonons()[0][gp]
            for j, sigma in enumerate(sigmas):
                if gamma_isotope is not None:
                    gamma_isotope_at_sigma = gamma_isotope[j, i]
                else:
                    gamma_isotope_at_sigma = None
                if gamma_N is None:
                    gamma_N_at_sigma = None
                else:
                    gamma_N_at_sigma = gamma_N[j, :, i]
                if gamma_U is None:
                    gamma_U_at_sigma = None
                else:
                    gamma_U_at_sigma = gamma_U[j, :, i]

                write_kappa_to_hdf5(
                    temperatures,
                    mesh,
                    bz_grid=bz_grid,
                    frequency=frequencies,
                    group_velocity=group_velocities[i],
                    gv_by_gv=gv_by_gv[i],
                    heat_capacity=mode_heat_capacities[:, i],
                    gamma=gamma[j, :, i],
                    gamma_isotope=gamma_isotope_at_sigma,
                    gamma_N=gamma_N_at_sigma,
                    gamma_U=gamma_U_at_sigma,
                    averaged_pp_interaction=ave_pp_i,
                    grid_point=gp,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    kappa_unit_conversion=unit_to_WmK / volume,
                    compression=compression,
                    filename=filename,
                    verbose=verbose,
                )
        else:
            for j, sigma in enumerate(sigmas):
                for k, bi in enumerate(interaction.band_indices):
                    if ave_pp is None:
                        ave_pp_ik = None
                    else:
                        ave_pp_ik = ave_pp[i, k]
                    frequencies = interaction.get_phonons()[0][gp, bi]
                    if gamma_isotope is not None:
                        gamma_isotope_at_sigma = gamma_isotope[j, i, k]
                    else:
                        gamma_isotope_at_sigma = None
                    if gamma_N is None:
                        gamma_N_at_sigma = None
                    else:
                        gamma_N_at_sigma = gamma_N[j, :, i, k]
                    if gamma_U is None:
                        gamma_U_at_sigma = None
                    else:
                        gamma_U_at_sigma = gamma_U[j, :, i, k]
                    write_kappa_to_hdf5(
                        temperatures,
                        mesh,
                        bz_grid=bz_grid,
                        frequency=frequencies,
                        group_velocity=group_velocities[i, k],
                        gv_by_gv=gv_by_gv[i, k],
                        heat_capacity=mode_heat_capacities[:, i, k],
                        gamma=gamma[j, :, i, k],
                        gamma_isotope=gamma_isotope_at_sigma,
                        gamma_N=gamma_N_at_sigma,
                        gamma_U=gamma_U_at_sigma,
                        averaged_pp_interaction=ave_pp_ik,
                        grid_point=gp,
                        band_index=bi,
                        sigma=sigma,
                        sigma_cutoff=sigma_cutoff,
                        kappa_unit_conversion=unit_to_WmK / volume,
                        compression=compression,
                        filename=filename,
                        verbose=verbose,
                    )

    @staticmethod
    def write_kappa(
        br: Union["ConductivityRTA", "ConductivityWignerRTA"],
        volume: float,
        compression: str = "gzip",
        filename: Optional[str] = None,
        log_level: int = 0,
    ):
        """Write kappa related properties into a hdf5 file."""
        temperatures = br.temperatures
        sigmas = br.sigmas
        sigma_cutoff = br.sigma_cutoff_width
        gamma = br.gamma
        gamma_isotope = br.gamma_isotope
        gamma_N, gamma_U = br.get_gamma_N_U()
        mesh = br.mesh_numbers
        bz_grid = br.bz_grid
        frequencies = br.frequencies

        try:
            kappa = br.kappa
        except AttributeError:
            kappa = None
        try:
            mode_kappa = br.mode_kappa
        except AttributeError:
            mode_kappa = None
        try:
            gv = br.group_velocities
        except AttributeError:
            gv = None
        try:
            gv_by_gv = br.gv_by_gv
        except AttributeError:
            gv_by_gv = None

        mode_cv = br.mode_heat_capacities
        ave_pp = br.averaged_pp_interaction
        qpoints = br.qpoints
        grid_points = br.grid_points
        weights = br.grid_weights

        for i, sigma in enumerate(sigmas):
            if kappa is None:
                kappa_at_sigma = None
            else:
                kappa_at_sigma = kappa[i]
            if mode_kappa is None:
                mode_kappa_at_sigma = None
            else:
                mode_kappa_at_sigma = mode_kappa[i]
            if gamma_isotope is not None:
                gamma_isotope_at_sigma = gamma_isotope[i]
            else:
                gamma_isotope_at_sigma = None
            if gamma_N is None:
                gamma_N_at_sigma = None
            else:
                gamma_N_at_sigma = gamma_N[i]
            if gamma_U is None:
                gamma_U_at_sigma = None
            else:
                gamma_U_at_sigma = gamma_U[i]

            write_kappa_to_hdf5(
                temperatures,
                mesh,
                bz_grid=bz_grid,
                frequency=frequencies,
                group_velocity=gv,
                gv_by_gv=gv_by_gv,
                heat_capacity=mode_cv,
                kappa=kappa_at_sigma,
                mode_kappa=mode_kappa_at_sigma,
                gamma=gamma[i],
                gamma_isotope=gamma_isotope_at_sigma,
                gamma_N=gamma_N_at_sigma,
                gamma_U=gamma_U_at_sigma,
                averaged_pp_interaction=ave_pp,
                qpoint=qpoints,
                grid_point=grid_points,
                weight=weights,
                sigma=sigma,
                sigma_cutoff=sigma_cutoff,
                kappa_unit_conversion=unit_to_WmK / volume,
                compression=compression,
                filename=filename,
                verbose=log_level,
            )

    @staticmethod
    def write_gamma_detail(
        br: Union["ConductivityRTA", "ConductivityWignerRTA"],
        interaction: Interaction,
        i: int,
        compression: str = "gzip",
        filename: Optional[str] = None,
        verbose: bool = True,
    ):
        """Write detailed Gamma values to hdf5 files."""
        gamma_detail = br.get_gamma_detail_at_q()
        temperatures = br.temperatures
        mesh = br.mesh_numbers
        bz_grid = br.bz_grid
        grid_points = br.grid_points
        gp = grid_points[i]
        sigmas = br.sigmas
        sigma_cutoff = br.sigma_cutoff_width
        triplets, weights, _, _ = interaction.get_triplets_at_q()
        all_triplets = get_all_triplets(gp, interaction.bz_grid)

        if all_bands_exist(interaction):
            for j, sigma in enumerate(sigmas):
                write_gamma_detail_to_hdf5(
                    temperatures,
                    mesh,
                    bz_grid=bz_grid,
                    gamma_detail=gamma_detail,
                    grid_point=gp,
                    triplet=triplets,
                    weight=weights,
                    triplet_all=all_triplets,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    compression=compression,
                    filename=filename,
                    verbose=verbose,
                )
        else:
            for j, sigma in enumerate(sigmas):
                for k, bi in enumerate(interaction.get_band_indices()):
                    write_gamma_detail_to_hdf5(
                        temperatures,
                        mesh,
                        bz_grid=bz_grid,
                        gamma_detail=gamma_detail[:, :, k, :, :],
                        grid_point=gp,
                        triplet=triplets,
                        weight=weights,
                        band_index=bi,
                        sigma=sigma,
                        sigma_cutoff=sigma_cutoff,
                        compression=compression,
                        filename=filename,
                        verbose=verbose,
                    )


def _set_gamma_from_file(br, filename=None, verbose=True):
    """Read kappa-*.hdf5 files for thermal conductivity calculation."""
    sigmas = br.get_sigmas()
    sigma_cutoff = br.get_sigma_cutoff_width()
    mesh = br.get_mesh_numbers()
    grid_points = br.get_grid_points()
    temperatures = br.get_temperatures()
    num_band = br.get_frequencies().shape[1]

    gamma = np.zeros(
        (len(sigmas), len(temperatures), len(grid_points), num_band), dtype="double"
    )
    gamma_N = np.zeros_like(gamma)
    gamma_U = np.zeros_like(gamma)
    gamma_iso = np.zeros((len(sigmas), len(grid_points), num_band), dtype="double")
    ave_pp = np.zeros((len(grid_points), num_band), dtype="double")

    is_gamma_N_U_in = False
    is_ave_pp_in = False
    read_succeeded = True

    for j, sigma in enumerate(sigmas):
        data = read_gamma_from_hdf5(
            mesh,
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
            verbose=verbose,
        )
        if data:
            gamma[j] = data["gamma"]
            if "gamma_isotope" in data:
                gamma_iso[j] = data["gamma_isotope"]
            if "gamma_N" in data:
                is_gamma_N_U_in = True
                gamma_N[j] = data["gamma_N"]
                gamma_U[j] = data["gamma_U"]
            if "ave_pp" in data:
                is_ave_pp_in = True
                ave_pp[:] = data["ave_pp"]
        else:
            for i, gp in enumerate(grid_points):
                data_gp = read_gamma_from_hdf5(
                    mesh,
                    grid_point=gp,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    filename=filename,
                    verbose=verbose,
                )
                if data_gp:
                    gamma[j, :, i] = data_gp["gamma"]
                    if "gamma_iso" in data_gp:
                        gamma_iso[j, i] = data_gp["gamma_iso"]
                    if "gamma_N" in data_gp:
                        is_gamma_N_U_in = True
                        gamma_N[j, :, i] = data_gp["gamma_N"]
                        gamma_U[j, :, i] = data_gp["gamma_U"]
                    if "ave_pp" in data_gp:
                        is_ave_pp_in = True
                        ave_pp[i] = data_gp["ave_pp"]
                else:
                    for bi in range(num_band):
                        data_band = read_gamma_from_hdf5(
                            mesh,
                            grid_point=gp,
                            band_index=bi,
                            sigma=sigma,
                            sigma_cutoff=sigma_cutoff,
                            filename=filename,
                            verbose=verbose,
                        )
                        if data_band:
                            gamma[j, :, i, bi] = data_band["gamma"]
                            if "gamma_iso" in data_band:
                                gamma_iso[j, i, bi] = data_band["gamma_iso"]
                            if "gamma_N" in data_band:
                                is_gamma_N_U_in = True
                                gamma_N[j, :, i, bi] = data_band["gamma_N"]
                                gamma_U[j, :, i, bi] = data_band["gamma_U"]
                            if "ave_pp" in data_band:
                                is_ave_pp_in = True
                                ave_pp[i, bi] = data_band["ave_pp"]
                        else:
                            read_succeeded = False

    if read_succeeded:
        br.set_gamma(gamma)
        if is_ave_pp_in:
            br.set_averaged_pp_interaction(ave_pp)
        if is_gamma_N_U_in:
            br.set_gamma_N_U(gamma_N, gamma_U)
        return True
    else:
        return False


class ShowCalcProgress:
    """Show calculation progress."""

    @staticmethod
    def kappa_RTA(br, log_level):
        """Show RTA calculation progess."""
        temperatures = br.temperatures
        sigmas = br.sigmas
        kappa = br.kappa
        num_ignored_phonon_modes = br.get_number_of_ignored_phonon_modes()
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.get_number_of_sampling_grid_points() * num_band
        for i, sigma in enumerate(sigmas):
            text = "----------- Thermal conductivity (W/m-k) "
            if sigma:
                text += "for sigma=%s -----------" % sigma
            else:
                text += "with tetrahedron method -----------"
            print(text)
            if log_level > 1:
                print(
                    ("#%6s       " + " %-10s" * 6 + "#ipm")
                    % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                for j, (t, k) in enumerate(zip(temperatures, kappa[i])):
                    print(
                        ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % (
                            (t,)
                            + tuple(k)
                            + (num_ignored_phonon_modes[i, j], num_phonon_modes)
                        )
                    )
            else:
                print(
                    ("#%6s       " + " %-10s" * 6)
                    % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                for j, (t, k) in enumerate(zip(temperatures, kappa[i])):
                    print(("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("")

    @staticmethod
    def kappa_Wigner_RTA(br, log_level):
        """Show Wigner-RTA calculation progess."""
        temperatures = br.temperatures
        sigmas = br.sigmas
        kappa_TOT_RTA = br.kappa_TOT_RTA
        kappa_P_RTA = br.kappa_P_RTA
        kappa_C = br.kappa_C
        num_ignored_phonon_modes = br.get_number_of_ignored_phonon_modes()
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.get_number_of_sampling_grid_points() * num_band
        for i, sigma in enumerate(sigmas):
            text = "----------- Thermal conductivity (W/m-k) "
            if sigma:
                text += "for sigma=%s -----------" % sigma
            else:
                text += "with tetrahedron method -----------"
            print(text)
            if log_level > 1:
                print(
                    ("#%6s       " + " %-10s" * 6 + "#ipm")
                    % ("      \t   T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                for j, (t, k) in enumerate(zip(temperatures, kappa_P_RTA[i])):
                    print(
                        "K_P\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % (
                            (t,)
                            + tuple(k)
                            + (num_ignored_phonon_modes[i, j], num_phonon_modes)
                        )
                    )
                print(" ")
                for j, (t, k) in enumerate(zip(temperatures, kappa_C[i])):
                    print(
                        "K_C\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % (
                            (t,)
                            + tuple(k)
                            + (num_ignored_phonon_modes[i, j], num_phonon_modes)
                        )
                    )
                print(" ")
                for j, (t, k) in enumerate(zip(temperatures, kappa_TOT_RTA[i])):
                    print(
                        "K_T\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % (
                            (t,)
                            + tuple(k)
                            + (num_ignored_phonon_modes[i, j], num_phonon_modes)
                        )
                    )
            else:
                print(
                    ("#%6s       " + " %-10s" * 6)
                    % ("      \t   T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                if kappa_P_RTA is not None:
                    for j, (t, k) in enumerate(zip(temperatures, kappa_P_RTA[i])):
                        print("K_P\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                    print(" ")
                    for j, (t, k) in enumerate(zip(temperatures, kappa_C[i])):
                        print("K_C\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for j, (t, k) in enumerate(zip(temperatures, kappa_TOT_RTA[i])):
                    print("K_T\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("")


def all_bands_exist(interaction: Interaction):
    """Return if all bands are selected or not."""
    band_indices = interaction.band_indices
    num_band = len(interaction.primitive) * 3
    if len(band_indices) == num_band:
        if (band_indices - np.arange(num_band) == 0).all():
            return True
    return False


def write_pp(
    conductivity: "ConductivityBase",
    pp: Interaction,
    i,
    filename=None,
    compression="gzip",
):
    """Write ph-ph interaction strength in hdf5 file."""
    grid_point = conductivity.grid_points[i]
    sigmas = conductivity.sigmas
    sigma_cutoff = conductivity.sigma_cutoff_width
    mesh = conductivity.mesh_numbers
    triplets, weights, _, _ = pp.get_triplets_at_q()
    all_triplets = get_all_triplets(grid_point, pp.bz_grid)

    if len(sigmas) > 1:
        print("Multiple smearing parameters were given. The last one in ")
        print("ph-ph interaction calculations was written in the file.")

    write_pp_to_hdf5(
        mesh,
        pp=pp.interaction_strength,
        g_zero=pp.zero_value_positions,
        grid_point=grid_point,
        triplet=triplets,
        weight=weights,
        triplet_all=all_triplets,
        sigma=sigmas[-1],
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        compression=compression,
    )
