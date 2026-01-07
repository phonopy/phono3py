"""Init lattice thermal conductivity classes with RTA."""

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

from collections.abc import Sequence
from typing import Literal, Optional, Union, cast

import numpy as np
from numpy.typing import ArrayLike

from phono3py.conductivity.base import get_unit_to_WmK
from phono3py.conductivity.kubo_rta import ConductivityKuboRTA
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.conductivity.utils import write_pp_interaction
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA
from phono3py.file_IO import (
    read_gamma_from_hdf5,
    write_gamma_detail_to_hdf5,
    write_kappa_to_hdf5,
)
from phono3py.phonon3.interaction import Interaction, all_bands_exist
from phono3py.phonon3.triplets import get_all_triplets

cond_RTA_type = Union[ConductivityRTA, ConductivityWignerRTA, ConductivityKuboRTA]


def get_thermal_conductivity_RTA(
    interaction: Interaction,
    temperatures: ArrayLike | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    mass_variances: ArrayLike | None = None,
    grid_points: ArrayLike | None = None,
    is_isotope: bool = False,
    boundary_mfp: float | None = None,  # in micrometer
    use_ave_pp: bool = False,
    is_kappa_star: bool = True,
    gv_delta_q: float | None = None,
    is_full_pp: bool = False,
    is_N_U: bool = False,
    conductivity_type: Literal["wigner", "kubo"] | None = None,
    write_gamma: bool = False,
    read_gamma: bool = False,
    write_kappa: bool = False,
    write_pp: bool = False,
    read_pp: bool = False,
    write_gamma_detail: bool = False,
    compression: Literal["gzip", "lzf"] | int | None = "gzip",
    input_filename: str | None = None,
    output_filename: str | None = None,
    log_level: int = 0,
) -> ConductivityRTA | ConductivityKuboRTA | ConductivityWignerRTA:
    """Run RTA thermal conductivity calculation."""
    if temperatures is None:
        _temperatures = np.arange(0, 1001, 10, dtype="double")
    else:
        _temperatures = temperatures

    if conductivity_type == "wigner":
        conductivity_RTA_class = ConductivityWignerRTA
    elif conductivity_type == "kubo":
        conductivity_RTA_class = ConductivityKuboRTA
    else:
        conductivity_RTA_class = ConductivityRTA

    if log_level:
        print(
            "-------------------- Lattice thermal conductivity (RTA) "
            "--------------------"
        )

    br = conductivity_RTA_class(
        interaction,
        grid_points=grid_points,
        temperatures=_temperatures,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        store_pp=write_pp,
        pp_filename=input_filename,
        is_N_U=is_N_U,
        is_gamma_detail=write_gamma_detail,
        log_level=log_level,
    )

    if read_gamma:
        if not _set_gamma_from_file(br, filename=input_filename):
            print("Reading collisions failed.")
            return False

    for i in br:
        if write_pp:
            write_pp_interaction(
                br, interaction, i, compression=compression, filename=output_filename
            )
        if write_gamma:
            ConductivityRTAWriter.write_gamma(
                br,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
                verbose=log_level,
            )
        if write_gamma_detail:
            ConductivityRTAWriter.write_gamma_detail(
                br,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
                verbose=log_level,
            )

    if grid_points is None and all_bands_exist(interaction):
        br.set_kappa_at_sigmas()
        if log_level:
            if conductivity_type == "wigner":
                ShowCalcProgress.kappa_Wigner_RTA(
                    cast(ConductivityWignerRTA, br), log_level
                )
            else:
                ShowCalcProgress.kappa_RTA(cast(ConductivityRTA, br), log_level)
        if write_kappa:
            ConductivityRTAWriter.write_kappa(
                br,
                interaction.primitive.volume,
                compression=compression,
                filename=output_filename,
                log_level=log_level,
            )

    return br


class ShowCalcProgress:
    """Show calculation progress."""

    @staticmethod
    def kappa_RTA(br: ConductivityRTA, log_level):
        """Show RTA calculation progress."""
        temperatures = br.temperatures
        sigmas = br.sigmas
        kappa = br.kappa
        num_ignored_phonon_modes = br.number_of_ignored_phonon_modes
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.number_of_sampling_grid_points * num_band
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
                for j, (t, k) in enumerate(zip(temperatures, kappa[i], strict=True)):
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
                for t, k in zip(temperatures, kappa[i], strict=True):
                    print(("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("", flush=True)

    @staticmethod
    def kappa_Wigner_RTA(br: ConductivityWignerRTA, log_level):
        """Show Wigner-RTA calculation progress."""
        temperatures = br.temperatures
        sigmas = br.sigmas
        kappa_TOT_RTA = br.kappa_TOT_RTA
        kappa_P_RTA = br.kappa_P_RTA
        kappa_C = br.kappa_C
        num_ignored_phonon_modes = br.number_of_ignored_phonon_modes
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.number_of_sampling_grid_points * num_band
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
                for j, (t, k) in enumerate(
                    zip(temperatures, kappa_P_RTA[i], strict=True)
                ):
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
                for j, (t, k) in enumerate(zip(temperatures, kappa_C[i], strict=True)):
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
                for j, (t, k) in enumerate(
                    zip(temperatures, kappa_TOT_RTA[i], strict=True)
                ):
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
                    for t, k in zip(temperatures, kappa_P_RTA[i], strict=True):
                        print("K_P\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                    print(" ")
                    for t, k in zip(temperatures, kappa_C[i], strict=True):
                        print("K_C\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_TOT_RTA[i], strict=True):
                    print("K_T\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("", flush=True)


class ConductivityRTAWriter:
    """Collection of result writers."""

    @staticmethod
    def write_gamma(
        br: cond_RTA_type,
        interaction: Interaction,
        i: int,
        compression: str = "gzip",
        filename: Optional[str] = None,
        verbose: bool = True,
    ):
        """Write mode kappa related properties into a hdf5 file."""
        grid_points = br.grid_points
        if isinstance(br, ConductivityRTA):
            group_velocities_i = br.group_velocities[i]
            gv_by_gv_i = br.gv_by_gv[i]
        else:
            group_velocities_i = None
            gv_by_gv_i = None
        if isinstance(br, ConductivityWignerRTA):
            velocity_operator_i = br.velocity_operator[i]
        else:
            velocity_operator_i = None
        if isinstance(br, (ConductivityRTA, ConductivityWignerRTA)):
            mode_heat_capacities = br.mode_heat_capacities
        else:
            mode_heat_capacities = None
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
                if gamma_isotope is None:
                    gamma_isotope_at_sigma = None
                else:
                    gamma_isotope_at_sigma = gamma_isotope[j, i]
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
                    group_velocity=group_velocities_i,
                    gv_by_gv=gv_by_gv_i,
                    velocity_operator=velocity_operator_i,
                    heat_capacity=mode_heat_capacities[:, i],
                    gamma=gamma[j, :, i],
                    gamma_isotope=gamma_isotope_at_sigma,
                    gamma_N=gamma_N_at_sigma,
                    gamma_U=gamma_U_at_sigma,
                    averaged_pp_interaction=ave_pp_i,
                    grid_point=gp,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    kappa_unit_conversion=get_unit_to_WmK() / volume,
                    compression=compression,
                    filename=filename,
                    verbose=verbose,
                )
        else:
            for j, sigma in enumerate(sigmas):
                for k, bi in enumerate(interaction.band_indices):
                    if group_velocities_i is None:
                        group_velocities_ik = None
                    else:
                        group_velocities_ik = group_velocities_i[k]
                    if velocity_operator_i is None:
                        velocity_operator_ik = None
                    else:
                        velocity_operator_ik = velocity_operator_i[k]
                    if gv_by_gv_i is None:
                        gv_by_gv_ik = None
                    else:
                        gv_by_gv_ik = gv_by_gv_i[k]
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
                        group_velocity=group_velocities_ik,
                        gv_by_gv=gv_by_gv_ik,
                        velocity_operator=velocity_operator_ik,
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
                        kappa_unit_conversion=get_unit_to_WmK() / volume,
                        compression=compression,
                        filename=filename,
                        verbose=verbose,
                    )

    @staticmethod
    def write_kappa(
        br: cond_RTA_type,
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

        if isinstance(br, ConductivityRTA):
            kappa = br.kappa
            mode_kappa = br.mode_kappa
            gv = br.group_velocities
            gv_by_gv = br.gv_by_gv
        else:
            kappa = None
            mode_kappa = None
            gv = None
            gv_by_gv = None

        if isinstance(br, ConductivityWignerRTA):
            kappa_TOT_RTA = br.kappa_TOT_RTA
            kappa_P_RTA = br.kappa_P_RTA
            kappa_C = br.kappa_C
            mode_kappa_P_RTA = br.mode_kappa_P_RTA
            mode_kappa_C = br.mode_kappa_C
        else:
            kappa_TOT_RTA = None
            kappa_P_RTA = None
            kappa_C = None
            mode_kappa_P_RTA = None
            mode_kappa_C = None

        if isinstance(br, (ConductivityRTA, ConductivityWignerRTA)):
            mode_cv = br.mode_heat_capacities
        else:
            mode_cv = None
        ave_pp = br.averaged_pp_interaction
        qpoints = br.qpoints
        grid_points = br.grid_points
        weights = br.grid_weights
        boundary_mfp = br.boundary_mfp

        for i, sigma in enumerate(sigmas):
            if kappa is None:
                kappa_at_sigma = None
            else:
                kappa_at_sigma = kappa[i]
            if mode_kappa is None:
                mode_kappa_at_sigma = None
            else:
                mode_kappa_at_sigma = mode_kappa[i]
            if kappa_TOT_RTA is None:
                kappa_TOT_RTA_at_sigma = None
            else:
                kappa_TOT_RTA_at_sigma = kappa_TOT_RTA[i]
            if kappa_P_RTA is None:
                kappa_P_RTA_at_sigma = None
            else:
                kappa_P_RTA_at_sigma = kappa_P_RTA[i]
            if kappa_C is None:
                kappa_C_at_sigma = None
            else:
                kappa_C_at_sigma = kappa_C[i]
            if mode_kappa_P_RTA is None:
                mode_kappa_P_RTA_at_sigma = None
            else:
                mode_kappa_P_RTA_at_sigma = mode_kappa_P_RTA[i]
            if mode_kappa_C is None:
                mode_kappa_C_at_sigma = None
            else:
                mode_kappa_C_at_sigma = mode_kappa_C[i]
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
                boundary_mfp=boundary_mfp,
                bz_grid=bz_grid,
                frequency=frequencies,
                group_velocity=gv,
                gv_by_gv=gv_by_gv,
                heat_capacity=mode_cv,
                kappa=kappa_at_sigma,
                mode_kappa=mode_kappa_at_sigma,
                kappa_TOT_RTA=kappa_TOT_RTA_at_sigma,
                kappa_P_RTA=kappa_P_RTA_at_sigma,
                kappa_C=kappa_C_at_sigma,
                mode_kappa_P_RTA=mode_kappa_P_RTA_at_sigma,
                mode_kappa_C=mode_kappa_C_at_sigma,
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
                kappa_unit_conversion=get_unit_to_WmK() / volume,
                compression=compression,
                filename=filename,
                verbose=log_level,
            )

    @staticmethod
    def write_gamma_detail(
        br: cond_RTA_type,
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
            for sigma in sigmas:
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
            for sigma in sigmas:
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


def _set_gamma_from_file(
    br: ConductivityRTABase, filename: Optional[str] = None, verbose: bool = True
):
    """Read kappa-*.hdf5 files for thermal conductivity calculation.

    If kappa-m*.hdf5 that contains all data is not found, kappa-m*-gp*.hdf5
    files at grid points are searched. If any of those files are not found,
    kappa-m*-gp*-b*.hdf5 files at grid points and bands are searched. If any
    of those files are not found, it fails.

    br : ConductivityRTABase
        RTA lattice thermal conductivity instance.
    filename : str, optional
        This string is inserted in the filename as kappa-m*.{filename}.hdf5.
    verbose : bool, optional
        Show text output or not.

    """
    sigmas = br.sigmas
    sigma_cutoff = br.sigma_cutoff_width
    mesh = br.mesh_numbers
    grid_points = br.grid_points
    temperatures = br.temperatures
    num_band = br.frequencies.shape[1]

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
        data, full_filename = read_gamma_from_hdf5(
            mesh,
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
        )
        if data:
            if verbose:
                print(f"Read gamma from {full_filename}.")
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
            if verbose:
                print(
                    "%s not found. Look for hdf5 files at grid points." % full_filename
                )
            for i, gp in enumerate(grid_points):
                data_gp, full_filename = read_gamma_from_hdf5(
                    mesh,
                    grid_point=gp,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    filename=filename,
                )
                if data_gp:
                    if verbose:
                        print("Read data from %s." % full_filename)
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
                    if verbose:
                        print(
                            "%s not found. Look for hdf5 files at bands."
                            % full_filename
                        )
                    for bi in range(num_band):
                        data_band, full_filename = read_gamma_from_hdf5(
                            mesh,
                            grid_point=gp,
                            band_index=bi,
                            sigma=sigma,
                            sigma_cutoff=sigma_cutoff,
                            filename=filename,
                        )
                        if data_band:
                            if verbose:
                                print("Read data from %s." % full_filename)
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
                            if verbose:
                                print("%s not found." % full_filename)
                            read_succeeded = False

    if read_succeeded:
        br.gamma = gamma
        if is_ave_pp_in:
            br.set_averaged_pp_interaction(ave_pp)
        if is_gamma_N_U_in:
            br.set_gamma_N_U(gamma_N, gamma_U)
        return True
    else:
        return False
