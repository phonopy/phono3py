"""RTA output helpers (progress display and file writers)."""

from __future__ import annotations

from typing import Literal, Optional, Union, cast

from phono3py.conductivity.base import get_unit_to_WmK
from phono3py.conductivity.kubo_rta import ConductivityKuboRTA
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.conductivity.type_dispatch import (
    get_rta_writer_grid_data,
    get_rta_writer_kappa_data,
)
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA
from phono3py.file_IO import write_gamma_detail_to_hdf5, write_kappa_to_hdf5
from phono3py.phonon3.interaction import Interaction, all_bands_exist
from phono3py.phonon3.triplets import get_all_triplets

cond_RTA_type = Union[ConductivityRTA, ConductivityWignerRTA, ConductivityKuboRTA]


def show_rta_progress(
    br: cond_RTA_type,
    conductivity_type: Literal["wigner", "kubo"] | None,
    log_level: int,
):
    """Show progress for selected conductivity type in RTA run."""
    _RTA_PROGRESS_HANDLERS[conductivity_type](br, log_level)


def _show_rta_progress_default(br: cond_RTA_type, log_level: int):
    ShowCalcProgress.kappa_RTA(cast(ConductivityRTA, br), log_level)


def _show_rta_progress_wigner(br: cond_RTA_type, log_level: int):
    ShowCalcProgress.kappa_Wigner_RTA(cast(ConductivityWignerRTA, br), log_level)


_RTA_PROGRESS_HANDLERS = {
    None: _show_rta_progress_default,
    "kubo": _show_rta_progress_default,
    "wigner": _show_rta_progress_wigner,
}


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
        (
            group_velocities_i,
            gv_by_gv_i,
            velocity_operator_i,
            mode_heat_capacities,
        ) = get_rta_writer_grid_data(br, i)
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
        (
            kappa,
            mode_kappa,
            gv,
            gv_by_gv,
            kappa_TOT_RTA,
            kappa_P_RTA,
            kappa_C,
            mode_kappa_P_RTA,
            mode_kappa_C,
            mode_cv,
        ) = get_rta_writer_kappa_data(br)
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
