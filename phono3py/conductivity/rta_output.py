"""RTA output helpers (progress display and file writers)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal, TypeAlias, TypeVar, cast

import numpy as np

from phono3py.conductivity.base import get_unit_to_WmK
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.conductivity.type_dispatch import (
    get_rta_progress_mode,
    get_rta_writer_grid_payload,
    get_rta_writer_kappa_payload,
)
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA
from phono3py.file_IO import write_gamma_detail_to_hdf5, write_kappa_to_hdf5
from phono3py.phonon3.interaction import Interaction, all_bands_exist
from phono3py.phonon3.triplets import get_all_triplets

cond_RTA_type: TypeAlias = ConductivityRTABase
_RTAProgressMode: TypeAlias = Literal["default", "wigner"]
_RTAProgressHandler: TypeAlias = Callable[[cond_RTA_type, int], None]
_T = TypeVar("_T")


def _require_not_none(value: _T | None, name: str) -> _T:
    """Return non-None value, otherwise fail fast with assertion."""
    assert value is not None, f"{name} must not be None"
    return value


def show_rta_progress(
    br: cond_RTA_type,
    conductivity_type: Literal["wigner", "kubo"] | None,
    log_level: int,
) -> None:
    """Show progress for selected conductivity type in RTA run."""
    mode = cast(_RTAProgressMode, get_rta_progress_mode(conductivity_type))
    _RTA_PROGRESS_HANDLERS[mode](br, log_level)


def _show_rta_progress_default(br: cond_RTA_type, log_level: int) -> None:
    ShowCalcProgress.kappa_RTA(cast(ConductivityRTA, br), log_level)


def _show_rta_progress_wigner(br: cond_RTA_type, log_level: int) -> None:
    ShowCalcProgress.kappa_Wigner_RTA(cast(ConductivityWignerRTA, br), log_level)


_RTA_PROGRESS_HANDLERS: dict[_RTAProgressMode, _RTAProgressHandler] = {
    "default": _show_rta_progress_default,
    "wigner": _show_rta_progress_wigner,
}


def _pick_sigma_item(values, sigma_index):
    """Return values at sigma index, or None when values is None."""
    if values is None:
        return None
    return values[sigma_index]


def _pick_optional_item(values, *indices):
    """Return indexed values, or None when values is None."""
    if values is None:
        return None
    return values[indices]


class ShowCalcProgress:
    """Show calculation progress."""

    @staticmethod
    def kappa_RTA(br: ConductivityRTA, log_level: int) -> None:
        """Show RTA calculation progress."""
        temperatures = br.temperatures
        sigmas = br.sigmas
        kappa = _require_not_none(br.kappa, "br.kappa")
        num_ignored_phonon_modes = _require_not_none(
            br.number_of_ignored_phonon_modes,
            "br.number_of_ignored_phonon_modes",
        )
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.number_of_sampling_grid_points * num_band
        for i, sigma in enumerate(sigmas):
            kappa_i = np.asarray(_require_not_none(kappa[i], "kappa[i]"))
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
                for j, (t, k) in enumerate(zip(temperatures, kappa_i, strict=True)):
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
                for t, k in zip(temperatures, kappa_i, strict=True):
                    print(("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("", flush=True)

    @staticmethod
    def kappa_Wigner_RTA(br: ConductivityWignerRTA, log_level: int) -> None:
        """Show Wigner-RTA calculation progress."""
        temperatures = br.temperatures
        sigmas = br.sigmas
        kappa_TOT_RTA = _require_not_none(br.kappa_TOT_RTA, "br.kappa_TOT_RTA")
        kappa_P_RTA = _require_not_none(br.kappa_P_RTA, "br.kappa_P_RTA")
        kappa_C = _require_not_none(br.kappa_C, "br.kappa_C")
        num_ignored_phonon_modes = _require_not_none(
            br.number_of_ignored_phonon_modes,
            "br.number_of_ignored_phonon_modes",
        )
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.number_of_sampling_grid_points * num_band
        for i, sigma in enumerate(sigmas):
            kappa_p_rta_i = np.asarray(
                _require_not_none(kappa_P_RTA[i], "kappa_P_RTA[i]"),
            )
            kappa_c_i = np.asarray(_require_not_none(kappa_C[i], "kappa_C[i]"))
            kappa_tot_rta_i = np.asarray(
                _require_not_none(kappa_TOT_RTA[i], "kappa_TOT_RTA[i]"),
            )
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
                    zip(temperatures, kappa_p_rta_i, strict=True)
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
                for j, (t, k) in enumerate(zip(temperatures, kappa_c_i, strict=True)):
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
                    zip(temperatures, kappa_tot_rta_i, strict=True)
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
                for t, k in zip(temperatures, kappa_p_rta_i, strict=True):
                    print("K_P\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_c_i, strict=True):
                    print("K_C\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_tot_rta_i, strict=True):
                    print("K_T\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("", flush=True)


class ConductivityRTAWriter:
    """Collection of result writers."""

    @staticmethod
    def write_gamma(
        br: cond_RTA_type,
        interaction: Interaction,
        i: int,
        compression: Literal["gzip", "lzf"] = "gzip",
        filename: str | os.PathLike | None = None,
        verbose: bool = True,
    ) -> None:
        """Write mode kappa related properties into a hdf5 file."""
        grid_points = br.grid_points
        grid_payload = get_rta_writer_grid_payload(br, i)
        group_velocities_i = grid_payload["group_velocities_i"]
        gv_by_gv_i = grid_payload["gv_by_gv_i"]
        velocity_operator_i = grid_payload["velocity_operator_i"]
        mode_heat_capacities = grid_payload["mode_heat_capacities"]
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
        phonons = _require_not_none(interaction.get_phonons()[0], "interaction phonons")
        mode_heat_capacities = _require_not_none(
            mode_heat_capacities,
            "mode_heat_capacities",
        )
        if all_bands_exist(interaction):
            ave_pp_i = _pick_optional_item(ave_pp, i)
            frequencies = phonons[gp]
            for j, sigma in enumerate(sigmas):
                gamma_isotope_at_sigma = _pick_optional_item(gamma_isotope, j, i)
                gamma_N_at_sigma = _pick_optional_item(gamma_N, j, slice(None), i)
                gamma_U_at_sigma = _pick_optional_item(gamma_U, j, slice(None), i)

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
                    group_velocities_ik = _pick_optional_item(group_velocities_i, k)
                    velocity_operator_ik = _pick_optional_item(velocity_operator_i, k)
                    gv_by_gv_ik = _pick_optional_item(gv_by_gv_i, k)
                    ave_pp_ik = _pick_optional_item(ave_pp, i, k)
                    frequencies = phonons[gp, bi]
                    gamma_isotope_at_sigma = _pick_optional_item(gamma_isotope, j, i, k)
                    gamma_N_at_sigma = _pick_optional_item(
                        gamma_N, j, slice(None), i, k
                    )
                    gamma_U_at_sigma = _pick_optional_item(
                        gamma_U, j, slice(None), i, k
                    )
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
        compression: Literal["gzip", "lzf"],
        filename: str | None = None,
        log_level: int = 0,
    ) -> None:
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
        payload = get_rta_writer_kappa_payload(br)
        kappa = payload["kappa"]
        mode_kappa = payload["mode_kappa"]
        gv = payload["group_velocities"]
        gv_by_gv = payload["gv_by_gv"]
        kappa_TOT_RTA = payload["kappa_TOT_RTA"]
        kappa_P_RTA = payload["kappa_P_RTA"]
        kappa_C = payload["kappa_C"]
        mode_kappa_P_RTA = payload["mode_kappa_P_RTA"]
        mode_kappa_C = payload["mode_kappa_C"]
        mode_cv = payload["mode_heat_capacities"]
        ave_pp = br.averaged_pp_interaction
        qpoints = br.qpoints
        grid_points = br.grid_points
        weights = br.grid_weights
        boundary_mfp = br.boundary_mfp

        for i, sigma in enumerate(sigmas):
            kappa_at_sigma = _pick_sigma_item(kappa, i)
            mode_kappa_at_sigma = _pick_sigma_item(mode_kappa, i)
            kappa_TOT_RTA_at_sigma = _pick_sigma_item(kappa_TOT_RTA, i)
            kappa_P_RTA_at_sigma = _pick_sigma_item(kappa_P_RTA, i)
            kappa_C_at_sigma = _pick_sigma_item(kappa_C, i)
            mode_kappa_P_RTA_at_sigma = _pick_sigma_item(mode_kappa_P_RTA, i)
            mode_kappa_C_at_sigma = _pick_sigma_item(mode_kappa_C, i)
            gamma_isotope_at_sigma = _pick_sigma_item(gamma_isotope, i)
            gamma_N_at_sigma = _pick_sigma_item(gamma_N, i)
            gamma_U_at_sigma = _pick_sigma_item(gamma_U, i)

            write_kappa_to_hdf5(
                temperatures,
                mesh,
                boundary_mfp=cast(float, boundary_mfp),
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
                verbose=bool(log_level),
            )

    @staticmethod
    def write_gamma_detail(
        br: cond_RTA_type,
        interaction: Interaction,
        i: int,
        compression: Literal["gzip", "lzf"] = "gzip",
        filename: str | os.PathLike | None = None,
        verbose: bool = True,
    ) -> None:
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
        gamma_detail = _require_not_none(gamma_detail, "gamma_detail")

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
            band_indices = interaction.band_indices
            for sigma in sigmas:
                for k, bi in enumerate(band_indices):
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
