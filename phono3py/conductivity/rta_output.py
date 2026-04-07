"""RTA output helpers (file writers)."""

from __future__ import annotations

import os
from typing import Any, Literal, cast

from numpy.typing import NDArray

from phono3py.conductivity.rta_calculator import RTACalculator
from phono3py.conductivity.utils import get_unit_to_WmK
from phono3py.file_IO import write_gamma_detail_to_hdf5, write_kappa_to_hdf5
from phono3py.phonon3.interaction import Interaction, all_bands_exist
from phono3py.phonon3.triplets import get_all_triplets


def _require_ndarray_not_none(value: NDArray[Any] | None, name: str) -> NDArray[Any]:
    """Return non-None ndarray, otherwise fail fast with assertion."""
    assert value is not None, f"{name} must not be None"
    return value


class ConductivityRTAWriter:
    """Collection of result writers."""

    @staticmethod
    def write_gamma(
        br: RTACalculator,
        interaction: Interaction,
        i: int,
        compression: Literal["gzip", "lzf"] | int | None = "gzip",
        filename: str | os.PathLike | None = None,
        verbose: bool = True,
    ) -> None:
        """Write mode kappa related properties into a hdf5 file."""
        grid_points = br.grid_points
        group_velocities_i = br.group_velocities[i]
        gv_by_gv_i = br.gv_by_gv[i] if br.gv_by_gv is not None else None
        extra_gp_full = br.get_extra_grid_point_output()
        extra_gp_data: dict[str, Any] | None = (
            {k: v[i] for k, v in extra_gp_full.items()}
            if extra_gp_full is not None
            else None
        )
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
        phonons = _require_ndarray_not_none(
            interaction.get_phonons()[0], "interaction phonons"
        )
        if all_bands_exist(interaction):
            ave_pp_i = ave_pp[i] if ave_pp is not None else None
            frequencies = phonons[gp]
            for j, sigma in enumerate(sigmas):
                gamma_isotope_at_sigma = (
                    gamma_isotope[j, i] if gamma_isotope is not None else None
                )
                gamma_N_at_sigma = gamma_N[j, :, i] if gamma_N is not None else None
                gamma_U_at_sigma = gamma_U[j, :, i] if gamma_U is not None else None

                write_kappa_to_hdf5(
                    temperatures,
                    mesh,
                    bz_grid=bz_grid,
                    frequency=frequencies,
                    group_velocity=group_velocities_i,
                    gv_by_gv=gv_by_gv_i,
                    heat_capacity=mode_heat_capacities[:, i],
                    extra_datasets=extra_gp_data,
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
                    group_velocities_ik = group_velocities_i[k]
                    gv_by_gv_ik = gv_by_gv_i[k]
                    ave_pp_ik = ave_pp[i, k] if ave_pp is not None else None
                    frequencies = phonons[gp, bi]
                    gamma_isotope_at_sigma = (
                        gamma_isotope[j, i, k] if gamma_isotope is not None else None
                    )
                    gamma_N_at_sigma = (
                        gamma_N[j, :, i, k] if gamma_N is not None else None
                    )
                    gamma_U_at_sigma = (
                        gamma_U[j, :, i, k] if gamma_U is not None else None
                    )
                    extra_gp_band: dict[str, Any] | None = (
                        {key: val[k] for key, val in extra_gp_data.items()}
                        if extra_gp_data is not None
                        else None
                    )
                    write_kappa_to_hdf5(
                        temperatures,
                        mesh,
                        bz_grid=bz_grid,
                        frequency=frequencies,
                        group_velocity=group_velocities_ik,
                        gv_by_gv=gv_by_gv_ik,
                        heat_capacity=mode_heat_capacities[:, i, k],
                        extra_datasets=extra_gp_band,
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
        br: RTACalculator,
        volume: float,
        compression: Literal["gzip", "lzf"] | int | None,
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
        gamma_elph = br.gamma_elph
        mesh = br.mesh_numbers
        bz_grid = br.bz_grid
        frequencies = br.frequencies
        kappa = br.kappa
        gv = br.group_velocities
        gv_by_gv = br.gv_by_gv
        mode_cv = br.mode_heat_capacities
        ave_pp = br.averaged_pp_interaction
        qpoints = br.qpoints
        grid_points = br.grid_points
        weights = br.grid_weights
        boundary_mfp = br.boundary_mfp
        extra_full: dict[str, Any] | None = br.get_extra_kappa_output()
        num_sigma = len(sigmas)

        for i, sigma in enumerate(sigmas):
            gamma_isotope_at_sigma = (
                gamma_isotope[i] if gamma_isotope is not None else None
            )
            gamma_N_at_sigma = gamma_N[i] if gamma_N is not None else None
            gamma_U_at_sigma = gamma_U[i] if gamma_U is not None else None
            extra_at_sigma: dict[str, Any] | None = (
                {
                    k: v[i] if v is not None and len(v) == num_sigma else v
                    for k, v in extra_full.items()
                }
                if extra_full is not None
                else None
            )

            write_kappa_to_hdf5(
                temperatures,
                mesh,
                boundary_mfp=cast(float, boundary_mfp),
                bz_grid=bz_grid,
                frequency=frequencies,
                group_velocity=gv,
                gv_by_gv=gv_by_gv,
                heat_capacity=mode_cv,
                kappa=kappa[i],
                extra_datasets=extra_at_sigma,
                gamma=gamma[i],
                gamma_isotope=gamma_isotope_at_sigma,
                gamma_N=gamma_N_at_sigma,
                gamma_U=gamma_U_at_sigma,
                gamma_elph=gamma_elph,
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
        br: RTACalculator,
        interaction: Interaction,
        i: int,
        compression: Literal["gzip", "lzf"] | int | None = "gzip",
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
        gamma_detail = _require_ndarray_not_none(gamma_detail, "gamma_detail")

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
