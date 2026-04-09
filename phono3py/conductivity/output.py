"""Output helpers (file writers) for conductivity calculations."""

from __future__ import annotations

import os
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.calculators import LBTECalculator, RTACalculator
from phono3py.conductivity.utils import get_unit_to_WmK, select_colmat_solver
from phono3py.file_IO import (
    write_collision_eigenvalues_to_hdf5,
    write_collision_to_hdf5,
    write_gamma_detail_to_hdf5,
    write_kappa_to_hdf5,
    write_unitary_matrix_to_hdf5,
)
from phono3py.phonon3.interaction import Interaction, all_bands_exist
from phono3py.phonon3.triplets import get_all_triplets


def _require_ndarray_not_none(value: NDArray[Any] | None, name: str) -> NDArray[Any]:
    """Return non-None ndarray, otherwise fail fast with assertion."""
    assert value is not None, f"{name} must not be None"
    return value


def _slice_at_sigma(
    i: int,
    gamma_isotope: NDArray[np.double] | None,
    extra_full: dict[str, Any] | None,
    num_sigma: int,
) -> tuple[NDArray[np.double] | None, dict[str, Any] | None]:
    """Extract per-sigma slices of gamma_isotope and extra datasets."""
    gamma_isotope_at_sigma = gamma_isotope[i] if gamma_isotope is not None else None
    extra_at_sigma: dict[str, Any] | None = (
        {
            k: v[i] if v is not None and len(v) == num_sigma else v
            for k, v in extra_full.items()
        }
        if extra_full is not None
        else None
    )
    return gamma_isotope_at_sigma, extra_at_sigma


class ConductivityRTAWriter:
    """Collection of RTA result writers."""

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
            gamma_isotope_at_sigma, extra_at_sigma = _slice_at_sigma(
                i,
                gamma_isotope,
                extra_full,
                num_sigma,
            )
            gamma_N_at_sigma = gamma_N[i] if gamma_N is not None else None
            gamma_U_at_sigma = gamma_U[i] if gamma_U is not None else None

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


class ConductivityLBTEWriter:
    """Collection of LBTE result writers."""

    @staticmethod
    def write_collision(
        lbte: LBTECalculator,
        interaction: Interaction,
        i: int | None = None,
        is_reducible_collision_matrix: bool = False,
        is_one_gp_colmat: bool = False,
        filename: str | os.PathLike | None = None,
    ) -> None:
        """Write collision matrix into hdf5 file."""
        grid_points = lbte.grid_points
        temperatures = lbte.temperatures
        sigmas = lbte.sigmas
        sigma_cutoff = lbte.sigma_cutoff_width
        gamma = lbte.gamma
        gamma_isotope = lbte.gamma_isotope
        collision_matrix = lbte.collision_matrix
        assert collision_matrix is not None
        mesh = lbte.mesh_numbers

        if i is not None:
            gp = grid_points[i]
            if is_one_gp_colmat:
                igp = 0
            else:
                if is_reducible_collision_matrix:
                    igp = interaction.bz_grid.bzg2grg[gp]
                else:
                    igp = i
            if all_bands_exist(interaction):
                for j, sigma in enumerate(sigmas):
                    if gamma_isotope is not None:
                        gamma_isotope_at_sigma = gamma_isotope[j, igp]
                    else:
                        gamma_isotope_at_sigma = None
                    write_collision_to_hdf5(
                        temperatures,
                        mesh,
                        gamma=gamma[j, :, igp],
                        gamma_isotope=gamma_isotope_at_sigma,
                        collision_matrix=collision_matrix[j, :, igp],
                        grid_point=gp,
                        sigma=sigma,
                        sigma_cutoff=sigma_cutoff,
                        filename=filename,
                    )
            else:
                for j, sigma in enumerate(sigmas):
                    for k, bi in enumerate(interaction.band_indices):
                        if gamma_isotope is not None:
                            gamma_isotope_at_sigma = gamma_isotope[j, igp, k]
                        else:
                            gamma_isotope_at_sigma = None
                        write_collision_to_hdf5(
                            temperatures,
                            mesh,
                            gamma=gamma[j, :, igp, k],
                            gamma_isotope=gamma_isotope_at_sigma,
                            collision_matrix=collision_matrix[j, :, igp, k],
                            grid_point=gp,
                            band_index=bi,
                            sigma=sigma,
                            sigma_cutoff=sigma_cutoff,
                            filename=filename,
                        )
        else:
            for j, sigma in enumerate(sigmas):
                if gamma_isotope is not None:
                    gamma_isotope_at_sigma = gamma_isotope[j]
                else:
                    gamma_isotope_at_sigma = None
                write_collision_to_hdf5(
                    temperatures,
                    mesh,
                    gamma=gamma[j],
                    gamma_isotope=gamma_isotope_at_sigma,
                    collision_matrix=collision_matrix[j],
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    filename=filename,
                )

    @staticmethod
    def write_kappa(
        lbte: LBTECalculator,
        volume: float,
        is_reducible_collision_matrix: bool = False,
        write_LBTE_solution: bool = False,
        pinv_solver: int | None = None,
        compression: Literal["gzip", "lzf"] | int | None = "gzip",
        filename: str | os.PathLike | None = None,
        log_level: int = 0,
    ) -> None:
        """Write kappa related properties into a hdf5 file."""
        kappa = lbte.kappa
        kappa_RTA = lbte.kappa_RTA
        mode_kappa_RTA = lbte.mode_kappa_RTA
        gv = lbte.group_velocities
        extra_full: dict[str, Any] | None = lbte.get_extra_kappa_output()

        temperatures = lbte.temperatures
        sigmas = lbte.sigmas
        sigma_cutoff = lbte.sigma_cutoff_width
        mesh = lbte.mesh_numbers
        bz_grid = lbte.bz_grid
        grid_points = lbte.grid_points
        weights = lbte.grid_weights
        ave_pp = lbte.averaged_pp_interaction
        qpoints = lbte.qpoints
        gamma = lbte.gamma
        gamma_isotope = lbte.gamma_isotope
        f_vector = lbte.f_vectors
        mode_cv = lbte.mode_heat_capacities
        mfp = lbte.mfp
        assert mfp is not None
        boundary_mfp = lbte.boundary_mfp

        coleigs = lbte.collision_eigenvalues
        unitary_matrix = lbte.collision_matrix

        if is_reducible_collision_matrix:
            frequencies = lbte.get_frequencies_all()
        else:
            frequencies = lbte.frequencies

        num_sigma = len(sigmas)

        for i, sigma in enumerate(sigmas):
            gamma_isotope_at_sigma, extra_at_sigma = _slice_at_sigma(
                i,
                gamma_isotope,
                extra_full,
                num_sigma,
            )
            write_kappa_to_hdf5(
                temperatures,
                mesh,
                boundary_mfp=boundary_mfp,
                bz_grid=bz_grid,
                frequency=frequencies,
                group_velocity=gv,
                mean_free_path=mfp[i],
                heat_capacity=mode_cv,
                kappa=kappa[i],
                kappa_RTA=kappa_RTA[i],
                mode_kappa_RTA=mode_kappa_RTA[i],
                extra_datasets=extra_at_sigma,
                f_vector=f_vector,
                gamma=gamma[i],
                gamma_isotope=gamma_isotope_at_sigma,
                averaged_pp_interaction=ave_pp,
                qpoint=qpoints,
                grid_point=grid_points,
                weight=weights,
                sigma=sigma,
                sigma_cutoff=sigma_cutoff,
                kappa_unit_conversion=get_unit_to_WmK() / volume,
                compression=compression,
                filename=filename,
                verbose=log_level > 0,
            )

            if coleigs is not None:
                write_collision_eigenvalues_to_hdf5(
                    temperatures,
                    mesh,
                    coleigs[i],
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    filename=filename,
                    verbose=log_level > 0,
                )

                if write_LBTE_solution:
                    if pinv_solver is not None:
                        solver = select_colmat_solver(pinv_solver)
                        if solver in [1, 2, 3, 4, 5]:
                            write_unitary_matrix_to_hdf5(
                                temperatures,
                                mesh,
                                unitary_matrix=unitary_matrix,
                                sigma=sigma,
                                sigma_cutoff=sigma_cutoff,
                                solver=solver,
                                filename=filename,
                                verbose=log_level > 0,
                            )
