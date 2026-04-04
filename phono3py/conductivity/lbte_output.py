"""Direct-solution output helpers (file writers)."""

from __future__ import annotations

import os
from typing import Any, Literal

from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.utils import get_unit_to_WmK, select_colmat_solver
from phono3py.file_IO import (
    write_collision_eigenvalues_to_hdf5,
    write_collision_to_hdf5,
    write_kappa_to_hdf5,
    write_unitary_matrix_to_hdf5,
)
from phono3py.phonon3.interaction import Interaction, all_bands_exist


class ConductivityLBTEWriter:
    """Collection of result writers."""

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
        mode_kappa = lbte.mode_kappa
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
            gamma_isotope_at_sigma = (
                gamma_isotope[i] if gamma_isotope is not None else None
            )
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
                boundary_mfp=boundary_mfp,
                bz_grid=bz_grid,
                frequency=frequencies,
                group_velocity=gv,
                mean_free_path=mfp[i],
                heat_capacity=mode_cv,
                kappa=kappa[i],
                mode_kappa=mode_kappa[i],
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
