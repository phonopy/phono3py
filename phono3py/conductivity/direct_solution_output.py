"""Direct-solution output helpers (file writers)."""

from __future__ import annotations

import os
from typing import Any, Literal, TypeAlias

from numpy.typing import NDArray

from phono3py.conductivity.base import get_unit_to_WmK
from phono3py.conductivity.direct_solution_base import ConductivityLBTEBase
from phono3py.conductivity.type_dispatch import get_lbte_writer_kappa_data
from phono3py.conductivity.utils import select_colmat_solver
from phono3py.file_IO import (
    write_collision_eigenvalues_to_hdf5,
    write_collision_to_hdf5,
    write_kappa_to_hdf5,
    write_unitary_matrix_to_hdf5,
)
from phono3py.phonon3.interaction import Interaction, all_bands_exist

cond_LBTE_type: TypeAlias = ConductivityLBTEBase


def _pick_sigma_item(
    values: NDArray[Any] | None, sigma_index: int
) -> NDArray[Any] | None:
    """Return values at sigma index, or None when values is None."""
    if values is None:
        return None
    return values[sigma_index]


class ConductivityLBTEWriter:
    """Collection of result writers."""

    @staticmethod
    def write_collision(
        lbte: cond_LBTE_type,
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
        lbte: cond_LBTE_type,
        volume: float,
        is_reducible_collision_matrix: bool = False,
        write_LBTE_solution: bool = False,
        pinv_solver: int | None = None,
        compression: Literal["gzip", "lzf"] | int | None = "gzip",
        filename: str | os.PathLike | None = None,
        log_level: int = 0,
    ) -> None:
        """Write kappa related properties into a hdf5 file."""
        kappa_data = get_lbte_writer_kappa_data(lbte)
        kappa = kappa_data["kappa"]
        mode_kappa = kappa_data["mode_kappa"]
        kappa_RTA = kappa_data["kappa_RTA"]
        mode_kappa_RTA = kappa_data["mode_kappa_RTA"]
        gv = kappa_data["group_velocities"]
        gv_by_gv = kappa_data["gv_by_gv"]
        kappa_P_exact = kappa_data["kappa_P_exact"]
        kappa_P_RTA = kappa_data["kappa_P_RTA"]
        kappa_C = kappa_data["kappa_C"]
        mode_kappa_P_exact = kappa_data["mode_kappa_P_exact"]
        mode_kappa_P_RTA = kappa_data["mode_kappa_P_RTA"]
        mode_kappa_C = kappa_data["mode_kappa_C"]

        temperatures = lbte.temperatures
        sigmas = lbte.sigmas
        sigma_cutoff = lbte.sigma_cutoff_width
        mesh = lbte.mesh_numbers
        bz_grid = lbte.bz_grid
        grid_points = lbte.grid_points
        weights = lbte.grid_weights
        frequencies = lbte.frequencies
        ave_pp = lbte.averaged_pp_interaction
        qpoints = lbte.qpoints
        gamma = lbte.gamma
        gamma_isotope = lbte.gamma_isotope
        f_vector = lbte.get_f_vectors()
        mode_cv = lbte.mode_heat_capacities
        mfp = lbte.get_mean_free_path()
        assert mfp is not None
        boundary_mfp = lbte.boundary_mfp

        coleigs = lbte.collision_eigenvalues
        unitary_matrix = lbte.collision_matrix

        if is_reducible_collision_matrix:
            frequencies = lbte.get_frequencies_all()
        else:
            frequencies = lbte.frequencies

        for i, sigma in enumerate(sigmas):
            kappa_at_sigma = _pick_sigma_item(kappa, i)
            mode_kappa_at_sigma = _pick_sigma_item(mode_kappa, i)
            kappa_RTA_at_sigma = _pick_sigma_item(kappa_RTA, i)
            mode_kappa_RTA_at_sigma = _pick_sigma_item(mode_kappa_RTA, i)
            kappa_P_exact_at_sigma = _pick_sigma_item(kappa_P_exact, i)
            kappa_P_RTA_at_sigma = _pick_sigma_item(kappa_P_RTA, i)
            kappa_C_at_sigma = _pick_sigma_item(kappa_C, i)
            if kappa_P_exact_at_sigma is None or kappa_C_at_sigma is None:
                kappa_TOT_exact_at_sigma = None
            else:
                kappa_TOT_exact_at_sigma = kappa_P_exact_at_sigma + kappa_C_at_sigma
            if kappa_P_RTA_at_sigma is None or kappa_C_at_sigma is None:
                kappa_TOT_RTA_at_sigma = None
            else:
                kappa_TOT_RTA_at_sigma = kappa_P_RTA_at_sigma + kappa_C_at_sigma
            mode_kappa_P_exact_at_sigma = _pick_sigma_item(mode_kappa_P_exact, i)
            mode_kappa_P_RTA_at_sigma = _pick_sigma_item(mode_kappa_P_RTA, i)
            mode_kappa_C_at_sigma = _pick_sigma_item(mode_kappa_C, i)
            gamma_isotope_at_sigma = _pick_sigma_item(gamma_isotope, i)
            write_kappa_to_hdf5(
                temperatures,
                mesh,
                boundary_mfp=boundary_mfp,
                bz_grid=bz_grid,
                frequency=frequencies,
                group_velocity=gv,
                gv_by_gv=gv_by_gv,
                mean_free_path=mfp[i],
                heat_capacity=mode_cv,
                kappa=kappa_at_sigma,
                mode_kappa=mode_kappa_at_sigma,
                kappa_RTA=kappa_RTA_at_sigma,
                mode_kappa_RTA=mode_kappa_RTA_at_sigma,
                kappa_P_exact=kappa_P_exact_at_sigma,
                kappa_P_RTA=kappa_P_RTA_at_sigma,
                kappa_C=kappa_C_at_sigma,
                kappa_TOT_exact=kappa_TOT_exact_at_sigma,
                kappa_TOT_RTA=kappa_TOT_RTA_at_sigma,
                mode_kappa_P_exact=mode_kappa_P_exact_at_sigma,
                mode_kappa_P_RTA=mode_kappa_P_RTA_at_sigma,
                mode_kappa_C=mode_kappa_C_at_sigma,
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
