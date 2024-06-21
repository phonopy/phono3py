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

import sys
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from phono3py.conductivity.base import unit_to_WmK
from phono3py.file_IO import (
    read_collision_from_hdf5,
    read_gamma_from_hdf5,
    write_collision_eigenvalues_to_hdf5,
    write_collision_to_hdf5,
    write_gamma_detail_to_hdf5,
    write_kappa_to_hdf5,
    write_pp_to_hdf5,
    write_unitary_matrix_to_hdf5,
)
from phono3py.phonon3.interaction import all_bands_exist
from phono3py.phonon3.triplets import get_all_triplets

if TYPE_CHECKING:
    from phono3py.conductivity.base import ConductivityBase
    from phono3py.conductivity.direct_solution import (
        ConductivityLBTE,
        ConductivityLBTEBase,
        ConductivityWignerLBTE,
    )
    from phono3py.conductivity.rta import (
        ConductivityKuboRTA,
        ConductivityRTA,
        ConductivityRTABase,
        ConductivityWignerRTA,
    )

    cond_RTA_type = Union[
        "ConductivityRTA", "ConductivityWignerRTA", "ConductivityKuboRTA"
    ]
    cond_LBTE_type = Union["ConductivityLBTE", "ConductivityWignerLBTE"]

from phono3py.phonon3.interaction import Interaction


class ConductivityRTAWriter:
    """Collection of result writers."""

    @staticmethod
    def write_gamma(
        br: "cond_RTA_type",
        interaction: Interaction,
        i: int,
        compression: str = "gzip",
        filename: Optional[str] = None,
        verbose: bool = True,
    ):
        """Write mode kappa related properties into a hdf5 file."""
        from phono3py.conductivity.rta import ConductivityRTA, ConductivityWignerRTA

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
                    kappa_unit_conversion=unit_to_WmK / volume,
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
                        kappa_unit_conversion=unit_to_WmK / volume,
                        compression=compression,
                        filename=filename,
                        verbose=verbose,
                    )

    @staticmethod
    def write_kappa(
        br: "cond_RTA_type",
        volume: float,
        compression: str = "gzip",
        filename: Optional[str] = None,
        log_level: int = 0,
    ):
        """Write kappa related properties into a hdf5 file."""
        from phono3py.conductivity.rta import ConductivityRTA, ConductivityWignerRTA

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
                kappa_unit_conversion=unit_to_WmK / volume,
                compression=compression,
                filename=filename,
                verbose=log_level,
            )

    @staticmethod
    def write_gamma_detail(
        br: "cond_RTA_type",
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


class ConductivityLBTEWriter:
    """Collection of result writers."""

    @staticmethod
    def write_collision(
        lbte: "cond_LBTE_type",
        interaction: Interaction,
        i=None,
        is_reducible_collision_matrix=False,
        is_one_gp_colmat=False,
        filename=None,
    ):
        """Write collision matrix into hdf5 file."""
        grid_points = lbte.grid_points
        temperatures = lbte.temperatures
        sigmas = lbte.sigmas
        sigma_cutoff = lbte.sigma_cutoff_width
        gamma = lbte.gamma
        gamma_isotope = lbte.gamma_isotope
        collision_matrix = lbte.collision_matrix
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
        lbte: "cond_LBTE_type",
        volume: float,
        is_reducible_collision_matrix: bool = False,
        write_LBTE_solution: bool = False,
        pinv_solver: Optional[int] = None,
        compression: str = "gzip",
        filename: Optional[str] = None,
        log_level: int = 0,
    ):
        """Write kappa related properties into a hdf5 file."""
        from phono3py.conductivity.direct_solution import (
            ConductivityLBTE,
            ConductivityWignerLBTE,
        )

        if isinstance(lbte, ConductivityLBTE):
            kappa = lbte.kappa
            mode_kappa = lbte.mode_kappa
            kappa_RTA = lbte.kappa_RTA
            mode_kappa_RTA = lbte.mode_kappa_RTA
            gv = lbte.group_velocities
            gv_by_gv = lbte.gv_by_gv
        else:
            kappa = None
            mode_kappa = None
            kappa_RTA = None
            mode_kappa_RTA = None
            gv = None
            gv_by_gv = None

        if isinstance(lbte, ConductivityWignerLBTE):
            kappa_P_exact = lbte.kappa_P_exact
            kappa_P_RTA = lbte.kappa_P_RTA
            kappa_C = lbte.kappa_C
            mode_kappa_P_exact = lbte.mode_kappa_P_exact
            mode_kappa_P_RTA = lbte.mode_kappa_P_RTA
            mode_kappa_C = lbte.mode_kappa_C
        else:
            kappa_P_exact = None
            kappa_P_RTA = None
            kappa_C = None
            mode_kappa_P_exact = None
            mode_kappa_P_RTA = None
            mode_kappa_C = None

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
        boundary_mfp = lbte.boundary_mfp

        coleigs = lbte.get_collision_eigenvalues()
        # After kappa calculation, the variable is overwritten by unitary matrix
        unitary_matrix = lbte.collision_matrix

        if is_reducible_collision_matrix:
            frequencies = lbte.get_frequencies_all()
        else:
            frequencies = lbte.frequencies

        for i, sigma in enumerate(sigmas):
            if kappa is None:
                kappa_at_sigma = None
            else:
                kappa_at_sigma = kappa[i]
            if mode_kappa is None:
                mode_kappa_at_sigma = None
            else:
                mode_kappa_at_sigma = mode_kappa[i]
            if kappa_RTA is None:
                kappa_RTA_at_sigma = None
            else:
                kappa_RTA_at_sigma = kappa_RTA[i]
            if mode_kappa_RTA is None:
                mode_kappa_RTA_at_sigma = None
            else:
                mode_kappa_RTA_at_sigma = mode_kappa_RTA[i]
            if kappa_P_exact is None:
                kappa_P_exact_at_sigma = None
            else:
                kappa_P_exact_at_sigma = kappa_P_exact[i]
            if kappa_P_RTA is None:
                kappa_P_RTA_at_sigma = None
            else:
                kappa_P_RTA_at_sigma = kappa_P_RTA[i]
            if kappa_C is None:
                kappa_C_at_sigma = None
            else:
                kappa_C_at_sigma = kappa_C[i]
            if kappa_P_exact_at_sigma is None or kappa_C_at_sigma is None:
                kappa_TOT_exact_at_sigma = None
            else:
                kappa_TOT_exact_at_sigma = kappa_P_exact_at_sigma + kappa_C_at_sigma
            if kappa_P_RTA_at_sigma is None or kappa_C_at_sigma is None:
                kappa_TOT_RTA_at_sigma = None
            else:
                kappa_TOT_RTA_at_sigma = kappa_P_RTA_at_sigma + kappa_C_at_sigma
            if mode_kappa_P_exact is None:
                mode_kappa_P_exact_at_sigma = None
            else:
                mode_kappa_P_exact_at_sigma = mode_kappa_P_exact[i]
            if mode_kappa_P_RTA is None:
                mode_kappa_P_RTA_at_sigma = None
            else:
                mode_kappa_P_RTA_at_sigma = mode_kappa_P_RTA[i]
            if mode_kappa_C is None:
                mode_kappa_C_at_sigma = None
            else:
                mode_kappa_C_at_sigma = mode_kappa_C[i]
            if gamma_isotope is None:
                gamma_isotope_at_sigma = None
            else:
                gamma_isotope_at_sigma = gamma_isotope[i]
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
                kappa_unit_conversion=unit_to_WmK / volume,
                compression=compression,
                filename=filename,
                verbose=log_level,
            )

            if coleigs is not None:
                write_collision_eigenvalues_to_hdf5(
                    temperatures,
                    mesh,
                    coleigs[i],
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    filename=filename,
                    verbose=log_level,
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
                                verbose=log_level,
                            )


def select_colmat_solver(pinv_solver):
    """Return collision matrix solver id."""
    try:
        import phono3py._phono3py as phono3c

        default_solver = phono3c.default_colmat_solver()
    except ImportError:
        print("Phono3py C-routine is not compiled correctly.")
        default_solver = 4

    solver_numbers = (1, 2, 3, 4, 5, 6)

    solver = pinv_solver
    if solver == 0:  # default solver
        if default_solver in (4, 5, 6):
            try:
                import scipy.linalg  # noqa F401
            except ImportError:
                solver = 1
            else:
                solver = default_solver
        else:
            solver = default_solver
    elif solver not in solver_numbers:
        solver = default_solver

    return solver


def set_gamma_from_file(
    br: "ConductivityRTABase", filename: Optional[str] = None, verbose: bool = True
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
                print("Read data from %s." % full_filename)
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


class ShowCalcProgress:
    """Show calculation progress."""

    @staticmethod
    def kappa_RTA(br: "ConductivityRTA", log_level):
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
                for t, k in zip(temperatures, kappa[i]):
                    print(("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("")

    @staticmethod
    def kappa_Wigner_RTA(br: "ConductivityWignerRTA", log_level):
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
                    for t, k in zip(temperatures, kappa_P_RTA[i]):
                        print("K_P\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                    print(" ")
                    for t, k in zip(temperatures, kappa_C[i]):
                        print("K_C\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_TOT_RTA[i]):
                    print("K_T\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("")


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


def set_collision_from_file(
    lbte: "ConductivityLBTEBase",
    indices="all",
    is_reducible_collision_matrix=False,
    filename=None,
    log_level=0,
):
    """Set collision matrix from that read from files.

    If collision-m*.hdf5 that contains all data is not found,
    collision-m*-gp*.hdf5 files at grid points are searched. If any of those
    files are not found, collision-m*-gp*-b*.hdf5 files at grid points and bands
    are searched. If any of those files are not found, it fails.

    lbte : ConductivityLBTEBase
        RTA lattice thermal conductivity instance.
    filename : str, optional
        This string is inserted in the filename as collision-m*.{filename}.hdf5.
    verbose : bool, optional
        Show text output or not.

    """
    bz_grid = lbte.bz_grid
    sigmas = lbte.sigmas
    sigma_cutoff = lbte.sigma_cutoff_width
    mesh = lbte.mesh_numbers
    grid_points = lbte.grid_points

    read_from = None

    if log_level:
        print(
            "---------------------- Reading collision data from file "
            "----------------------"
        )
        sys.stdout.flush()

    arrays_allocated = False
    for i_sigma, sigma in enumerate(sigmas):
        collisions = read_collision_from_hdf5(
            mesh,
            indices=indices,
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
            verbose=(log_level > 0),
        )
        if log_level:
            sys.stdout.flush()

        if collisions:
            (colmat_at_sigma, gamma_at_sigma, temperatures) = collisions
            if not arrays_allocated:
                arrays_allocated = True
                # The following invokes self._allocate_values()
                lbte.temperatures = temperatures
            lbte.collision_matrix[i_sigma] = colmat_at_sigma[0]
            lbte.gamma[i_sigma] = gamma_at_sigma[0]
            read_from = "full_matrix"
        else:
            vals = _allocate_collision(
                True,
                mesh,
                sigma,
                sigma_cutoff,
                grid_points,
                indices,
                filename,
            )
            if vals:
                colmat_at_sigma, gamma_at_sigma, temperatures = vals
            else:
                if log_level:
                    print("Collision at grid point %d doesn't exist." % grid_points[0])
                vals = _allocate_collision(
                    False,
                    mesh,
                    sigma,
                    sigma_cutoff,
                    grid_points,
                    indices,
                    filename,
                )
                if vals:
                    colmat_at_sigma, gamma_at_sigma, temperatures = vals
                else:
                    if log_level:
                        print(
                            "Collision at (grid point %d, band index %d) "
                            "doesn't exist." % (grid_points[0], 1)
                        )
                    return False

            if not arrays_allocated:
                arrays_allocated = True
                # The following invokes self._allocate_values()
                lbte.temperatures = temperatures

            for i, gp in enumerate(grid_points):
                if not _collect_collision_gp(
                    lbte.collision_matrix[i_sigma],
                    lbte.gamma[i_sigma],
                    temperatures,
                    mesh,
                    sigma,
                    sigma_cutoff,
                    i,
                    gp,
                    bz_grid.bzg2grg,
                    indices,
                    is_reducible_collision_matrix,
                    filename,
                    log_level,
                ):
                    num_band = lbte.collision_matrix.shape[3]
                    for i_band in range(num_band):
                        if not _collect_collision_band(
                            lbte.collision_matrix[i_sigma],
                            lbte.gamma[i_sigma],
                            temperatures,
                            mesh,
                            sigma,
                            sigma_cutoff,
                            i,
                            gp,
                            bz_grid.bzg2grg,
                            i_band,
                            indices,
                            is_reducible_collision_matrix,
                            filename,
                            log_level,
                        ):
                            return False
            read_from = "grid_points"

    return read_from


def _allocate_collision(
    for_gps,
    mesh,
    sigma,
    sigma_cutoff,
    grid_points,
    indices,
    filename,
):
    if for_gps:
        collision = read_collision_from_hdf5(
            mesh,
            indices=indices,
            grid_point=grid_points[0],
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
            only_temperatures=True,
            verbose=False,
        )
    else:
        collision = read_collision_from_hdf5(
            mesh,
            indices=indices,
            grid_point=grid_points[0],
            band_index=0,
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
            only_temperatures=True,
            verbose=False,
        )
    if collision is None:
        return False

    temperatures = collision[2]
    return None, None, temperatures


def _collect_collision_gp(
    colmat_at_sigma,
    gamma_at_sigma,
    temperatures,
    mesh,
    sigma,
    sigma_cutoff,
    i,
    gp,
    bzg2grg,
    indices,
    is_reducible_collision_matrix,
    filename,
    log_level,
):
    collision_gp = read_collision_from_hdf5(
        mesh,
        indices=indices,
        grid_point=gp,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        verbose=(log_level > 0),
    )
    if log_level:
        sys.stdout.flush()

    if not collision_gp:
        return False

    (colmat_at_gp, gamma_at_gp, temperatures_at_gp) = collision_gp
    if is_reducible_collision_matrix:
        igp = bzg2grg[gp]
    else:
        igp = i
    gamma_at_sigma[:, igp] = gamma_at_gp
    colmat_at_sigma[:, igp] = colmat_at_gp[0]
    temperatures[:] = temperatures_at_gp

    return True


def _collect_collision_band(
    colmat_at_sigma,
    gamma_at_sigma,
    temperatures,
    mesh,
    sigma,
    sigma_cutoff,
    i,
    gp,
    bzg2grg,
    j,
    indices,
    is_reducible_collision_matrix,
    filename,
    log_level,
):
    collision_band = read_collision_from_hdf5(
        mesh,
        indices=indices,
        grid_point=gp,
        band_index=j,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        verbose=(log_level > 0),
    )
    if log_level:
        sys.stdout.flush()

    if collision_band is False:
        return False

    (colmat_at_band, gamma_at_band, temperatures_at_band) = collision_band
    if is_reducible_collision_matrix:
        igp = bzg2grg[gp]
    else:
        igp = i
    gamma_at_sigma[:, igp, j] = gamma_at_band[0]
    colmat_at_sigma[:, igp, j] = colmat_at_band[0]
    temperatures[:] = temperatures_at_band

    return True
