"""Init lattice thermal conductivity classes with direct solution."""

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

import os
import sys
from collections.abc import Sequence
from typing import Literal, Union

from numpy.typing import ArrayLike

from phono3py.conductivity.base import get_unit_to_WmK
from phono3py.conductivity.direct_solution import ConductivityLBTE
from phono3py.conductivity.direct_solution_base import ConductivityLBTEBase
from phono3py.conductivity.utils import (
    select_colmat_solver,
    write_pp_interaction,
)
from phono3py.conductivity.wigner_direct_solution import ConductivityWignerLBTE
from phono3py.file_IO import (
    read_collision_from_hdf5,
    write_collision_eigenvalues_to_hdf5,
    write_collision_to_hdf5,
    write_kappa_to_hdf5,
    write_unitary_matrix_to_hdf5,
)
from phono3py.phonon3.interaction import Interaction, all_bands_exist

cond_LBTE_type = Union[ConductivityLBTE, ConductivityWignerLBTE]


def get_thermal_conductivity_LBTE(
    interaction: Interaction,
    temperatures: Sequence | None = None,
    sigmas: Sequence | None = None,
    sigma_cutoff: float | None = None,
    is_isotope: bool = False,
    mass_variances: Sequence | None = None,
    grid_points: ArrayLike | None = None,
    boundary_mfp: float | None = None,  # in micrometer
    solve_collective_phonon: bool = False,
    is_reducible_collision_matrix: bool = False,
    is_kappa_star: bool = True,
    gv_delta_q: float | None = None,
    is_full_pp: bool = False,
    conductivity_type: Literal["wigner", "kubo"] | None = None,
    pinv_cutoff: float = 1.0e-8,
    pinv_solver: int = 0,  # default: dsyev in lapacke
    pinv_method: int = 0,  # default: abs(eig) < cutoff
    write_collision: bool = False,
    read_collision: str | Sequence | None = None,
    write_kappa: bool = False,
    write_pp: bool = False,
    read_pp: bool = False,
    write_LBTE_solution: bool = False,
    compression: Literal["gzip", "lzf"] | int | None = "gzip",
    input_filename: str | os.PathLike | None = None,
    output_filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> ConductivityLBTE | ConductivityWignerLBTE:
    """Calculate lattice thermal conductivity by direct solution."""
    if temperatures is None:
        _temperatures = [
            300,
        ]
    else:
        _temperatures = temperatures
    if sigmas is None:
        sigmas = []
    if log_level:
        print("-" * 19 + " Lattice thermal conductivity (LBTE) " + "-" * 19)
        print(
            "Cutoff frequency of pseudo inversion of collision matrix: %s" % pinv_cutoff
        )

    if read_collision:
        temps = None
    else:
        temps = _temperatures

    if conductivity_type == "wigner":
        conductivity_LBTE_class = ConductivityWignerLBTE
    else:
        conductivity_LBTE_class = ConductivityLBTE

    lbte = conductivity_LBTE_class(
        interaction,
        grid_points=grid_points,
        temperatures=temps,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        solve_collective_phonon=solve_collective_phonon,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        pp_filename=input_filename,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        log_level=log_level,
    )

    if read_collision:
        read_from = _set_collision_from_file(
            lbte,
            indices=read_collision,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            filename=input_filename,
            log_level=log_level,
        )
        if not read_from:
            print("Reading collision failed.")
            return False
        if log_level:
            temps_read = lbte.temperatures
            if len(temps_read) > 5:
                text = (" %.1f " * 5 + "...") % tuple(temps_read[:5])
                text += " %.1f" % temps_read[-1]
            else:
                text = (" %.1f " * len(temps_read)) % tuple(temps_read)
            print("Temperature: " + text)

    # This computes pieces of collision matrix sequentially.
    for i in lbte:
        if write_pp:
            write_pp_interaction(
                lbte, interaction, i, filename=output_filename, compression=compression
            )

        if write_collision:
            ConductivityLBTEWriter.write_collision(
                lbte,
                interaction,
                i=i,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                is_one_gp_colmat=(grid_points is not None),
                filename=output_filename,
            )

        lbte.delete_gp_collision_and_pp()

    # Write full collision matrix
    if write_LBTE_solution:
        if (
            read_collision
            and all_bands_exist(interaction)
            and read_from == "grid_points"
            and grid_points is None
        ) or (not read_collision):
            ConductivityLBTEWriter.write_collision(
                lbte, interaction, filename=output_filename
            )

    if grid_points is None and all_bands_exist(interaction):
        lbte.set_kappa_at_sigmas()

        if write_kappa:
            ConductivityLBTEWriter.write_kappa(
                lbte,
                interaction.primitive.volume,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                write_LBTE_solution=write_LBTE_solution,
                pinv_solver=pinv_solver,
                compression=compression,
                filename=output_filename,
                log_level=log_level,
            )

    return lbte


class ConductivityLBTEWriter:
    """Collection of result writers."""

    @staticmethod
    def write_collision(
        lbte: cond_LBTE_type,
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
        lbte: cond_LBTE_type,
        volume: float,
        is_reducible_collision_matrix: bool = False,
        write_LBTE_solution: bool = False,
        pinv_solver: int | None = None,
        compression: str = "gzip",
        filename: str | os.PathLike | None = None,
        log_level: int = 0,
    ):
        """Write kappa related properties into a hdf5 file."""
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

        coleigs = lbte.collision_eigenvalues
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
                kappa_unit_conversion=get_unit_to_WmK() / volume,
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


def _set_collision_from_file(
    lbte: ConductivityLBTEBase,
    indices: str | Sequence | None = "all",
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
            "----------------------",
            flush=True,
        )

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
