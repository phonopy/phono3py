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
from typing import Any, Literal, TypeAlias, TypedDict, cast, overload

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.exceptions import LBTECollisionReadError
from phono3py.conductivity.factory import make_conductivity_calculator
from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.lbte_output import ConductivityLBTEWriter
from phono3py.conductivity.utils import build_options, write_pp_interaction
from phono3py.file_IO import read_collision_from_hdf5
from phono3py.phonon3.interaction import Interaction, all_bands_exist


class _LBTEFullCollisionOptions(TypedDict):
    write_LBTE_solution: bool
    read_collision: str | Sequence | None
    read_from: Literal["full_matrix", "grid_points"] | None
    grid_points: Sequence[int] | NDArray[np.int64] | None
    output_filename: str | os.PathLike | None


class _CollisionReadContext(TypedDict):
    mesh: NDArray[np.int64]
    indices: str | Sequence[int]
    sigma: float | None
    sigma_cutoff: float | None
    filename: str | os.PathLike | None
    log_level: int


_CollisionMatrixData: TypeAlias = tuple[
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
]
_CollisionReadSource: TypeAlias = Literal["full_matrix", "grid_points"]
_AllocatedCollisionData: TypeAlias = tuple[None, None, NDArray[np.double]]
_CollisionData: TypeAlias = _CollisionMatrixData | _AllocatedCollisionData


def _build_lbte_full_collision_options(
    *,
    write_LBTE_solution: bool,
    read_collision: str | Sequence[int] | None,
    read_from: _CollisionReadSource | None,
    grid_points: Sequence[int] | NDArray[np.int64] | None,
    output_filename: str | os.PathLike | None,
) -> _LBTEFullCollisionOptions:
    return build_options(
        _LBTEFullCollisionOptions,
        write_LBTE_solution=write_LBTE_solution,
        read_collision=read_collision,
        read_from=read_from,
        grid_points=grid_points,
        output_filename=output_filename,
    )


def _build_collision_read_context(
    *,
    mesh: NDArray[np.int64],
    indices: str | Sequence[int],
    sigma: float | None,
    sigma_cutoff: float | None,
    filename: str | os.PathLike | None,
    log_level: int,
) -> _CollisionReadContext:
    return build_options(
        _CollisionReadContext,
        mesh=mesh,
        indices=indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        log_level=log_level,
    )


@overload
def _read_collision_data(
    context: _CollisionReadContext,
    *,
    grid_point: int | None = None,
    band_index: int | None = None,
    only_temperatures: Literal[False] = False,
) -> _CollisionMatrixData: ...


@overload
def _read_collision_data(
    context: _CollisionReadContext,
    *,
    grid_point: int | None = None,
    band_index: int | None = None,
    only_temperatures: Literal[True],
) -> _AllocatedCollisionData: ...


def _read_collision_data(
    context: _CollisionReadContext,
    *,
    grid_point: int | None = None,
    band_index: int | None = None,
    only_temperatures: bool = False,
) -> _CollisionData:
    return read_collision_from_hdf5(
        context["mesh"],
        indices=context["indices"],
        grid_point=grid_point,
        band_index=band_index,
        sigma=context["sigma"],
        sigma_cutoff=context["sigma_cutoff"],
        filename=context["filename"],
        only_temperatures=only_temperatures,
        verbose=(False if only_temperatures else (context["log_level"] > 0)),
    )


def get_thermal_conductivity_LBTE(
    interaction: Interaction,
    temperatures: Sequence[float] | NDArray[np.double] | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    is_isotope: bool = False,
    mass_variances: Sequence[float] | NDArray[np.double] | None = None,
    grid_points: Sequence[int] | NDArray[np.int64] | None = None,
    boundary_mfp: float | None = None,  # in micrometer
    solve_collective_phonon: bool = False,
    is_reducible_collision_matrix: bool = False,
    is_kappa_star: bool = True,
    gv_delta_q: float | None = None,
    is_full_pp: bool = False,
    transport_type: str | None = None,
    pinv_cutoff: float = 1.0e-8,
    pinv_solver: int = 0,  # default: dsyev in lapacke
    pinv_method: int = 0,  # default: abs(eig) < cutoff
    write_collision: bool = False,
    read_collision: str | Sequence[int] | None = None,
    write_kappa: bool = False,
    write_pp: bool = False,
    read_pp: bool = False,
    write_LBTE_solution: bool = False,
    compression: Literal["gzip", "lzf"] | int | None = "gzip",
    input_filename: str | os.PathLike | None = None,
    output_filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> LBTECalculator:
    """Calculate lattice thermal conductivity by direct solution."""
    _temperatures = _normalize_lbte_temperatures(temperatures)
    if sigmas is None:
        sigmas = []
    if log_level:
        print("-" * 19 + " Lattice thermal conductivity (LBTE) " + "-" * 19)
        print(
            "Cutoff frequency of pseudo inversion of collision matrix: %s" % pinv_cutoff
        )

    method = f"{transport_type}-lbte" if transport_type else "lbte"
    return _run_standard_lbte(
        interaction,
        method=method,
        temperatures=_temperatures,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        grid_points=grid_points,
        boundary_mfp=boundary_mfp,
        solve_collective_phonon=solve_collective_phonon,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        write_collision=write_collision,
        read_collision=read_collision,
        write_kappa=write_kappa,
        write_pp=write_pp,
        read_pp=read_pp,
        write_LBTE_solution=write_LBTE_solution,
        compression=compression,
        input_filename=input_filename,
        output_filename=output_filename,
        log_level=log_level,
    )


def _run_standard_lbte(
    interaction: Interaction,
    *,
    method: str = "lbte",
    temperatures: Sequence[float] | NDArray[np.double],
    sigmas: Sequence[float | None],
    sigma_cutoff: float | None,
    is_isotope: bool,
    mass_variances: Sequence[float] | NDArray[np.double] | None,
    grid_points: Sequence[int] | NDArray[np.int64] | None,
    boundary_mfp: float | None,
    solve_collective_phonon: bool,
    is_reducible_collision_matrix: bool,
    is_kappa_star: bool,
    gv_delta_q: float | None,
    is_full_pp: bool,
    pinv_cutoff: float,
    pinv_solver: int,
    pinv_method: int,
    write_collision: bool,
    read_collision: str | Sequence[int] | None,
    write_kappa: bool,
    write_pp: bool,
    read_pp: bool,
    write_LBTE_solution: bool,
    compression: Literal["gzip", "lzf"] | int | None,
    input_filename: str | os.PathLike | None,
    output_filename: str | os.PathLike | None,
    log_level: int,
) -> LBTECalculator:
    """Build and run an LBTECalculator."""
    temps = _get_lbte_initial_temperatures(temperatures, read_collision)

    lbte = cast(
        LBTECalculator,
        make_conductivity_calculator(
            interaction,
            method=method,
            temperatures=temps,
            sigmas=sigmas,
            sigma_cutoff=sigma_cutoff,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            boundary_mfp=boundary_mfp,
            is_kappa_star=is_kappa_star,
            gv_delta_q=gv_delta_q,
            is_full_pp=is_full_pp,
            read_pp=read_pp,
            pp_filename=input_filename,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            solve_collective_phonon=solve_collective_phonon,
            pinv_cutoff=pinv_cutoff,
            pinv_solver=pinv_solver,
            pinv_method=pinv_method,
            log_level=log_level,
        ),
    )

    read_from, read_collision_failed = _read_lbte_collision_if_requested(
        cast(Any, lbte),
        read_collision=read_collision,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        input_filename=input_filename,
        log_level=log_level,
    )
    if read_collision_failed:
        raise LBTECollisionReadError("Reading collision failed.")

    if not read_collision:
        compression_pp: Literal["gzip", "lzf"] = (
            compression if isinstance(compression, str) else "gzip"
        )

        def _on_grid_point(i: int) -> None:
            if write_pp:
                write_pp_interaction(
                    cast(Any, lbte),
                    interaction,
                    i,
                    filename=output_filename,
                    compression=compression_pp,
                )
            if write_collision:
                ConductivityLBTEWriter.write_collision(
                    cast(Any, lbte),
                    interaction,
                    i=i,
                    is_reducible_collision_matrix=is_reducible_collision_matrix,
                    is_one_gp_colmat=(grid_points is not None),
                    filename=output_filename,
                )

        lbte.run(
            on_grid_point=_on_grid_point if (write_pp or write_collision) else None
        )

    full_collision_options = _build_lbte_full_collision_options(
        write_LBTE_solution=write_LBTE_solution,
        read_collision=read_collision,
        read_from=read_from,
        grid_points=grid_points,
        output_filename=output_filename,
    )
    _write_full_collision_if_requested(
        cast(Any, lbte), interaction, **full_collision_options
    )

    if grid_points is None and all_bands_exist(interaction):
        if read_collision:
            lbte.set_kappa_at_sigmas()
        if write_kappa:
            compression_kappa: Literal["gzip", "lzf"] = (
                compression if isinstance(compression, str) else "gzip"
            )
            ConductivityLBTEWriter.write_kappa(
                cast(Any, lbte),
                interaction.primitive.volume,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                write_LBTE_solution=write_LBTE_solution,
                pinv_solver=pinv_solver,
                compression=compression_kappa,
                filename=output_filename,
                log_level=log_level,
            )

    return lbte


def _normalize_lbte_temperatures(
    temperatures: Sequence[float] | NDArray | None,
) -> Sequence[float] | NDArray:
    if temperatures is None:
        return [300]
    return temperatures


def _format_lbte_temperatures_log(
    temperatures: Sequence[float] | NDArray[np.double],
) -> str:
    if len(temperatures) > 5:
        text = (" %.1f " * 5 + "...") % tuple(temperatures[:5])
        text += " %.1f" % temperatures[-1]
        return text
    return (" %.1f " * len(temperatures)) % tuple(temperatures)


def _read_lbte_collision_if_requested(
    lbte: Any,
    *,
    read_collision: str | Sequence[int] | None,
    is_reducible_collision_matrix: bool,
    input_filename: str | os.PathLike | None,
    log_level: int,
) -> tuple[_CollisionReadSource | None, bool]:
    if not read_collision:
        return None, False

    read_from = _set_collision_from_file(
        lbte,
        indices=read_collision,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        filename=input_filename,
        log_level=log_level,
    )
    if not read_from:
        return None, True

    if log_level:
        temps_read = lbte.temperatures
        assert temps_read is not None
        print("Temperature: " + _format_lbte_temperatures_log(temps_read))

    return read_from, False


def _get_lbte_initial_temperatures(
    _temperatures: Sequence[float] | NDArray, read_collision: str | Sequence[int] | None
) -> Sequence[float] | NDArray | None:
    if read_collision is not None:
        return None
    return _temperatures


def _write_full_collision_if_requested(
    lbte: Any,
    interaction: Interaction,
    *,
    write_LBTE_solution: bool,
    read_collision: str | Sequence[int] | None,
    read_from: _CollisionReadSource | None,
    grid_points: Sequence[int] | NDArray[np.int64] | None,
    output_filename: str | os.PathLike | None,
) -> None:
    # Write full collision matrix
    if not write_LBTE_solution:
        return

    if (
        read_collision
        and all_bands_exist(interaction)
        and read_from == "grid_points"
        and grid_points is None
    ) or (not read_collision):
        ConductivityLBTEWriter.write_collision(
            lbte, interaction, filename=output_filename
        )


def _set_collision_from_file(
    lbte: Any,
    indices: str | Sequence[int] = "all",
    is_reducible_collision_matrix: bool = False,
    filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> _CollisionReadSource | Literal[False] | None:
    """Set collision matrix from that read from files.

    If collision-m*.hdf5 that contains all data is not found,
    collision-m*-gp*.hdf5 files at grid points are searched. If any of those
    files are not found, collision-m*-gp*-b*.hdf5 files at grid points and bands
    are searched. If any of those files are not found, it fails.

    lbte : LBTE calculator instance.
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
        context = _build_collision_read_context(
            mesh=mesh,
            indices=indices,
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
            log_level=log_level,
        )
        collision_data = _read_collision_data(context)
        if log_level:
            sys.stdout.flush()

        if _set_collision_from_full_matrix_if_available(
            lbte,
            collision_data,
            i_sigma,
            arrays_allocated,
        ):
            if not arrays_allocated:
                arrays_allocated = True
            read_from = "full_matrix"
        else:
            vals = _allocate_collision_with_fallback(
                grid_points,
                context,
                log_level,
            )
            if not vals:
                return False
            colmat_at_sigma, gamma_at_sigma, temperatures = vals

            if not arrays_allocated:
                arrays_allocated = True
                # The following invokes self._allocate_values()
                lbte.temperatures = temperatures

            collision_matrix = lbte.collision_matrix
            gamma = lbte.gamma
            assert collision_matrix is not None
            assert gamma is not None

            for i, gp in enumerate(grid_points):
                if not _collect_collision_with_band_fallback(
                    collision_matrix[i_sigma],
                    gamma[i_sigma],
                    temperatures,
                    context,
                    i,
                    gp,
                    bz_grid.bzg2grg,
                    is_reducible_collision_matrix,
                ):
                    return False
            read_from = "grid_points"

    return read_from


def _set_collision_from_full_matrix_if_available(
    lbte: Any,
    collision_data: _CollisionMatrixData | None,
    i_sigma: int,
    arrays_allocated: bool,
) -> bool:
    if not collision_data:
        return False

    colmat_at_sigma, gamma_at_sigma, temperatures = collision_data
    if not arrays_allocated:
        # The following invokes self._allocate_values()
        lbte.temperatures = temperatures
    collision_matrix = lbte.collision_matrix
    gamma = lbte.gamma
    assert collision_matrix is not None
    assert gamma is not None
    collision_matrix[i_sigma] = colmat_at_sigma[0]
    gamma[i_sigma] = gamma_at_sigma[0]
    return True


def _allocate_collision(
    for_gps: bool,
    grid_points: Sequence[int] | NDArray[np.int64],
    context: _CollisionReadContext,
) -> _AllocatedCollisionData | Literal[False]:
    if for_gps:
        collision = _read_collision_data(
            context,
            grid_point=grid_points[0],
            only_temperatures=True,
        )
    else:
        collision = _read_collision_data(
            context,
            grid_point=grid_points[0],
            band_index=0,
            only_temperatures=True,
        )
    if collision is None:
        return False

    temperatures = collision[2]
    return None, None, temperatures


def _allocate_collision_with_fallback(
    grid_points: Sequence[int] | NDArray[np.int64],
    context: _CollisionReadContext,
    log_level: int,
) -> _AllocatedCollisionData | Literal[False]:
    vals = _allocate_collision(
        True,
        grid_points,
        context,
    )
    if vals:
        return vals

    if log_level:
        print("Collision at grid point %d doesn't exist." % grid_points[0])

    vals = _allocate_collision(
        False,
        grid_points,
        context,
    )
    if vals:
        return vals

    if log_level:
        print(
            "Collision at (grid point %d, band index %d) "
            "doesn't exist." % (grid_points[0], 1)
        )
    return False


def _collect_collision_gp(
    colmat_at_sigma: NDArray[np.double],
    gamma_at_sigma: NDArray[np.double],
    temperatures: NDArray[np.double],
    context: _CollisionReadContext,
    i: int,
    gp: int,
    bzg2grg: NDArray[np.int64],
    is_reducible_collision_matrix: bool,
) -> bool:
    collision_gp = _read_collision_data(context, grid_point=gp)
    if context["log_level"]:
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


def _collect_collision_with_band_fallback(
    colmat_at_sigma: NDArray[np.double],
    gamma_at_sigma: NDArray[np.double],
    temperatures: NDArray[np.double],
    context: _CollisionReadContext,
    i: int,
    gp: int,
    bzg2grg: NDArray[np.int64],
    is_reducible_collision_matrix: bool,
) -> bool:
    if _collect_collision_gp(
        colmat_at_sigma,
        gamma_at_sigma,
        temperatures,
        context,
        i,
        gp,
        bzg2grg,
        is_reducible_collision_matrix,
    ):
        return True

    num_band = colmat_at_sigma.shape[2]
    for i_band in range(num_band):
        if not _collect_collision_band(
            colmat_at_sigma,
            gamma_at_sigma,
            temperatures,
            context,
            i,
            gp,
            bzg2grg,
            i_band,
            is_reducible_collision_matrix,
        ):
            return False
    return True


def _collect_collision_band(
    colmat_at_sigma: NDArray[np.double],
    gamma_at_sigma: NDArray[np.double],
    temperatures: NDArray[np.double],
    context: _CollisionReadContext,
    i: int,
    gp: int,
    bzg2grg: NDArray[np.int64],
    j: int,
    is_reducible_collision_matrix: bool,
) -> bool:
    collision_band = _read_collision_data(
        context,
        grid_point=gp,
        band_index=j,
    )
    if context["log_level"]:
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
