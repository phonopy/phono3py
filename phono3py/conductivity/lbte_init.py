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
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.calculators import LBTECalculator
from phono3py.conductivity.exceptions import LBTECollisionReadError
from phono3py.conductivity.factory import conductivity_calculator
from phono3py.conductivity.output import ConductivityLBTEWriter
from phono3py.conductivity.utils import write_pp_interaction
from phono3py.file_IO import read_collision_from_hdf5
from phono3py.phonon.grid import get_ir_grid_points
from phono3py.phonon3.interaction import Interaction, all_bands_exist

_CollisionReadSource: TypeAlias = Literal["full_matrix", "grid_points"]


class CollisionFileReader:
    """Read collision data from HDF5 with 3-level fallback (full -> GP -> band)."""

    def __init__(
        self,
        mesh: NDArray[np.int64],
        indices: str | Sequence[int],
        sigma_cutoff: float | None,
        filename: str | os.PathLike | None,
        log_level: int,
    ):
        self._mesh = mesh
        self._indices = indices
        self._sigma_cutoff = sigma_cutoff
        self._filename = filename
        self._log_level = log_level

    def read(
        self,
        sigma: float | None,
        *,
        grid_point: int | None = None,
        band_index: int | None = None,
        only_temperatures: bool = False,
    ) -> tuple:
        """Thin wrapper around read_collision_from_hdf5."""
        return read_collision_from_hdf5(
            self._mesh,
            indices=self._indices,
            grid_point=grid_point,
            band_index=band_index,
            sigma=sigma,
            sigma_cutoff=self._sigma_cutoff,
            filename=self._filename,
            only_temperatures=only_temperatures,
            verbose=(False if only_temperatures else (self._log_level > 0)),
        )

    def peek_temperatures(
        self, sigma: float | None, first_gp: int
    ) -> NDArray[np.double]:
        """Read only temperatures, trying full -> GP -> band files."""
        try:
            _, _, temperatures = self.read(sigma, only_temperatures=True)
            return temperatures
        except FileNotFoundError:
            pass
        try:
            _, _, temperatures = self.read(
                sigma, grid_point=first_gp, only_temperatures=True
            )
            return temperatures
        except FileNotFoundError:
            pass
        try:
            _, _, temperatures = self.read(
                sigma, grid_point=first_gp, band_index=0, only_temperatures=True
            )
            return temperatures
        except FileNotFoundError:
            pass
        raise LBTECollisionReadError(
            "No collision file found for reading temperatures."
        )

    def try_full_matrix(
        self,
        sigma: float | None,
        collision_matrix: NDArray[np.double],
        gamma: NDArray[np.double],
        i_sigma: int,
    ) -> bool:
        """Try reading full collision matrix.  Returns True if successful."""
        try:
            data = self.read(sigma)
        except FileNotFoundError:
            return False
        if self._log_level:
            sys.stdout.flush()
        colmat_at_sigma, gamma_at_sigma, _temperatures = data
        collision_matrix[i_sigma] = colmat_at_sigma[0]
        gamma[i_sigma] = gamma_at_sigma[0]
        return True

    def allocate_with_fallback(
        self, sigma: float | None, grid_points: NDArray[np.int64]
    ) -> NDArray[np.double] | Literal[False]:
        """Get temperatures from per-GP or per-band file."""
        try:
            collision = self.read(
                sigma, grid_point=grid_points[0], only_temperatures=True
            )
            return collision[2]
        except FileNotFoundError:
            pass

        if self._log_level:
            print("Collision at grid point %d doesn't exist." % grid_points[0])

        try:
            collision = self.read(
                sigma, grid_point=grid_points[0], band_index=0, only_temperatures=True
            )
            return collision[2]
        except FileNotFoundError:
            pass

        if self._log_level:
            print(
                "Collision at (grid point %d, band index %d) "
                "doesn't exist." % (grid_points[0], 1)
            )
        return False

    def collect_gp(
        self,
        sigma: float | None,
        colmat_at_sigma: NDArray[np.double],
        gamma_at_sigma: NDArray[np.double],
        temperatures: NDArray[np.double],
        i: int,
        gp: int,
        bzg2grg: NDArray[np.int64],
        is_reducible: bool,
    ) -> bool:
        """Collect collision data for one grid point."""
        try:
            data = self.read(sigma, grid_point=gp)
        except FileNotFoundError:
            return False
        if self._log_level:
            sys.stdout.flush()
        colmat_at_gp, gamma_at_gp, temperatures_at_gp = data
        igp = bzg2grg[gp] if is_reducible else i
        gamma_at_sigma[:, igp] = gamma_at_gp
        colmat_at_sigma[:, igp] = colmat_at_gp[0]
        temperatures[:] = temperatures_at_gp
        return True

    def collect_band(
        self,
        sigma: float | None,
        colmat_at_sigma: NDArray[np.double],
        gamma_at_sigma: NDArray[np.double],
        temperatures: NDArray[np.double],
        i: int,
        gp: int,
        bzg2grg: NDArray[np.int64],
        j: int,
        is_reducible: bool,
    ) -> bool:
        """Collect collision data for one band."""
        try:
            data = self.read(sigma, grid_point=gp, band_index=j)
        except FileNotFoundError:
            return False
        if self._log_level:
            sys.stdout.flush()
        colmat_at_band, gamma_at_band, temperatures_at_band = data
        igp = bzg2grg[gp] if is_reducible else i
        gamma_at_sigma[:, igp, j] = gamma_at_band[0]
        colmat_at_sigma[:, igp, j] = colmat_at_band[0]
        temperatures[:] = temperatures_at_band
        return True

    def collect_with_band_fallback(
        self,
        sigma: float | None,
        colmat_at_sigma: NDArray[np.double],
        gamma_at_sigma: NDArray[np.double],
        temperatures: NDArray[np.double],
        i: int,
        gp: int,
        bzg2grg: NDArray[np.int64],
        is_reducible: bool,
    ) -> bool:
        """Collect for one GP, falling back to per-band if needed."""
        if self.collect_gp(
            sigma,
            colmat_at_sigma,
            gamma_at_sigma,
            temperatures,
            i,
            gp,
            bzg2grg,
            is_reducible,
        ):
            return True
        num_band = colmat_at_sigma.shape[2]
        for j in range(num_band):
            if not self.collect_band(
                sigma,
                colmat_at_sigma,
                gamma_at_sigma,
                temperatures,
                i,
                gp,
                bzg2grg,
                j,
                is_reducible,
            ):
                return False
        return True


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
    pinv_cutoff: float | None = None,
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
    lang: Literal["C", "Python", "Rust"] = "C",
) -> LBTECalculator:
    """Calculate lattice thermal conductivity by direct solution."""
    _sigmas = [None] if sigmas is None else list(sigmas)
    if read_collision:
        _temperatures = _peek_temperatures_from_file(
            interaction,
            _sigmas,
            sigma_cutoff,
            input_filename,
        )
    else:
        _temperatures = _normalize_lbte_temperatures(temperatures)
    _mass_variances = (
        np.asarray(mass_variances, dtype="double")
        if mass_variances is not None
        else None
    )
    _grid_points = (
        np.asarray(grid_points, dtype="int64") if grid_points is not None else None
    )
    if pinv_cutoff is None:
        _pinv_cutoff = 1.0e-8
    else:
        _pinv_cutoff = pinv_cutoff

    if log_level:
        print("-" * 19 + " Lattice thermal conductivity (LBTE) " + "-" * 19)

    method = f"{transport_type}-lbte" if transport_type else "std-lbte"
    return _run_standard_lbte(
        interaction,
        method=method,
        temperatures=_temperatures,
        sigmas=_sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=_mass_variances,
        grid_points=_grid_points,
        boundary_mfp=boundary_mfp,
        solve_collective_phonon=solve_collective_phonon,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        pinv_cutoff=_pinv_cutoff,
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
        lang=lang,
    )


def _run_standard_lbte(
    interaction: Interaction,
    *,
    method: str = "std-lbte",
    temperatures: NDArray[np.double],
    sigmas: Sequence[float | None],
    sigma_cutoff: float | None,
    is_isotope: bool,
    mass_variances: NDArray[np.double] | None,
    grid_points: NDArray[np.int64] | None,
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
    lang: Literal["C", "Python", "Rust"],
) -> LBTECalculator:
    """Build and run an LBTECalculator."""
    lbte = conductivity_calculator(
        interaction,
        temperatures,
        sigmas,
        method=method,
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
        lang=lang,
    )
    assert isinstance(lbte, LBTECalculator)

    read_from, read_collision_failed = _read_lbte_collision_if_requested(
        lbte,
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
                    lbte,
                    interaction,
                    i,
                    filename=output_filename,
                    compression=compression_pp,
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

        lbte.run(
            on_grid_point=_on_grid_point if (write_pp or write_collision) else None
        )

    _write_full_collision_if_requested(
        lbte,
        interaction,
        write_LBTE_solution=write_LBTE_solution,
        read_collision=read_collision,
        read_from=read_from,
        grid_points=grid_points,
        output_filename=output_filename,
    )

    if grid_points is None and all_bands_exist(interaction):
        if read_collision:
            lbte.set_kappa_at_sigmas()
        if write_kappa:
            compression_kappa: Literal["gzip", "lzf"] = (
                compression if isinstance(compression, str) else "gzip"
            )
            ConductivityLBTEWriter.write_kappa(
                lbte,
                interaction.primitive.volume,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                write_LBTE_solution=write_LBTE_solution,
                pinv_solver=pinv_solver,
                compression=compression_kappa,
                filename=output_filename,
                log_level=log_level,
            )

    return lbte


def _peek_temperatures_from_file(
    interaction: Interaction,
    sigmas: Sequence[float | None],
    sigma_cutoff: float | None,
    filename: str | os.PathLike | None,
) -> NDArray[np.double]:
    """Read temperatures from the first available collision HDF5 file."""
    reader = CollisionFileReader(
        mesh=interaction.mesh_numbers,
        indices="all",
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        log_level=0,
    )
    ir_grg, _, _ = get_ir_grid_points(interaction.bz_grid)
    first_gp = int(interaction.bz_grid.grg2bzg[ir_grg[0]])
    return reader.peek_temperatures(sigmas[0], first_gp)


def _normalize_lbte_temperatures(
    temperatures: Sequence[float] | NDArray | None,
) -> NDArray[np.double]:
    if temperatures is None:
        return np.array([300.0], dtype="double")
    return np.asarray(temperatures, dtype="double")


def _format_lbte_temperatures_log(
    temperatures: NDArray[np.double],
) -> str:
    if len(temperatures) > 5:
        text = (" %.1f " * 5 + "...") % tuple(temperatures[:5])
        text += " %.1f" % temperatures[-1]
        return text
    return (" %.1f " * len(temperatures)) % tuple(temperatures)


def _read_lbte_collision_if_requested(
    lbte: LBTECalculator,
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


def _write_full_collision_if_requested(
    lbte: LBTECalculator,
    interaction: Interaction,
    *,
    write_LBTE_solution: bool,
    read_collision: str | Sequence[int] | None,
    read_from: _CollisionReadSource | None,
    grid_points: NDArray[np.int64] | None,
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
    lbte: LBTECalculator,
    indices: str | Sequence[int] = "all",
    is_reducible_collision_matrix: bool = False,
    filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> _CollisionReadSource | Literal[False] | None:
    """Set collision matrix from HDF5 files.

    Tries collision-m*.hdf5 (full matrix) first, then per-grid-point files,
    then per-band files.
    """
    if log_level:
        print(
            "---------------------- Reading collision data from file "
            "----------------------",
            flush=True,
        )

    read_from = None
    for i_sigma, sigma in enumerate(lbte.sigmas):
        reader = CollisionFileReader(
            mesh=lbte.mesh_numbers,
            indices=indices,
            sigma_cutoff=lbte.sigma_cutoff_width,
            filename=filename,
            log_level=log_level,
        )

        collision_matrix = lbte.collision_matrix
        gamma = lbte.gamma
        assert collision_matrix is not None
        assert gamma is not None

        if reader.try_full_matrix(sigma, collision_matrix, gamma, i_sigma):
            read_from = "full_matrix"
        else:
            temperatures = reader.allocate_with_fallback(sigma, lbte.grid_points)
            if temperatures is False:
                return False

            for i, gp in enumerate(lbte.grid_points):
                if not reader.collect_with_band_fallback(
                    sigma,
                    collision_matrix[i_sigma],
                    gamma[i_sigma],
                    temperatures,
                    i,
                    gp,
                    lbte.bz_grid.bzg2grg,
                    is_reducible_collision_matrix,
                ):
                    return False
            read_from = "grid_points"

    return read_from
