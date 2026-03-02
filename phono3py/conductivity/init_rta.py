"""Init lattice thermal conductivity classes with RTA."""

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
import pathlib
from collections.abc import Sequence
from typing import Literal, TypeAlias, TypedDict, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.conductivity.rta_output import ConductivityRTAWriter, show_rta_progress
from phono3py.conductivity.type_dispatch import get_rta_conductivity_class
from phono3py.conductivity.utils import build_options, write_pp_interaction
from phono3py.file_IO import read_gamma_from_hdf5
from phono3py.phonon3.interaction import Interaction, all_bands_exist


class _OptionalGammaFlags(TypedDict):
    has_gamma_N_U: bool
    has_ave_pp: bool


class _GammaFileData(TypedDict, total=False):
    gamma: NDArray[np.double]
    gamma_isotope: NDArray[np.double]
    gamma_N: NDArray[np.double]
    gamma_U: NDArray[np.double]
    ave_pp: NDArray[np.double] | float


class _GammaReadContext(TypedDict):
    mesh: NDArray[np.int64]
    sigma_cutoff: float | None
    filename: str | None
    grid_points: Sequence[int] | NDArray[np.int64]
    num_band: int
    gamma: NDArray[np.double]
    gamma_iso: NDArray[np.double]
    gamma_N: NDArray[np.double]
    gamma_U: NDArray[np.double]
    ave_pp: NDArray[np.double]
    optional_flags: _OptionalGammaFlags
    verbose: bool


class _RTAInitOptions(TypedDict):
    grid_points: Sequence[int] | NDArray[np.int64] | None
    temperatures: Sequence[float] | NDArray[np.double] | None
    sigmas: Sequence[float | None] | None
    sigma_cutoff: float | None
    is_isotope: bool
    mass_variances: Sequence[float] | NDArray[np.double] | None
    boundary_mfp: float | None
    use_ave_pp: bool
    is_kappa_star: bool
    gv_delta_q: float | None
    is_full_pp: bool
    read_pp: bool
    store_pp: bool
    pp_filename: str | None
    is_N_U: bool
    is_gamma_detail: bool
    log_level: int


class _RTARunOptions(TypedDict):
    write_pp: bool
    write_gamma: bool
    write_gamma_detail: bool
    compression: Literal["gzip", "lzf"] | int | None
    output_filename: str | None
    log_level: int


class _RTAFinalizeOptions(TypedDict):
    grid_points: Sequence[int] | NDArray[np.int64] | None
    conductivity_type: Literal["wigner", "kubo"] | None
    write_kappa: bool
    compression: Literal["gzip", "lzf"] | int | None
    output_filename: str | None
    log_level: int


class _RTAInputReadOptions(TypedDict):
    read_gamma: bool
    read_elph: int | None
    input_filename: str | None


_GammaReadArrays: TypeAlias = tuple[
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
]

_GammaReadResult: TypeAlias = tuple[_GammaFileData | None, str]

_GammaReadParams: TypeAlias = tuple[
    NDArray[np.int64],
    float | None,
    str | None,
    bool,
]

_GammaDataTargets: TypeAlias = tuple[
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    _OptionalGammaFlags,
]


def _allocate_gamma_read_arrays(
    sigmas: Sequence[float | None],
    temperatures: Sequence[float] | NDArray[np.double],
    grid_points: Sequence[int] | NDArray[np.int64],
    num_band: int,
) -> _GammaReadArrays:
    """Allocate arrays used while collecting gamma data from files."""
    gamma = np.zeros(
        (len(sigmas), len(temperatures), len(grid_points), num_band), dtype="double"
    )
    gamma_N = np.zeros_like(gamma)
    gamma_U = np.zeros_like(gamma)
    gamma_iso = np.zeros((len(sigmas), len(grid_points), num_band), dtype="double")
    ave_pp = np.zeros((len(grid_points), num_band), dtype="double")
    return gamma, gamma_N, gamma_U, gamma_iso, ave_pp


def _build_gamma_read_context(
    *,
    mesh: NDArray[np.int64],
    sigma_cutoff: float | None,
    filename: str | None,
    grid_points: Sequence[int] | NDArray[np.int64],
    num_band: int,
    gamma: NDArray[np.double],
    gamma_iso: NDArray[np.double],
    gamma_N: NDArray[np.double],
    gamma_U: NDArray[np.double],
    ave_pp: NDArray[np.double],
    optional_flags: _OptionalGammaFlags,
    verbose: bool,
) -> _GammaReadContext:
    """Build immutable-like context passed to gamma read helpers."""
    return build_options(
        _GammaReadContext,
        mesh=mesh,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        grid_points=grid_points,
        num_band=num_band,
        gamma=gamma,
        gamma_iso=gamma_iso,
        gamma_N=gamma_N,
        gamma_U=gamma_U,
        ave_pp=ave_pp,
        optional_flags=optional_flags,
        verbose=verbose,
    )


def _apply_loaded_gamma_results(
    br: ConductivityRTABase,
    *,
    gamma: NDArray[np.double],
    gamma_N: NDArray[np.double],
    gamma_U: NDArray[np.double],
    ave_pp: NDArray[np.double],
    optional_flags: _OptionalGammaFlags,
) -> None:
    """Apply loaded gamma arrays and optional data to conductivity object."""
    br.gamma = gamma
    if optional_flags["has_ave_pp"]:
        br.set_averaged_pp_interaction(ave_pp)
    if optional_flags["has_gamma_N_U"]:
        br.set_gamma_N_U(gamma_N, gamma_U)


def _build_rta_init_options(
    *,
    grid_points: Sequence[int] | NDArray[np.int64] | None,
    temperatures: Sequence[float] | NDArray[np.double] | None,
    sigmas: Sequence[float | None] | None,
    sigma_cutoff: float | None,
    is_isotope: bool,
    mass_variances: Sequence[float] | NDArray[np.double] | None,
    boundary_mfp: float | None,
    use_ave_pp: bool,
    is_kappa_star: bool,
    gv_delta_q: float | None,
    is_full_pp: bool,
    read_pp: bool,
    write_pp: bool,
    input_filename: str | None,
    is_N_U: bool,
    write_gamma_detail: bool,
    log_level: int,
) -> _RTAInitOptions:
    return build_options(
        _RTAInitOptions,
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        store_pp=write_pp,
        pp_filename=input_filename,
        is_N_U=is_N_U,
        is_gamma_detail=write_gamma_detail,
        log_level=log_level,
    )


def _build_rta_run_options(
    *,
    write_pp: bool,
    write_gamma: bool,
    write_gamma_detail: bool,
    compression: Literal["gzip", "lzf"] | int | None,
    output_filename: str | None,
    log_level: int,
) -> _RTARunOptions:
    return build_options(
        _RTARunOptions,
        write_pp=write_pp,
        write_gamma=write_gamma,
        write_gamma_detail=write_gamma_detail,
        compression=compression,
        output_filename=output_filename,
        log_level=log_level,
    )


def _build_rta_finalize_options(
    *,
    grid_points: Sequence[int] | NDArray[np.int64] | None,
    conductivity_type: Literal["wigner", "kubo"] | None,
    write_kappa: bool,
    compression: Literal["gzip", "lzf"] | int | None,
    output_filename: str | None,
    log_level: int,
) -> _RTAFinalizeOptions:
    return build_options(
        _RTAFinalizeOptions,
        grid_points=grid_points,
        conductivity_type=conductivity_type,
        write_kappa=write_kappa,
        compression=compression,
        output_filename=output_filename,
        log_level=log_level,
    )


def _build_rta_input_read_options(
    *,
    read_gamma: bool,
    read_elph: int | None,
    input_filename: str | None,
) -> _RTAInputReadOptions:
    return build_options(
        _RTAInputReadOptions,
        read_gamma=read_gamma,
        read_elph=read_elph,
        input_filename=input_filename,
    )


def _apply_rta_input_reads(
    br: ConductivityRTABase,
    *,
    read_gamma: bool,
    read_elph: int | None,
    input_filename: str | None,
    verbose: bool = False,
) -> None:
    if read_gamma:
        _set_gamma_from_file(br, filename=input_filename, verbose=verbose)

    if read_elph is not None:
        _set_gamma_elph_from_file(br, read_elph, verbose=verbose)


def get_thermal_conductivity_RTA(
    interaction: Interaction,
    temperatures: Sequence[float] | NDArray[np.double] | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    mass_variances: Sequence[float] | NDArray[np.double] | None = None,
    grid_points: Sequence[int] | NDArray[np.int64] | None = None,
    is_isotope: bool = False,
    boundary_mfp: float | None = None,  # in micrometer
    use_ave_pp: bool = False,
    is_kappa_star: bool = True,
    gv_delta_q: float | None = None,
    is_full_pp: bool = False,
    is_N_U: bool = False,
    conductivity_type: Literal["wigner", "kubo"] | None = None,
    write_gamma: bool = False,
    read_gamma: bool = False,
    write_kappa: bool = False,
    write_pp: bool = False,
    read_pp: bool = False,
    read_elph: int | None = None,
    write_gamma_detail: bool = False,
    compression: Literal["gzip", "lzf"] | int | None = "gzip",
    input_filename: str | None = None,
    output_filename: str | None = None,
    log_level: int = 0,
) -> ConductivityRTABase:
    """Run RTA thermal conductivity calculation."""
    _temperatures = _normalize_rta_temperatures(temperatures)

    conductivity_RTA_class = get_rta_conductivity_class(conductivity_type)

    if log_level:
        print(
            "-------------------- Lattice thermal conductivity (RTA) "
            "--------------------"
        )

    init_options = _build_rta_init_options(
        grid_points=grid_points,
        temperatures=_temperatures,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        write_pp=write_pp,
        input_filename=input_filename,
        is_N_U=is_N_U,
        write_gamma_detail=write_gamma_detail,
        log_level=log_level,
    )

    br = conductivity_RTA_class(
        interaction,
        **init_options,
    )

    input_read_options = _build_rta_input_read_options(
        read_gamma=read_gamma,
        read_elph=read_elph,
        input_filename=input_filename,
    )
    _apply_rta_input_reads(br, **input_read_options, verbose=log_level > 0)

    run_options = _build_rta_run_options(
        write_pp=write_pp,
        write_gamma=write_gamma,
        write_gamma_detail=write_gamma_detail,
        compression=compression,
        output_filename=output_filename,
        log_level=log_level,
    )
    _run_rta_grid_point_outputs(br, interaction, **run_options)

    finalize_options = _build_rta_finalize_options(
        grid_points=grid_points,
        conductivity_type=conductivity_type,
        write_kappa=write_kappa,
        compression=compression,
        output_filename=output_filename,
        log_level=log_level,
    )
    _finalize_rta_kappa(br, interaction, **finalize_options)

    return br


def _normalize_rta_temperatures(
    temperatures: Sequence[float] | NDArray[np.double] | None,
) -> Sequence[float] | NDArray[np.double]:
    if temperatures is None:
        return np.arange(0, 1001, 10, dtype="double")
    return temperatures


def _set_gamma_elph_from_file(
    br: ConductivityRTABase, read_elph: int, verbose: bool = False
) -> None:
    if br.temperatures is None:
        raise RuntimeError(
            "br.temperatures must be set to read gamma of el-ph interaction."
        )

    mesh_str = "".join(map(str, br.mesh_numbers))
    filename = pathlib.Path(f"gamma_elph-m{mesh_str}.hdf5")
    if not filename.is_file():
        raise RuntimeError(f'"{filename}" not found for gammas of el-ph interaction.')

    _log_if_verbose(verbose, f'Read gamma of el-ph interaction from "{filename}".')
    with h5py.File(filename, "r") as f:
        gamma_key = f"gamma_{read_elph}"
        if gamma_key not in f:
            raise RuntimeError(f"{gamma_key} not found in phono3py_elph.hdf5.")

        # Check consistency between br.temperatures and file temperatures
        if f"temperature_{read_elph}" in f:
            file_temperatures = f[f"temperature_{read_elph}"][:]  # type: ignore
            if not np.allclose(br.temperatures, file_temperatures):
                raise RuntimeError(
                    f"Temperature mismatch: ph-ph {br.temperatures} "
                    f"el-ph {file_temperatures}."
                )
        else:
            raise RuntimeError('"temperature" dataset not found in gamma el-ph file.')

        br.gamma_elph = f[gamma_key][:]  # type: ignore


def _run_rta_grid_point_outputs(
    br: ConductivityRTABase,
    interaction: Interaction,
    *,
    write_pp: bool,
    write_gamma: bool,
    write_gamma_detail: bool,
    compression: Literal["gzip", "lzf"] | int | None,
    output_filename: str | os.PathLike | None,
    log_level: int,
) -> None:
    for i in br:
        if write_pp:
            write_pp_interaction(
                br, interaction, i, compression=compression, filename=output_filename
            )
        if write_gamma:
            ConductivityRTAWriter.write_gamma(
                br,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
                verbose=log_level > 0,
            )
        if write_gamma_detail:
            ConductivityRTAWriter.write_gamma_detail(
                br,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
                verbose=log_level > 0,
            )


def _finalize_rta_kappa(
    br: ConductivityRTABase,
    interaction: Interaction,
    *,
    grid_points: Sequence[int] | NDArray[np.int64] | None,
    conductivity_type: Literal["wigner", "kubo"] | None,
    write_kappa: bool,
    compression: Literal["gzip", "lzf"] | int | None,
    output_filename: str | None,
    log_level: int,
) -> None:
    if grid_points is not None or not all_bands_exist(interaction):
        return

    br.set_kappa_at_sigmas()
    if log_level:
        show_rta_progress(br, conductivity_type, log_level)
    if write_kappa:
        ConductivityRTAWriter.write_kappa(
            br,
            interaction.primitive.volume,
            compression=compression,
            filename=output_filename,
            log_level=log_level,
        )


def _log_if_verbose(verbose: bool, text: str) -> None:
    """Print text only when verbose mode is enabled."""
    if verbose:
        print(text)


def _get_gamma_read_params(context: _GammaReadContext) -> _GammaReadParams:
    """Extract common file-read settings from gamma read context."""
    return (
        context["mesh"],
        context["sigma_cutoff"],
        context["filename"],
        context["verbose"],
    )


def _get_gamma_data_targets(context: _GammaReadContext) -> _GammaDataTargets:
    """Extract mutable data arrays and flags from gamma read context."""
    return (
        context["gamma"],
        context["gamma_iso"],
        context["gamma_N"],
        context["gamma_U"],
        context["ave_pp"],
        context["optional_flags"],
    )


def _read_gamma_data(
    context: _GammaReadContext,
    *,
    sigma: float | None,
    grid_point: int | None = None,
    band_index: int | None = None,
) -> _GammaReadResult:
    """Read gamma data from HDF5 using shared context parameters."""
    mesh, sigma_cutoff, filename, _ = _get_gamma_read_params(context)
    return cast(
        _GammaReadResult,
        read_gamma_from_hdf5(
            mesh,
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
            grid_point=grid_point,
            band_index=band_index,
        ),
    )


def _store_gamma_data(
    gamma_data: _GammaFileData,
    *,
    gamma_target: NDArray[np.double],
    gamma_iso_target: NDArray[np.double],
    gamma_N_target: NDArray[np.double],
    gamma_U_target: NDArray[np.double],
    ave_pp_target: NDArray[np.double],
    optional_flags: _OptionalGammaFlags,
) -> None:
    """Store one gamma data block into target arrays and optional fields."""
    if "gamma" not in gamma_data:
        raise KeyError("'gamma' is missing in gamma data.")

    if gamma_target.shape != gamma_data["gamma"].shape:
        raise ValueError(
            f"Shape mismatch for 'gamma': target {gamma_target.shape} - "
            f"from file {gamma_data['gamma'].shape}."
        )

    gamma_target[...] = gamma_data["gamma"]
    _update_optional_gamma_data(
        gamma_data=gamma_data,
        gamma_iso_target=gamma_iso_target,
        gamma_N_target=gamma_N_target,
        gamma_U_target=gamma_U_target,
        ave_pp_target=ave_pp_target,
        optional_flags=optional_flags,
    )


def _load_gamma_for_band(
    i_sigma: int,
    sigma: float | None,
    context: _GammaReadContext,
    i_gp: int,
    gp: int,
    bi: int,
) -> bool:
    """Load per-band gamma data for one grid point and sigma."""
    _, _, _, verbose = _get_gamma_read_params(context)
    gamma, gamma_iso, gamma_N, gamma_U, ave_pp, optional_flags = (
        _get_gamma_data_targets(context)
    )

    gamma_data_band, full_filename_band = _read_gamma_data(
        context,
        sigma=sigma,
        grid_point=gp,
        band_index=bi,
    )
    if gamma_data_band:
        _log_if_verbose(verbose, "Read data from %s." % full_filename_band)
        _store_gamma_data(
            gamma_data_band,
            gamma_target=gamma[i_sigma, :, i_gp, bi],
            gamma_iso_target=gamma_iso[i_sigma, i_gp, bi],
            gamma_N_target=gamma_N[i_sigma, :, i_gp, bi],
            gamma_U_target=gamma_U[i_sigma, :, i_gp, bi],
            ave_pp_target=ave_pp[i_gp, bi],
            optional_flags=optional_flags,
        )
        return True

    _log_if_verbose(verbose, "%s not found." % full_filename_band)
    return False


def _load_gamma_for_grid_point(
    i_sigma: int,
    sigma: float | None,
    context: _GammaReadContext,
    i_gp: int,
    gp: int,
) -> bool:
    """Load per-grid-point gamma data, with band-level fallback when missing."""
    _, _, _, verbose = _get_gamma_read_params(context)
    num_band = context["num_band"]
    gamma, gamma_iso, gamma_N, gamma_U, ave_pp, optional_flags = (
        _get_gamma_data_targets(context)
    )

    gamma_data_gp, full_filename_gp = _read_gamma_data(
        context,
        sigma=sigma,
        grid_point=gp,
    )
    if gamma_data_gp:
        _log_if_verbose(verbose, "Read data from %s." % full_filename_gp)
        _store_gamma_data(
            gamma_data_gp,
            gamma_target=gamma[i_sigma, :, i_gp],
            gamma_iso_target=gamma_iso[i_sigma, i_gp],
            gamma_N_target=gamma_N[i_sigma, :, i_gp],
            gamma_U_target=gamma_U[i_sigma, :, i_gp],
            ave_pp_target=ave_pp[i_gp],
            optional_flags=optional_flags,
        )
        return True

    _log_if_verbose(
        verbose,
        "%s not found. Look for hdf5 files at bands." % full_filename_gp,
    )
    return all(
        _load_gamma_for_band(i_sigma, sigma, context, i_gp, gp, bi)
        for bi in range(num_band)
    )


def _load_gamma_for_sigma(
    i_sigma: int,
    sigma: float | None,
    context: _GammaReadContext,
) -> bool:
    """Load gamma data for one sigma, trying full then grid-point files."""
    _, _, _, verbose = _get_gamma_read_params(context)
    grid_points = context["grid_points"]
    gamma, gamma_iso, gamma_N, gamma_U, ave_pp, optional_flags = (
        _get_gamma_data_targets(context)
    )

    gamma_data, full_filename = _read_gamma_data(
        context,
        sigma=sigma,
    )
    if gamma_data:
        _log_if_verbose(
            verbose, f'Read gamma of ph-ph interaction from "{full_filename}".'
        )
        _store_gamma_data(
            gamma_data,
            gamma_target=gamma[i_sigma],
            gamma_iso_target=gamma_iso[i_sigma],
            gamma_N_target=gamma_N[i_sigma],
            gamma_U_target=gamma_U[i_sigma],
            ave_pp_target=ave_pp[:],
            optional_flags=optional_flags,
        )
        return True

    _log_if_verbose(
        verbose,
        "%s not found. Look for hdf5 files at grid points." % full_filename,
    )

    return all(
        _load_gamma_for_grid_point(i_sigma, sigma, context, i, gp)
        for i, gp in enumerate(grid_points)
    )


def _set_gamma_from_file(
    br: ConductivityRTABase, filename: str | None = None, verbose: bool = False
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
    assert temperatures is not None

    gamma, gamma_N, gamma_U, gamma_iso, ave_pp = _allocate_gamma_read_arrays(
        sigmas,
        temperatures,
        grid_points,
        num_band,
    )

    optional_flags: _OptionalGammaFlags = {
        "has_gamma_N_U": False,
        "has_ave_pp": False,
    }
    context = _build_gamma_read_context(
        mesh=mesh,
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        grid_points=grid_points,
        num_band=num_band,
        gamma=gamma,
        gamma_iso=gamma_iso,
        gamma_N=gamma_N,
        gamma_U=gamma_U,
        ave_pp=ave_pp,
        optional_flags=optional_flags,
        verbose=verbose,
    )

    read_succeeded = all(
        _load_gamma_for_sigma(j, sigma, context) for j, sigma in enumerate(sigmas)
    )

    if not read_succeeded:
        raise RuntimeError("Reading collisions failed.")

    _apply_loaded_gamma_results(
        br,
        gamma=gamma,
        gamma_N=gamma_N,
        gamma_U=gamma_U,
        ave_pp=ave_pp,
        optional_flags=optional_flags,
    )


def _update_optional_gamma_data(
    gamma_data: _GammaFileData,
    *,
    gamma_iso_target: NDArray[np.double],
    gamma_N_target: NDArray[np.double],
    gamma_U_target: NDArray[np.double],
    ave_pp_target: NDArray[np.double],
    optional_flags: _OptionalGammaFlags,
) -> None:
    """Update optional gamma-derived arrays if those keys exist."""
    gamma_isotope = gamma_data.get("gamma_isotope")
    if gamma_isotope is not None:
        gamma_iso_target[...] = gamma_isotope

    gamma_N = gamma_data.get("gamma_N")
    if gamma_N is not None:
        gamma_U = gamma_data.get("gamma_U")
        if gamma_U is None:
            raise KeyError("'gamma_U' is missing in gamma data.")
        optional_flags["has_gamma_N_U"] = True
        gamma_N_target[...] = gamma_N
        gamma_U_target[...] = gamma_U

    ave_pp = gamma_data.get("ave_pp")
    if ave_pp is not None:
        optional_flags["has_ave_pp"] = True
        ave_pp_target[...] = ave_pp
