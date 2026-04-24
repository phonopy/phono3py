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

import pathlib
from collections.abc import Sequence
from typing import Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.calculators import RTACalculator
from phono3py.conductivity.factory import conductivity_calculator
from phono3py.conductivity.output import ConductivityRTAWriter
from phono3py.conductivity.utils import write_pp_interaction
from phono3py.file_IO import read_gamma_from_hdf5
from phono3py.phonon3.interaction import Interaction, all_bands_exist


class GammaFileReader:
    """Read gamma data from HDF5 with 3-level fallback (full -> GP -> band).

    Tries kappa-m*.hdf5 (full dataset) first.  If not found, falls back to
    kappa-m*-gp*.hdf5 (per grid point), then kappa-m*-gp*-b*.hdf5 (per band).
    """

    def __init__(
        self,
        sigmas: Sequence[float | None],
        temperatures: Sequence[float] | NDArray[np.double],
        grid_points: Sequence[int] | NDArray[np.int64],
        num_band: int,
        mesh: NDArray[np.int64],
        sigma_cutoff: float | None,
        filename: str | None,
        verbose: bool,
    ):
        self._mesh = mesh
        self._sigma_cutoff = sigma_cutoff
        self._filename = filename
        self._grid_points = grid_points
        self._num_band = num_band
        self._verbose = verbose

        self.gamma = np.zeros(
            (len(sigmas), len(temperatures), len(grid_points), num_band),
            dtype="double",
        )
        self.gamma_N = np.zeros_like(self.gamma)
        self.gamma_U = np.zeros_like(self.gamma)
        self.gamma_iso = np.zeros(
            (len(sigmas), len(grid_points), num_band), dtype="double"
        )
        self.ave_pp = np.zeros((len(grid_points), num_band), dtype="double")
        self.has_gamma_N_U: bool = False
        self.has_ave_pp: bool = False

    def read_all(self, sigmas: Sequence[float | None]) -> bool:
        """Read gamma for all sigmas.  Returns True on success."""
        return all(self._read_for_sigma(j, sigma) for j, sigma in enumerate(sigmas))

    def apply_to(self, br: RTACalculator) -> None:
        """Apply loaded results to an RTACalculator."""
        br.gamma = self.gamma
        if self.has_ave_pp:
            br.set_averaged_pp_interaction(self.ave_pp)
        if self.has_gamma_N_U:
            br.set_gamma_N_U(self.gamma_N, self.gamma_U)

    # ----- 3-level fallback chain -----

    def _read_for_sigma(self, i_sigma: int, sigma: float | None) -> bool:
        """Level 1: full dataset (kappa-m*.hdf5)."""
        data, fname = self._read_hdf5(sigma=sigma)
        if data:
            self._log(f'Read gamma of ph-ph interaction from "{fname}".')
            self._store(data, i_sigma)
            return True
        self._log(f"{fname} not found. Look for hdf5 files at grid points.")
        return all(
            self._read_for_grid_point(i_sigma, sigma, i, gp)
            for i, gp in enumerate(self._grid_points)
        )

    def _read_for_grid_point(
        self, i_sigma: int, sigma: float | None, i_gp: int, gp: int
    ) -> bool:
        """Level 2: per grid point (kappa-m*-gp*.hdf5)."""
        data, fname = self._read_hdf5(sigma=sigma, grid_point=gp)
        if data:
            self._log("Read data from %s." % fname)
            self._store(data, i_sigma, i_gp=i_gp)
            return True
        self._log("%s not found. Look for hdf5 files at bands." % fname)
        return all(
            self._read_for_band(i_sigma, sigma, i_gp, gp, bi)
            for bi in range(self._num_band)
        )

    def _read_for_band(
        self, i_sigma: int, sigma: float | None, i_gp: int, gp: int, bi: int
    ) -> bool:
        """Level 3: per band (kappa-m*-gp*-b*.hdf5)."""
        data, fname = self._read_hdf5(sigma=sigma, grid_point=gp, band_index=bi)
        if data:
            self._log("Read data from %s." % fname)
            self._store(data, i_sigma, i_gp=i_gp, bi=bi)
            return True
        self._log("%s not found." % fname)
        return False

    # ----- low-level helpers -----

    def _read_hdf5(
        self,
        *,
        sigma: float | None,
        grid_point: int | None = None,
        band_index: int | None = None,
    ) -> tuple[dict | None, str]:
        """Thin wrapper around read_gamma_from_hdf5."""
        return read_gamma_from_hdf5(
            self._mesh,
            sigma=sigma,
            sigma_cutoff=self._sigma_cutoff,
            filename=self._filename,
            grid_point=grid_point,
            band_index=band_index,
        )

    def _store(
        self,
        data: dict,
        i_sigma: int,
        i_gp: int | None = None,
        bi: int | None = None,
    ) -> None:
        """Store loaded data into the appropriate array slices."""
        if i_gp is None:
            g = self.gamma[i_sigma]
            gi = self.gamma_iso[i_sigma]
            gn = self.gamma_N[i_sigma]
            gu = self.gamma_U[i_sigma]
            ap = self.ave_pp[:]
        elif bi is None:
            g = self.gamma[i_sigma, :, i_gp]
            gi = self.gamma_iso[i_sigma, i_gp]
            gn = self.gamma_N[i_sigma, :, i_gp]
            gu = self.gamma_U[i_sigma, :, i_gp]
            ap = self.ave_pp[i_gp]
        else:
            g = self.gamma[i_sigma, :, i_gp, bi]
            gi = self.gamma_iso[i_sigma, i_gp, bi]
            gn = self.gamma_N[i_sigma, :, i_gp, bi]
            gu = self.gamma_U[i_sigma, :, i_gp, bi]
            ap = self.ave_pp[i_gp, bi]

        if "gamma" not in data:
            raise KeyError("'gamma' is missing in gamma data.")
        if g.shape != data["gamma"].shape:
            raise ValueError(
                f"Shape mismatch for 'gamma': target {g.shape} - "
                f"from file {data['gamma'].shape}."
            )
        g[...] = data["gamma"]

        gamma_isotope = data.get("gamma_isotope")
        if gamma_isotope is not None:
            gi[...] = gamma_isotope

        gamma_N = data.get("gamma_N")
        if gamma_N is not None:
            gamma_U = data.get("gamma_U")
            if gamma_U is None:
                raise KeyError("'gamma_U' is missing in gamma data.")
            self.has_gamma_N_U = True
            gn[...] = gamma_N
            gu[...] = gamma_U

        ave_pp = data.get("ave_pp")
        if ave_pp is not None:
            self.has_ave_pp = True
            ap[...] = ave_pp

    def _log(self, text: str) -> None:
        if self._verbose:
            print(text)


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
    transport_type: str | None = None,
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
    lang: Literal["C", "Python", "Rust"] = "C",
) -> RTACalculator:
    """Run RTA thermal conductivity calculation."""
    _sigmas = [None] if sigmas is None else list(sigmas)
    _temperatures = _normalize_rta_temperatures(temperatures)
    _mass_variances = (
        np.asarray(mass_variances, dtype="double")
        if mass_variances is not None
        else None
    )
    _grid_points = (
        np.asarray(grid_points, dtype="int64") if grid_points is not None else None
    )

    if log_level:
        print(
            "-------------------- Lattice thermal conductivity (RTA) "
            "--------------------"
        )

    method = f"{transport_type}-rta" if transport_type else "std-rta"
    return _run_standard_rta(
        interaction,
        method=method,
        temperatures=_temperatures,
        sigmas=_sigmas,
        sigma_cutoff=sigma_cutoff,
        mass_variances=_mass_variances,
        grid_points=_grid_points,
        is_isotope=is_isotope,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        is_N_U=is_N_U,
        write_gamma=write_gamma,
        read_gamma=read_gamma,
        write_kappa=write_kappa,
        write_pp=write_pp,
        read_pp=read_pp,
        read_elph=read_elph,
        write_gamma_detail=write_gamma_detail,
        compression=compression,
        input_filename=input_filename,
        output_filename=output_filename,
        log_level=log_level,
        lang=lang,
    )


def _run_standard_rta(
    interaction: Interaction,
    *,
    method: str = "std-rta",
    temperatures: NDArray[np.double],
    sigmas: Sequence[float | None],
    sigma_cutoff: float | None,
    mass_variances: NDArray[np.double] | None,
    grid_points: NDArray[np.int64] | None,
    is_isotope: bool,
    boundary_mfp: float | None,
    use_ave_pp: bool,
    is_kappa_star: bool,
    gv_delta_q: float | None,
    is_full_pp: bool,
    is_N_U: bool,
    write_gamma: bool,
    read_gamma: bool,
    write_kappa: bool,
    write_pp: bool,
    read_pp: bool,
    read_elph: int | None,
    write_gamma_detail: bool,
    compression: Literal["gzip", "lzf"] | int | None,
    input_filename: str | None,
    output_filename: str | None,
    log_level: int,
    lang: Literal["C", "Python", "Rust"],
) -> RTACalculator:
    """Run RTA (standard or its variants) using RTACalculator."""
    calc = conductivity_calculator(
        interaction,
        temperatures,
        sigmas,
        method=method,
        grid_points=grid_points,
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
        lang=lang,
    )

    if read_gamma:
        _set_gamma_from_file(
            calc,
            filename=input_filename,
            verbose=log_level > 0,
        )
    if read_elph is not None:
        _set_gamma_elph_from_file(
            calc,
            read_elph,
            verbose=log_level > 0,
        )

    def _on_grid_point(i: int) -> None:
        if write_pp:
            write_pp_interaction(
                calc,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
            )
        if write_gamma:
            ConductivityRTAWriter.write_gamma(
                calc,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
                verbose=log_level > 0,
            )
        if write_gamma_detail:
            ConductivityRTAWriter.write_gamma_detail(
                calc,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
                verbose=log_level > 0,
            )

    calc.run(on_grid_point=_on_grid_point)

    if grid_points is None and all_bands_exist(interaction):
        calc.log_kappa()
        if write_kappa:
            ConductivityRTAWriter.write_kappa(
                calc,
                interaction.primitive.volume,
                compression=compression,
                filename=output_filename,
                log_level=log_level,
            )

    return calc


def _normalize_rta_temperatures(
    temperatures: Sequence[float] | NDArray[np.double] | None,
) -> NDArray[np.double]:
    if temperatures is None:
        return np.arange(0, 1001, 10, dtype="double")
    return np.asarray(temperatures, dtype="double")


def _set_gamma_elph_from_file(
    br: RTACalculator, read_elph: int, verbose: bool = False
) -> None:
    if br.temperatures is None:
        raise RuntimeError(
            "br.temperatures must be set to read gamma of el-ph interaction."
        )

    mesh_str = "".join(map(str, br.mesh_numbers))
    filename = pathlib.Path(f"gamma_elph-m{mesh_str}.hdf5")
    if not filename.is_file():
        raise RuntimeError(f'"{filename}" not found for gammas of el-ph interaction.')

    if verbose:
        print(f'Read gamma of el-ph interaction from "{filename}".')
    with h5py.File(filename, "r") as f:
        gamma_key = f"gamma_{read_elph}"
        if gamma_key not in f:
            raise RuntimeError(f"{gamma_key} not found in phono3py_elph.hdf5.")

        # Check consistency between br.temperatures and file temperatures
        if f"temperature_{read_elph}" in f:
            file_temperatures: NDArray[np.double] = f[f"temperature_{read_elph}"][:]  # type: ignore
            if not np.allclose(br.temperatures, file_temperatures):
                raise RuntimeError(
                    f"Temperature mismatch: ph-ph {br.temperatures} "
                    f"el-ph {file_temperatures}."
                )
        else:
            raise RuntimeError('"temperature" dataset not found in gamma el-ph file.')

        br.gamma_elph = f[gamma_key][:]  # type: ignore


def _set_gamma_from_file(
    br: RTACalculator, filename: str | None = None, verbose: bool = False
) -> None:
    """Read kappa-*.hdf5 files for thermal conductivity calculation.

    Tries kappa-m*.hdf5 (full dataset) first, then kappa-m*-gp*.hdf5
    (per grid point), then kappa-m*-gp*-b*.hdf5 (per band).
    """
    assert br.temperatures is not None
    reader = GammaFileReader(
        sigmas=br.sigmas,
        temperatures=br.temperatures,
        grid_points=br.grid_points,
        num_band=br.frequencies.shape[1],
        mesh=br.mesh_numbers,
        sigma_cutoff=br.sigma_cutoff_width,
        filename=filename,
        verbose=verbose,
    )
    if not reader.read_all(br.sigmas):
        raise RuntimeError("Reading collisions failed.")
    reader.apply_to(br)
