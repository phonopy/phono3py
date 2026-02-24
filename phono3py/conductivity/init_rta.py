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

from collections.abc import Sequence
from typing import Literal, Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.kubo_rta import ConductivityKuboRTA
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.conductivity.rta_output import ConductivityRTAWriter, show_rta_progress
from phono3py.conductivity.type_dispatch import get_rta_conductivity_class
from phono3py.conductivity.utils import write_pp_interaction
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA
from phono3py.file_IO import read_gamma_from_hdf5
from phono3py.phonon3.interaction import Interaction, all_bands_exist


def get_thermal_conductivity_RTA(
    interaction: Interaction,
    temperatures: Sequence[float] | NDArray | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    mass_variances: Sequence[float] | NDArray | None = None,
    grid_points: Sequence[int] | NDArray | None = None,
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
) -> ConductivityRTA | ConductivityKuboRTA | ConductivityWignerRTA:
    """Run RTA thermal conductivity calculation."""
    if temperatures is None:
        _temperatures = np.arange(0, 1001, 10, dtype="double")
    else:
        _temperatures = temperatures

    conductivity_RTA_class = get_rta_conductivity_class(conductivity_type)

    if log_level:
        print(
            "-------------------- Lattice thermal conductivity (RTA) "
            "--------------------"
        )

    br = conductivity_RTA_class(
        interaction,
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
        store_pp=write_pp,
        pp_filename=input_filename,
        is_N_U=is_N_U,
        is_gamma_detail=write_gamma_detail,
        log_level=log_level,
    )

    if read_gamma:
        if not _set_gamma_from_file(br, filename=input_filename):
            raise RuntimeError("Reading collisions failed.")

    if read_elph is not None:
        with h5py.File("phono3py_elph.hdf5", "r") as f:
            gamma_key = f"gamma_elph_{read_elph}"
            if gamma_key not in f:
                raise RuntimeError(f"{gamma_key} not found in phono3py_elph.hdf5.")
            br.gamma_elph = f[gamma_key][:]  # type: ignore

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
                verbose=log_level,
            )
        if write_gamma_detail:
            ConductivityRTAWriter.write_gamma_detail(
                br,
                interaction,
                i,
                compression=compression,
                filename=output_filename,
                verbose=log_level,
            )

    if grid_points is None and all_bands_exist(interaction):
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

    return br


def _set_gamma_from_file(
    br: ConductivityRTABase, filename: Optional[str] = None, verbose: bool = True
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
                print(f"Read gamma from {full_filename}.")
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
