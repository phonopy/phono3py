"""Phono3py command option and conf file parser."""

# Copyright (C) 2015 Atsushi Togo
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

import argparse
import os
from typing import Literal

import numpy as np
from phonopy.cui.settings import ConfParser, Settings, fracval


class Phono3pySettings(Settings):
    """Setting parameter container."""

    def __init__(self, load_phono3py_yaml: bool = False):
        """Init method."""
        super().__init__(load_phonopy_yaml=load_phono3py_yaml)
        self.boundary_mfp = 1.0e6  # In micrometer. Value is for avoiding divergence.
        self.conductivity_type = None
        self.constant_averaged_pp_interaction = None
        self.create_forces_fc2: list[str] | None = None
        self.create_forces_fc3: list[str] | None = None
        self.create_forces_fc3_file = None
        self.cutoff_fc3_distance = None
        self.cutoff_pair_distance = None
        self.grid_addresses = None
        self.grid_points = None
        self.grid_matrix = None
        self.ion_clamped = False
        self.is_bterta = False
        self.is_compact_fc = False
        self.is_fc3_r0_average = True
        self.is_full_pp = False
        self.is_gruneisen = False
        self.is_imag_self_energy = False
        self.is_isotope = False
        self.is_joint_dos = False
        self.is_kappa_star = True
        self.is_lbte = False
        self.is_N_U = False
        self.is_plusminus_displacement_fc2 = "auto"
        self.is_real_self_energy = False
        self.is_reducible_collision_matrix = False
        self.is_spectral_function = False
        self.is_symmetrize_fc2 = False
        self.is_symmetrize_fc3_q = False
        self.is_symmetrize_fc3_r = False
        self.is_tetrahedron_method = False
        self.lapack_zheev_uplo: Literal["L", "U"] = "L"
        self.mass_variances = None
        self.max_freepath = None
        self.num_points_in_batch = None
        self.read_collision = None
        if load_phono3py_yaml:
            self.read_fc2 = True
            self.read_fc3 = True
        else:
            self.read_fc2 = False
            self.read_fc3 = False
        self.read_gamma = False
        self.read_phonon = False
        self.read_pp = False
        self.output_yaml_filename = None
        self.phonon_supercell_matrix = None
        self.pinv_cutoff = 1.0e-8
        self.pinv_solver = 0
        self.pinv_method = 0
        self.pp_conversion_factor = None
        self.random_displacements_fc2 = None
        self.scattering_event_class = None  # scattering event class 1 or 2
        self.sigma_cutoff_width = None
        self.solve_collective_phonon = False
        self.show_symfc_memory_usage = False
        self.symfc_memory_size = None
        self.subtract_forces = None
        self.subtract_forces_fc2 = None
        self.temperatures = None
        self.use_ave_pp = False
        self.use_grg = False
        self.write_collision = False
        self.write_gamma_detail = False
        self.write_gamma = False
        self.write_phonon = False
        self.write_pp = False
        self.write_LBTE_solution = False


class Phono3pyConfParser(ConfParser):
    """Phonopy conf parser.

    Attributes
    ----------
    settings : Phono3pySettings
        Phono3py settings container.
    confs : dict
        Dictionary of settings read from conf file or command options.

    """

    def __init__(
        self,
        filename: str | os.PathLike | None = None,
        args: argparse.Namespace | None = None,
        load_phono3py_yaml: bool = False,
    ):
        """Init method."""
        super().__init__()
        if filename is not None:
            self._read_file(filename)
        if args is not None:
            self._read_options(args)
        self._parse_conf()
        self.settings = Phono3pySettings(load_phono3py_yaml=load_phono3py_yaml)
        self._set_settings(self.settings)

    def _read_options(self, args: argparse.Namespace):
        super()._read_options(args)  # store data in self._confs
        if "phonon_supercell_dimension" in args:
            dim_fc2 = args.phonon_supercell_dimension
            if dim_fc2 is not None:
                self._confs["dim_fc2"] = " ".join(dim_fc2)

        if "boundary_mfp" in args:
            if args.boundary_mfp is not None:
                self._confs["boundary_mfp"] = args.boundary_mfp

        if "const_ave_pp" in args:
            const_ave_pp = args.const_ave_pp
            if const_ave_pp is not None:
                self._confs["const_ave_pp"] = const_ave_pp

        if "create_forces_fc2" in args:
            if args.create_forces_fc2:
                self._confs["create_forces_fc2"] = args.create_forces_fc2

        if "create_forces_fc3" in args:
            if args.create_forces_fc3:
                self._confs["create_forces_fc3"] = args.create_forces_fc3

        if "create_forces_fc3_file" in args:
            if args.create_forces_fc3_file:
                cfc3_file = args.create_forces_fc3_file
                self._confs["create_forces_fc3_file"] = cfc3_file

        if "cutoff_fc3_distance" in args:
            cutoff_fc3 = args.cutoff_fc3_distance
            if cutoff_fc3 is not None:
                self._confs["cutoff_fc3_distance"] = cutoff_fc3

        if "cutoff_pair_distance" in args:
            cutoff_pair = args.cutoff_pair_distance
            if cutoff_pair is not None:
                self._confs["cutoff_pair_distance"] = cutoff_pair

        if "grid_addresses" in args:
            grid_adrs = args.grid_addresses
            if grid_adrs is not None:
                self._confs["grid_addresses"] = " ".join(grid_adrs)

        if "grid_points" in args:
            if args.grid_points is not None:
                self._confs["grid_points"] = " ".join(args.grid_points)

        if "grid_matrix" in args:
            if args.grid_matrix is not None:
                self._confs["grid_matrix"] = " ".join(args.grid_matrix)

        if "ion_clamped" in args:
            if args.ion_clamped:
                self._confs["ion_clamped"] = ".true."
            elif args.ion_clamped is False:
                self._confs["ion_clamped"] = ".false."

        if "is_bterta" in args:
            if args.is_bterta:
                self._confs["bterta"] = ".true."
            elif args.is_bterta is False:
                self._confs["bterta"] = ".false."

        if "is_compact_fc" in args:
            if args.is_compact_fc:
                self._confs["compact_fc"] = ".true."
            elif args.is_compact_fc is False:
                self._confs["compact_fc"] = ".false."

        if "is_gruneisen" in args:
            if args.is_gruneisen:
                self._confs["gruneisen"] = ".true."
            elif args.is_gruneisen is False:
                self._confs["gruneisen"] = ".false."

        if "is_fc3_r0_average" in args:
            if args.is_fc3_r0_average:
                self._confs["fc3_r0_average"] = ".true."
            elif args.is_fc3_r0_average is False:
                self._confs["fc3_r0_average"] = ".false."

        if "is_full_pp" in args:
            if args.is_full_pp:
                self._confs["full_pp"] = ".true."
            elif args.is_full_pp is False:
                self._confs["full_pp"] = ".false."

        if "is_imag_self_energy" in args:
            if args.is_imag_self_energy:
                self._confs["imag_self_energy"] = ".true."
            elif args.is_imag_self_energy is False:
                self._confs["imag_self_energy"] = ".false."

        if "is_isotope" in args:
            if args.is_isotope:
                self._confs["isotope"] = ".true."
            elif args.is_isotope is False:
                self._confs["isotope"] = ".false."

        if "is_joint_dos" in args:
            if args.is_joint_dos:
                self._confs["joint_dos"] = ".true."
            elif args.is_joint_dos is False:
                self._confs["joint_dos"] = ".false."

        if "no_kappa_star" in args:
            if args.kappa_star:
                self._confs["kappa_star"] = ".true."
            elif args.kappa_star is False:
                self._confs["kappa_star"] = ".false."

        if "is_lbte" in args:
            if args.is_lbte:
                self._confs["lbte"] = ".true."
            elif args.is_lbte is False:
                self._confs["lbte"] = ".false."

        if "is_N_U" in args:
            if args.is_N_U:
                self._confs["N_U"] = ".true."
            elif args.is_N_U is False:
                self._confs["N_U"] = ".false."

        if "is_plusminus_displacements_fc2" in args:
            if args.is_plusminus_displacements_fc2:
                self._confs["pm_fc2"] = ".true."
            elif args.is_plusminus_displacements_fc2 is False:
                self._confs["pm_fc2"] = ".false."

        if "is_real_self_energy" in args:
            if args.is_real_self_energy:
                self._confs["real_self_energy"] = ".true."
            elif args.is_real_self_energy is False:
                self._confs["real_self_energy"] = ".false."

        if "is_reducible_collision_matrix" in args:
            if args.is_reducible_collision_matrix:
                self._confs["reducible_collision_matrix"] = ".true."
            elif args.is_reducible_collision_matrix is False:
                self._confs["reducible_collision_matrix"] = ".false."

        if "is_spectral_function" in args:
            if args.is_spectral_function:
                self._confs["spectral_function"] = ".true."
            elif args.is_spectral_function is False:
                self._confs["spectral_function"] = ".false."

        if "is_symmetrize_fc2" in args:
            if args.is_symmetrize_fc2:
                self._confs["symmetrize_fc2"] = ".true."
            elif args.is_symmetrize_fc2 is False:
                self._confs["symmetrize_fc2"] = ".false."

        if "is_symmetrize_fc3_q" in args:
            if args.is_symmetrize_fc3_q:
                self._confs["symmetrize_fc3_q"] = ".true."
            elif args.is_symmetrize_fc3_q is False:
                self._confs["symmetrize_fc3_q"] = ".false."

        if "is_symmetrize_fc3_r" in args:
            if args.is_symmetrize_fc3_r:
                self._confs["symmetrize_fc3_r"] = ".true."
            elif args.is_symmetrize_fc3_r is False:
                self._confs["symmetrize_fc3_r"] = ".false."

        if "is_tetrahedron_method" in args:
            if args.is_tetrahedron_method:
                self._confs["tetrahedron"] = ".true."
            elif args.is_tetrahedron_method is False:
                self._confs["tetrahedron"] = ".false."

        if "is_wigner_kappa" in args:
            if args.is_wigner_kappa:
                self._confs["conductivity_type"] = "wigner"

        if "is_kubo_kappa" in args:
            if args.is_kubo_kappa:
                self._confs["conductivity_type"] = "kubo"

        if "lapack_zheev_uplo" in args:
            if args.lapack_zheev_uplo is not None:
                self._confs["lapack_zheev_uplo"] = args.lapack_zheev_uplo

        if "mass_variances" in args:
            mass_variances = args.mass_variances
            if mass_variances is not None:
                self._confs["mass_variances"] = " ".join(mass_variances)

        if "max_freepath" in args:
            if args.max_freepath is not None:
                self._confs["max_freepath"] = args.max_freepath

        if "num_points_in_batch" in args:
            num_points_in_batch = args.num_points_in_batch
            if num_points_in_batch is not None:
                self._confs["num_points_in_batch"] = num_points_in_batch

        if "output_yaml_filename" in args:
            if args.output_yaml_filename is not None:
                self._confs["output_yaml_filename"] = args.output_yaml_filename

        if "pinv_cutoff" in args:
            if args.pinv_cutoff is not None:
                self._confs["pinv_cutoff"] = args.pinv_cutoff

        if "pinv_method" in args:
            if args.pinv_method is not None:
                self._confs["pinv_method"] = args.pinv_method

        if "pinv_solver" in args:
            if args.pinv_solver is not None:
                self._confs["pinv_solver"] = args.pinv_solver

        if "pp_conversion_factor" in args:
            pp_conv_factor = args.pp_conversion_factor
            if pp_conv_factor is not None:
                self._confs["pp_conversion_factor"] = pp_conv_factor

        if "random_displacements_fc2" in args:
            rd_fc2 = args.random_displacements_fc2
            if rd_fc2 is not None:
                self._confs["random_displacements_fc2"] = rd_fc2

        if "read_fc2" in args:
            if args.read_fc2:
                self._confs["read_fc2"] = ".true."
            elif args.read_fc2 is False:
                self._confs["read_fc2"] = ".false."

        if "read_fc3" in args:
            if args.read_fc3:
                self._confs["read_fc3"] = ".true."
            elif args.read_fc3 is False:
                self._confs["read_fc3"] = ".false."

        if "read_gamma" in args:
            if args.read_gamma:
                self._confs["read_gamma"] = ".true."
            elif args.read_gamma is False:
                self._confs["read_gamma"] = ".false."

        if "read_phonon" in args:
            if args.read_phonon:
                self._confs["read_phonon"] = ".true."
            elif args.read_phonon is False:
                self._confs["read_phonon"] = ".false."

        if "read_pp" in args:
            if args.read_pp:
                self._confs["read_pp"] = ".true."
            elif args.read_pp is False:
                self._confs["read_pp"] = ".false."

        if "read_collision" in args:
            if args.read_collision is not None:
                self._confs["read_collision"] = args.read_collision

        if "scattering_event_class" in args:
            scatt_class = args.scattering_event_class
            if scatt_class is not None:
                self._confs["scattering_event_class"] = scatt_class

        if "sigma_cutoff_width" in args:
            if args.sigma_cutoff_width is not None:
                self._confs["sigma_cutoff_width"] = args.sigma_cutoff_width

        if "solve_collective_phonon" in args:
            if args.solve_collective_phonon:
                self._confs["collective_phonon"] = ".true."
            elif args.solve_collective_phonon is False:
                self._confs["collective_phonon"] = ".false."

        if "show_symfc_memory_usage" in args:
            if args.show_symfc_memory_usage:
                self._confs["show_symfc_memory_usage"] = ".true."
            elif args.show_symfc_memory_usage is False:
                self._confs["show_symfc_memory_usage"] = ".false."

        if "subtract_forces" in args:
            if args.subtract_forces:
                self._confs["subtract_forces"] = args.subtract_forces

        if "subtract_forces_fc2" in args:
            if args.subtract_forces_fc2:
                self._confs["subtract_forces_fc2"] = args.subtract_forces_fc2

        if "symfc_memory_size" in args:
            if args.symfc_memory_size is not None:
                self._confs["symfc_memory_size"] = args.symfc_memory_size

        if "temperatures" in args:
            if args.temperatures is not None:
                self._confs["temperatures"] = " ".join(args.temperatures)

        if "use_ave_pp" in args:
            if args.use_ave_pp:
                self._confs["use_ave_pp"] = ".true."
            elif args.use_ave_pp is False:
                self._confs["use_ave_pp"] = ".false."

        if "use_grg" in args:
            if args.use_grg:
                self._confs["use_grg"] = ".true."
            elif args.use_grg is False:
                self._confs["use_grg"] = ".false."

        if "write_gamma_detail" in args:
            if args.write_gamma_detail:
                self._confs["write_gamma_detail"] = ".true."
            elif args.write_gamma_detail is False:
                self._confs["write_gamma_detail"] = ".false."

        if "write_gamma" in args:
            if args.write_gamma:
                self._confs["write_gamma"] = ".true."
            elif args.write_gamma is False:
                self._confs["write_gamma"] = ".false."

        if "write_collision" in args:
            if args.write_collision:
                self._confs["write_collision"] = ".true."
            elif args.write_collision is False:
                self._confs["write_collision"] = ".false."

        if "write_phonon" in args:
            if args.write_phonon:
                self._confs["write_phonon"] = ".true."
            elif args.write_phonon is False:
                self._confs["write_phonon"] = ".false."

        if "write_pp" in args:
            if args.write_pp:
                self._confs["write_pp"] = ".true."
            elif args.write_pp is False:
                self._confs["write_pp"] = ".false."

        if "write_LBTE_solution" in args:
            if args.write_LBTE_solution:
                self._confs["write_LBTE_solution"] = ".true."
            elif args.write_LBTE_solution is False:
                self._confs["write_LBTE_solution"] = ".false."

    def _parse_conf(self):
        super()._parse_conf()
        confs = self._confs

        for conf_key in confs.keys():
            # Boolean
            if conf_key in (
                "read_fc2",
                "read_fc3",
                "read_gamma",
                "read_phonon",
                "read_pp",
                "use_ave_pp",
                "use_grg",
                "collective_phonon",
                "write_gamma_detail",
                "write_gamma",
                "write_collision",
                "write_phonon",
                "write_pp",
                "write_LBTE_solution",
                "full_pp",
                "ion_clamped",
                "bterta",
                "compact_fc",
                "fc3_r0_average",
                "real_self_energy",
                "gruneisen",
                "imag_self_energy",
                "isotope",
                "joint_dos",
                "lbte",
                "N_U",
                "spectral_function",
                "reducible_collision_matrix",
                "show_symfc_memory_usage",
                "symmetrize_fc2",
                "symmetrize_fc3_q",
                "symmetrize_fc3_r",
                "kappa_star",
            ):
                if confs[conf_key].lower() == ".true.":
                    self._set_parameter(conf_key, True)
                elif confs[conf_key].lower() == ".false.":
                    self._set_parameter(conf_key, False)

            # float
            if conf_key in (
                "boundary_mfp",
                "const_ave_pp",
                "cutoff_fc3_distance",
                "cutoff_pair_distance",
                "max_freepath",
                "pinv_cutoff",
                "pp_conversion_factor",
                "sigma_cutoff_width",
                "symfc_memory_size",
            ):
                self._set_parameter(conf_key, float(confs[conf_key]))

            # int
            if conf_key in (
                "pinv_method",
                "pinv_solver",
                "num_points_in_batch",
                "scattering_event_class",
            ):
                self._set_parameter(conf_key, int(confs[conf_key]))

            # string
            if conf_key in (
                "conductivity_type",
                "create_forces_fc3_file",
                "output_yaml_filename",
                "subtract_forces",
                "subtract_forces_fc2",
            ):
                self._set_parameter(conf_key, confs[conf_key])

            # specials
            if conf_key in ("create_forces_fc2", "create_forces_fc3"):
                if isinstance(confs[conf_key], str):
                    fnames = confs[conf_key].split()
                else:
                    fnames = confs[conf_key]
                self._set_parameter(conf_key, fnames)

            if conf_key == "dim_fc2":
                matrix = [int(x) for x in confs["dim_fc2"].split()]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of dim-fc2 has to be 3 or 9."
                    )

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            "Determinant of supercell matrix has " + "to be positive."
                        )
                    else:
                        self._set_parameter("dim_fc2", matrix)

            if conf_key == "grid_addresses":
                vals = [
                    int(x) for x in confs["grid_addresses"].replace(",", " ").split()
                ]
                if len(vals) % 3 == 0 and len(vals) > 0:
                    self._set_parameter("grid_addresses", np.reshape(vals, (-1, 3)))
                else:
                    self.setting_error("Grid addresses are incorrectly set.")

            if conf_key == "grid_points":
                vals = [int(x) for x in confs["grid_points"].replace(",", " ").split()]
                self._set_parameter("grid_points", vals)

            if conf_key == "grid_matrix":
                vals = [int(x) for x in confs["grid_matrix"].replace(",", " ").split()]
                if len(vals) == 9:
                    self._set_parameter("grid_matrix", np.reshape(vals, (3, 3)))
                else:
                    self.setting_error("Grid matrix are incorrectly set.")

            if conf_key == "lapack_zheev_uplo":
                self._set_parameter(
                    "lapack_zheev_uplo", confs["lapack_zheev_uplo"].upper()
                )

            if conf_key == "mass_variances":
                vals = [fracval(x) for x in confs["mass_variances"].split()]
                if len(vals) < 1:
                    self.setting_error("Mass variance parameters are incorrectly set.")
                else:
                    self._set_parameter("mass_variances", vals)

            if conf_key == "read_collision":
                if confs["read_collision"] == "all":
                    self._set_parameter("read_collision", "all")
                else:
                    vals = [int(x) for x in confs["read_collision"].split()]
                    self._set_parameter("read_collision", vals)

            # For multiple T values.
            if conf_key == "temperatures":
                vals = [fracval(x) for x in confs["temperatures"].split()]
                if len(vals) < 1:
                    self.setting_error("Temperatures are incorrectly set.")
                else:
                    self._set_parameter("temperatures", vals)

            if conf_key == "random_displacements_fc2":
                rd = confs["random_displacements_fc2"]
                if rd.lower() == "auto":
                    self._set_parameter("random_displacements_fc2", "auto")
                else:
                    try:
                        self._set_parameter("random_displacements_fc2", int(rd))
                    except ValueError:
                        self.setting_error(f"{conf_key.upper()} is incorrectly set.")

            if conf_key == "pm_fc2":
                if confs["pm_fc2"].lower() == ".false.":
                    self._set_parameter("pm_fc2", False)
                elif confs["pm_fc2"].lower() == ".true.":
                    self._set_parameter("pm_fc2", True)

    def _set_settings(self, settings: Phono3pySettings):
        super()._set_settings(settings)
        params = self._parameters

        # Supercell dimension for fc2
        if "dim_fc2" in params:
            settings.phonon_supercell_matrix = params["dim_fc2"]

        # Boundary mean free path for thermal conductivity calculation
        if "boundary_mfp" in params:
            settings.boundary_mfp = params["boundary_mfp"]

        # Calculate thermal conductivity in BTE-RTA
        if "bterta" in params:
            settings.is_bterta = params["bterta"]

        # Choice of thermal conductivity type
        if "conductivity_type" in params:
            settings.conductivity_type = params["conductivity_type"]

        # Solve collective phonons
        if "collective_phonon" in params:
            settings.solve_collective_phonon = params["collective_phonon"]

        # Compact force constants or full force constants
        if "compact_fc" in params:
            settings.is_compact_fc = params["compact_fc"]

        # Peierls type approximation for squared ph-ph interaction strength
        if "const_ave_pp" in params:
            settings.constant_averaged_pp_interaction = params["const_ave_pp"]

        # Trigger to create FORCES_FC2 and FORCES_FC3
        if "create_forces_fc2" in params:
            settings.create_forces_fc2 = params["create_forces_fc2"]

        if "create_forces_fc3" in params:
            settings.create_forces_fc3 = params["create_forces_fc3"]

        if "create_forces_fc3_file" in params:
            settings.create_forces_fc3_file = params["create_forces_fc3_file"]

        # Cutoff distance of third-order force constants. Elements where any
        # pair of atoms has larger distance than cut-off distance are set zero.
        if "cutoff_fc3_distance" in params:
            settings.cutoff_fc3_distance = params["cutoff_fc3_distance"]

        # Cutoff distance between pairs of displaced atoms used for supercell
        # creation with displacements and making third-order force constants
        if "cutoff_pair_distance" in params:
            settings.cutoff_pair_distance = params["cutoff_pair_distance"]

        # Grid addresses (sets of three integer values)
        if "grid_addresses" in params:
            settings.grid_addresses = params["grid_addresses"]

        # Grid points
        if "grid_points" in params:
            settings.grid_points = params["grid_points"]

        # Grid matrix
        if "grid_matrix" in params:
            settings.mesh_numbers = params["grid_matrix"]

        # Atoms are clamped under applied strain in Gruneisen parameter
        # calculation.
        if "ion_clamped" in params:
            settings.ion_clamped = params["ion_clamped"]

        # Take average in fc3-r2q transformation around three atoms
        if "fc3_r0_average" in params:
            settings.is_fc3_r0_average = params["fc3_r0_average"]

        # Calculate full ph-ph interaction strength for RTA conductivity
        if "full_pp" in params:
            settings.is_full_pp = params["full_pp"]

        # Calculate phonon-Gruneisen parameters
        if "gruneisen" in params:
            settings.is_gruneisen = params["gruneisen"]

        # Calculate imaginary part of self energy
        if "imag_self_energy" in params:
            settings.is_imag_self_energy = params["imag_self_energy"]

        # Calculate lifetime due to isotope scattering
        if "isotope" in params:
            settings.is_isotope = params["isotope"]

        # Calculate joint-DOS
        if "joint_dos" in params:
            settings.is_joint_dos = params["joint_dos"]

        # Sum partial kappa at q-stars
        if "kappa_star" in params:
            settings.is_kappa_star = params["kappa_star"]

        if "lapack_zheev_uplo" in params:
            settings.lapack_zheev_uplo = params["lapack_zheev_uplo"]

        # Calculate thermal conductivity in LBTE with Chaput's method
        if "lbte" in params:
            settings.is_lbte = params["lbte"]

        # Number of frequency points in a batch.
        if "num_points_in_batch" in params:
            settings.num_points_in_batch = params["num_points_in_batch"]

        # Calculate Normal and Umklapp processes
        if "N_U" in params:
            settings.is_N_U = params["N_U"]

        # Plus minus displacement for fc2
        if "pm_fc2" in params:
            settings.is_plusminus_displacement_fc2 = params["pm_fc2"]

        # Solve reducible collision matrix but not reduced matrix
        if "reducible_collision_matrix" in params:
            settings.is_reducible_collision_matrix = params[
                "reducible_collision_matrix"
            ]

        # Symmetrize fc2 by index exchange
        if "symmetrize_fc2" in params:
            settings.is_symmetrize_fc2 = params["symmetrize_fc2"]

        # Symmetrize phonon fc3 by index exchange
        if "symmetrize_fc3_q" in params:
            settings.is_symmetrize_fc3_q = params["symmetrize_fc3_q"]

        # Symmetrize fc3 by index exchange
        if "symmetrize_fc3_r" in params:
            settings.is_symmetrize_fc3_r = params["symmetrize_fc3_r"]

        # Mass variance parameters
        if "mass_variances" in params:
            settings.mass_variances = params["mass_variances"]

        # Maximum mean free path
        if "max_freepath" in params:
            settings.max_freepath = params["max_freepath"]

        # Output yaml filename instead of default filename of phono3py.yaml.
        if "output_yaml_filename" in params:
            settings.output_yaml_filename = params["output_yaml_filename"]

        # Cutoff frequency for pseudo inversion of collision matrix
        if "pinv_cutoff" in params:
            settings.pinv_cutoff = params["pinv_cutoff"]

        # Switch for pseudo-inverse method either taking abs or not.
        if "pinv_method" in params:
            settings.pinv_method = params["pinv_method"]

        # Switch for pseudo-inverse solver
        if "pinv_solver" in params:
            settings.pinv_solver = params["pinv_solver"]

        # Ph-ph interaction unit conversion factor
        if "pp_conversion_factor" in params:
            settings.pp_conversion_factor = params["pp_conversion_factor"]

        # Random displacements for fc2
        if "random_displacements_fc2" in params:
            settings.random_displacements_fc2 = params["random_displacements_fc2"]

        # Calculate real_self_energys
        if "real_self_energy" in params:
            settings.is_real_self_energy = params["real_self_energy"]

            # Read collision matrix and gammas from hdf5
        if "read_collision" in params:
            settings.read_collision = params["read_collision"]

        # Read fc2 from hdf5
        if "read_fc2" in params:
            settings.read_fc2 = params["read_fc2"]

        # Read fc3 from hdf5
        if "read_fc3" in params:
            settings.read_fc3 = params["read_fc3"]

        # Read gammas from hdf5
        if "read_gamma" in params:
            settings.read_gamma = params["read_gamma"]

        # Read phonons from hdf5
        if "read_phonon" in params:
            settings.read_phonon = params["read_phonon"]

        # Read ph-ph interaction strength from hdf5
        if "read_pp" in params:
            settings.read_pp = params["read_pp"]

        # Scattering event class 1 or 2
        if "scattering_event_class" in params:
            settings.scattering_event_class = params["scattering_event_class"]

        # Show symfc memory usage
        if "show_symfc_memory_usage" in params:
            settings.show_symfc_memory_usage = params["show_symfc_memory_usage"]

        # Cutoff width of smearing function (ratio to sigma value)
        if "sigma_cutoff_width" in params:
            settings.sigma_cutoff_width = params["sigma_cutoff_width"]

        # Calculate spectral_functions
        if "spectral_function" in params:
            settings.is_spectral_function = params["spectral_function"]

        # Subtract residual forces to create FORCES_FC2 and FORCES_FC3
        if "subtract_forces" in params:
            settings.subtract_forces = params["subtract_forces"]

        # Subtract residual forces to create FORCES_FC2
        if "subtract_forces_fc2" in params:
            settings.subtract_forces_fc2 = params["subtract_forces_fc2"]

        if "symfc_memory_size" in params:
            settings.symfc_memory_size = params["symfc_memory_size"]

        # Temperatures for scatterings
        if "temperatures" in params:
            settings.temperatures = params["temperatures"]

        # Use averaged ph-ph interaction
        if "use_ave_pp" in params:
            settings.use_ave_pp = params["use_ave_pp"]

        # Use generalized regular grid
        if "use_grg" in params:
            settings.use_grg = params["use_grg"]

        # Write detailed imag-part of self energy to hdf5
        if "write_gamma_detail" in params:
            settings.write_gamma_detail = params["write_gamma_detail"]

        # Write imag-part of self energy to hdf5
        if "write_gamma" in params:
            settings.write_gamma = params["write_gamma"]

        # Write collision matrix and gammas to hdf5
        if "write_collision" in params:
            settings.write_collision = params["write_collision"]

        # Write all phonons on grid points to hdf5
        if "write_phonon" in params:
            settings.write_phonon = params["write_phonon"]

        # Write phonon-phonon interaction amplitudes to hdf5
        if "write_pp" in params:
            settings.write_pp = params["write_pp"]

        # Write direct solution of LBTE to hdf5 files
        if "write_LBTE_solution" in params:
            settings.write_LBTE_solution = params["write_LBTE_solution"]
