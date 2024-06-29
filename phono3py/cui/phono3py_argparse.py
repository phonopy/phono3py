"""Phono3py command option argument parser."""

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

import sys

from phonopy.cui.phonopy_argparse import fix_deprecated_option_names


def get_parser(fc_symmetry=False, is_nac=False, load_phono3py_yaml=False):
    """Return ArgumentParser instance."""
    deprecated = fix_deprecated_option_names(sys.argv)
    import argparse

    from phonopy.interface.calculator import add_arguments_of_calculators

    from phono3py.interface.calculator import calculator_info

    parser = argparse.ArgumentParser(description="Phono3py command-line-tool")

    add_arguments_of_calculators(parser, calculator_info)

    parser.add_argument(
        "--alm",
        dest="use_alm",
        action="store_true",
        default=False,
        help=("Use ALM for generating 2nd and 3rd force constants " "in one fitting"),
    )
    parser.add_argument(
        "--amplitude",
        dest="displacement_distance",
        type=float,
        default=None,
        help="Distance of displacements",
    )
    parser.add_argument(
        "--ave-pp",
        dest="use_ave_pp",
        action="store_true",
        default=False,
        help="Use averaged ph-ph interaction",
    )
    parser.add_argument(
        "--band",
        nargs="+",
        dest="band_paths",
        default=None,
        help="Band structure paths calculated for Gruneisen parameter",
    )
    parser.add_argument(
        "--band-points",
        dest="band_points",
        type=int,
        default=None,
        help=(
            "Number of points calculated on a band segment in the band "
            "structure Gruneisen parameter calculation"
        ),
    )
    parser.add_argument(
        "--bi",
        "--band-indices",
        nargs="+",
        dest="band_indices",
        default=None,
        help="Band indices where life time is calculated",
    )
    parser.add_argument(
        "--boundary-mfp",
        "--bmfp",
        dest="boundary_mfp",
        type=float,
        default=None,
        help=(
            "Boundary mean free path in micrometre for thermal conductivity "
            "calculation"
        ),
    )
    parser.add_argument(
        "--br",
        "--bterta",
        dest="is_bterta",
        action="store_true",
        default=False,
        help="Calculate thermal conductivity in BTE-RTA",
    )
    if not load_phono3py_yaml:
        parser.add_argument(
            "-c",
            "--cell",
            dest="cell_filename",
            metavar="FILE",
            default=None,
            help="Read unit cell",
        )
    parser.add_argument(
        "--cf2",
        "--create-f2",
        dest="create_forces_fc2",
        nargs="+",
        default=None,
        help="Create FORCES_FC2",
    )
    parser.add_argument(
        "--cf3",
        "--create-f3",
        dest="create_forces_fc3",
        nargs="+",
        default=None,
        help="Create FORCES_FC3",
    )
    parser.add_argument(
        "--cf3-file",
        "--create-f3-from-file",
        metavar="FILE",
        dest="create_forces_fc3_file",
        default=None,
        help="Create FORCES_FC3 from file name list",
    )
    parser.add_argument(
        "--cfz",
        "--subtract-forces",
        metavar="FILE",
        dest="subtract_forces",
        default=None,
        help="Subtract recidual forces from supercell forces",
    )
    parser.add_argument(
        "--cfc",
        "--compact-fc",
        dest="is_compact_fc",
        action="store_true",
        default=False,
        help="Use compact force cosntants",
    )
    parser.add_argument(
        "--cfs",
        "--create-force-sets",
        dest="force_sets_mode",
        action="store_true",
        default=False,
        help="Create phonopy FORCE_SETS from FORCES_FC2",
    )
    parser.add_argument(
        "--cph",
        "--collective-phonon",
        dest="solve_collective_phonon",
        action="store_true",
        default=False,
        help="Solve collective phonons",
    )
    if load_phono3py_yaml:
        parser.add_argument(
            "--config",
            dest="conf_filename",
            metavar="FILE",
            default=None,
            help="Phono3py configuration file",
        )
    parser.add_argument(
        "--const-ave-pp",
        dest="const_ave_pp",
        type=float,
        default=None,
        help="Set constant averaged ph-ph interaction (Pqj)",
    )
    parser.add_argument(
        "--cutoff-fc3",
        "--cutoff-fc3-distance",
        dest="cutoff_fc3_distance",
        type=float,
        default=None,
        help=(
            "Cutoff distance of third-order force constants. Elements where "
            "any pair of atoms has larger distance than cut-off distance "
            "are set zero."
        ),
    )
    parser.add_argument(
        "--cutoff-freq",
        "--cutoff-frequency",
        dest="cutoff_frequency",
        type=float,
        default=None,
        help="Phonon modes below this frequency are ignored.",
    )
    parser.add_argument(
        "--cutoff-pair",
        "--cutoff-pair-distance",
        dest="cutoff_pair_distance",
        type=float,
        default=None,
        help=(
            "Cutoff distance between pairs of displaced atoms used for "
            "supercell creation with displacements and making third-order "
            "force constants"
        ),
    )
    if not load_phono3py_yaml:
        parser.add_argument(
            "-d",
            "--disp",
            dest="is_displacement",
            action="store_true",
            default=False,
            help="As first stage, get least displacements",
        )
        parser.add_argument(
            "--dim",
            nargs="+",
            dest="supercell_dimension",
            default=None,
            help="Supercell dimension",
        )
        parser.add_argument(
            "--dim-fc2",
            nargs="+",
            dest="phonon_supercell_dimension",
            default=None,
            help="Supercell dimension for extra fc2",
        )
    # parser.add_argument(
    #     "--emulate-v2",
    #     dest="emulate_v2",
    #     action="store_true",
    #     default=False,
    #     help="Emulate v2.x behaviour.",
    # )
    parser.add_argument(
        "--factor",
        dest="frequency_conversion_factor",
        type=float,
        default=None,
        help="Frequency unit conversion factor",
    )
    if not load_phono3py_yaml:
        parser.add_argument(
            "--fc2",
            dest="read_fc2",
            action="store_true",
            default=False,
            help="Read second order force constants",
        )
        parser.add_argument(
            "--fc3",
            dest="read_fc3",
            action="store_true",
            default=False,
            help="Read third order force constants",
        )
    parser.add_argument(
        "--v2",
        dest="is_fc3_r0_average",
        action="store_false",
        default=True,
        help="Take average in fc3-r2q transformation around three atoms",
    )
    parser.add_argument(
        "--fc-calc",
        "--fc-calculator",
        dest="fc_calculator",
        default=None,
        help=("Force constants calculator"),
    )
    parser.add_argument(
        "--fc-calc-opt",
        "--fc-calculator-options",
        dest="fc_calculator_options",
        default=None,
        help=(
            "Options for force constants calculator as comma separated "
            "string with the style of key = values"
        ),
    )
    if not fc_symmetry:
        parser.add_argument(
            "--fc-symmetry",
            "--sym-fc",
            dest="fc_symmetry",
            action="store_true",
            default=None,
            help="Symmetrize force constants",
        )
    parser.add_argument(
        "--freq-scale",
        dest="frequency_scale_factor",
        type=float,
        default=None,
        help=(
            "Factor multiplied as fc2 * factor^2 and fc3 * factor^2. "
            "Phonon frequency is changed but the contribution from NAC is "
            "not changed."
        ),
    )
    parser.add_argument(
        "--freq-pitch",
        dest="fpitch",
        type=float,
        default=None,
        help="Pitch in frequency for spectrum",
    )
    parser.add_argument(
        "--fs2f2",
        "--force-sets-to-forces-fc2",
        dest="force_sets_to_forces_fc2_mode",
        default=False,
        action="store_true",
        help="Create FORCES_FC2 from FORCE_SETS",
    )
    parser.add_argument(
        "--full-pp",
        dest="is_full_pp",
        action="store_true",
        default=False,
        help=(
            "Calculate full ph-ph interaction for RTA conductivity."
            "This may be activated when full elements of ph-ph interaction "
            "strength are needed, i.e., to calculate average ph-ph "
            "interaction strength."
        ),
    )
    parser.add_argument(
        "--ga",
        "--grid-addresses",
        nargs="+",
        dest="grid_addresses",
        default=None,
        help="Fixed grid addresses where anharmonic properties are calculated",
    )
    parser.add_argument(
        "--gm",
        "--grid-matrix",
        nargs="+",
        dest="grid_matrix",
        default=None,
        help="Grid generating matrix for generalized regular grid",
    )
    parser.add_argument(
        "--gp",
        "--grid-points",
        nargs="+",
        dest="grid_points",
        default=None,
        help="Fixed grid points where anharmonic properties are calculated",
    )
    parser.add_argument(
        "--grg",
        "--generalized-regular-grid",
        dest="use_grg",
        action="store_true",
        default=False,
        help="Use generalized regular grid.",
    )
    parser.add_argument(
        "--gruneisen",
        dest="is_gruneisen",
        action="store_true",
        default=False,
        help="Calculate phonon Gruneisen parameter",
    )
    parser.add_argument(
        "--gv-delta-q",
        dest="gv_delta_q",
        type=float,
        default=None,
        help="Delta-q distance used for group velocity calculation",
    )
    parser.add_argument(
        "--hdf5-compression",
        dest="hdf5_compression",
        default=None,
        help="hdf5 compression filter (default: gzip)",
    )
    if not load_phono3py_yaml:
        parser.add_argument(
            "-i", dest="input_filename", default=None, help="Input filename extension"
        )
        parser.add_argument(
            "--io",
            dest="input_output_filename",
            default=None,
            help="Input and output filename extension",
        )
    parser.add_argument(
        "--ion-clamped",
        dest="ion_clamped",
        action="store_true",
        default=False,
        help=(
            "Atoms are clamped under applied strain in Gruneisen parameter "
            "calculation"
        ),
    )
    parser.add_argument(
        "--ise",
        dest="is_imag_self_energy",
        action="store_true",
        default=False,
        help="Calculate imaginary part of self energy",
    )
    parser.add_argument(
        "--isotope",
        dest="is_isotope",
        action="store_true",
        default=False,
        help="Isotope scattering lifetime",
    )
    parser.add_argument(
        "--jdos",
        dest="is_joint_dos",
        action="store_true",
        default=False,
        help="Calculate joint density of states",
    )
    parser.add_argument(
        "--kubo",
        dest="is_kubo_kappa",
        action="store_true",
        default=False,
        help="Choose Kubo lattice thermal conductivity.",
    )
    parser.add_argument(
        "--lbte",
        dest="is_lbte",
        action="store_true",
        default=False,
        help="Calculate thermal conductivity LBTE with Chaput's method",
    )
    parser.add_argument(
        "--loglevel", dest="log_level", type=int, default=None, help="Log level"
    )
    parser.add_argument(
        "--mass", nargs="+", dest="masses", default=None, help="Same as MASS tag"
    )
    parser.add_argument(
        "--magmom", nargs="+", dest="magmoms", default=None, help="Same as MAGMOM tag"
    )
    parser.add_argument(
        "--mesh", nargs="+", dest="mesh_numbers", default=None, help="Mesh numbers"
    )
    parser.add_argument(
        "--mlp-params",
        dest="mlp_params",
        default=None,
        help=(
            "Parameters for machine learning potentials as comma separated "
            "string with the style of key = values"
        ),
    )
    parser.add_argument(
        "--mv",
        "--mass-variances",
        nargs="+",
        dest="mass_variances",
        default=None,
        help="Mass variance parameters for isotope scattering",
    )
    if not is_nac:
        parser.add_argument(
            "--nac",
            dest="is_nac",
            action="store_true",
            default=None,
            help="Non-analytical term correction",
        )
    parser.add_argument(
        "--nac-method",
        dest="nac_method",
        default=None,
        help="Non-analytical term correction method: Gonze (default) or Wang",
    )
    if fc_symmetry:
        parser.add_argument(
            "--no-fc-symmetry",
            "--no-sym-fc",
            dest="fc_symmetry",
            action="store_false",
            default=None,
            help="Do not symmetrize force constants",
        )
    parser.add_argument(
        "--nodiag",
        dest="is_nodiag",
        action="store_true",
        default=False,
        help="Set displacements parallel to axes",
    )
    parser.add_argument(
        "--noks",
        "--no-kappa-stars",
        dest="no_kappa_stars",
        action="store_true",
        default=False,
        help="Deactivate summation of partial kappa at q-stars",
    )
    parser.add_argument(
        "--nomeshsym",
        dest="is_nomeshsym",
        action="store_true",
        default=False,
        help="No symmetrization of triplets is made.",
    )
    if is_nac:
        parser.add_argument(
            "--nonac",
            dest="is_nac",
            action="store_false",
            default=None,
            help="Non-analytical term correction",
        )
    parser.add_argument(
        "--nosym",
        dest="is_nosym",
        action="store_true",
        default=False,
        help="Symmetry is not imposed.",
    )
    parser.add_argument(
        "--nu",
        dest="is_N_U",
        action="store_true",
        default=False,
        help="Split Gamma into Normal and Umklapp processes",
    )
    parser.add_argument(
        "--num-freq-points",
        dest="num_frequency_points",
        type=int,
        default=None,
        help="Number of sampling points for spectrum",
    )
    parser.add_argument(
        "--num-points-in-batch",
        dest="num_points_in_batch",
        type=int,
        default=None,
        help=(
            "Number of frequency points in a batch for the frequency "
            "sampling modes of imag-self-energy calculation"
        ),
    )
    if load_phono3py_yaml:
        parser.add_argument(
            "-o",
            dest="output_yaml_filename",
            default=None,
            help="Output yaml filename instead of default filename of phono3py.yaml",
        )
    else:
        parser.add_argument(
            "-o", dest="output_filename", default=None, help="Output filename extension"
        )
    parser.add_argument(
        "--pa",
        "--primitive-axis",
        "--primitive-axes",
        nargs="+",
        dest="primitive_axes",
        default=None,
        help="Same as PRIMITIVE_AXES tags",
    )
    parser.add_argument(
        "--pinv-cutoff",
        dest="pinv_cutoff",
        type=float,
        default=None,
        help="Cutoff frequency (THz) for pseudo inversion of collision matrix",
    )
    parser.add_argument(
        "--pinv-solver",
        dest="pinv_solver",
        type=int,
        default=None,
        help="Switch of LBTE pinv solver",
    )
    parser.add_argument(
        "--pinv-method",
        dest="pinv_method",
        type=int,
        default=None,
        help="Switch of LBTE pinv method",
    )
    parser.add_argument(
        "--pm",
        dest="is_plusminus_displacements",
        action="store_true",
        default=False,
        help="Set plus minus displacements",
    )
    parser.add_argument(
        "--pp-unit-conversion",
        dest="pp_unit_conversion",
        type=float,
        default=None,
        help="Conversion factor for ph-ph interaction",
    )
    parser.add_argument(
        "--pypolymlp",
        dest="use_pypolymlp",
        action="store_true",
        default=False,
        help="Use pypolymlp and symfc for generating force constants",
    )
    parser.add_argument(
        "--qpoints",
        nargs="+",
        dest="qpoints",
        default=None,
        help="Calculate at specified q-points",
    )
    parser.add_argument(
        "--q-direction",
        nargs="+",
        dest="nac_q_direction",
        default=None,
        help="q-vector direction at q->0 for non-analytical term correction",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Print out smallest information",
    )
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        type=int,
        default=None,
        help="Random seed by a 32 bit unsigned integer",
    )
    parser.add_argument(
        "--rd",
        "--random-displacements",
        dest="random_displacements",
        type=int,
        default=None,
        help="Number of supercells with random displacements",
    )
    parser.add_argument(
        "--rd-fc2",
        "--random-displacements-fc2",
        dest="random_displacements_fc2",
        type=int,
        default=None,
        help="Number of phonon supercells with random displacements",
    )
    parser.add_argument(
        "--read-collision",
        dest="read_collision",
        default=None,
        help="Read collision matrix and Gammas from files",
    )
    parser.add_argument(
        "--read-gamma",
        dest="read_gamma",
        action="store_true",
        default=False,
        help="Read Gammas from files",
    )
    parser.add_argument(
        "--read-phonon",
        dest="read_phonon",
        action="store_true",
        default=False,
        help="Read phonons from files",
    )
    parser.add_argument(
        "--read-pp",
        dest="read_pp",
        action="store_true",
        default=False,
        help="Read phonon-phonon interaction strength",
    )
    parser.add_argument(
        "--reducible-colmat",
        dest="is_reducible_collision_matrix",
        action="store_true",
        default=False,
        help="Solve reducible collision matrix",
    )
    parser.add_argument(
        "--rse",
        dest="is_real_self_energy",
        action="store_true",
        default=False,
        help="Calculate real part of self energy",
    )
    parser.add_argument(
        "--sp",
        "--save-params",
        dest="save_params",
        action="store_true",
        default=None,
        help="Save parameters that can run phono3py in phono3py_params.yaml.",
    )
    parser.add_argument(
        "--scattering-event-class",
        dest="scattering_event_class",
        type=int,
        default=None,
        help=("Scattering event class 1 or 2 to draw imaginary part of self " "energy"),
    )
    parser.add_argument(
        "--sigma",
        nargs="+",
        dest="sigma",
        default=None,
        help=(
            "A sigma value or multiple sigma values (separated by space) "
            "for smearing function"
        ),
    )
    parser.add_argument(
        "--sigma-cutoff",
        dest="sigma_cutoff_width",
        type=float,
        default=None,
        help="Cutoff width of smearing function (ratio to sigma value)",
    )
    parser.add_argument(
        "--symfc",
        dest="use_symfc",
        action="store_true",
        default=None,
        help="Use symfc for generating force constants",
    )
    parser.add_argument(
        "--spf",
        dest="is_spectral_function",
        action="store_true",
        default=False,
        help="Calculate spectral function",
    )
    parser.add_argument(
        "--stp",
        "--show-num-triplets",
        dest="show_num_triplets",
        action="store_true",
        default=False,
        help=(
            "Show reduced number of triplets to be calculated at "
            "specified grid points"
        ),
    )
    if not load_phono3py_yaml:
        parser.add_argument(
            "--sym-fc2",
            dest="is_symmetrize_fc2",
            action="store_true",
            default=False,
            help="Symmetrize fc2 by index exchange",
        )
        parser.add_argument(
            "--sym-fc3r",
            dest="is_symmetrize_fc3_r",
            action="store_true",
            default=False,
            help="Symmetrize fc3 in real space by index exchange",
        )
    parser.add_argument(
        "--sym-fc3q",
        dest="is_symmetrize_fc3_q",
        action="store_true",
        default=False,
        help="Symmetrize fc3 in reciprocal space by index exchange",
    )
    parser.add_argument(
        "--thm",
        "--tetrahedron-method",
        dest="is_tetrahedron_method",
        action="store_true",
        default=False,
        help="Use tetrahedron method.",
    )
    parser.add_argument(
        "--tmax", dest="tmax", default=None, help="Maximum calculated temperature"
    )
    parser.add_argument(
        "--tmin", dest="tmin", default=None, help="Minimum calculated temperature"
    )
    parser.add_argument(
        "--ts",
        nargs="+",
        dest="temperatures",
        default=None,
        help="Temperatures for damping functions",
    )
    parser.add_argument(
        "--tstep", dest="tstep", default=None, help="Calculated temperature step"
    )
    parser.add_argument(
        "--tolerance",
        dest="symmetry_tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance to search",
    )
    parser.add_argument(
        "--uplo",
        dest="lapack_zheev_uplo",
        default=None,
        help="Lapack zheev UPLO for phonon solver (default: L)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Detailed run-time information is displayed",
    )
    parser.add_argument(
        "--wgp",
        "--write-grid-points",
        dest="write_grid_points",
        action="store_true",
        default=False,
        help=(
            "Write grid address of irreducible grid points for specified "
            "mesh numbers to ir_grid_address.yaml"
        ),
    )
    parser.add_argument(
        "--wigner",
        dest="is_wigner_kappa",
        action="store_true",
        default=False,
        help="Choose Wigner lattice thermal conductivity.",
    )
    parser.add_argument(
        "--write-collision",
        dest="write_collision",
        action="store_true",
        default=False,
        help="Write collision matrix and Gammas to files",
    )
    parser.add_argument(
        "--write-gamma",
        dest="write_gamma",
        action="store_true",
        default=False,
        help="Write imag-part of self energy to files",
    )
    parser.add_argument(
        "--write-gamma-detail",
        "--write_detailed_gamma",
        dest="write_gamma_detail",
        action="store_true",
        default=False,
        help="Write out detailed imag-part of self energy",
    )
    parser.add_argument(
        "--write-phonon",
        dest="write_phonon",
        action="store_true",
        default=False,
        help="Write all phonons on grid points to files",
    )
    parser.add_argument(
        "--write-pp",
        dest="write_pp",
        action="store_true",
        default=False,
        help="Write phonon-phonon interaction strength",
    )
    parser.add_argument(
        "--write-lbte-solution",
        dest="write_LBTE_solution",
        action="store_true",
        default=False,
        help="Write direct solution of LBTE to hdf5 files",
    )
    if load_phono3py_yaml:
        parser.add_argument("filename", nargs="*", help="phono3py.yaml like file")
    else:
        parser.add_argument("filename", nargs="*", help="Phono3py configure file")

    return parser, deprecated
