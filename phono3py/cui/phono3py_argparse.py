# Copyright (C) 2015 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
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


def get_parser():
    deprecated = fix_deprecated_option_names(sys.argv)

    import argparse
    parser = argparse.ArgumentParser(
        description="Phono3py command-line-tool")
    parser.set_defaults(abinit_mode=False,
                        alm_options=None,
                        band_indices=None,
                        band_paths=None,
                        band_points=None,
                        cell_filename=None,
                        const_ave_pp=None,
                        crystal_mode=False,
                        cutoff_fc3_distance=None,
                        cutoff_frequency=None,
                        boundary_mfp=None,
                        cutoff_pair_distance=None,
                        delta_fc2=False,
                        delta_fc2_sets_mode=False,
                        displacement_distance=None,
                        fc_symmetry=False,
                        force_sets_to_forces_fc2_mode=None,
                        forces_fc2=None,
                        forces_fc3=None,
                        forces_fcz=None,
                        forces_fc3_file=None,
                        force_sets_mode=False,
                        frequency_conversion_factor=None,
                        fpitch=None,
                        frequency_scale_factor=None,
                        gamma_unit_conversion=None,
                        grid_addresses=None,
                        grid_points=None,
                        gv_delta_q=None,
                        hdf5_compression="gzip",
                        input_filename=None,
                        input_output_filename=None,
                        ion_clamped=False,
                        is_bterta=False,
                        is_compact_fc=False,
                        is_decay_channel=False,
                        is_displacement=False,
                        is_frequency_shift=False,
                        is_full_pp=False,
                        is_gruneisen=False,
                        is_isotope=False,
                        is_joint_dos=False,
                        is_lbte=False,
                        is_nac=False,
                        is_nodiag=False,
                        is_nomeshsym=False,
                        is_nosym=False,
                        is_N_U=False,
                        is_plusminus_displacements=False,
                        is_reducible_collision_matrix=False,
                        is_translational_symmetry=False,
                        is_symmetrize_fc2=False,
                        is_symmetrize_fc3_r=False,
                        is_symmetrize_fc3_q=False,
                        is_tetrahedron_method=False,
                        log_level=None,
                        max_freepath=None,
                        masses=None,
                        mass_variances=None,
                        mesh_numbers=None,
                        mesh_divisors=None,
                        nac_method=None,
                        nac_q_direction=None,
                        no_kappa_stars=False,
                        num_frequency_points=None,
                        output_filename=None,
                        phonon_supercell_dimension=None,
                        pinv_cutoff=None,
                        pinv_solver=None,
                        pp_unit_conversion=None,
                        primitive_axes=None,
                        qe_mode=False,
                        qpoints=None,
                        quiet=False,
                        read_collision=None,
                        read_fc2=False,
                        read_fc3=False,
                        read_gamma=False,
                        read_pp=False,
                        scattering_event_class=None,
                        show_num_triplets=False,
                        sigma=None,
                        sigma_cutoff=None,
                        solve_collective_phonon=False,
                        supercell_dimension=None,
                        symprec=1e-5,
                        temperatures=None,
                        tmax=None,
                        tmin=None,
                        tstep=None,
                        uplo='L',
                        use_alm_fc2=False,
                        use_alm_fc3=False,
                        use_ave_pp=False,
                        verbose=False,
                        write_collision=False,
                        write_gamma_detail=False,
                        write_gamma=False,
                        write_grid_points=False,
                        write_phonon=False,
                        write_pp=False,
                        write_LBTE_solution=False)
    parser.add_argument(
        "--abinit", dest="abinit_mode", action="store_true",
        help="Invoke Abinit mode")
    parser.add_argument(
        "--alm-fc2", dest="use_alm_fc2", action="store_true",
        help="Use ALM for creating 2nd order force constants")
    parser.add_argument(
        "--alm-fc3", dest="use_alm_fc3", action="store_true",
        help="Use ALM for creating 3rd order force constants")
    parser.add_argument(
        "--alm-options", dest="alm_options",
        help="List of ALM options as string separated by commas")
    parser.add_argument(
        "--amplitude", dest="displacement_distance", type=float,
        help="Distance of displacements")
    parser.add_argument(
        "--ave-pp", dest="use_ave_pp", action="store_true",
        help="Use averaged ph-ph interaction")
    parser.add_argument(
        "--band", nargs='+', dest="band_paths",
        help="Band structure paths calculated for Gruneisen parameter")
    parser.add_argument(
        "--band-points", dest="band_points", type=int,
        help=("Number of points calculated on a band segment in the band "
              "structure Gruneisen parameter calculation"))
    parser.add_argument(
        "--bi", "--band-indices", nargs='+', dest="band_indices",
        help="Band indices where life time is calculated")
    parser.add_argument(
        "--boundary-mfp", "--bmfp", dest="boundary_mfp", type=float,
        help=("Boundary mean free path in micrometre for thermal conductivity "
              "calculation"))
    parser.add_argument(
        "--br", "--bterta", dest="is_bterta", action="store_true",
        help="Calculate thermal conductivity in BTE-RTA")
    parser.add_argument(
        "-c", "--cell", dest="cell_filename", metavar="FILE",
        help="Read unit cell")
    parser.add_argument(
        "--cf2", "--create-f2", dest="forces_fc2", nargs='+',
        help="Create FORCES_FC2")
    parser.add_argument(
        "--cf3", "--create-f3", dest="forces_fc3", nargs='+',
        help="Create FORCES_FC3")
    parser.add_argument(
        "--cfz", "--subtract-forces", dest="forces_fcz",
        help="Subtract recidual forces from supercell forces")
    parser.add_argument(
        "--cf3-file", "--create-f3-from-file", dest="forces_fc3_file",
        help="Create FORCES_FC3 from file name list")
    parser.add_argument(
        "--cfc", "--compact-fc", dest="is_compact_fc", action="store_true",
        help="Use compact force cosntants")
    parser.add_argument(
        "--cfs", "--create-force-sets", dest="force_sets_mode",
        action="store_true",
        help="Create phonopy FORCE_SETS from FORCES_FC2")
    parser.add_argument(
        "--cph", "--collective-phonon", dest="solve_collective_phonon",
        action="store_true",
        help="Solve collective phonons")
    parser.add_argument(
        "--const-ave-pp", dest="const_ave_pp", type=float,
        help="Set constant averaged ph-ph interaction (Pqj)")
    parser.add_argument(
        "--crystal", dest="crystal_mode", action="store_true",
        help="Invoke CRYSTAL mode")
    parser.add_argument(
        "--cutoff-fc3", "--cutoff-fc3-distance", dest="cutoff_fc3_distance",
        type=float,
        help=("Cutoff distance of third-order force constants. Elements where "
              "any pair of atoms has larger distance than cut-off distance "
              "are set zero."))
    parser.add_argument(
        "--cutoff-freq", "--cutoff-frequency", dest="cutoff_frequency",
        type=float,
        help="Phonon modes below this frequency are ignored.")
    parser.add_argument(
        "--cutoff-pair", "--cutoff-pair-distance", dest="cutoff_pair_distance",
        type=float,
        help=("Cutoff distance between pairs of displaced atoms used for "
              "supercell creation with displacements and making third-order "
              "force constants"))
    parser.add_argument(
        "-d", "--disp", dest="is_displacement", action="store_true",
        help="As first stage, get least displacements")
    parser.add_argument(
        "--dim", nargs='+', dest="supercell_dimension",
        help="Supercell dimension")
    parser.add_argument(
        "--dim-fc2", nargs='+', dest="phonon_supercell_dimension",
        help="Supercell dimension for extra fc2")
    parser.add_argument(
        "--factor", dest="frequency_conversion_factor", type=float,
        help="Frequency unit conversion factor")
    parser.add_argument(
        "--fc2", dest="read_fc2", action="store_true",
        help="Read second order force constants")
    parser.add_argument(
        "--fc3", dest="read_fc3", action="store_true",
        help="Read third order force constants")
    parser.add_argument(
        "--fc-symmetry", "--sym-fc", dest="fc_symmetry", action="store_true",
        help="Symmetrize force constants")
    parser.add_argument(
        "--freq-scale", dest="frequency_scale_factor", type=float,
        help=("Factor multiplied as fc2 * factor^2 and fc3 * factor^2. "
              "Phonon frequency is changed but the contribution from NAC is "
              "not changed."))
    parser.add_argument(
        "--freq-pitch", dest="fpitch", type=float,
        help="Pitch in frequency for spectrum")
    parser.add_argument(
        "--fs2f2", "--force-sets-to-forces-fc2",
        dest="force_sets_to_forces_fc2_mode",
        action="store_true", help="Create FORCES_FC2 from FORCE_SETS")
    parser.add_argument(
        "--fst", "--frequency-shift", dest="is_frequency_shift",
        action="store_true", help="Calculate frequency shifts")
    parser.add_argument(
        "--full-pp", dest="is_full_pp", action="store_true",
        help=("Calculate full ph-ph interaction for RTA conductivity."
              "This may be activated when full elements of ph-ph interaction "
              "strength are needed, i.e., to calculate average ph-ph "
              "interaction strength."))
    parser.add_argument(
        "--ga", "--grid-addresses", nargs='+', dest="grid_addresses",
        help="Fixed grid addresses where anharmonic properties are calculated")
    parser.add_argument(
        "--gamma-unit-conversion", dest="gamma_unit_conversion", type=float,
        help="Conversion factor for gamma")
    parser.add_argument(
        "--gp", "--grid-points", nargs='+', dest="grid_points",
        help="Fixed grid points where anharmonic properties are calculated")
    parser.add_argument(
        "--gruneisen", dest="is_gruneisen", action="store_true",
        help="Calculate phonon Gruneisen parameter")
    parser.add_argument(
        "--gv-delta-q", dest="gv_delta_q", type=float,
        help="Delta-q distance used for group velocity calculation")
    parser.add_argument(
        "--hdf5-compression", dest="hdf5_compression",
        help="hdf5 compression filter")
    parser.add_argument(
        "-i", dest="input_filename",
        help="Input filename extension")
    parser.add_argument(
        "--io", dest="input_output_filename",
        help="Input and output filename extension")
    parser.add_argument(
        "--ion-clamped", dest="ion_clamped", action="store_true",
        help=("Atoms are clamped under applied strain in Gruneisen parameter "
              "calculation"))
    parser.add_argument(
        "--ise", dest="is_imag_self_energy", action="store_true",
        help="Calculate imaginary part of self energy")
    parser.add_argument(
        "--isotope", dest="is_isotope", action="store_true",
        help="Isotope scattering lifetime")
    parser.add_argument(
        "--jdos", dest="is_joint_dos", action="store_true",
        help="Calculate joint density of states")
    parser.add_argument(
        "--lbte", dest="is_lbte", action="store_true",
        help="Calculate thermal conductivity LBTE with Chaput's method")
    parser.add_argument(
        "--loglevel", dest="log_level", type=int,
        help="Log level")
    parser.add_argument(
        "--mass", nargs='+', dest="masses",
        help="Same as MASS tag")
    parser.add_argument(
        "--md", "--mesh-divisors", nargs='+', dest="mesh_divisors",
        help="Divisors for mesh numbers")
    parser.add_argument(
        "--mesh", nargs='+', dest="mesh_numbers",
        help="Mesh numbers")
    parser.add_argument(
        "--mv", "--mass-variances", nargs='+', dest="mass_variances",
        help="Mass variance parameters for isotope scattering")
    parser.add_argument(
        "--nac", dest="is_nac", action="store_true",
        help="Non-analytical term correction")
    parser.add_argument(
        "--nac-method", dest="nac_method",
        help="Non-analytical term correction method: Wang (default) or Gonze")
    parser.add_argument(
        "--nodiag", dest="is_nodiag", action="store_true",
        help="Set displacements parallel to axes")
    parser.add_argument(
        "--noks", "--no-kappa-stars", dest="no_kappa_stars",
        action="store_true",
        help="Deactivate summation of partial kappa at q-stars"),
    parser.add_argument(
        "--nomeshsym", dest="is_nomeshsym", action="store_true",
        help="No symmetrization of triplets is made.")
    parser.add_argument(
        "--nosym", dest="is_nosym", action="store_true",
        help="Symmetry is not imposed.")
    parser.add_argument(
        "--nu", dest="is_N_U", action="store_true",
        help="Split Gamma into Normal and Umklapp processes")
    parser.add_argument(
        "--num-freq-points", dest="num_frequency_points", type=int,
        help="Number of sampling points for spectrum")
    parser.add_argument(
        "-o", dest="output_filename",
        help="Output filename extension")
    parser.add_argument(
        "--pa", "--primitive-axis", "--primitive-axes", nargs='+',
        dest="primitive_axes",
        help="Same as PRIMITIVE_AXES tags")
    parser.add_argument(
        "--pinv-cutoff", dest="pinv_cutoff", type=float,
        help="Cutoff frequency (THz) for pseudo inversion of collision matrix")
    parser.add_argument(
        "--pinv-solver", dest="pinv_solver", type=int,
        help="Switch of LBTE pinv solver")
    parser.add_argument(
        "--pm", dest="is_plusminus_displacements", action="store_true",
        help="Set plus minus displacements")
    parser.add_argument(
        "--pp-unit-conversion", dest="pp_unit_conversion", type=float,
        help="Conversion factor for ph-ph interaction")
    parser.add_argument(
        "--qe", "--pwscf", dest="qe_mode",
        action="store_true", help="Invoke Quantum espresso (QE) mode")
    parser.add_argument(
        "--qpoints", nargs='+', dest="qpoints",
        help="Calculate at specified q-points")
    parser.add_argument(
        "--q-direction", nargs='+', dest="nac_q_direction",
        help="q-vector direction at q->0 for non-analytical term correction")
    parser.add_argument(
        "-q", "--quiet", dest="quiet", action="store_true",
        help="Print out smallest information")
    parser.add_argument(
        "--read-collision", dest="read_collision",
        help="Read collision matrix and Gammas from files")
    parser.add_argument(
        "--read-gamma", dest="read_gamma", action="store_true",
        help="Read Gammas from files")
    parser.add_argument(
        "--read-phonon", dest="read_phonon", action="store_true",
        help="Read phonons from files")
    parser.add_argument(
        "--read-pp", dest="read_pp", action="store_true",
        help="Read phonon-phonon interaction strength")
    parser.add_argument(
        "--reducible-colmat", dest="is_reducible_collision_matrix",
        action="store_true",
        help="Solve reducible collision matrix")
    parser.add_argument(
        "--scattering-event-class", dest="scattering_event_class", type=int,
        help=("Scattering event class 1 or 2 to draw imaginary part of self "
              "energy"))
    parser.add_argument(
        "--sigma", nargs='+', dest="sigma",
        help=("A sigma value or multiple sigma values (separated by space) "
              "for smearing function"))
    parser.add_argument(
        "--sigma-cutoff", dest="sigma_cutoff_width", type=float,
        help="Cutoff width of smearing function (ratio to sigma value)")
    parser.add_argument(
        "--stp", "--show-num-triplets", dest="show_num_triplets",
        action="store_true",
        help=("Show reduced number of triplets to be calculated at "
              "specified grid points"))
    parser.add_argument(
        "--sym-fc2", dest="is_symmetrize_fc2", action="store_true",
        help="Symmetrize fc2 by index exchange")
    parser.add_argument(
        "--sym-fc3r", dest="is_symmetrize_fc3_r", action="store_true",
        help="Symmetrize fc3 in real space by index exchange")
    parser.add_argument(
        "--sym-fc3q", dest="is_symmetrize_fc3_q", action="store_true",
        help="Symmetrize fc3 in reciprocal space by index exchange")
    parser.add_argument(
        "--thm", "--tetrahedron-method", dest="is_tetrahedron_method",
        action="store_true",
        help="Use tetrahedron method")
    parser.add_argument(
        "--tmax", dest="tmax",
        help="Maximum calculated temperature")
    parser.add_argument(
        "--tmin", dest="tmin",
        help="Minimum calculated temperature")
    parser.add_argument(
        "--ts", nargs='+', dest="temperatures",
        help="Temperatures for damping functions")
    parser.add_argument(
        "--tstep", dest="tstep",
        help="Calculated temperature step")
    parser.add_argument(
        "--tolerance", dest="symprec", type=float,
        help="Symmetry tolerance to search")
    parser.add_argument(
        "--uplo", dest="uplo",
        help="Lapack zheev UPLO")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true",
        help="Detailed run-time information is displayed")
    parser.add_argument(
        "--wgp", "--write-grid-points", dest="write_grid_points",
        action="store_true",
        help=("Write grid address of irreducible grid points for specified "
              "mesh numbers to ir_grid_address.yaml"))
    parser.add_argument(
        "--write-collision", dest="write_collision", action="store_true",
        help="Write collision matrix and Gammas to files")
    parser.add_argument(
        "--write-gamma", dest="write_gamma", action="store_true",
        help="Write imag-part of self energy to files")
    parser.add_argument(
        "--write-gamma-detail", "--write_detailed_gamma",
        dest="write_gamma_detail", action="store_true",
        help="Write out detailed imag-part of self energy")
    parser.add_argument(
        "--write-phonon", dest="write_phonon", action="store_true",
        help="Write all phonons on grid points to files")
    parser.add_argument(
        "--write-pp", dest="write_pp", action="store_true",
        help="Write phonon-phonon interaction strength")
    parser.add_argument(
        "--write-lbte-solution", dest="write_LBTE_solution",
        action="store_true",
        help="Write direct solution of LBTE to hdf5 files")
    parser.add_argument(
        "conf_file", nargs='*',
        help="Phono3py configure file")

    return parser, deprecated
