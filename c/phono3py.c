/* Copyright (C) 2021 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include "phono3py.h"

#include <stdio.h>
#include <stdlib.h>

#include "bzgrid.h"
#include "collision_matrix.h"
#include "fc3.h"
#include "grgrid.h"
#include "imag_self_energy_with_g.h"
#include "interaction.h"
#include "isotope.h"
#include "lagrid.h"
#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "pp_collision.h"
#include "real_self_energy.h"
#include "real_to_reciprocal.h"
#include "tetrahedron_method.h"
#include "triplet.h"
#include "triplet_iw.h"

#ifdef _OPENMP
#include <omp.h>
#endif

long ph3py_get_interaction(
    Darray *fc3_normal_squared, const char *g_zero, const Darray *frequencies,
    const _lapack_complex_double *eigenvectors, const long (*triplets)[3],
    const long num_triplets, const long (*bz_grid_addresses)[3],
    const long D_diag[3], const long Q[3][3], const double *fc3,
    const long is_compact_fc3, const double (*svecs)[3],
    const long multi_dims[2], const long (*multiplicity)[2],
    const double *masses, const long *p2s_map, const long *s2p_map,
    const long *band_indices, const long symmetrize_fc3_q,
    const long make_r0_average, const char *all_shortest,
    const double cutoff_frequency, const long openmp_per_triplets) {
    ConstBZGrid *bzgrid;
    AtomTriplets *atom_triplets;
    long i, j;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = 0;
        for (j = 0; j < 3; j++) {
            bzgrid->Q[i][j] = Q[i][j];
        }
    }

    if ((atom_triplets = (AtomTriplets *)malloc(sizeof(AtomTriplets))) ==
        NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    atom_triplets->svecs = svecs;
    atom_triplets->multi_dims[0] = multi_dims[0];
    atom_triplets->multi_dims[1] = multi_dims[1];
    atom_triplets->multiplicity = multiplicity;
    atom_triplets->p2s_map = p2s_map;
    atom_triplets->s2p_map = s2p_map;
    atom_triplets->make_r0_average = make_r0_average;
    atom_triplets->all_shortest = all_shortest;

    itr_get_interaction(fc3_normal_squared, g_zero, frequencies,
                        (lapack_complex_double *)eigenvectors, triplets,
                        num_triplets, bzgrid, fc3, is_compact_fc3,
                        atom_triplets, masses, band_indices, symmetrize_fc3_q,
                        cutoff_frequency, openmp_per_triplets);

    free(atom_triplets);
    atom_triplets = NULL;

    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

long ph3py_get_pp_collision(
    double *imag_self_energy,
    const long relative_grid_address[24][4][3], /* thm */
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const long (*bz_grid_addresses)[3], /* thm */
    const long *bz_map,                                              /* thm */
    const long bz_grid_type, const long D_diag[3], const long Q[3][3],
    const double *fc3, const long is_compact_fc3, const double (*svecs)[3],
    const long multi_dims[2], const long (*multiplicity)[2],
    const double *masses, const long *p2s_map, const long *s2p_map,
    const Larray *band_indices, const Darray *temperatures, const long is_NU,
    const long symmetrize_fc3_q, const long make_r0_average,
    const char *all_shortest, const double cutoff_frequency,
    const long openmp_per_triplets) {
    ConstBZGrid *bzgrid;
    AtomTriplets *atom_triplets;
    long i, j;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = 0;
        for (j = 0; j < 3; j++) {
            bzgrid->Q[i][j] = Q[i][j];
        }
    }

    if ((atom_triplets = (AtomTriplets *)malloc(sizeof(AtomTriplets))) ==
        NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    atom_triplets->svecs = svecs;
    atom_triplets->multi_dims[0] = multi_dims[0];
    atom_triplets->multi_dims[1] = multi_dims[1];
    atom_triplets->multiplicity = multiplicity;
    atom_triplets->p2s_map = p2s_map;
    atom_triplets->s2p_map = s2p_map;
    atom_triplets->make_r0_average = make_r0_average;
    atom_triplets->all_shortest = all_shortest;

    ppc_get_pp_collision(imag_self_energy, relative_grid_address, frequencies,
                         (lapack_complex_double *)eigenvectors, triplets,
                         num_triplets, triplet_weights, bzgrid, fc3,
                         is_compact_fc3, atom_triplets, masses, band_indices,
                         temperatures, is_NU, symmetrize_fc3_q,
                         cutoff_frequency, openmp_per_triplets);

    free(atom_triplets);
    atom_triplets = NULL;

    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

long ph3py_get_pp_collision_with_sigma(
    double *imag_self_energy, const double sigma, const double sigma_cutoff,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const long (*bz_grid_addresses)[3],
    const long D_diag[3], const long Q[3][3], const double *fc3,
    const long is_compact_fc3, const double (*svecs)[3],
    const long multi_dims[2], const long (*multiplicity)[2],
    const double *masses, const long *p2s_map, const long *s2p_map,
    const Larray *band_indices, const Darray *temperatures, const long is_NU,
    const long symmetrize_fc3_q, const long make_r0_average,
    const char *all_shortest, const double cutoff_frequency,
    const long openmp_per_triplets) {
    ConstBZGrid *bzgrid;
    AtomTriplets *atom_triplets;
    long i, j;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = 0;
        for (j = 0; j < 3; j++) {
            bzgrid->Q[i][j] = Q[i][j];
        }
    }

    if ((atom_triplets = (AtomTriplets *)malloc(sizeof(AtomTriplets))) ==
        NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    atom_triplets->svecs = svecs;
    atom_triplets->multi_dims[0] = multi_dims[0];
    atom_triplets->multi_dims[1] = multi_dims[1];
    atom_triplets->multiplicity = multiplicity;
    atom_triplets->p2s_map = p2s_map;
    atom_triplets->s2p_map = s2p_map;
    atom_triplets->make_r0_average = make_r0_average;
    atom_triplets->all_shortest = all_shortest;

    ppc_get_pp_collision_with_sigma(
        imag_self_energy, sigma, sigma_cutoff, frequencies,
        (lapack_complex_double *)eigenvectors, triplets, num_triplets,
        triplet_weights, bzgrid, fc3, is_compact_fc3, atom_triplets, masses,
        band_indices, temperatures, is_NU, symmetrize_fc3_q, cutoff_frequency,
        openmp_per_triplets);

    free(atom_triplets);
    atom_triplets = NULL;

    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

void ph3py_get_imag_self_energy_at_bands_with_g(
    double *imag_self_energy, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const double *g, const char *g_zero,
    const double temperature, const double cutoff_frequency,
    const long num_frequency_points, const long frequency_point_index) {
    ise_get_imag_self_energy_with_g(
        imag_self_energy, fc3_normal_squared, frequencies, triplets,
        triplet_weights, g, g_zero, temperature, cutoff_frequency,
        num_frequency_points, frequency_point_index);
}

void ph3py_get_detailed_imag_self_energy_at_bands_with_g(
    double *detailed_imag_self_energy, double *imag_self_energy_N,
    double *imag_self_energy_U, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const long (*bz_grid_addresses)[3],
    const double *g, const char *g_zero, const double temperature,
    const double cutoff_frequency) {
    ise_get_detailed_imag_self_energy_with_g(
        detailed_imag_self_energy, imag_self_energy_N, imag_self_energy_U,
        fc3_normal_squared, frequencies, triplets, triplet_weights,
        bz_grid_addresses, g, g_zero, temperature, cutoff_frequency);
}

void ph3py_get_real_self_energy_at_bands(
    double *real_self_energy, const Darray *fc3_normal_squared,
    const long *band_indices, const double *frequencies,
    const long (*triplets)[3], const long *triplet_weights,
    const double epsilon, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency) {
    rse_get_real_self_energy_at_bands(real_self_energy, fc3_normal_squared,
                                      band_indices, frequencies, triplets,
                                      triplet_weights, epsilon, temperature,
                                      unit_conversion_factor, cutoff_frequency);
}

void ph3py_get_real_self_energy_at_frequency_point(
    double *real_self_energy, const double frequency_point,
    const Darray *fc3_normal_squared, const long *band_indices,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const double epsilon, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency) {
    rse_get_real_self_energy_at_frequency_point(
        real_self_energy, frequency_point, fc3_normal_squared, band_indices,
        frequencies, triplets, triplet_weights, epsilon, temperature,
        unit_conversion_factor, cutoff_frequency);
}

void ph3py_get_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplets_map, const long *map_q,
    const long *rotated_grid_points, const double *rotations_cartesian,
    const double *g, const long num_ir_gp, const long num_gp,
    const long num_rot, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency) {
    col_get_collision_matrix(collision_matrix, fc3_normal_squared, frequencies,
                             triplets, triplets_map, map_q, rotated_grid_points,
                             rotations_cartesian, g, num_ir_gp, num_gp, num_rot,
                             temperature, unit_conversion_factor,
                             cutoff_frequency);
}

void ph3py_get_reducible_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplets_map, const long *map_q, const double *g,
    const long num_gp, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency) {
    col_get_reducible_collision_matrix(
        collision_matrix, fc3_normal_squared, frequencies, triplets,
        triplets_map, map_q, g, num_gp, temperature, unit_conversion_factor,
        cutoff_frequency);
}

void ph3py_get_isotope_scattering_strength(
    double *gamma, const long grid_point, const long *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long num_ir_grid_points, const long *band_indices,
    const long num_band, const long num_band0, const double sigma,
    const double cutoff_frequency) {
    iso_get_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        (lapack_complex_double *)eigenvectors, num_ir_grid_points, band_indices,
        num_band, num_band0, sigma, cutoff_frequency);
}

void ph3py_get_thm_isotope_scattering_strength(
    double *gamma, const long grid_point, const long *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long num_ir_grid_points, const long *band_indices,
    const long num_band, const long num_band0,
    const double *integration_weights, const double cutoff_frequency) {
    iso_get_thm_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        (lapack_complex_double *)eigenvectors, num_ir_grid_points, band_indices,
        num_band, num_band0, integration_weights, cutoff_frequency);
}

void ph3py_distribute_fc3(double *fc3, const long target, const long source,
                          const long *atom_mapping, const long num_atom,
                          const double *rot_cart) {
    fc3_distribute_fc3(fc3, target, source, atom_mapping, num_atom, rot_cart);
}

void ph3py_rotate_delta_fc2(double (*fc3)[3][3][3],
                            const double (*delta_fc2s)[3][3],
                            const double *inv_U,
                            const double (*site_sym_cart)[3][3],
                            const long *rot_map_syms, const long num_atom,
                            const long num_site_sym, const long num_disp) {
    fc3_rotate_delta_fc2(fc3, delta_fc2s, inv_U, site_sym_cart, rot_map_syms,
                         num_atom, num_site_sym, num_disp);
}

void ph3py_get_permutation_symmetry_fc3(double *fc3, const long num_atom) {
    fc3_set_permutation_symmetry_fc3(fc3, num_atom);
}

void ph3py_get_permutation_symmetry_compact_fc3(
    double *fc3, const long p2s[], const long s2pp[], const long nsym_list[],
    const long perms[], const long n_satom, const long n_patom) {
    fc3_set_permutation_symmetry_compact_fc3(fc3, p2s, s2pp, nsym_list, perms,
                                             n_satom, n_patom);
}

void ph3py_transpose_compact_fc3(double *fc3, const long p2s[],
                                 const long s2pp[], const long nsym_list[],
                                 const long perms[], const long n_satom,
                                 const long n_patom, const long t_type) {
    fc3_transpose_compact_fc3(fc3, p2s, s2pp, nsym_list, perms, n_satom,
                              n_patom, t_type);
}

long ph3py_get_triplets_reciprocal_mesh_at_q(
    long *map_triplets, long *map_q, const long grid_point,
    const long D_diag[3], const long is_time_reversal, const long num_rot,
    const long (*rec_rotations)[3][3], const long swappable) {
    return tpl_get_triplets_reciprocal_mesh_at_q(
        map_triplets, map_q, grid_point, D_diag, is_time_reversal, num_rot,
        rec_rotations, swappable);
}

long ph3py_get_BZ_triplets_at_q(long (*triplets)[3], const long grid_point,
                                const long (*bz_grid_addresses)[3],
                                const long *bz_map, const long *map_triplets,
                                const long num_map_triplets,
                                const long D_diag[3], const long Q[3][3],
                                const long bz_grid_type) {
    ConstBZGrid *bzgrid;
    long i, j, num_ir;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = 0;
        for (j = 0; j < 3; j++) {
            bzgrid->Q[i][j] = Q[i][j];
        }
    }
    bzgrid->size = num_map_triplets;

    num_ir =
        tpl_get_BZ_triplets_at_q(triplets, grid_point, bzgrid, map_triplets);
    free(bzgrid);
    bzgrid = NULL;

    return num_ir;
}

/* relative_grid_addresses are given as P multipled with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
long ph3py_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long relative_grid_address[24][4][3],
    const long D_diag[3], const long (*triplets)[3], const long num_triplets,
    const long (*bz_grid_addresses)[3], const long *bz_map,
    const long bz_grid_type, const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets) {
    ConstBZGrid *bzgrid;
    long i;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
    }

    tpl_get_integration_weight(iw, iw_zero, frequency_points, num_band0,
                               relative_grid_address, triplets, num_triplets,
                               bzgrid, frequencies1, num_band1, frequencies2,
                               num_band2, tp_type, openmp_per_triplets);
    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

void ph3py_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const long num_band0,
    const long (*triplets)[3], const long num_triplets,
    const double *frequencies, const long num_band, const long tp_type) {
    tpl_get_integration_weight_with_sigma(
        iw, iw_zero, sigma, sigma_cutoff, frequency_points, num_band0, triplets,
        num_triplets, frequencies, num_band, tp_type);
}

/* From single address to grid index */
long ph3py_get_grid_index_from_address(const long address[3],
                                       const long D_diag[3]) {
    return grg_get_grid_index(address, D_diag);
}

void ph3py_get_gr_grid_addresses(long gr_grid_addresses[][3],
                                 const long D_diag[3]) {
    grg_get_all_grid_addresses(gr_grid_addresses, D_diag);
}

long ph3py_get_reciprocal_rotations(long rec_rotations[48][3][3],
                                    const long (*rotations)[3][3],
                                    const long num_rot,
                                    const long is_time_reversal) {
    return grg_get_reciprocal_point_group(rec_rotations, rotations, num_rot,
                                          is_time_reversal, 1);
}

/* Rotation matrices with respect to reciprocal basis vectors are
 * transformed to those for GRGrid. This set of the rotations are
 * used always in GRGrid handling. */
long ph3py_transform_rotations(long (*transformed_rots)[3][3],
                               const long (*rotations)[3][3],
                               const long num_rot, const long D_diag[3],
                               const long Q[3][3]) {
    return grg_transform_rotations(transformed_rots, rotations, num_rot, D_diag,
                                   Q);
}

long ph3py_get_snf3x3(long D_diag[3], long P[3][3], long Q[3][3],
                      const long A[3][3]) {
    return grg_get_snf3x3(D_diag, P, Q, A);
}

/* The rotations are those after proper transformation in GRGrid. */
long ph3py_get_ir_grid_map(long *ir_grid_map, const long D_diag[3],
                           const long PS[3], const long (*grg_rotations)[3][3],
                           const long num_rot) {
    long num_ir, i;

    grg_get_ir_grid_map(ir_grid_map, grg_rotations, num_rot, D_diag, PS);

    num_ir = 0;
    for (i = 0; i < D_diag[0] * D_diag[1] * D_diag[2]; i++) {
        if (ir_grid_map[i] == i) {
            num_ir++;
        }
    }

    return num_ir;
}

long ph3py_get_bz_grid_addresses(long (*bz_grid_addresses)[3], long *bz_map,
                                 long *bzg2grg, const long D_diag[3],
                                 const long Q[3][3], const long PS[3],
                                 const double rec_lattice[3][3],
                                 const long type) {
    BZGrid *bzgrid;
    long i, j, size;

    if ((bzgrid = (BZGrid *)malloc(sizeof(BZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->bzg2grg = bzg2grg;
    bzgrid->type = type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = PS[i];
        for (j = 0; j < 3; j++) {
            bzgrid->Q[i][j] = Q[i][j];
            bzgrid->reclat[i][j] = rec_lattice[i][j];
        }
    }

    if (bzg_get_bz_grid_addresses(bzgrid)) {
        size = bzgrid->size;
    } else {
        size = 0;
    }

    free(bzgrid);
    bzgrid = NULL;

    return size;
}

long ph3py_rotate_bz_grid_index(const long bz_grid_index,
                                const long rotation[3][3],
                                const long (*bz_grid_addresses)[3],
                                const long *bz_map, const long D_diag[3],
                                const long PS[3], const long bz_grid_type) {
    ConstBZGrid *bzgrid;
    long i, rot_bz_gp;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = PS[i];
    }

    rot_bz_gp = bzg_rotate_grid_index(bz_grid_index, rotation, bzgrid);

    free(bzgrid);
    bzgrid = NULL;

    return rot_bz_gp;
}

void ph3py_symmetrize_collision_matrix(double *collision_matrix,
                                       const long num_column,
                                       const long num_temp,
                                       const long num_sigma) {
    double val;
    long i, j, k, l, adrs_shift;

    for (i = 0; i < num_sigma; i++) {
        for (j = 0; j < num_temp; j++) {
            adrs_shift = (i * num_column * num_column * num_temp +
                          j * num_column * num_column);
            /* show_colmat_info(py_collision_matrix, i, j, adrs_shift); */
#ifdef _OPENMP
#pragma omp parallel for schedule(guided) private(l, val)
#endif
            for (k = 0; k < num_column; k++) {
                for (l = k + 1; l < num_column; l++) {
                    val = (collision_matrix[adrs_shift + k * num_column + l] +
                           collision_matrix[adrs_shift + l * num_column + k]) /
                          2;
                    collision_matrix[adrs_shift + k * num_column + l] = val;
                    collision_matrix[adrs_shift + l * num_column + k] = val;
                }
            }
        }
    }
}

void ph3py_expand_collision_matrix(double *collision_matrix,
                                   const long *rot_grid_points,
                                   const long *ir_grid_points,
                                   const long num_ir_gp,
                                   const long num_grid_points,
                                   const long num_rot, const long num_sigma,
                                   const long num_temp, const long num_band)

{
    long i, j, k, l, m, n, p, adrs_shift, adrs_shift_plus, ir_gp, gp_r;
    long num_column, num_bgb;
    long *multi;
    double *colmat_copy;

    multi = (long *)malloc(sizeof(long) * num_ir_gp);
    colmat_copy = NULL;

    num_column = num_grid_points * num_band;
    num_bgb = num_band * num_grid_points * num_band;

#ifdef _OPENMP
#pragma omp parallel for schedule(guided) private(j, ir_gp)
#endif
    for (i = 0; i < num_ir_gp; i++) {
        ir_gp = ir_grid_points[i];
        multi[i] = 0;
        for (j = 0; j < num_rot; j++) {
            if (rot_grid_points[j * num_grid_points + ir_gp] == ir_gp) {
                multi[i]++;
            }
        }
    }

    for (i = 0; i < num_sigma; i++) {
        for (j = 0; j < num_temp; j++) {
            adrs_shift = (i * num_column * num_column * num_temp +
                          j * num_column * num_column);
#ifdef _OPENMP
#pragma omp parallel for private(ir_gp, adrs_shift_plus, colmat_copy, l, gp_r, \
                                     m, n, p)
#endif
            for (k = 0; k < num_ir_gp; k++) {
                ir_gp = ir_grid_points[k];
                adrs_shift_plus = adrs_shift + ir_gp * num_bgb;
                colmat_copy = (double *)malloc(sizeof(double) * num_bgb);
                for (l = 0; l < num_bgb; l++) {
                    colmat_copy[l] =
                        collision_matrix[adrs_shift_plus + l] / multi[k];
                    collision_matrix[adrs_shift_plus + l] = 0;
                }
                for (l = 0; l < num_rot; l++) {
                    gp_r = rot_grid_points[l * num_grid_points + ir_gp];
                    for (m = 0; m < num_band; m++) {
                        for (n = 0; n < num_grid_points; n++) {
                            for (p = 0; p < num_band; p++) {
                                collision_matrix
                                    [adrs_shift + gp_r * num_bgb +
                                     m * num_grid_points * num_band +
                                     rot_grid_points[l * num_grid_points + n] *
                                         num_band +
                                     p] +=
                                    colmat_copy[m * num_grid_points * num_band +
                                                n * num_band + p];
                            }
                        }
                    }
                }
                free(colmat_copy);
                colmat_copy = NULL;
            }
        }
    }

    free(multi);
    multi = NULL;
}

/**
 * @brief Get relative grid addresses needed for computing tetrahedron
 * integration weights
 *
 * @param relative_grid_address
 * @param reciprocal_lattice
 */
void ph3py_get_relative_grid_address(long relative_grid_address[24][4][3],
                                     const double reciprocal_lattice[3][3]) {
    thm_get_relative_grid_address(relative_grid_address, reciprocal_lattice);
}

/* tpi_get_neighboring_grid_points around multiple grid points for using
 * openmp
 *
 * relative_grid_addresses are given as P multipled with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
long ph3py_get_neighboring_gird_points(
    long *relative_grid_points, const long *grid_points,
    const long (*relative_grid_address)[3], const long D_diag[3],
    const long (*bz_grid_addresses)[3], const long *bz_map,
    const long bz_grid_type, const long num_grid_points,
    const long num_relative_grid_address) {
    long i;
    ConstBZGrid *bzgrid;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_grid_points; i++) {
        tpi_get_neighboring_grid_points(
            relative_grid_points + i * num_relative_grid_address,
            grid_points[i], relative_grid_address, num_relative_grid_address,
            bzgrid);
    }

    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

/* thm_get_integration_weight at multiple grid points for using openmp
 *
 * relative_grid_addresses are given as P multipled with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
long ph3py_get_thm_integration_weights_at_grid_points(
    double *iw, const double *frequency_points, const long num_frequency_points,
    const long num_band, const long num_gp,
    const long (*relative_grid_address)[4][3], const long D_diag[3],
    const long *grid_points, const long (*bz_grid_addresses)[3],
    const long *bz_map, const long bz_grid_type, const double *frequencies,
    const long *gp2irgp_map, const char function) {
    long i, j, k, bi;
    long vertices[24][4];
    double freq_vertices[24][4];
    ConstBZGrid *bzgrid;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
    }

#ifdef _OPENMP
#pragma omp parallel for private(j, k, bi, vertices, freq_vertices)
#endif
    for (i = 0; i < num_gp; i++) {
        for (j = 0; j < 24; j++) {
            tpi_get_neighboring_grid_points(vertices[j], grid_points[i],
                                            relative_grid_address[j], 4,
                                            bzgrid);
        }
        for (bi = 0; bi < num_band; bi++) {
            for (j = 0; j < 24; j++) {
                for (k = 0; k < 4; k++) {
                    freq_vertices[j][k] =
                        frequencies[gp2irgp_map[vertices[j][k]] * num_band +
                                    bi];
                }
            }
            for (j = 0; j < num_frequency_points; j++) {
                iw[i * num_frequency_points * num_band + j * num_band + bi] =
                    thm_get_integration_weight(frequency_points[j],
                                               freq_vertices, function);
            }
        }
    }

    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

long ph3py_phonopy_dsyev(double *data, double *eigvals, const long size,
                         const long algorithm) {
    return (long)phonopy_dsyev(data, eigvals, (int)size, (int)algorithm);
}

long ph3py_phonopy_pinv(double *data_out, const double *data_in, const long m,
                        const long n, const double cutoff) {
    return (long)phonopy_pinv(data_out, data_in, (int)m, (int)n, cutoff);
}

void ph3py_pinv_from_eigensolution(double *data, const double *eigvals,
                                   const long size, const double cutoff,
                                   const long pinv_method) {
    pinv_from_eigensolution(data, eigvals, size, cutoff, pinv_method);
}

long ph3py_get_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}
