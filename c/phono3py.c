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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "collision_matrix.h"
#include "fc3.h"
#include "imag_self_energy_with_g.h"
#include "interaction.h"
#include "isotope.h"
#include "lagrid.h"
#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "pp_collision.h"
#include "real_self_energy.h"
#include "real_to_reciprocal.h"
#include "recgrid.h"
#include "tetrahedron_method.h"
#include "triplet.h"
#include "triplet_iw.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int64_t ph3py_get_interaction(
    Darray *fc3_normal_squared, const char *g_zero, const Darray *frequencies,
    const _lapack_complex_double *eigenvectors, const int64_t (*triplets)[3],
    const int64_t num_triplets, const int64_t (*bz_grid_addresses)[3],
    const int64_t D_diag[3], const int64_t Q[3][3], const double *fc3,
    const char *fc3_nonzero_indices, const int64_t is_compact_fc3,
    const double (*svecs)[3], const int64_t multi_dims[2],
    const int64_t (*multiplicity)[2], const double *masses,
    const int64_t *p2s_map, const int64_t *s2p_map, const int64_t *band_indices,
    const int64_t symmetrize_fc3_q, const int64_t make_r0_average,
    const char *all_shortest, const double cutoff_frequency,
    const int64_t openmp_per_triplets) {
    RecgridConstBZGrid *bzgrid;
    AtomTriplets *atom_triplets;
    int64_t i, j;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
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
    atom_triplets->nonzero_indices = fc3_nonzero_indices;

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

int64_t ph3py_get_pp_collision(
    double *imag_self_energy,
    const int64_t relative_grid_address[24][4][3], /* thm */
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const int64_t *triplet_weights,
    const int64_t (*bz_grid_addresses)[3], /* thm */
    const int64_t *bz_map,                 /* thm */
    const int64_t bz_grid_type, const int64_t D_diag[3], const int64_t Q[3][3],
    const double *fc3, const char *fc3_nonzero_indices,
    const int64_t is_compact_fc3, const double (*svecs)[3],
    const int64_t multi_dims[2], const int64_t (*multiplicity)[2],
    const double *masses, const int64_t *p2s_map, const int64_t *s2p_map,
    const Larray *band_indices, const Darray *temperatures_THz,
    const int64_t is_NU, const int64_t symmetrize_fc3_q,
    const int64_t make_r0_average, const char *all_shortest,
    const double cutoff_frequency, const int64_t openmp_per_triplets) {
    RecgridConstBZGrid *bzgrid;
    AtomTriplets *atom_triplets;
    int64_t i, j;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
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
    atom_triplets->nonzero_indices = fc3_nonzero_indices;

    ppc_get_pp_collision(imag_self_energy, relative_grid_address, frequencies,
                         (lapack_complex_double *)eigenvectors, triplets,
                         num_triplets, triplet_weights, bzgrid, fc3,
                         is_compact_fc3, atom_triplets, masses, band_indices,
                         temperatures_THz, is_NU, symmetrize_fc3_q,
                         cutoff_frequency, openmp_per_triplets);

    free(atom_triplets);
    atom_triplets = NULL;

    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

int64_t ph3py_get_pp_collision_with_sigma(
    double *imag_self_energy, const double sigma, const double sigma_cutoff,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const int64_t *triplet_weights, const int64_t (*bz_grid_addresses)[3],
    const int64_t D_diag[3], const int64_t Q[3][3], const double *fc3,
    const char *fc3_nonzero_indices, const int64_t is_compact_fc3,
    const double (*svecs)[3], const int64_t multi_dims[2],
    const int64_t (*multiplicity)[2], const double *masses,
    const int64_t *p2s_map, const int64_t *s2p_map, const Larray *band_indices,
    const Darray *temperatures_THz, const int64_t is_NU,
    const int64_t symmetrize_fc3_q, const int64_t make_r0_average,
    const char *all_shortest, const double cutoff_frequency,
    const int64_t openmp_per_triplets) {
    RecgridConstBZGrid *bzgrid;
    AtomTriplets *atom_triplets;
    int64_t i, j;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
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
    atom_triplets->nonzero_indices = fc3_nonzero_indices;

    ppc_get_pp_collision_with_sigma(
        imag_self_energy, sigma, sigma_cutoff, frequencies,
        (lapack_complex_double *)eigenvectors, triplets, num_triplets,
        triplet_weights, bzgrid, fc3, is_compact_fc3, atom_triplets, masses,
        band_indices, temperatures_THz, is_NU, symmetrize_fc3_q,
        cutoff_frequency, openmp_per_triplets);

    free(atom_triplets);
    atom_triplets = NULL;

    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

void ph3py_get_imag_self_energy_at_bands_with_g(
    double *imag_self_energy, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const double *g, const char *g_zero,
    const double temperature_THz, const double cutoff_frequency,
    const int64_t num_frequency_points, const int64_t frequency_point_index) {
    ise_get_imag_self_energy_with_g(
        imag_self_energy, fc3_normal_squared, frequencies, triplets,
        triplet_weights, g, g_zero, temperature_THz, cutoff_frequency,
        num_frequency_points, frequency_point_index);
}

void ph3py_get_detailed_imag_self_energy_at_bands_with_g(
    double *detailed_imag_self_energy, double *imag_self_energy_N,
    double *imag_self_energy_U, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const int64_t (*bz_grid_addresses)[3],
    const double *g, const char *g_zero, const double temperature_THz,
    const double cutoff_frequency) {
    ise_get_detailed_imag_self_energy_with_g(
        detailed_imag_self_energy, imag_self_energy_N, imag_self_energy_U,
        fc3_normal_squared, frequencies, triplets, triplet_weights,
        bz_grid_addresses, g, g_zero, temperature_THz, cutoff_frequency);
}

void ph3py_get_real_self_energy_at_bands(
    double *real_self_energy, const Darray *fc3_normal_squared,
    const int64_t *band_indices, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplet_weights,
    const double epsilon, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency) {
    rse_get_real_self_energy_at_bands(real_self_energy, fc3_normal_squared,
                                      band_indices, frequencies, triplets,
                                      triplet_weights, epsilon, temperature_THz,
                                      unit_conversion_factor, cutoff_frequency);
}

void ph3py_get_real_self_energy_at_frequency_point(
    double *real_self_energy, const double frequency_point,
    const Darray *fc3_normal_squared, const int64_t *band_indices,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const double epsilon,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency) {
    rse_get_real_self_energy_at_frequency_point(
        real_self_energy, frequency_point, fc3_normal_squared, band_indices,
        frequencies, triplets, triplet_weights, epsilon, temperature_THz,
        unit_conversion_factor, cutoff_frequency);
}

void ph3py_get_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q,
    const int64_t *rotated_grid_points, const double *rotations_cartesian,
    const double *g, const int64_t num_ir_gp, const int64_t num_gp,
    const int64_t num_rot, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency) {
    col_get_collision_matrix(collision_matrix, fc3_normal_squared, frequencies,
                             triplets, triplets_map, map_q, rotated_grid_points,
                             rotations_cartesian, g, num_ir_gp, num_gp, num_rot,
                             temperature_THz, unit_conversion_factor,
                             cutoff_frequency);
}

void ph3py_get_reducible_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q, const double *g,
    const int64_t num_gp, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency) {
    col_get_reducible_collision_matrix(
        collision_matrix, fc3_normal_squared, frequencies, triplets,
        triplets_map, map_q, g, num_gp, temperature_THz, unit_conversion_factor,
        cutoff_frequency);
}

void ph3py_get_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const int64_t num_ir_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0, const double sigma,
    const double cutoff_frequency) {
    iso_get_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        (lapack_complex_double *)eigenvectors, num_ir_grid_points, band_indices,
        num_band, num_band0, sigma, cutoff_frequency);
}

void ph3py_get_thm_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const int64_t num_ir_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0,
    const double *integration_weights, const double cutoff_frequency) {
    iso_get_thm_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        (lapack_complex_double *)eigenvectors, num_ir_grid_points, band_indices,
        num_band, num_band0, integration_weights, cutoff_frequency);
}

void ph3py_distribute_fc3(double *fc3, const int64_t target,
                          const int64_t source, const int64_t *atom_mapping,
                          const int64_t num_atom, const double *rot_cart) {
    fc3_distribute_fc3(fc3, target, source, atom_mapping, num_atom, rot_cart);
}

void ph3py_rotate_delta_fc2(double (*fc3)[3][3][3],
                            const double (*delta_fc2s)[3][3],
                            const double *inv_U,
                            const double (*site_sym_cart)[3][3],
                            const int64_t *rot_map_syms, const int64_t num_atom,
                            const int64_t num_site_sym,
                            const int64_t num_disp) {
    fc3_rotate_delta_fc2(fc3, delta_fc2s, inv_U, site_sym_cart, rot_map_syms,
                         num_atom, num_site_sym, num_disp);
}

void ph3py_get_permutation_symmetry_fc3(double *fc3, const int64_t num_atom) {
    fc3_set_permutation_symmetry_fc3(fc3, num_atom);
}

void ph3py_get_permutation_symmetry_compact_fc3(
    double *fc3, const int64_t p2s[], const int64_t s2pp[],
    const int64_t nsym_list[], const int64_t perms[], const int64_t n_satom,
    const int64_t n_patom) {
    fc3_set_permutation_symmetry_compact_fc3(fc3, p2s, s2pp, nsym_list, perms,
                                             n_satom, n_patom);
}

void ph3py_transpose_compact_fc3(double *fc3, const int64_t p2s[],
                                 const int64_t s2pp[],
                                 const int64_t nsym_list[],
                                 const int64_t perms[], const int64_t n_satom,
                                 const int64_t n_patom, const int64_t t_type) {
    fc3_transpose_compact_fc3(fc3, p2s, s2pp, nsym_list, perms, n_satom,
                              n_patom, t_type);
}

int64_t ph3py_get_triplets_reciprocal_mesh_at_q(
    int64_t *map_triplets, int64_t *map_q, const int64_t grid_point,
    const int64_t D_diag[3], const int64_t is_time_reversal,
    const int64_t num_rot, const int64_t (*rec_rotations)[3][3],
    const int64_t swappable) {
    return tpl_get_triplets_reciprocal_mesh_at_q(
        map_triplets, map_q, grid_point, D_diag, is_time_reversal, num_rot,
        rec_rotations, swappable);
}

int64_t ph3py_get_BZ_triplets_at_q(
    int64_t (*triplets)[3], const int64_t grid_point,
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t *map_triplets, const int64_t num_map_triplets,
    const int64_t D_diag[3], const int64_t Q[3][3],
    const double reciprocal_lattice[3][3], const int64_t bz_grid_type) {
    RecgridConstBZGrid *bzgrid;
    int64_t i, j, num_ir;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
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
            bzgrid->reclat[i][j] = reciprocal_lattice[i][j];
        }
    }
    bzgrid->size = num_map_triplets;

    num_ir =
        tpl_get_BZ_triplets_at_q(triplets, grid_point, bzgrid, map_triplets);
    free(bzgrid);
    bzgrid = NULL;

    return num_ir;
}

/* relative_grid_addresses are given as P multiplied with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
int64_t ph3py_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const int64_t num_band0, const int64_t relative_grid_address[24][4][3],
    const int64_t D_diag[3], const int64_t (*triplets)[3],
    const int64_t num_triplets, const int64_t (*bz_grid_addresses)[3],
    const int64_t *bz_map, const int64_t bz_grid_type,
    const double *frequencies1, const int64_t num_band1,
    const double *frequencies2, const int64_t num_band2, const int64_t tp_type,
    const int64_t openmp_per_triplets) {
    RecgridConstBZGrid *bzgrid;
    int64_t i;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
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
    const double *frequency_points, const int64_t num_band0,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const double *frequencies, const int64_t num_band, const int64_t tp_type) {
    tpl_get_integration_weight_with_sigma(
        iw, iw_zero, sigma, sigma_cutoff, frequency_points, num_band0, triplets,
        num_triplets, frequencies, num_band, tp_type);
}

void ph3py_symmetrize_collision_matrix(double *collision_matrix,
                                       const int64_t num_column,
                                       const int64_t num_temp,
                                       const int64_t num_sigma) {
    double val;
    int64_t i, j, k, l, adrs_shift;

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

void ph3py_expand_collision_matrix(
    double *collision_matrix, const int64_t *rot_grid_points,
    const int64_t *ir_grid_points, const int64_t num_ir_gp,
    const int64_t num_grid_points, const int64_t num_rot,
    const int64_t num_sigma, const int64_t num_temp, const int64_t num_band)

{
    int64_t i, j, k, l, m, n, p, adrs_shift, adrs_shift_plus, ir_gp, gp_r;
    int64_t num_column, num_bgb;
    int64_t *multi;
    double *colmat_copy;

    multi = (int64_t *)malloc(sizeof(int64_t) * num_ir_gp);
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
void ph3py_get_relative_grid_address(int64_t relative_grid_address[24][4][3],
                                     const double reciprocal_lattice[3][3]) {
    thm_get_relative_grid_address(relative_grid_address, reciprocal_lattice);
}

/* tpi_get_neighboring_grid_points around multiple grid points for using
 * openmp
 *
 * relative_grid_addresses are given as P multiplied with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
int64_t ph3py_get_neighboring_gird_points(
    int64_t *relative_grid_points, const int64_t *grid_points,
    const int64_t (*relative_grid_address)[3], const int64_t D_diag[3],
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t bz_grid_type, const int64_t num_grid_points,
    const int64_t num_relative_grid_address) {
    int64_t i;
    RecgridConstBZGrid *bzgrid;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
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
 * relative_grid_addresses are given as P multiplied with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
int64_t ph3py_get_thm_integration_weights_at_grid_points(
    double *iw, const double *frequency_points,
    const int64_t num_frequency_points, const int64_t num_band,
    const int64_t num_gp, const int64_t (*relative_grid_address)[4][3],
    const int64_t D_diag[3], const int64_t *grid_points,
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t bz_grid_type, const double *frequencies,
    const int64_t *gp2irgp_map, const char function) {
    int64_t i, j, k, bi;
    int64_t vertices[24][4];
    double freq_vertices[24][4];
    RecgridConstBZGrid *bzgrid;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
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

int64_t ph3py_get_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

#ifndef NO_INCLUDE_LAPACKE
int64_t ph3py_phonopy_dsyev(double *data, double *eigvals, const int64_t size,
                            const int64_t algorithm) {
    return (int64_t)phonopy_dsyev(data, eigvals, (int)size, (int)algorithm);
}

int64_t ph3py_phonopy_pinv(double *data_out, const double *data_in,
                           const int64_t m, const int64_t n,
                           const double cutoff) {
    return (int64_t)phonopy_pinv(data_out, data_in, (int)m, (int)n, cutoff);
}

void ph3py_pinv_from_eigensolution(double *data, const double *eigvals,
                                   const int64_t size, const double cutoff,
                                   const int64_t pinv_method) {
    pinv_from_eigensolution(data, eigvals, size, cutoff, pinv_method);
}
#endif
