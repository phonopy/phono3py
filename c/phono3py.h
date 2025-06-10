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

#ifndef __phono3py_H__
#define __phono3py_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

#include "phonoc_array.h"

typedef struct {
    double re;
    double im;
} _lapack_complex_double;

int64_t ph3py_get_interaction(
    Darray *fc3_normal_squared, const char *g_zero, const Darray *frequencies,
    const _lapack_complex_double *eigenvectors, const int64_t (*triplets)[3],
    const int64_t num_triplets, const int64_t (*bz_grid_addresses)[3],
    const int64_t D_diag[3], const int64_t Q[3][3], const double *fc3,
    const char *fc3_nonzero_indices, const int64_t is_compact_fc3,
    const double (*svecs)[3], const int64_t multi_dims[2],
    const int64_t (*multi)[2], const double *masses, const int64_t *p2s_map,
    const int64_t *s2p_map, const int64_t *band_indices,
    const int64_t symmetrize_fc3_q, const int64_t make_r0_average,
    const char *all_shortest, const double cutoff_frequency,
    const int64_t openmp_per_triplets);
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
    const int64_t multi_dims[2], const int64_t (*multi)[2],
    const double *masses, const int64_t *p2s_map, const int64_t *s2p_map,
    const Larray *band_indices, const Darray *temperatures_THz,
    const int64_t is_NU, const int64_t symmetrize_fc3_q,
    const int64_t make_r0_average, const char *all_shortest,
    const double cutoff_frequency, const int64_t openmp_per_triplets);
int64_t ph3py_get_pp_collision_with_sigma(
    double *imag_self_energy, const double sigma, const double sigma_cutoff,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const int64_t *triplet_weights, const int64_t (*bz_grid_addresses)[3],
    const int64_t D_diag[3], const int64_t Q[3][3], const double *fc3,
    const char *fc3_nonzero_indices, const int64_t is_compact_fc3,
    const double (*svecs)[3], const int64_t multi_dims[2],
    const int64_t (*multi)[2], const double *masses, const int64_t *p2s_map,
    const int64_t *s2p_map, const Larray *band_indices,
    const Darray *temperatures_THz, const int64_t is_NU,
    const int64_t symmetrize_fc3_q, const int64_t make_r0_average,
    const char *all_shortest, const double cutoff_frequency,
    const int64_t openmp_per_triplets);
void ph3py_get_imag_self_energy_at_bands_with_g(
    double *imag_self_energy, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const double *g, const char *g_zero,
    const double temperature_THz, const double cutoff_frequency,
    const int64_t num_frequency_points, const int64_t frequency_point_index);
void ph3py_get_detailed_imag_self_energy_at_bands_with_g(
    double *detailed_imag_self_energy, double *imag_self_energy_N,
    double *imag_self_energy_U, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const int64_t (*bz_grid_addresses)[3],
    const double *g, const char *g_zero, const double temperature_THz,
    const double cutoff_frequency);
void ph3py_get_real_self_energy_at_bands(
    double *real_self_energy, const Darray *fc3_normal_squared,
    const int64_t *band_indices, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplet_weights,
    const double epsilon, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency);
void ph3py_get_real_self_energy_at_frequency_point(
    double *real_self_energy, const double frequency_point,
    const Darray *fc3_normal_squared, const int64_t *band_indices,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const double epsilon,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency);
void ph3py_get_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q,
    const int64_t *rotated_grid_points, const double *rotations_cartesian,
    const double *g, const int64_t num_ir_gp, const int64_t num_gp,
    const int64_t num_rot, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency);
void ph3py_get_reducible_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q, const double *g,
    const int64_t num_gp, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency);
void ph3py_get_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const int64_t num_ir_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0, const double sigma,
    const double cutoff_frequency);
void ph3py_get_thm_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const int64_t num_ir_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0,
    const double *integration_weights, const double cutoff_frequency);
void ph3py_distribute_fc3(double *fc3, const int64_t target,
                          const int64_t source, const int64_t *atom_mapping,
                          const int64_t num_atom, const double *rot_cart);
void ph3py_rotate_delta_fc2(double (*fc3)[3][3][3],
                            const double (*delta_fc2s)[3][3],
                            const double *inv_U,
                            const double (*site_sym_cart)[3][3],
                            const int64_t *rot_map_syms, const int64_t num_atom,
                            const int64_t num_site_sym, const int64_t num_disp);
void ph3py_get_permutation_symmetry_fc3(double *fc3, const int64_t num_atom);
void ph3py_get_permutation_symmetry_compact_fc3(
    double *fc3, const int64_t p2s[], const int64_t s2pp[],
    const int64_t nsym_list[], const int64_t perms[], const int64_t n_satom,
    const int64_t n_patom);
void ph3py_transpose_compact_fc3(double *fc3, const int64_t p2s[],
                                 const int64_t s2pp[],
                                 const int64_t nsym_list[],
                                 const int64_t perms[], const int64_t n_satom,
                                 const int64_t n_patom, const int64_t t_type);
int64_t ph3py_get_triplets_reciprocal_mesh_at_q(
    int64_t *map_triplets, int64_t *map_q, const int64_t grid_point,
    const int64_t mesh[3], const int64_t is_time_reversal,
    const int64_t num_rot, const int64_t (*rec_rotations)[3][3],
    const int64_t swappable);
int64_t ph3py_get_BZ_triplets_at_q(
    int64_t (*triplets)[3], const int64_t grid_point,
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t *map_triplets, const int64_t num_map_triplets,
    const int64_t D_diag[3], const int64_t Q[3][3],
    const double reciprocal_lattice[3][3], const int64_t bz_grid_type);
int64_t ph3py_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const int64_t num_band0, const int64_t relative_grid_address[24][4][3],
    const int64_t mesh[3], const int64_t (*triplets)[3],
    const int64_t num_triplets, const int64_t (*bz_grid_addresses)[3],
    const int64_t *bz_map, const int64_t bz_grid_type,
    const double *frequencies1, const int64_t num_band1,
    const double *frequencies2, const int64_t num_band2, const int64_t tp_type,
    const int64_t openmp_per_triplets);
void ph3py_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const int64_t num_band0,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const double *frequencies, const int64_t num_band, const int64_t tp_type);
void ph3py_symmetrize_collision_matrix(double *collision_matrix,
                                       const int64_t num_column,
                                       const int64_t num_temp,
                                       const int64_t num_sigma);
void ph3py_expand_collision_matrix(
    double *collision_matrix, const int64_t *rot_grid_points,
    const int64_t *ir_grid_points, const int64_t num_ir_gp,
    const int64_t num_grid_points, const int64_t num_rot,
    const int64_t num_sigma, const int64_t num_temp, const int64_t num_band);
void ph3py_get_relative_grid_address(int64_t relative_grid_address[24][4][3],
                                     const double reciprocal_lattice[3][3]);
int64_t ph3py_get_neighboring_gird_points(
    int64_t *relative_grid_points, const int64_t *grid_points,
    const int64_t (*relative_grid_address)[3], const int64_t mesh[3],
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t bz_grid_type, const int64_t num_grid_points,
    const int64_t num_relative_grid_address);
int64_t ph3py_get_thm_integration_weights_at_grid_points(
    double *iw, const double *frequency_points, const int64_t num_band0,
    const int64_t num_band, const int64_t num_gp,
    const int64_t (*relative_grid_address)[4][3], const int64_t D_diag[3],
    const int64_t *grid_points, const int64_t (*bz_grid_addresses)[3],
    const int64_t *bz_map, const int64_t bz_grid_type,
    const double *frequencies, const int64_t *gp2irgp_map, const char function);
int64_t ph3py_get_max_threads(void);

#ifndef NO_INCLUDE_LAPACKE
int64_t ph3py_phonopy_dsyev(double *data, double *eigvals, const int64_t size,
                            const int64_t algorithm);
int64_t ph3py_phonopy_pinv(double *data_out, const double *data_in,
                           const int64_t m, const int64_t n,
                           const double cutoff);
void ph3py_pinv_from_eigensolution(double *data, const double *eigvals,
                                   const int64_t size, const double cutoff,
                                   const int64_t pinv_method);
#endif

#ifdef __cplusplus
}
#endif

#endif
