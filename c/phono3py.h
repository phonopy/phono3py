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

#include "phonoc_array.h"

typedef struct {
    double re;
    double im;
} _lapack_complex_double;

long ph3py_get_interaction(
    Darray *fc3_normal_squared, const char *g_zero, const Darray *frequencies,
    const _lapack_complex_double *eigenvectors, const long (*triplets)[3],
    const long num_triplets, const long (*bz_grid_addresses)[3],
    const long D_diag[3], const long Q[3][3], const double *fc3,
    const long is_compact_fc3, const double (*svecs)[3],
    const long multi_dims[2], const long (*multi)[2], const double *masses,
    const long *p2s_map, const long *s2p_map, const long *band_indices,
    const long symmetrize_fc3_q, const long make_r0_average,
    const char *all_shortest, const double cutoff_frequency,
    const long openmp_per_triplets);
long ph3py_get_pp_collision(
    double *imag_self_energy,
    const long relative_grid_address[24][4][3], /* thm */
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const long (*bz_grid_addresses)[3], /* thm */
    const long *bz_map,                                              /* thm */
    const long bz_grid_type, const long D_diag[3], const long Q[3][3],
    const double *fc3, const long is_compact_fc3, const double (*svecs)[3],
    const long multi_dims[2], const long (*multi)[2], const double *masses,
    const long *p2s_map, const long *s2p_map, const Larray *band_indices,
    const Darray *temperatures, const long is_NU, const long symmetrize_fc3_q,
    const long make_r0_average, const char *all_shortest,
    const double cutoff_frequency, const long openmp_per_triplets);
long ph3py_get_pp_collision_with_sigma(
    double *imag_self_energy, const double sigma, const double sigma_cutoff,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const long (*bz_grid_addresses)[3],
    const long D_diag[3], const long Q[3][3], const double *fc3,
    const long is_compact_fc3, const double (*svecs)[3],
    const long multi_dims[2], const long (*multi)[2], const double *masses,
    const long *p2s_map, const long *s2p_map, const Larray *band_indices,
    const Darray *temperatures, const long is_NU, const long symmetrize_fc3_q,
    const long make_r0_average, const char *all_shortest,
    const double cutoff_frequency, const long openmp_per_triplets);
void ph3py_get_imag_self_energy_at_bands_with_g(
    double *imag_self_energy, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const double *g, const char *g_zero,
    const double temperature, const double cutoff_frequency,
    const long num_frequency_points, const long frequency_point_index);
void ph3py_get_detailed_imag_self_energy_at_bands_with_g(
    double *detailed_imag_self_energy, double *imag_self_energy_N,
    double *imag_self_energy_U, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const long (*bz_grid_addresses)[3],
    const double *g, const char *g_zero, const double temperature,
    const double cutoff_frequency);
void ph3py_get_real_self_energy_at_bands(
    double *real_self_energy, const Darray *fc3_normal_squared,
    const long *band_indices, const double *frequencies,
    const long (*triplets)[3], const long *triplet_weights,
    const double epsilon, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency);
void ph3py_get_real_self_energy_at_frequency_point(
    double *real_self_energy, const double frequency_point,
    const Darray *fc3_normal_squared, const long *band_indices,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const double epsilon, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency);
void ph3py_get_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplets_map, const long *map_q,
    const long *rotated_grid_points, const double *rotations_cartesian,
    const double *g, const long num_ir_gp, const long num_gp,
    const long num_rot, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency);
void ph3py_get_reducible_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplets_map, const long *map_q, const double *g,
    const long num_gp, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency);
void ph3py_get_isotope_scattering_strength(
    double *gamma, const long grid_point, const long *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long num_ir_grid_points, const long *band_indices,
    const long num_band, const long num_band0, const double sigma,
    const double cutoff_frequency);
void ph3py_get_thm_isotope_scattering_strength(
    double *gamma, const long grid_point, const long *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const _lapack_complex_double *eigenvectors,
    const long num_ir_grid_points, const long *band_indices,
    const long num_band, const long num_band0,
    const double *integration_weights, const double cutoff_frequency);
void ph3py_distribute_fc3(double *fc3, const long target, const long source,
                          const long *atom_mapping, const long num_atom,
                          const double *rot_cart);
void ph3py_rotate_delta_fc2(double (*fc3)[3][3][3],
                            const double (*delta_fc2s)[3][3],
                            const double *inv_U,
                            const double (*site_sym_cart)[3][3],
                            const long *rot_map_syms, const long num_atom,
                            const long num_site_sym, const long num_disp);
void ph3py_get_permutation_symmetry_fc3(double *fc3, const long num_atom);
void ph3py_get_permutation_symmetry_compact_fc3(
    double *fc3, const long p2s[], const long s2pp[], const long nsym_list[],
    const long perms[], const long n_satom, const long n_patom);
void ph3py_transpose_compact_fc3(double *fc3, const long p2s[],
                                 const long s2pp[], const long nsym_list[],
                                 const long perms[], const long n_satom,
                                 const long n_patom, const long t_type);
long ph3py_get_triplets_reciprocal_mesh_at_q(
    long *map_triplets, long *map_q, const long grid_point, const long mesh[3],
    const long is_time_reversal, const long num_rot,
    const long (*rec_rotations)[3][3], const long swappable);
long ph3py_get_BZ_triplets_at_q(long (*triplets)[3], const long grid_point,
                                const long (*bz_grid_addresses)[3],
                                const long *bz_map, const long *map_triplets,
                                const long num_map_triplets,
                                const long D_diag[3], const long Q[3][3],
                                const long bz_grid_type);
long ph3py_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long relative_grid_address[24][4][3],
    const long mesh[3], const long (*triplets)[3], const long num_triplets,
    const long (*bz_grid_addresses)[3], const long *bz_map,
    const long bz_grid_type, const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets);
void ph3py_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const long num_band0,
    const long (*triplets)[3], const long num_triplets,
    const double *frequencies, const long num_band, const long tp_type);
long ph3py_get_grid_index_from_address(const long address[3],
                                       const long mesh[3]);
void ph3py_get_gr_grid_addresses(long gr_grid_addresses[][3],
                                 const long D_diag[3]);
long ph3py_get_reciprocal_rotations(long rec_rotations[48][3][3],
                                    const long (*rotations)[3][3],
                                    const long num_rot,
                                    const long is_time_reversal);
long ph3py_transform_rotations(long (*transformed_rots)[3][3],
                               const long (*rotations)[3][3],
                               const long num_rot, const long D_diag[3],
                               const long Q[3][3]);
long ph3py_get_snf3x3(long D_diag[3], long P[3][3], long Q[3][3],
                      const long A[3][3]);
long ph3py_transform_rotations(long (*transformed_rots)[3][3],
                               const long (*rotations)[3][3],
                               const long num_rot, const long D_diag[3],
                               const long Q[3][3]);
long ph3py_get_ir_grid_map(long *ir_grid_map, const long D_diag[3],
                           const long PS[3], const long (*grg_rotations)[3][3],
                           const long num_rot);
long ph3py_get_bz_grid_addresses(long (*bz_grid_addresses)[3], long *bz_map,
                                 long *bzg2grg, const long D_diag[3],
                                 const long Q[3][3], const long PS[3],
                                 const double rec_lattice[3][3],
                                 const long type);
long ph3py_rotate_bz_grid_index(const long bz_grid_index,
                                const long rotation[3][3],
                                const long (*bz_grid_addresses)[3],
                                const long *bz_map, const long D_diag[3],
                                const long PS[3], const long bz_grid_type);
void ph3py_symmetrize_collision_matrix(double *collision_matrix,
                                       const long num_column,
                                       const long num_temp,
                                       const long num_sigma);
void ph3py_expand_collision_matrix(double *collision_matrix,
                                   const long *rot_grid_points,
                                   const long *ir_grid_points,
                                   const long num_ir_gp,
                                   const long num_grid_points,
                                   const long num_rot, const long num_sigma,
                                   const long num_temp, const long num_band);
void ph3py_get_relative_grid_address(long relative_grid_address[24][4][3],
                                     const double reciprocal_lattice[3][3]);
long ph3py_get_neighboring_gird_points(
    long *relative_grid_points, const long *grid_points,
    const long (*relative_grid_address)[3], const long mesh[3],
    const long (*bz_grid_addresses)[3], const long *bz_map,
    const long bz_grid_type, const long num_grid_points,
    const long num_relative_grid_address);
long ph3py_get_thm_integration_weights_at_grid_points(
    double *iw, const double *frequency_points, const long num_band0,
    const long num_band, const long num_gp,
    const long (*relative_grid_address)[4][3], const long D_diag[3],
    const long *grid_points, const long (*bz_grid_addresses)[3],
    const long *bz_map, const long bz_grid_type, const double *frequencies,
    const long *gp2irgp_map, const char function);
long ph3py_phonopy_dsyev(double *data, double *eigvals, const long size,
                         const long algorithm);
long ph3py_phonopy_pinv(double *data_out, const double *data_in, const long m,
                        const long n, const double cutoff);
void ph3py_pinv_from_eigensolution(double *data, const double *eigvals,
                                   const long size, const double cutoff,
                                   const long pinv_method);
long ph3py_get_max_threads(void);

#ifdef __cplusplus
}
#endif

#endif
