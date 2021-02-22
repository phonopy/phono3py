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

#ifndef PHPYCONST
#define PHPYCONST
#endif

#include "lapack_wrapper.h"
#include "phonoc_array.h"

void ph3py_get_interaction(Darray *fc3_normal_squared,
                           const char *g_zero,
                           const Darray *frequencies,
                           const lapack_complex_double *eigenvectors,
                           const size_t (*triplets)[3],
                           const size_t num_triplets,
                           const int *grid_address,
                           const int *mesh,
                           const double *fc3,
                           const int is_compact_fc3,
                           const double *shortest_vectors,
                           const int svecs_dims[3],
                           const int *multiplicity,
                           const double *masses,
                           const int *p2s_map,
                           const int *s2p_map,
                           const int *band_indices,
                           const int symmetrize_fc3_q,
                           const double cutoff_frequency);
void ph3py_get_pp_collision(double *imag_self_energy,
                            PHPYCONST int relative_grid_address[24][4][3], /* thm */
                            const double *frequencies,
                            const lapack_complex_double *eigenvectors,
                            const size_t (*triplets)[3],
                            const size_t num_triplets,
                            const int *triplet_weights,
                            const int *grid_address, /* thm */
                            const size_t *bz_map, /* thm */
                            const int *mesh, /* thm */
                            const double *fc3,
                            const int is_compact_fc3,
                            const double *shortest_vectors,
                            const int svecs_dims[3],
                            const int *multiplicity,
                            const double *masses,
                            const int *p2s_map,
                            const int *s2p_map,
                            const Iarray *band_indices,
                            const Darray *temperatures,
                            const int is_NU,
                            const int symmetrize_fc3_q,
                            const double cutoff_frequency);
void ph3py_get_pp_collision_with_sigma(
  double *imag_self_energy,
  const double sigma,
  const double sigma_cutoff,
  const double *frequencies,
  const lapack_complex_double *eigenvectors,
  const size_t (*triplets)[3],
  const size_t num_triplets,
  const int *triplet_weights,
  const int *grid_address,
  const int *mesh,
  const double *fc3,
  const int is_compact_fc3,
  const double *shortest_vectors,
  const int svecs_dims[3],
  const int *multiplicity,
  const double *masses,
  const int *p2s_map,
  const int *s2p_map,
  const Iarray *band_indices,
  const Darray *temperatures,
  const int is_NU,
  const int symmetrize_fc3_q,
  const double cutoff_frequency);
void ph3py_get_imag_self_energy_at_bands_with_g(
  double *imag_self_energy,
  const Darray *fc3_normal_squared,
  const double *frequencies,
  const size_t (*triplets)[3],
  const int *triplet_weights,
  const double *g,
  const char *g_zero,
  const double temperature,
  const double cutoff_frequency,
  const int num_frequency_points,
  const int frequency_point_index);
void ph3py_get_detailed_imag_self_energy_at_bands_with_g(
  double *detailed_imag_self_energy,
  double *imag_self_energy_N,
  double *imag_self_energy_U,
  const Darray *fc3_normal_squared,
  const double *frequencies,
  const size_t (*triplets)[3],
  const int *triplet_weights,
  const int *grid_address,
  const double *g,
  const char *g_zero,
  const double temperature,
  const double cutoff_frequency);
void ph3py_get_real_self_energy_at_bands(double *real_self_energy,
                                         const Darray *fc3_normal_squared,
                                         const int *band_indices,
                                         const double *frequencies,
                                         const size_t (*triplets)[3],
                                         const int *triplet_weights,
                                         const double epsilon,
                                         const double temperature,
                                         const double unit_conversion_factor,
                                         const double cutoff_frequency);
void ph3py_get_real_self_energy_at_frequency_point(
  double *real_self_energy,
  const double frequency_point,
  const Darray *fc3_normal_squared,
  const int *band_indices,
  const double *frequencies,
  const size_t (*triplets)[3],
  const int *triplet_weights,
  const double epsilon,
  const double temperature,
  const double unit_conversion_factor,
  const double cutoff_frequency);
void ph3py_get_collision_matrix(double *collision_matrix,
                                const Darray *fc3_normal_squared,
                                const double *frequencies,
                                const size_t (*triplets)[3],
                                const size_t *triplets_map,
                                const size_t *map_q,
                                const size_t *rotated_grid_points,
                                const double *rotations_cartesian,
                                const double *g,
                                const size_t num_ir_gp,
                                const size_t num_gp,
                                const size_t num_rot,
                                const double temperature,
                                const double unit_conversion_factor,
                                const double cutoff_frequency);
void ph3py_get_reducible_collision_matrix(double *collision_matrix,
                                          const Darray *fc3_normal_squared,
                                          const double *frequencies,
                                          const size_t (*triplets)[3],
                                          const size_t *triplets_map,
                                          const size_t *map_q,
                                          const double *g,
                                          const size_t num_gp,
                                          const double temperature,
                                          const double unit_conversion_factor,
                                          const double cutoff_frequency);
void ph3py_get_isotope_scattering_strength(
  double *gamma,
  const size_t grid_point,
  const double *mass_variances,
  const double *frequencies,
  const lapack_complex_double *eigenvectors,
  const size_t num_grid_points,
  const int *band_indices,
  const size_t num_band,
  const size_t num_band0,
  const double sigma,
  const double cutoff_frequency);
void ph3py_get_thm_isotope_scattering_strength(
  double *gamma,
  const size_t grid_point,
  const size_t *ir_grid_points,
  const int *weights,
  const double *mass_variances,
  const double *frequencies,
  const lapack_complex_double *eigenvectors,
  const size_t num_ir_grid_points,
  const int *band_indices,
  const size_t num_band,
  const size_t num_band0,
  const double *integration_weights,
  const double cutoff_frequency);
void ph3py_distribute_fc3(double *fc3,
                          const int target,
                          const int source,
                          const int *atom_mapping,
                          const size_t num_atom,
                          const double *rot_cart);
void ph3py_rotate_delta_fc2(double (*fc3)[3][3][3],
                            PHPYCONST double (*delta_fc2s)[3][3],
                            const double *inv_U,
                            PHPYCONST double (*site_sym_cart)[3][3],
                            const int *rot_map_syms,
                            const size_t num_atom,
                            const size_t num_site_sym,
                            const size_t num_disp);
void ph3py_set_permutation_symmetry_fc3(double *fc3, const size_t num_atom);
void ph3py_set_permutation_symmetry_compact_fc3(double * fc3,
                                                const int p2s[],
                                                const int s2pp[],
                                                const int nsym_list[],
                                                const int perms[],
                                                const size_t n_satom,
                                                const size_t n_patom);
void ph3py_transpose_compact_fc3(double * fc3,
                                 const int p2s[],
                                 const int s2pp[],
                                 const int nsym_list[],
                                 const int perms[],
                                 const size_t n_satom,
                                 const size_t n_patom,
                                 const int t_type);
size_t ph3py_get_triplets_reciprocal_mesh_at_q(size_t *map_triplets,
                                               size_t *map_q,
                                               int (*grid_address)[3],
                                               const size_t grid_point,
                                               const int mesh[3],
                                               const int is_time_reversal,
                                               const int num_rot,
                                               PHPYCONST int (*rotations)[3][3],
                                               const int swappable);
size_t ph3py_get_BZ_triplets_at_q(size_t (*triplets)[3],
                                  const size_t grid_point,
                                  PHPYCONST int (*bz_grid_address)[3],
                                  const size_t *bz_map,
                                  const size_t *map_triplets,
                                  const size_t num_map_triplets,
                                  const int mesh[3]);
void ph3py_get_integration_weight(double *iw,
                                  char *iw_zero,
                                  const double *frequency_points,
                                  const size_t num_band0,
                                  PHPYCONST int relative_grid_address[24][4][3],
                                  const int mesh[3],
                                  PHPYCONST size_t (*triplets)[3],
                                  const size_t num_triplets,
                                  PHPYCONST int (*bz_grid_address)[3],
                                  const size_t *bz_map,
                                  const double *frequencies1,
                                  const size_t num_band1,
                                  const double *frequencies2,
                                  const size_t num_band2,
                                  const size_t tp_type,
                                  const int openmp_per_triplets,
                                  const int openmp_per_bands);
void ph3py_get_integration_weight_with_sigma(double *iw,
                                             char *iw_zero,
                                             const double sigma,
                                             const double sigma_cutoff,
                                             const double *frequency_points,
                                             const size_t num_band0,
                                             PHPYCONST size_t (*triplets)[3],
                                             const size_t num_triplets,
                                             const double *frequencies,
                                             const size_t num_band,
                                             const size_t tp_type);


void ph3py_symmetrize_collision_matrix(double *collision_matrix,
                                       const long num_column,
                                       const long num_temp,
                                       const long num_sigma);
void ph3py_expand_collision_matrix(double *collision_matrix,
                                   const size_t *rot_grid_points,
                                   const size_t *ir_grid_points,
                                   const long num_ir_gp,
                                   const long num_grid_points,
                                   const long num_rot,
                                   const long num_sigma,
                                   const long num_temp,
                                   const long num_band);
void ph3py_get_neighboring_gird_points(size_t *relative_grid_points,
                                       const size_t *grid_points,
                                       PHPYCONST int (*relative_grid_address)[3],
                                       const int mesh[3],
                                       PHPYCONST int (*bz_grid_address)[3],
                                       const size_t *bz_map,
                                       const long num_grid_points,
                                       const long num_relative_grid_address);
void ph3py_set_integration_weights(double *iw,
                                   const double *frequency_points,
                                   const long num_band0,
                                   const long num_band,
                                   const long num_gp,
                                   PHPYCONST int (*relative_grid_address)[4][3],
                                   const int mesh[3],
                                   const size_t *grid_points,
                                   PHPYCONST int (*bz_grid_address)[3],
                                   const size_t *bz_map,
                                   const double *frequencies);

#endif
