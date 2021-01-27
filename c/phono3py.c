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
#include "lapack_wrapper.h"
#include "phonoc_array.h"

#include "phonon.h"
#include "interaction.h"
#include "pp_collision.h"
#include "imag_self_energy_with_g.h"
#include "real_self_energy.h"
#include "collision_matrix.h"
#include "isotope.h"

#include <stdio.h>

void ph3py_get_phonons_at_gridpoints(double *frequencies,
                                     lapack_complex_double *eigenvectors,
                                     char *phonon_done,
                                     const size_t num_phonons,
                                     const size_t *grid_points,
                                     const size_t num_grid_points,
                                     PHPYCONST int (*grid_address)[3],
                                     const int mesh[3],
                                     const double *fc2,
                                     PHPYCONST double(*svecs_fc2)[27][3],
                                     const int *multi_fc2,
                                     PHPYCONST double (*positions_fc2)[3],
                                     const size_t num_patom,
                                     const size_t num_satom,
                                     const double *masses_fc2,
                                     const int *p2s_fc2,
                                     const int *s2p_fc2,
                                     const double unit_conversion_factor,
                                     PHPYCONST double (*born)[3][3],
                                     PHPYCONST double dielectric[3][3],
                                     PHPYCONST double reciprocal_lattice[3][3],
                                     const double *q_direction, /* pointer */
                                     const double nac_factor,
                                     const double *dd_q0,
                                     PHPYCONST double(*G_list)[3],
                                     const size_t num_G_points,
                                     const double lambda,
                                     const char uplo)
{
  if (!dd_q0) {
    phn_get_phonons_at_gridpoints(frequencies,
                                  eigenvectors,
                                  phonon_done,
                                  num_phonons,
                                  grid_points,
                                  num_grid_points,
                                  grid_address,
                                  mesh,
                                  fc2,
                                  svecs_fc2,
                                  multi_fc2,
                                  num_patom,
                                  num_satom,
                                  masses_fc2,
                                  p2s_fc2,
                                  s2p_fc2,
                                  unit_conversion_factor,
                                  born,
                                  dielectric,
                                  reciprocal_lattice,
                                  q_direction,
                                  nac_factor,
                                  uplo);
  } else {
    phn_get_gonze_phonons_at_gridpoints(frequencies,
                                        eigenvectors,
                                        phonon_done,
                                        num_phonons,
                                        grid_points,
                                        num_grid_points,
                                        grid_address,
                                        mesh,
                                        fc2,
                                        svecs_fc2,
                                        multi_fc2,
                                        positions_fc2,
                                        num_patom,
                                        num_satom,
                                        masses_fc2,
                                        p2s_fc2,
                                        s2p_fc2,
                                        unit_conversion_factor,
                                        born,
                                        dielectric,
                                        reciprocal_lattice,
                                        q_direction,
                                        nac_factor,
                                        dd_q0,
                                        G_list,
                                        num_G_points,
                                        lambda,
                                        uplo);
  }
}


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
                           const double cutoff_frequency)
{
  itr_get_interaction(fc3_normal_squared,
                      g_zero,
                      frequencies,
                      eigenvectors,
                      triplets,
                      num_triplets,
                      grid_address,
                      mesh,
                      fc3,
                      is_compact_fc3,
                      shortest_vectors,
                      svecs_dims,
                      multiplicity,
                      masses,
                      p2s_map,
                      s2p_map,
                      band_indices,
                      symmetrize_fc3_q,
                      cutoff_frequency);
}


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
                            const double cutoff_frequency)
{
  ppc_get_pp_collision(imag_self_energy,
                       relative_grid_address,
                       frequencies,
                       eigenvectors,
                       triplets,
                       num_triplets,
                       triplet_weights,
                       grid_address,
                       bz_map,
                       mesh,
                       fc3,
                       is_compact_fc3,
                       shortest_vectors,
                       svecs_dims,
                       multiplicity,
                       masses,
                       p2s_map,
                       s2p_map,
                       band_indices,
                       temperatures,
                       is_NU,
                       symmetrize_fc3_q,
                       cutoff_frequency);
}


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
  const double cutoff_frequency)
{
  ppc_get_pp_collision_with_sigma(imag_self_energy,
                                  sigma,
                                  sigma_cutoff,
                                  frequencies,
                                  eigenvectors,
                                  triplets,
                                  num_triplets,
                                  triplet_weights,
                                  grid_address,
                                  mesh,
                                  fc3,
                                  is_compact_fc3,
                                  shortest_vectors,
                                  svecs_dims,
                                  multiplicity,
                                  masses,
                                  p2s_map,
                                  s2p_map,
                                  band_indices,
                                  temperatures,
                                  is_NU,
                                  symmetrize_fc3_q,
                                  cutoff_frequency);
}


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
  const int frequency_point_index)
{
  ise_get_imag_self_energy_at_bands_with_g(imag_self_energy,
                                           fc3_normal_squared,
                                           frequencies,
                                           triplets,
                                           triplet_weights,
                                           g,
                                           g_zero,
                                           temperature,
                                           cutoff_frequency,
                                           num_frequency_points,
                                           frequency_point_index);
}


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
  const double cutoff_frequency)
{
  ise_get_detailed_imag_self_energy_at_bands_with_g(detailed_imag_self_energy,
                                                    imag_self_energy_N,
                                                    imag_self_energy_U,
                                                    fc3_normal_squared,
                                                    frequencies,
                                                    triplets,
                                                    triplet_weights,
                                                    grid_address,
                                                    g,
                                                    g_zero,
                                                    temperature,
                                                    cutoff_frequency);
}


void ph3py_get_real_self_energy_at_bands(double *real_self_energy,
                                         const Darray *fc3_normal_squared,
                                         const int *band_indices,
                                         const double *frequencies,
                                         const size_t (*triplets)[3],
                                         const int *triplet_weights,
                                         const double epsilon,
                                         const double temperature,
                                         const double unit_conversion_factor,
                                         const double cutoff_frequency)
{
  rse_get_real_self_energy_at_bands(real_self_energy,
                                    fc3_normal_squared,
                                    band_indices,
                                    frequencies,
                                    triplets,
                                    triplet_weights,
                                    epsilon,
                                    temperature,
                                    unit_conversion_factor,
                                    cutoff_frequency);
}


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
  const double cutoff_frequency)
{
  rse_get_real_self_energy_at_frequency_point(real_self_energy,
                                              frequency_point,
                                              fc3_normal_squared,
                                              band_indices,
                                              frequencies,
                                              triplets,
                                              triplet_weights,
                                              epsilon,
                                              temperature,
                                              unit_conversion_factor,
                                              cutoff_frequency);
}


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
                                const double cutoff_frequency)
{
  col_get_collision_matrix(collision_matrix,
                           fc3_normal_squared,
                           frequencies,
                           triplets,
                           triplets_map,
                           map_q,
                           rotated_grid_points,
                           rotations_cartesian,
                           g,
                           num_ir_gp,
                           num_gp,
                           num_rot,
                           temperature,
                           unit_conversion_factor,
                           cutoff_frequency);
}


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
                                          const double cutoff_frequency)
{
  col_get_reducible_collision_matrix(collision_matrix,
                                     fc3_normal_squared,
                                     frequencies,
                                     triplets,
                                     triplets_map,
                                     map_q,
                                     g,
                                     num_gp,
                                     temperature,
                                     unit_conversion_factor,
                                     cutoff_frequency);
}


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
  const double cutoff_frequency)
{
  iso_get_isotope_scattering_strength(gamma,
                                      grid_point,
                                      mass_variances,
                                      frequencies,
                                      eigenvectors,
                                      num_grid_points,
                                      band_indices,
                                      num_band,
                                      num_band0,
                                      sigma,
                                      cutoff_frequency);
}


void ph3py_get_thm_isotope_scattering_strength
(double *gamma,
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
 const double cutoff_frequency)
{
  iso_get_thm_isotope_scattering_strength(gamma,
                                          grid_point,
                                          ir_grid_points,
                                          weights,
                                          mass_variances,
                                          frequencies,
                                          eigenvectors,
                                          num_ir_grid_points,
                                          band_indices,
                                          num_band,
                                          num_band0,
                                          integration_weights,
                                          cutoff_frequency);
}

/* void fc3_distribute_fc3(double *fc3,
 *                         const int target,
 *                         const int source,
 *                         const int *atom_mapping,
 *                         const size_t num_atom,
 *                         const double *rot_cart)
 * void fc3_rotate_delta_fc2(double (*fc3)[3][3][3],
 *                           PHPYCONST double (*delta_fc2s)[3][3],
 *                           const double *inv_U,
 *                           PHPYCONST double (*site_sym_cart)[3][3],
 *                           const int *rot_map_syms,
 *                           const size_t num_atom,
 *                           const size_t num_site_sym,
 *                           const size_t num_disp)
 * void fc3_set_permutation_symmetry_fc3(double *fc3, const size_t num_atom)
 * void fc3_set_permutation_symmetry_compact_fc3(double * fc3,
 *                                               const int p2s[],
 *                                               const int s2pp[],
 *                                               const int nsym_list[],
 *                                               const int perms[],
 *                                               const size_t n_satom,
 *                                               const size_t n_patom)
 * void fc3_transpose_compact_fc3(double * fc3,
 *                                const int p2s[],
 *                                const int s2pp[],
 *                                const int nsym_list[],
 *                                const int perms[],
 *                                const size_t n_satom,
 *                                const size_t n_patom,
 *                                const int t_type) */



void ph3py_symmetrize_collision_matrix(double *collision_matrix,
                                       const long num_column,
                                       const long num_temp,
                                       const long num_sigma)
{
  double val;
  long i, j, k, l, adrs_shift;

  for (i = 0; i < num_sigma; i++) {
    for (j = 0; j < num_temp; j++) {
      adrs_shift = (i * num_column * num_column * num_temp +
                    j * num_column * num_column);
      /* show_colmat_info(py_collision_matrix, i, j, adrs_shift); */
#pragma omp parallel for schedule(guided) private(l, val)
      for (k = 0; k < num_column; k++) {
        for (l = k + 1; l < num_column; l++) {
          val = (collision_matrix[adrs_shift + k * num_column + l] +
                 collision_matrix[adrs_shift + l * num_column + k]) / 2;
          collision_matrix[adrs_shift + k * num_column + l] = val;
          collision_matrix[adrs_shift + l * num_column + k] = val;
        }
      }
    }
  }
}


void ph3py_expand_collision_matrix(double *collision_matrix,
                                   const size_t *rot_grid_points,
                                   const size_t *ir_grid_points,
                                   const long num_ir_gp,
                                   const long num_grid_points,
                                   const long num_rot,
                                   const long num_sigma,
                                   const long num_temp,
                                   const long num_band)

{
  long i, j, k, l, m, n, p, adrs_shift, adrs_shift_plus, ir_gp, gp_r;
  long num_column, num_bgb;
  long *multi;
  double *colmat_copy;

  multi = (long*)malloc(sizeof(long) * num_ir_gp);
  colmat_copy = NULL;

  num_column = num_grid_points * num_band;
  num_bgb = num_band * num_grid_points * num_band;

#pragma omp parallel for schedule(guided) private(j, ir_gp)
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
#pragma omp parallel for private(ir_gp, adrs_shift_plus, colmat_copy, l, gp_r, m, n, p)
      for (k = 0; k < num_ir_gp; k++) {
        ir_gp = ir_grid_points[k];
        adrs_shift_plus = adrs_shift + ir_gp * num_bgb;
        colmat_copy = (double*)malloc(sizeof(double) * num_bgb);
        for (l = 0; l < num_bgb; l++) {
          colmat_copy[l] = collision_matrix[adrs_shift_plus + l] / multi[k];
          collision_matrix[adrs_shift_plus + l] = 0;
        }
        for (l = 0; l < num_rot; l++) {
          gp_r = rot_grid_points[l * num_grid_points + ir_gp];
          for (m = 0; m < num_band; m++) {
            for (n = 0; n < num_grid_points; n++) {
              for (p = 0; p < num_band; p++) {
                collision_matrix[
                  adrs_shift + gp_r * num_bgb + m * num_grid_points * num_band
                  + rot_grid_points[l * num_grid_points + n] * num_band + p] +=
                  colmat_copy[m * num_grid_points * num_band + n * num_band + p];
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
