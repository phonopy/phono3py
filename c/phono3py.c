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
