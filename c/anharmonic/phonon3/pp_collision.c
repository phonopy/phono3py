/* Copyright (C) 2017 Atsushi Togo */
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

#include <stdio.h>
#include <stdlib.h>
#include <phonoc_array.h>
#include <phonoc_const.h>
#include <phonoc_utils.h>
#include <phonon3_h/imag_self_energy_with_g.h>
#include <phonon3_h/pp_collision.h>
#include <phonon3_h/interaction.h>
#include <triplet_h/triplet.h>
#include <triplet_h/triplet_iw.h>
#include <lapack_wrapper.h>

static void get_collision(double *ise,
                          const size_t num_band0,
                          const size_t num_band,
                          const size_t num_temps,
                          const double *temperatures,
                          const double *g,
                          const char *g_zero,
                          const double *frequencies,
                          const lapack_complex_double *eigenvectors,
                          const size_t triplet[3],
                          const int weight,
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
                          const double cutoff_frequency,
                          const int openmp_per_triplets);
static void finalize_ise(double *imag_self_energy,
                         const double *ise,
                         const int *grid_address,
                         const size_t (*triplets)[3],
                         const size_t num_triplets,
                         const size_t num_temps,
                         const size_t num_band0,
                         const int is_NU);

void ppc_get_pp_collision(double *imag_self_energy,
                          PHPYCONST int relative_grid_address[24][4][3], /* thm */
                          const double *frequencies,
                          const lapack_complex_double *eigenvectors,
                          const size_t (*triplets)[3],
                          const size_t num_triplets,
                          const int *weights,
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
  size_t i;
  size_t num_band, num_band0, num_band_prod, num_temps;
  int openmp_per_triplets;
  double *ise, *freqs_at_gp, *g;
  char *g_zero;
  int tp_relative_grid_address[2][24][4][3];

  ise = NULL;
  freqs_at_gp = NULL;
  g = NULL;
  g_zero = NULL;

  num_band0 = band_indices->dims[0];
  num_band = svecs_dims[1] * 3;
  num_band_prod = num_band0 * num_band * num_band;
  num_temps = temperatures->dims[0];
  ise = (double*)malloc(sizeof(double) * num_triplets * num_temps * num_band0);
  freqs_at_gp = (double*)malloc(sizeof(double) * num_band0);
  for (i = 0; i < num_band0; i++) {
    freqs_at_gp[i] = frequencies[triplets[0][0] * num_band
                                 + band_indices->data[i]];
  }

  if (num_triplets > num_band) {
    openmp_per_triplets = 1;
  } else {
    openmp_per_triplets = 0;
  }

  tpl_set_relative_grid_address(tp_relative_grid_address,
                                relative_grid_address);

#pragma omp parallel for schedule(guided) private(g, g_zero) if (openmp_per_triplets)
  for (i = 0; i < num_triplets; i++) {
    g = (double*)malloc(sizeof(double) * 2 * num_band_prod);
    g_zero = (char*)malloc(sizeof(char) * num_band_prod);
    tpi_get_integration_weight(g,
                               g_zero,
                               freqs_at_gp,
                               num_band0,
                               tp_relative_grid_address,
                               mesh,
                               triplets[i],
                               1,
                               (int(*)[3])grid_address,
                               bz_map,
                               frequencies,
                               num_band,
                               2,
                               1 - openmp_per_triplets);

    get_collision(ise + i * num_temps * num_band0,
                  num_band0,
                  num_band,
                  num_temps,
                  temperatures->data,
                  g,
                  g_zero,
                  frequencies,
                  eigenvectors,
                  triplets[i],
                  weights[i],
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
                  band_indices->data,
                  symmetrize_fc3_q,
                  cutoff_frequency,
                  openmp_per_triplets);

    free(g_zero);
    g_zero = NULL;
    free(g);
    g = NULL;
  }

  finalize_ise(imag_self_energy,
               ise,
               grid_address,
               triplets,
               num_triplets,
               num_temps,
               num_band0,
               is_NU);

  free(freqs_at_gp);
  freqs_at_gp = NULL;
  free(ise);
  ise = NULL;
}

void ppc_get_pp_collision_with_sigma(
  double *imag_self_energy,
  const double sigma,
  const double sigma_cutoff,
  const double *frequencies,
  const lapack_complex_double *eigenvectors,
  const size_t (*triplets)[3],
  const size_t num_triplets,
  const int *weights,
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
  size_t i;
  size_t num_band, num_band0, num_band_prod, num_temps;
  int openmp_per_triplets, const_adrs_shift;
  double cutoff;
  double *ise, *freqs_at_gp, *g;
  char *g_zero;

  ise = NULL;
  freqs_at_gp = NULL;
  g = NULL;
  g_zero = NULL;

  num_band0 = band_indices->dims[0];
  num_band = svecs_dims[1] * 3;
  num_band_prod = num_band0 * num_band * num_band;
  num_temps = temperatures->dims[0];
  const_adrs_shift = num_band_prod;

  ise = (double*)malloc(sizeof(double) * num_triplets * num_temps * num_band0);
  freqs_at_gp = (double*)malloc(sizeof(double) * num_band0);
  for (i = 0; i < num_band0; i++) {
    freqs_at_gp[i] = frequencies[triplets[0][0] * num_band +
                                 band_indices->data[i]];
  }

  if (num_triplets > num_band) {
    openmp_per_triplets = 1;
  } else {
    openmp_per_triplets = 0;
  }

  cutoff = sigma * sigma_cutoff;

#pragma omp parallel for schedule(guided) private(g, g_zero) if (openmp_per_triplets)
  for (i = 0; i < num_triplets; i++) {
    g = (double*)malloc(sizeof(double) * 2 * num_band_prod);
    g_zero = (char*)malloc(sizeof(char) * num_band_prod);
    tpi_get_integration_weight_with_sigma(g,
                                          g_zero,
                                          sigma,
                                          cutoff,
                                          freqs_at_gp,
                                          num_band0,
                                          triplets[i],
                                          const_adrs_shift,
                                          frequencies,
                                          num_band,
                                          2,
                                          0);

    get_collision(ise + i * num_temps * num_band0,
                  num_band0,
                  num_band,
                  num_temps,
                  temperatures->data,
                  g,
                  g_zero,
                  frequencies,
                  eigenvectors,
                  triplets[i],
                  weights[i],
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
                  band_indices->data,
                  symmetrize_fc3_q,
                  cutoff_frequency,
                  openmp_per_triplets);

    free(g_zero);
    g_zero = NULL;
    free(g);
    g = NULL;
  }

  finalize_ise(imag_self_energy,
               ise,
               grid_address,
               triplets,
               num_triplets,
               num_temps,
               num_band0,
               is_NU);

  free(freqs_at_gp);
  freqs_at_gp = NULL;
  free(ise);
  ise = NULL;
}

static void get_collision(double *ise,
                          const size_t num_band0,
                          const size_t num_band,
                          const size_t num_temps,
                          const double *temperatures,
                          const double *g,
                          const char *g_zero,
                          const double *frequencies,
                          const lapack_complex_double *eigenvectors,
                          const size_t triplet[3],
                          const int weight,
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
                          const double cutoff_frequency,
                          const int openmp_per_triplets)
{
  size_t i;
  size_t num_band_prod, num_g_pos;
  double *fc3_normal_squared;
  int (*g_pos)[4];

  fc3_normal_squared = NULL;
  g_pos = NULL;

  num_band_prod = num_band0 * num_band * num_band;
  fc3_normal_squared = (double*)malloc(sizeof(double) * num_band_prod);
  g_pos = (int(*)[4])malloc(sizeof(int[4]) * num_band_prod);

  for (i = 0; i < num_band_prod; i++) {
    fc3_normal_squared[i] = 0;
  }

  num_g_pos = ise_set_g_pos(g_pos,
                            num_band0,
                            num_band,
                            g_zero);

  itr_get_interaction_at_triplet(
    fc3_normal_squared,
    num_band0,
    num_band,
    g_pos,
    num_g_pos,
    frequencies,
    eigenvectors,
    triplet,
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
    cutoff_frequency,
    0,
    0,
    1 - openmp_per_triplets);

  ise_imag_self_energy_at_triplet(
    ise,
    num_band0,
    num_band,
    fc3_normal_squared,
    frequencies,
    triplet,
    weight,
    g,
    g + num_band_prod,
    g_pos,
    num_g_pos,
    temperatures,
    num_temps,
    cutoff_frequency,
    1 - openmp_per_triplets);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;
  free(g_pos);
  g_pos = NULL;
}

static void finalize_ise(double *imag_self_energy,
                         const double *ise,
                         const int *grid_address,
                         const size_t (*triplets)[3],
                         const size_t num_triplets,
                         const size_t num_temps,
                         const size_t num_band0,
                         const int is_NU)
{
  size_t i, j, k;
  int is_N;

  if (is_NU) {
    for (i = 0; i < 2 * num_temps * num_band0; i++) {
      imag_self_energy[i] = 0;
    }
    for (i = 0; i < num_triplets; i++) {
      is_N = tpl_is_N(triplets[i], grid_address);
      for (j = 0; j < num_temps; j++) {
        for (k = 0; k < num_band0; k++) {
          if (is_N) {
            imag_self_energy[j * num_band0 + k] +=
              ise[i * num_temps * num_band0 + j * num_band0 + k];
          } else {
            imag_self_energy[num_temps * num_band0 + j * num_band0 + k] +=
              ise[i * num_temps * num_band0 + j * num_band0 + k];
          }
        }
      }
    }
  } else {
    for (i = 0; i < num_temps * num_band0; i++) {
      imag_self_energy[i] = 0;
    }
    for (i = 0; i < num_triplets; i++) {
      for (j = 0; j < num_temps; j++) {
        for (k = 0; k < num_band0; k++) {
          imag_self_energy[j * num_band0 + k] +=
            ise[i * num_temps * num_band0 + j * num_band0 + k];
        }
      }
    }
  }
}
