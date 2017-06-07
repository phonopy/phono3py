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

#ifdef MKL_KAPACKE
#include <mkl.h>
#else
#include <lapacke.h>
#endif

void get_pp_collision_with_g(double *imag_self_energy,
                             PHPYCONST int relative_grid_address[24][4][3],
                             const double *frequencies,
                             const lapack_complex_double *eigenvectors,
                             const Iarray *triplets,
                             const int *weights,
                             const int *grid_address,
                             const int *bz_map,
                             const int *mesh,
                             const double *fc3,
                             const Darray *shortest_vectors,
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
  int i, j, k, l, jkl, num_band, num_band0, num_band_prod, num_triplets;
  int num_temps, num_g_pos, is_N;
  int openmp_per_triplets;
  double *fc3_normal_squared, *ise, *freqs_at_gp, *g;
  char *g_zero;
  int (*g_pos)[4];

  fc3_normal_squared = NULL;
  ise = NULL;
  freqs_at_gp = NULL;
  g = NULL;
  g_zero = NULL;
  g_pos = NULL;

  num_band0 = band_indices->dims[0];
  num_band = shortest_vectors->dims[1] * 3;
  num_band_prod = num_band0 * num_band * num_band;
  num_triplets = triplets->dims[0];
  num_temps = temperatures->dims[0];
  ise = (double*)malloc(sizeof(double) * num_triplets * num_temps * num_band0);
  freqs_at_gp = (double*)malloc(sizeof(double) * num_band0);
  for (i = 0; i < num_band0; i++) {
    freqs_at_gp[i] = frequencies[triplets->data[0] * num_band +
                                 band_indices->data[i]];
  }

  if (num_triplets > num_band) {
    openmp_per_triplets = 1;
  } else {
    openmp_per_triplets = 0;
  }

#pragma omp parallel for schedule(guided) private(j, k, l, jkl, fc3_normal_squared, g, g_zero, g_pos, num_g_pos) if (openmp_per_triplets)
  for (i = 0; i < num_triplets; i++) {
    g = (double*)malloc(sizeof(double) * 2 * num_band_prod);
    g_zero = (char*)malloc(sizeof(char) * num_band_prod);
    tpl_get_integration_weight(g,
                               g_zero,
                               freqs_at_gp,
                               num_band0,
                               relative_grid_address,
                               mesh,
                               (int(*)[3])(triplets->data + i * 3),
                               1,
                               (int(*)[3])grid_address,
                               bz_map,
                               frequencies,
                               num_band,
                               2,
                               0,
                               1 - openmp_per_triplets);

    fc3_normal_squared = (double*)malloc(sizeof(double) * num_band_prod);
    num_g_pos = 0;
    jkl = 0;
    g_pos = (int(*)[4])malloc(sizeof(int[4]) * num_band_prod);
    for (j = 0; j < num_band0; j++) {
      for (k = 0; k < num_band; k++) {
        for (l = 0; l < num_band; l++) {
          if (!g_zero[jkl]) {
            g_pos[num_g_pos][0] = j;
            g_pos[num_g_pos][1] = k;
            g_pos[num_g_pos][2] = l;
            g_pos[num_g_pos][3] = jkl;
            num_g_pos++;
          }
          fc3_normal_squared[jkl] = 0;
          jkl++;
        }
      }
    }

    get_interaction_at_triplet(
      fc3_normal_squared,
      num_band0,
      num_band,
      g_pos,
      num_g_pos,
      frequencies,
      eigenvectors,
      triplets->data + i * 3,
      grid_address,
      mesh,
      fc3,
      shortest_vectors,
      multiplicity,
      masses,
      p2s_map,
      s2p_map,
      band_indices->data,
      symmetrize_fc3_q,
      cutoff_frequency,
      i,
      num_triplets,
      1 - openmp_per_triplets);

    imag_self_energy_at_triplet(
      ise + i * num_temps * num_band0,
      num_band0,
      num_band,
      fc3_normal_squared,
      frequencies,
      triplets->data + i * 3,
      weights[i],
      g,
      g + num_band_prod,
      g_pos,
      num_g_pos,
      temperatures->data,
      num_temps,
      cutoff_frequency,
      1 - openmp_per_triplets);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
    free(g_pos);
    g_pos = NULL;
    free(g_zero);
    g_zero = NULL;
    free(g);
    g = NULL;
  }

  if (is_NU) {
    for (i = 0; i < 2 * num_temps * num_band0; i++) {
      imag_self_energy[i] = 0;
    }
    for (i = 0; i < num_triplets; i++) {
      is_N = tpl_is_N(triplets->data + i * 3, grid_address);
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

  free(freqs_at_gp);
  freqs_at_gp = NULL;
  free(ise);
  ise = NULL;
}
