/* Copyright (C) 2016 Atsushi Togo */
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

#include <stddef.h>
#include <math.h>
#include <phonoc_utils.h>
#include <triplet_h/triplet.h>
#include <triplet_h/triplet_iw.h>
#include <tetrahedron_method.h>

static void set_freq_vertices(double freq_vertices[3][24][4],
                              const double *frequencies,
                              TPLCONST size_t vertices[2][24][4],
                              const int num_band,
                              const int b1,
                              const int b2);
static int set_g(double g[3],
                 const double f0,
                 TPLCONST double freq_vertices[3][24][4]);
static int in_tetrahedra(const double f0, TPLCONST double freq_vertices[24][4]);
static void get_triplet_tetrahedra_vertices(
  size_t vertices[2][24][4],
  TPLCONST int tp_relative_grid_address[2][24][4][3],
  const int mesh[3],
  const size_t triplet[3],
  TPLCONST int (*bz_grid_address)[3],
  const size_t *bz_map);

void
tpi_get_integration_weight(double *iw,
                           char *iw_zero,
                           const double *frequency_points,
                           const size_t num_band0,
                           TPLCONST int tp_relative_grid_address[2][24][4][3],
                           const int mesh[3],
                           const size_t triplets[3],
                           const size_t num_triplets,
                           TPLCONST int (*bz_grid_address)[3],
                           const size_t *bz_map,
                           const double *frequencies,
                           const size_t num_band,
                           const size_t num_iw,
                           const int openmp_per_bands)
{
  size_t j, b1, b2, b12, num_band_prod, adrs_shift;
  size_t vertices[2][24][4];
  double g[3];
  double freq_vertices[3][24][4];

  get_triplet_tetrahedra_vertices(vertices,
                                  tp_relative_grid_address,
                                  mesh,
                                  triplets,
                                  bz_grid_address,
                                  bz_map);

  num_band_prod = num_triplets * num_band0 * num_band * num_band;

#pragma omp parallel for private(j, b1, b2, adrs_shift, g, freq_vertices) if (openmp_per_bands)
  for (b12 = 0; b12 < num_band * num_band; b12++) {
    b1 = b12 / num_band;
    b2 = b12 % num_band;
    set_freq_vertices
      (freq_vertices, frequencies, vertices, num_band, b1, b2);
    for (j = 0; j < num_band0; j++) {
      adrs_shift = j * num_band * num_band + b1 * num_band + b2;
      iw_zero[adrs_shift] = set_g(g, frequency_points[j], freq_vertices);
      iw[adrs_shift] = g[0];
      adrs_shift += num_band_prod;
      iw[adrs_shift] = g[1] - g[2];
      if (num_iw == 3) {
        adrs_shift += num_band_prod;
        iw[adrs_shift] = g[0] + g[1] + g[2];
      }
    }
  }
}

void tpi_get_integration_weight_with_sigma(double *iw,
                                           char *iw_zero,
                                           const double sigma,
                                           const double cutoff,
                                           const double *frequency_points,
                                           const size_t num_band0,
                                           const size_t triplet[3],
                                           const size_t const_adrs_shift,
                                           const double *frequencies,
                                           const size_t num_band,
                                           const size_t num_iw,
                                           const int openmp_per_bands)
{
  size_t j, b12, b1, b2, adrs_shift;
  double f0, f1, f2, g0, g1, g2;

#pragma omp parallel for private(j, b1, b2, f0, f1, f2, g0, g1, g2, adrs_shift) if (openmp_per_bands)
  for (b12 = 0; b12 < num_band * num_band; b12++) {
    b1 = b12 / num_band;
    b2 = b12 % num_band;
    f1 = frequencies[triplet[1] * num_band + b1];
    f2 = frequencies[triplet[2] * num_band + b2];
    for (j = 0; j < num_band0; j++) {
      f0 = frequency_points[j];
      adrs_shift = j * num_band * num_band + b1 * num_band + b2;

      if (cutoff > 0 &&
          fabs(f0 - f1 - f2) > cutoff &&
          fabs(f0 + f1 - f2) > cutoff &&
          fabs(f0 - f1 + f2) > cutoff) {
        iw_zero[adrs_shift] = 1;
        g0 = 0;
        g1 = 0;
        g2 = 0;
      } else {
        iw_zero[adrs_shift] = 0;
        g0 = gaussian(f0 - f1 - f2, sigma);
        g1 = gaussian(f0 + f1 - f2, sigma);
        g2 = gaussian(f0 - f1 + f2, sigma);
      }
      iw[adrs_shift] = g0;
      adrs_shift += const_adrs_shift;
      iw[adrs_shift] = g1 - g2;
      if (num_iw == 3) {
        adrs_shift += const_adrs_shift;
        iw[adrs_shift] = g0 + g1 + g2;
      }
    }
  }
}

static void set_freq_vertices(double freq_vertices[3][24][4],
                              const double *frequencies,
                              TPLCONST size_t vertices[2][24][4],
                              const int num_band,
                              const int b1,
                              const int b2)
{
  int i, j;
  double f1, f2;

  for (i = 0; i < 24; i++) {
    for (j = 0; j < 4; j++) {
      f1 = frequencies[vertices[0][i][j] * num_band + b1];
      f2 = frequencies[vertices[1][i][j] * num_band + b2];
      if (f1 < 0) {f1 = 0;}
      if (f2 < 0) {f2 = 0;}
      freq_vertices[0][i][j] = f1 + f2;
      freq_vertices[1][i][j] = -f1 + f2;
      freq_vertices[2][i][j] = f1 - f2;
    }
  }
}

static int set_g(double g[3],
                 const double f0,
                 TPLCONST double freq_vertices[3][24][4])
{
  int iw_zero;

  iw_zero = 1;

  if (in_tetrahedra(f0, freq_vertices[0])) {
    g[0] = thm_get_integration_weight(f0, freq_vertices[0], 'I');
    iw_zero = 0;
  } else {
    g[0] = 0;
  }
  if (in_tetrahedra(f0, freq_vertices[1])) {
    g[1] = thm_get_integration_weight(f0, freq_vertices[1], 'I');
    iw_zero = 0;
  } else {
    g[1] = 0;
  }
  if (in_tetrahedra(f0, freq_vertices[2])) {
    g[2] = thm_get_integration_weight(f0, freq_vertices[2], 'I');
    iw_zero = 0;
  } else {
    g[2] = 0;
  }

  return iw_zero;
}

static int in_tetrahedra(const double f0, TPLCONST double freq_vertices[24][4])
{
  int i, j;
  double fmin, fmax;

  fmin = freq_vertices[0][0];
  fmax = freq_vertices[0][0];

  for (i = 0; i < 24; i++) {
    for (j = 0; j < 4; j++) {
      if (fmin > freq_vertices[i][j]) {
        fmin = freq_vertices[i][j];
      }
      if (fmax < freq_vertices[i][j]) {
        fmax = freq_vertices[i][j];
      }
    }
  }

  if (fmin > f0 || fmax < f0) {
    return 0;
  } else {
    return 1;
  }
}

static void get_triplet_tetrahedra_vertices(
  size_t vertices[2][24][4],
  TPLCONST int tp_relative_grid_address[2][24][4][3],
  const int mesh[3],
  const size_t triplet[3],
  TPLCONST int (*bz_grid_address)[3],
  const size_t *bz_map)
{
  int i, j;

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 24; j++) {
      thm_get_dense_neighboring_grid_points(vertices[i][j],
                                            triplet[i + 1],
                                            tp_relative_grid_address[i][j],
                                            4,
                                            mesh,
                                            bz_grid_address,
                                            bz_map);
    }
  }
}
