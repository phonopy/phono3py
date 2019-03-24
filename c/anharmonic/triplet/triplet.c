/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* These codes were originally parts of spglib, but only develped */
/* and used for phono3py. Therefore these were moved from spglib to */
/* phono3py. This file is part of phonopy. */

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
#include <mathfunc.h>
#include <triplet_h/triplet.h>
#include <triplet_h/triplet_iw.h>
#include <triplet_h/triplet_kpoint.h>

static size_t get_triplets_reciprocal_mesh_at_q(size_t *map_triplets,
                                                size_t *map_q,
                                                int (*grid_address)[3],
                                                const int grid_point,
                                                const int mesh[3],
                                                const int is_time_reversal,
                                                const int num_rot,
                                                TPLCONST int (*rotations)[3][3],
                                                const int swappable);

size_t tpl_get_BZ_triplets_at_q(size_t (*triplets)[3],
                                const size_t grid_point,
                                TPLCONST int (*bz_grid_address)[3],
                                const size_t *bz_map,
                                const size_t *map_triplets,
                                const size_t num_map_triplets,
                                const int mesh[3])
{
  return tpk_get_BZ_triplets_at_q(triplets,
                                  grid_point,
                                  bz_grid_address,
                                  bz_map,
                                  map_triplets,
                                  num_map_triplets,
                                  mesh);
}

size_t tpl_get_triplets_reciprocal_mesh_at_q(size_t *map_triplets,
                                             size_t *map_q,
                                             int (*grid_address)[3],
                                             const size_t grid_point,
                                             const int mesh[3],
                                             const int is_time_reversal,
                                             const int num_rot,
                                             TPLCONST int (*rotations)[3][3],
                                             const int swappable)
{
  return get_triplets_reciprocal_mesh_at_q(map_triplets,
                                           map_q,
                                           grid_address,
                                           grid_point,
                                           mesh,
                                           is_time_reversal,
                                           num_rot,
                                           rotations,
                                           swappable);
}

void tpl_get_integration_weight(double *iw,
                                char *iw_zero,
                                const double *frequency_points,
                                const size_t num_band0,
                                TPLCONST int relative_grid_address[24][4][3],
                                const int mesh[3],
                                TPLCONST size_t (*triplets)[3],
                                const size_t num_triplets,
                                TPLCONST int (*bz_grid_address)[3],
                                const size_t *bz_map,
                                const double *frequencies,
                                const size_t num_band,
                                const size_t num_iw,
                                const int openmp_per_triplets,
                                const int openmp_per_bands)
{
  size_t i, num_band_prod;
  int tp_relative_grid_address[2][24][4][3];

  tpl_set_relative_grid_address(tp_relative_grid_address,
                                relative_grid_address);
  num_band_prod = num_band0 * num_band * num_band;

#pragma omp parallel for if (openmp_per_triplets)
  for (i = 0; i < num_triplets; i++) {
    tpi_get_integration_weight(iw + i * num_band_prod,
                               iw_zero + i * num_band_prod,
                               frequency_points,
                               num_band0,
                               tp_relative_grid_address,
                               mesh,
                               triplets[i],
                               num_triplets,
                               bz_grid_address,
                               bz_map,
                               frequencies,
                               num_band,
                               num_iw,
                               openmp_per_bands);
  }
}


void tpl_get_integration_weight_with_sigma(double *iw,
                                           char *iw_zero,
                                           const double sigma,
                                           const double sigma_cutoff,
                                           const double *frequency_points,
                                           const size_t num_band0,
                                           TPLCONST size_t (*triplets)[3],
                                           const size_t num_triplets,
                                           const double *frequencies,
                                           const size_t num_band,
                                           const size_t num_iw)
{
  size_t i, num_band_prod, const_adrs_shift;
  double cutoff;

  cutoff = sigma * sigma_cutoff;
  num_band_prod = num_band0 * num_band * num_band;
  const_adrs_shift = num_triplets * num_band0 * num_band * num_band;

#pragma omp parallel for
  for (i = 0; i < num_triplets; i++) {
    tpi_get_integration_weight_with_sigma(
      iw + i * num_band_prod,
      iw_zero + i * num_band_prod,
      sigma,
      cutoff,
      frequency_points,
      num_band0,
      triplets[i],
      const_adrs_shift,
      frequencies,
      num_band,
      num_iw,
      0);
  }
}


int tpl_is_N(const size_t triplet[3], const int *grid_address)
{
  int i, j, sum_q, is_N;

  is_N = 1;
  for (i = 0; i < 3; i++) {
    sum_q = 0;
    for (j = 0; j < 3; j++) { /* 1st, 2nd, 3rd triplet */
      sum_q += grid_address[triplet[j] * 3 + i];
    }
    if (sum_q) {
      is_N = 0;
      break;
    }
  }
  return is_N;
}

void tpl_set_relative_grid_address(
  int tp_relative_grid_address[2][24][4][3],
  TPLCONST int relative_grid_address[24][4][3])
{
  int i, j, k, l, sign;

  for (i = 0; i < 2; i++) {
    sign = 1 - i * 2;
    for (j = 0; j < 24; j++) {
      for (k = 0; k < 4; k++) {
        for (l = 0; l < 3; l++) {
          tp_relative_grid_address[i][j][k][l] =
            relative_grid_address[j][k][l] * sign;
        }
      }
    }
  }
}

static size_t get_triplets_reciprocal_mesh_at_q(size_t *map_triplets,
                                                size_t *map_q,
                                                int (*grid_address)[3],
                                                const int grid_point,
                                                const int mesh[3],
                                                const int is_time_reversal,
                                                const int num_rot,
                                                TPLCONST int (*rotations)[3][3],
                                                const int swappable)
{
  MatINT *rot_real;
  int i;
  size_t num_ir;

  rot_real = mat_alloc_MatINT(num_rot);
  for (i = 0; i < num_rot; i++) {
    mat_copy_matrix_i3(rot_real->mat[i], rotations[i]);
  }

  num_ir = tpk_get_ir_triplets_at_q(map_triplets,
                                    map_q,
                                    grid_address,
                                    grid_point,
                                    mesh,
                                    is_time_reversal,
                                    rot_real,
                                    swappable);

  mat_free_MatINT(rot_real);

  return num_ir;
}
