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

#include "gridsys.h"

#include "bzgrid.h"
#include "lagrid.h"
#include "grgrid.h"
#include "tetrahedron_method.h"
#include "triplet.h"
#include "triplet_iw.h"

#include <stdio.h>
#include <stdlib.h>

long
gridsys_get_triplets_reciprocal_mesh_at_q(long *map_triplets,
                                          long *map_q,
                                          const long grid_point,
                                          const long D_diag[3],
                                          const long is_time_reversal,
                                          const long num_rot,
                                          const long (*rec_rotations)[3][3],
                                          const long swappable)
{
  return tpl_get_triplets_reciprocal_mesh_at_q(map_triplets,
                                               map_q,
                                               grid_point,
                                               D_diag,
                                               is_time_reversal,
                                               num_rot,
                                               rec_rotations,
                                               swappable);
}


long gridsys_get_BZ_triplets_at_q(long (*triplets)[3],
                                  const long grid_point,
                                  const long (*bz_grid_addresses)[3],
                                  const long *bz_map,
                                  const long *map_triplets,
                                  const long num_map_triplets,
                                  const long D_diag[3],
                                  const long Q[3][3],
                                  const long bz_grid_type)
{
  ConstBZGrid *bzgrid;
  long i, j, num_ir;

  if ((bzgrid = (ConstBZGrid*) malloc(sizeof(ConstBZGrid))) == NULL) {
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
  bzgrid->size = num_map_triplets;

  num_ir = tpl_get_BZ_triplets_at_q(triplets,
                                    grid_point,
                                    bzgrid,
                                    map_triplets);
  free(bzgrid);
  bzgrid = NULL;

  return num_ir;
}

/* relative_grid_addresses are given as P multipled with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
long gridsys_get_integration_weight(double *iw,
                                    char *iw_zero,
                                    const double *frequency_points,
                                    const long num_band0,
                                    const long relative_grid_address[24][4][3],
                                    const long D_diag[3],
                                    const long (*triplets)[3],
                                    const long num_triplets,
                                    const long (*bz_grid_addresses)[3],
                                    const long *bz_map,
                                    const long bz_grid_type,
                                    const double *frequencies1,
                                    const long num_band1,
                                    const double *frequencies2,
                                    const long num_band2,
                                    const long tp_type,
                                    const long openmp_per_triplets,
                                    const long openmp_per_bands)
{
  ConstBZGrid *bzgrid;
  long i;

  if ((bzgrid = (ConstBZGrid*) malloc(sizeof(ConstBZGrid))) == NULL) {
    warning_print("Memory could not be allocated.");
    return 0;
  }

  bzgrid->addresses = bz_grid_addresses;
  bzgrid->gp_map = bz_map;
  bzgrid->type = bz_grid_type;
  for (i = 0; i < 3; i++) {
    bzgrid->D_diag[i] = D_diag[i];
  }

  tpl_get_integration_weight(iw,
                             iw_zero,
                             frequency_points,
                             num_band0,
                             relative_grid_address,
                             triplets,
                             num_triplets,
                             bzgrid,
                             frequencies1,
                             num_band1,
                             frequencies2,
                             num_band2,
                             tp_type,
                             openmp_per_triplets,
                             openmp_per_bands);
  free(bzgrid);
  bzgrid = NULL;

  return 1;
}

void gridsys_get_integration_weight_with_sigma(double *iw,
                                               char *iw_zero,
                                               const double sigma,
                                               const double sigma_cutoff,
                                               const double *frequency_points,
                                               const long num_band0,
                                               const long (*triplets)[3],
                                               const long num_triplets,
                                               const double *frequencies,
                                               const long num_band,
                                               const long tp_type)
{
  tpl_get_integration_weight_with_sigma(iw,
                                        iw_zero,
                                        sigma,
                                        sigma_cutoff,
                                        frequency_points,
                                        num_band0,
                                        triplets,
                                        num_triplets,
                                        frequencies,
                                        num_band,
                                        tp_type);
}


/* From single address to grid index */
long gridsys_get_grid_index_from_address(const long address[3],
                                         const long D_diag[3])
{
  return grg_get_grid_index(address, D_diag);
}


void gridsys_get_gr_grid_addresses(long gr_grid_addresses[][3],
                                   const long D_diag[3])
{
  grg_get_all_grid_addresses(gr_grid_addresses, D_diag);
}

/* Rotation matrices with respect to reciprocal basis vectors are
 * transformed to those for GRGrid. This set of the rotations are
 * used always in GRGrid handling. */
long gridsys_transform_rotations(long (*transformed_rots)[3][3],
                                 const long (*rotations)[3][3],
                                 const long num_rot,
                                 const long D_diag[3],
                                 const long Q[3][3])
{
  return grg_transform_rotations(transformed_rots,
                                 rotations,
                                 num_rot,
                                 D_diag,
                                 Q);
}

long gridsys_get_snf3x3(long D_diag[3],
                        long P[3][3],
                        long Q[3][3],
                        const long A[3][3])
{
  return grg_get_snf3x3(D_diag, P, Q, A);
}

/* The rotations are those after proper transformation in GRGrid. */
long gridsys_get_ir_reciprocal_mesh(long *ir_mapping_table,
                                    const long D_diag[3],
                                    const long PS[3],
                                    const long is_time_reversal,
                                    const long (*rec_rotations)[3][3],
                                    const long num_rot)
{
  long num_ir;

  num_ir = bzg_get_ir_reciprocal_mesh(ir_mapping_table,
                                      D_diag,
                                      PS,
                                      is_time_reversal,
                                      rec_rotations,
                                      num_rot);
  return num_ir;
}

long gridsys_get_bz_grid_address(long (*bz_grid_addresses)[3],
                                 long *bz_map,
                                 long *bzg2grg,
                                 const long (*grid_address)[3],
                                 const long D_diag[3],
                                 const long Q[3][3],
                                 const long PS[3],
                                 const double rec_lattice[3][3],
                                 const long type)
{
  BZGrid *bzgrid;
  long i, j, size;

  if ((bzgrid = (BZGrid*) malloc(sizeof(BZGrid))) == NULL) {
    warning_print("Memory could not be allocated.");
    return 0;
  }

  bzgrid->addresses = bz_grid_addresses;
  bzgrid->gp_map = bz_map;
  bzgrid->bzg2grg = bzg2grg;
  bzgrid->type = type;
  for (i = 0; i < 3; i++) {
    bzgrid->D_diag[i] = D_diag[i];
    bzgrid->PS[i] = PS[i];
    for (j = 0; j < 3; j++) {
      bzgrid->Q[i][j] = Q[i][j];
      bzgrid->reclat[i][j] = rec_lattice[i][j];
    }
  }

  if (bzg_get_bz_grid_addresses(bzgrid, grid_address)) {
    size = bzgrid->size;
  } else {
    size = 0;
  }

  free(bzgrid);
  bzgrid = NULL;

  return size;
}

/* relative_grid_addresses are given as P multipled with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
long
gridsys_get_neighboring_gird_points(long *relative_grid_points,
                                    const long *grid_points,
                                    const long (*relative_grid_address)[3],
                                    const long D_diag[3],
                                    const long (*bz_grid_addresses)[3],
                                    const long *bz_map,
                                    const long bz_grid_type,
                                    const long num_grid_points,
                                    const long num_relative_grid_address)
{
  long i;
  ConstBZGrid *bzgrid;

  if ((bzgrid = (ConstBZGrid*) malloc(sizeof(ConstBZGrid))) == NULL) {
    warning_print("Memory could not be allocated.");
    return 0;
  }

  bzgrid->addresses = bz_grid_addresses;
  bzgrid->gp_map = bz_map;
  bzgrid->type = bz_grid_type;
  for (i = 0; i < 3; i++) {
    bzgrid->D_diag[i] = D_diag[i];
  }

#pragma omp parallel for
  for (i = 0; i < num_grid_points; i++) {
    tpi_get_neighboring_grid_points
      (relative_grid_points + i * num_relative_grid_address,
       grid_points[i],
       relative_grid_address,
       num_relative_grid_address,
       bzgrid);
  }

  free(bzgrid);
  bzgrid = NULL;

  return 1;
}


/* relative_grid_addresses are given as P multipled with those from dataset,
 * i.e.,
 *     np.dot(relative_grid_addresses, P.T) */
long gridsys_set_integration_weights(double *iw,
                                     const double *frequency_points,
                                     const long num_band0,
                                     const long num_band,
                                     const long num_gp,
                                     const long (*relative_grid_address)[4][3],
                                     const long D_diag[3],
                                     const long *grid_points,
                                     const long (*bz_grid_addresses)[3],
                                     const long *bz_map,
                                     const long bz_grid_type,
                                     const double *frequencies)
{
  long i, j, k, bi;
  long vertices[24][4];
  double freq_vertices[24][4];
  ConstBZGrid *bzgrid;

  if ((bzgrid = (ConstBZGrid*) malloc(sizeof(ConstBZGrid))) == NULL) {
    warning_print("Memory could not be allocated.");
    return 0;
  }

  bzgrid->addresses = bz_grid_addresses;
  bzgrid->gp_map = bz_map;
  bzgrid->type = bz_grid_type;
  for (i = 0; i < 3; i++) {
    bzgrid->D_diag[i] = D_diag[i];
  }

#pragma omp parallel for private(j, k, bi, vertices, freq_vertices)
  for (i = 0; i < num_gp; i++) {
    for (j = 0; j < 24; j++) {
      tpi_get_neighboring_grid_points(vertices[j],
                                      grid_points[i],
                                      relative_grid_address[j],
                                      4,
                                      bzgrid);
    }
    for (bi = 0; bi < num_band; bi++) {
      for (j = 0; j < 24; j++) {
        for (k = 0; k < 4; k++) {
          freq_vertices[j][k] = frequencies[vertices[j][k] * num_band + bi];
        }
      }
      for (j = 0; j < num_band0; j++) {
        iw[i * num_band0 * num_band + j * num_band + bi] =
          thm_get_integration_weight(frequency_points[j], freq_vertices, 'I');
      }
    }
  }

  free(bzgrid);
  bzgrid = NULL;

  return 1;
}
