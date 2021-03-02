/* Copyright (C) 2008 Atsushi Togo */
/* All rights reserved. */

/* This file was originally part of spglib. */

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
#include <stddef.h>
#include "kpoint.h"
#include "grgrid.h"

#ifdef KPTWARNING
#include <stdio.h>
#define warning_print(...) fprintf(stderr,__VA_ARGS__)
#else
#define warning_print(...)
#endif

#define KPT_NUM_BZ_SEARCH_SPACE 125
static long bz_search_space[KPT_NUM_BZ_SEARCH_SPACE][3] = {
  { 0,  0,  0},
  { 0,  0,  1},
  { 0,  0,  2},
  { 0,  0, -2},
  { 0,  0, -1},
  { 0,  1,  0},
  { 0,  1,  1},
  { 0,  1,  2},
  { 0,  1, -2},
  { 0,  1, -1},
  { 0,  2,  0},
  { 0,  2,  1},
  { 0,  2,  2},
  { 0,  2, -2},
  { 0,  2, -1},
  { 0, -2,  0},
  { 0, -2,  1},
  { 0, -2,  2},
  { 0, -2, -2},
  { 0, -2, -1},
  { 0, -1,  0},
  { 0, -1,  1},
  { 0, -1,  2},
  { 0, -1, -2},
  { 0, -1, -1},
  { 1,  0,  0},
  { 1,  0,  1},
  { 1,  0,  2},
  { 1,  0, -2},
  { 1,  0, -1},
  { 1,  1,  0},
  { 1,  1,  1},
  { 1,  1,  2},
  { 1,  1, -2},
  { 1,  1, -1},
  { 1,  2,  0},
  { 1,  2,  1},
  { 1,  2,  2},
  { 1,  2, -2},
  { 1,  2, -1},
  { 1, -2,  0},
  { 1, -2,  1},
  { 1, -2,  2},
  { 1, -2, -2},
  { 1, -2, -1},
  { 1, -1,  0},
  { 1, -1,  1},
  { 1, -1,  2},
  { 1, -1, -2},
  { 1, -1, -1},
  { 2,  0,  0},
  { 2,  0,  1},
  { 2,  0,  2},
  { 2,  0, -2},
  { 2,  0, -1},
  { 2,  1,  0},
  { 2,  1,  1},
  { 2,  1,  2},
  { 2,  1, -2},
  { 2,  1, -1},
  { 2,  2,  0},
  { 2,  2,  1},
  { 2,  2,  2},
  { 2,  2, -2},
  { 2,  2, -1},
  { 2, -2,  0},
  { 2, -2,  1},
  { 2, -2,  2},
  { 2, -2, -2},
  { 2, -2, -1},
  { 2, -1,  0},
  { 2, -1,  1},
  { 2, -1,  2},
  { 2, -1, -2},
  { 2, -1, -1},
  {-2,  0,  0},
  {-2,  0,  1},
  {-2,  0,  2},
  {-2,  0, -2},
  {-2,  0, -1},
  {-2,  1,  0},
  {-2,  1,  1},
  {-2,  1,  2},
  {-2,  1, -2},
  {-2,  1, -1},
  {-2,  2,  0},
  {-2,  2,  1},
  {-2,  2,  2},
  {-2,  2, -2},
  {-2,  2, -1},
  {-2, -2,  0},
  {-2, -2,  1},
  {-2, -2,  2},
  {-2, -2, -2},
  {-2, -2, -1},
  {-2, -1,  0},
  {-2, -1,  1},
  {-2, -1,  2},
  {-2, -1, -2},
  {-2, -1, -1},
  {-1,  0,  0},
  {-1,  0,  1},
  {-1,  0,  2},
  {-1,  0, -2},
  {-1,  0, -1},
  {-1,  1,  0},
  {-1,  1,  1},
  {-1,  1,  2},
  {-1,  1, -2},
  {-1,  1, -1},
  {-1,  2,  0},
  {-1,  2,  1},
  {-1,  2,  2},
  {-1,  2, -2},
  {-1,  2, -1},
  {-1, -2,  0},
  {-1, -2,  1},
  {-1, -2,  2},
  {-1, -2, -2},
  {-1, -2, -1},
  {-1, -1,  0},
  {-1, -1,  1},
  {-1, -1,  2},
  {-1, -1, -2},
  {-1, -1, -1}
};

static MatLONG *get_point_group_reciprocal(const MatLONG * rotations,
                                           const long is_time_reversal);
static MatLONG *get_point_group_reciprocal_with_q(const MatLONG * rot_reciprocal,
                                                  const double symprec,
                                                  const long num_q,
                                                  KPTCONST double qpoints[][3]);
static long get_ir_reciprocal_mesh(long grid_address[][3],
                                   long ir_mapping_table[],
                                   const long mesh[3],
                                   const long is_shift[3],
                                   const MatLONG *rot_reciprocal);
static long get_ir_reciprocal_mesh_normal(long grid_address[][3],
                                          long ir_mapping_table[],
                                          const long mesh[3],
                                          const long is_shift[3],
                                          const MatLONG *rot_reciprocal);
static long get_ir_reciprocal_mesh_distortion(long grid_address[][3],
                                              long ir_mapping_table[],
                                              const long mesh[3],
                                              const long is_shift[3],
                                              const MatLONG *rot_reciprocal);
static long relocate_BZ_grid_address(long bz_grid_address[][3],
                                     long bz_map[],
                                     KPTCONST long grid_address[][3],
                                     const long mesh[3],
                                     KPTCONST double rec_lattice[3][3],
                                     const long is_shift[3]);
static long get_bz_grid_addresses(long bz_grid_address[][3],
                                  long bz_map[][2],
                                  KPTCONST long grid_address[][3],
                                  const long mesh[3],
                                  KPTCONST double rec_lattice[3][3],
                                  const long is_shift[3]);
static double get_tolerance_for_BZ_reduction(KPTCONST double rec_lattice[3][3],
                                             const long mesh[3]);
static long get_num_ir(long ir_mapping_table[], const long mesh[3]);
static long check_mesh_symmetry(const long mesh[3],
                                const long is_shift[3],
                                const MatLONG *rot_reciprocal);
static long Nint(const double a);
static double Dabs(const double a);
static void transpose_matrix_l3(long a[3][3], KPTCONST long b[3][3]);
static void multiply_matrix_l3(long m[3][3],
                               KPTCONST long a[3][3], KPTCONST long b[3][3]);
static long check_identity_matrix_l3(KPTCONST long a[3][3],
                                     KPTCONST long b[3][3]);
static void multiply_matrix_vector_ld3(double v[3],
                                       KPTCONST long a[3][3],
                                       const double b[3]);
static void multiply_matrix_vector_l3(long v[3],
                                      KPTCONST long a[3][3],
                                      const long b[3]);
static void multiply_matrix_vector_d3(double v[3],
                                      KPTCONST double a[3][3],
                                      const double b[3]);
static double norm_squared_d3(const double a[3]);


long kpt_get_irreducible_reciprocal_mesh(long grid_address[][3],
                                         long ir_mapping_table[],
                                         const long mesh[3],
                                         const long is_shift[3],
                                         const MatLONG *rot_reciprocal)
{
  long num_ir;

  num_ir = get_ir_reciprocal_mesh(grid_address,
                                  ir_mapping_table,
                                  mesh,
                                  is_shift,
                                  rot_reciprocal);

  return num_ir;
}

MatLONG *kpt_get_point_group_reciprocal(const MatLONG * rotations,
                                        const long is_time_reversal)
{
  return get_point_group_reciprocal(rotations, is_time_reversal);
}

MatLONG *kpt_get_point_group_reciprocal_with_q(const MatLONG * rot_reciprocal,
                                               const double symprec,
                                               const long num_q,
                                               KPTCONST double qpoints[][3])
{
  return get_point_group_reciprocal_with_q(rot_reciprocal,
                                           symprec,
                                           num_q,
                                           qpoints);
}

long kpt_get_stabilized_reciprocal_mesh(long grid_address[][3],
                                        long ir_mapping_table[],
                                        const long mesh[3],
                                        const long is_shift[3],
                                        const long is_time_reversal,
                                        const MatLONG * rotations,
                                        const long num_q,
                                        KPTCONST double qpoints[][3])
{
  long num_ir;
  MatLONG *rot_reciprocal, *rot_reciprocal_q;
  double tolerance;

  rot_reciprocal = NULL;
  rot_reciprocal_q = NULL;

  rot_reciprocal = get_point_group_reciprocal(rotations, is_time_reversal);
  tolerance = 0.01 / (mesh[0] + mesh[1] + mesh[2]);
  rot_reciprocal_q = get_point_group_reciprocal_with_q(rot_reciprocal,
                                                       tolerance,
                                                       num_q,
                                                       qpoints);

  num_ir = get_ir_reciprocal_mesh(grid_address,
                                  ir_mapping_table,
                                  mesh,
                                  is_shift,
                                  rot_reciprocal_q);

  kpt_free_MatLONG(rot_reciprocal_q);
  rot_reciprocal_q = NULL;
  kpt_free_MatLONG(rot_reciprocal);
  rot_reciprocal = NULL;
  return num_ir;
}

long kpt_relocate_BZ_grid_address(long bz_grid_address[][3],
                                  long bz_map[],
                                  KPTCONST long grid_address[][3],
                                  const long mesh[3],
                                  KPTCONST double rec_lattice[3][3],
                                  const long is_shift[3])
{
  return relocate_BZ_grid_address(bz_grid_address,
                                  bz_map,
                                  grid_address,
                                  mesh,
                                  rec_lattice,
                                  is_shift);
}

long kpt_get_bz_grid_addresses(long bz_grid_address[][3],
                               long bz_map[][2],
                               KPTCONST long grid_address[][3],
                               const long mesh[3],
                               KPTCONST double rec_lattice[3][3],
                               const long is_shift[3])
{
  return get_bz_grid_addresses(bz_grid_address,
                               bz_map,
                               grid_address,
                               mesh,
                               rec_lattice,
                               is_shift);
}

void kpt_copy_matrix_l3(long a[3][3], KPTCONST long b[3][3])
{
  a[0][0] = b[0][0];
  a[0][1] = b[0][1];
  a[0][2] = b[0][2];
  a[1][0] = b[1][0];
  a[1][1] = b[1][1];
  a[1][2] = b[1][2];
  a[2][0] = b[2][0];
  a[2][1] = b[2][1];
  a[2][2] = b[2][2];
}

MatLONG * kpt_alloc_MatLONG(const long size)
{
  MatLONG *matlong;

  matlong = NULL;

  if ((matlong = (MatLONG*) malloc(sizeof(MatLONG))) == NULL) {
    warning_print("Memory could not be allocated.");
    return NULL;
  }

  matlong->size = size;
  if (size > 0) {
    if ((matlong->mat = (long (*)[3][3]) malloc(sizeof(long[3][3]) * size))
        == NULL) {
      warning_print("Memory could not be allocated ");
      warning_print("(MatLONG, line %d, %s).\n", __LINE__, __FILE__);
      free(matlong);
      matlong = NULL;
      return NULL;
    }
  }
  return matlong;
}

void kpt_free_MatLONG(MatLONG * matlong)
{
  if (matlong->size > 0) {
    free(matlong->mat);
    matlong->mat = NULL;
  }
  free(matlong);
}


/* Return NULL if failed */
static MatLONG *get_point_group_reciprocal(const MatLONG * rotations,
                                           const long is_time_reversal)
{
  long i, j, num_rot;
  MatLONG *rot_reciprocal, *rot_return;
  long *unique_rot;
  KPTCONST long inversion[3][3] = {
    {-1, 0, 0 },
    { 0,-1, 0 },
    { 0, 0,-1 }
  };

  rot_reciprocal = NULL;
  rot_return = NULL;
  unique_rot = NULL;

  if (is_time_reversal) {
    if ((rot_reciprocal = kpt_alloc_MatLONG(rotations->size * 2)) == NULL) {
      return NULL;
    }
  } else {
    if ((rot_reciprocal = kpt_alloc_MatLONG(rotations->size)) == NULL) {
      return NULL;
    }
  }

  if ((unique_rot = (long*)malloc(sizeof(long) * rot_reciprocal->size)) == NULL) {
    warning_print("Memory of unique_rot could not be allocated.");
    kpt_free_MatLONG(rot_reciprocal);
    rot_reciprocal = NULL;
    return NULL;
  }

  for (i = 0; i < rot_reciprocal->size; i++) {
    unique_rot[i] = -1;
  }

  for (i = 0; i < rotations->size; i++) {
    transpose_matrix_l3(rot_reciprocal->mat[i], rotations->mat[i]);

    if (is_time_reversal) {
      multiply_matrix_l3(rot_reciprocal->mat[rotations->size+i],
                         inversion,
                         rot_reciprocal->mat[i]);
    }
  }

  num_rot = 0;
  for (i = 0; i < rot_reciprocal->size; i++) {
    for (j = 0; j < num_rot; j++) {
      if (check_identity_matrix_l3(rot_reciprocal->mat[unique_rot[j]],
                                   rot_reciprocal->mat[i])) {
        goto escape;
      }
    }
    unique_rot[num_rot] = i;
    num_rot++;
  escape:
    ;
  }

  if ((rot_return = kpt_alloc_MatLONG(num_rot)) != NULL) {
    for (i = 0; i < num_rot; i++) {
      kpt_copy_matrix_l3(rot_return->mat[i], rot_reciprocal->mat[unique_rot[i]]);
    }
  }

  free(unique_rot);
  unique_rot = NULL;
  kpt_free_MatLONG(rot_reciprocal);
  rot_reciprocal = NULL;

  return rot_return;
}

/* Return NULL if failed */
static MatLONG *get_point_group_reciprocal_with_q(const MatLONG * rot_reciprocal,
                                                  const double symprec,
                                                  const long num_q,
                                                  KPTCONST double qpoints[][3])
{
  long i, j, k, l, is_all_ok, num_rot;
  long *ir_rot;
  double q_rot[3], diff[3];
  MatLONG * rot_reciprocal_q;

  ir_rot = NULL;
  rot_reciprocal_q = NULL;
  is_all_ok = 0;
  num_rot = 0;

  if ((ir_rot = (long*)malloc(sizeof(long) * rot_reciprocal->size)) == NULL) {
    warning_print("Memory of ir_rot could not be allocated.");
    return NULL;
  }

  for (i = 0; i < rot_reciprocal->size; i++) {
    ir_rot[i] = -1;
  }
  for (i = 0; i < rot_reciprocal->size; i++) {
    for (j = 0; j < num_q; j++) {
      is_all_ok = 0;
      multiply_matrix_vector_ld3(q_rot,
                                 rot_reciprocal->mat[i],
                                 qpoints[j]);

      for (k = 0; k < num_q; k++) {
        for (l = 0; l < 3; l++) {
          diff[l] = q_rot[l] - qpoints[k][l];
          diff[l] -= Nint(diff[l]);
        }

        if (Dabs(diff[0]) < symprec &&
            Dabs(diff[1]) < symprec &&
            Dabs(diff[2]) < symprec) {
          is_all_ok = 1;
          break;
        }
      }

      if (! is_all_ok) {
        break;
      }
    }

    if (is_all_ok) {
      ir_rot[num_rot] = i;
      num_rot++;
    }
  }

  if ((rot_reciprocal_q = kpt_alloc_MatLONG(num_rot)) != NULL) {
    for (i = 0; i < num_rot; i++) {
      kpt_copy_matrix_l3(rot_reciprocal_q->mat[i],
                         rot_reciprocal->mat[ir_rot[i]]);
    }
  }

  free(ir_rot);
  ir_rot = NULL;

  return rot_reciprocal_q;
}

static long get_ir_reciprocal_mesh(long grid_address[][3],
                                   long ir_mapping_table[],
                                   const long mesh[3],
                                   const long is_shift[3],
                                   const MatLONG *rot_reciprocal)
{
  if (check_mesh_symmetry(mesh, is_shift, rot_reciprocal)) {
    return get_ir_reciprocal_mesh_normal(grid_address,
                                         ir_mapping_table,
                                         mesh,
                                         is_shift,
                                         rot_reciprocal);
  } else {
    return get_ir_reciprocal_mesh_distortion(grid_address,
                                             ir_mapping_table,
                                             mesh,
                                             is_shift,
                                             rot_reciprocal);
  }
}

static long get_ir_reciprocal_mesh_normal(long grid_address[][3],
                                          long ir_mapping_table[],
                                          const long mesh[3],
                                          const long is_shift[3],
                                          const MatLONG *rot_reciprocal)
{
  /* In the following loop, mesh is doubled. */
  /* Even and odd mesh numbers correspond to */
  /* is_shift[i] are 0 or 1, respectively. */
  /* is_shift = [0,0,0] gives Gamma center mesh. */
  /* grid: reducible grid points */
  /* ir_mapping_table: the mapping from each point to ir-point. */

  long i, grid_point_rot;
  long j;
  long address_double[3], address_double_rot[3];

  grg_get_all_grid_addresses(grid_address, mesh);

#pragma omp parallel for private(j, grid_point_rot, address_double, address_double_rot)
  for (i = 0; i < mesh[0] * mesh[1] * (long)(mesh[2]); i++) {
    grg_get_double_grid_address(address_double,
                                grid_address[i],
                                mesh,
                                is_shift);
    ir_mapping_table[i] = i;
    for (j = 0; j < rot_reciprocal->size; j++) {
      multiply_matrix_vector_l3(address_double_rot,
                                rot_reciprocal->mat[j],
                                address_double);
      grid_point_rot = grg_get_double_grid_index(address_double_rot, mesh, is_shift);
      if (grid_point_rot < ir_mapping_table[i]) {
#ifdef _OPENMP
        ir_mapping_table[i] = grid_point_rot;
#else
        ir_mapping_table[i] = ir_mapping_table[grid_point_rot];
        break;
#endif
      }
    }
  }

  return get_num_ir(ir_mapping_table, mesh);
}

static long
get_ir_reciprocal_mesh_distortion(long grid_address[][3],
                                  long ir_mapping_table[],
                                  const long mesh[3],
                                  const long is_shift[3],
                                  const MatLONG *rot_reciprocal)
{
  long i, grid_point_rot;
  long j, k, indivisible;
  long address_double[3], address_double_rot[3];
  long long_address_double[3], long_address_double_rot[3], divisor[3];

  /* divisor, long_address_double, and long_address_double_rot have */
  /* long integer type to treat dense mesh. */

  grg_get_all_grid_addresses(grid_address, mesh);

  for (j = 0; j < 3; j++) {
    divisor[j] = mesh[(j + 1) % 3] * mesh[(j + 2) % 3];
  }

#pragma omp parallel for private(j, k, grid_point_rot, address_double, address_double_rot, long_address_double, long_address_double_rot)
  for (i = 0; i < mesh[0] * mesh[1] * (long)(mesh[2]); i++) {
    grg_get_double_grid_address(address_double,
                                grid_address[i],
                                mesh,
                                is_shift);
    for (j = 0; j < 3; j++) {
      long_address_double[j] = address_double[j] * divisor[j];
    }
    ir_mapping_table[i] = i;
    for (j = 0; j < rot_reciprocal->size; j++) {

      /* Equivalent to mat_multiply_matrix_vector_i3 except for data type */
      for (k = 0; k < 3; k++) {
        long_address_double_rot[k] =
          rot_reciprocal->mat[j][k][0] * long_address_double[0] +
          rot_reciprocal->mat[j][k][1] * long_address_double[1] +
          rot_reciprocal->mat[j][k][2] * long_address_double[2];
      }

      for (k = 0; k < 3; k++) {
        indivisible = long_address_double_rot[k] % divisor[k];
        if (indivisible) {break;}
        address_double_rot[k] = long_address_double_rot[k] / divisor[k];
        if ((address_double_rot[k] % 2 != 0 && is_shift[k] == 0) ||
            (address_double_rot[k] % 2 == 0 && is_shift[k] == 1)) {
          indivisible = 1;
          break;
        }
      }
      if (indivisible) {continue;}
      grid_point_rot =
        grg_get_double_grid_index(address_double_rot, mesh, is_shift);
      if (grid_point_rot < ir_mapping_table[i]) {
#ifdef _OPENMP
        ir_mapping_table[i] = grid_point_rot;
#else
        ir_mapping_table[i] = ir_mapping_table[grid_point_rot];
        break;
#endif
      }
    }
  }

  return get_num_ir(ir_mapping_table, mesh);
}

static long relocate_BZ_grid_address(long bz_grid_address[][3],
                                     long bz_map[],
                                     KPTCONST long grid_address[][3],
                                     const long mesh[3],
                                     KPTCONST double rec_lattice[3][3],
                                     const long is_shift[3])
{
  double tolerance, min_distance;
  double q_vector[3], distance[KPT_NUM_BZ_SEARCH_SPACE];
  long bzmesh[3], bz_address_double[3];
  long i, boundary_num_gp, total_num_gp, bzgp, gp, num_bzmesh;
  long j, k, min_index;

  tolerance = get_tolerance_for_BZ_reduction(rec_lattice, mesh);
  for (j = 0; j < 3; j++) {
    bzmesh[j] = mesh[j] * 2;
  }

  num_bzmesh = bzmesh[0] * bzmesh[1] * bzmesh[2];
  for (i = 0; i < num_bzmesh; i++) {
    bz_map[i] = num_bzmesh;
  }

  boundary_num_gp = 0;
  total_num_gp = mesh[0] * mesh[1] * mesh[2];

  /* Multithreading doesn't work for this loop since gp calculated */
  /* with boundary_num_gp is unstable to store bz_grid_address. */
  for (i = 0; i < total_num_gp; i++) {
    for (j = 0; j < KPT_NUM_BZ_SEARCH_SPACE; j++) {
      for (k = 0; k < 3; k++) {
        q_vector[k] =
          ((grid_address[i][k] + bz_search_space[j][k] * mesh[k]) * 2 +
           is_shift[k]) / ((double)mesh[k]) / 2;
      }
      multiply_matrix_vector_d3(q_vector, rec_lattice, q_vector);
      distance[j] = norm_squared_d3(q_vector);
    }
    min_distance = distance[0];
    min_index = 0;
    for (j = 1; j < KPT_NUM_BZ_SEARCH_SPACE; j++) {
      if (distance[j] < min_distance) {
        min_distance = distance[j];
        min_index = j;
      }
    }

    for (j = 0; j < KPT_NUM_BZ_SEARCH_SPACE; j++) {
      if (distance[j] < min_distance + tolerance) {
        if (j == min_index) {
          gp = i;
        } else {
          gp = boundary_num_gp + total_num_gp;
        }

        for (k = 0; k < 3; k++) {
          bz_grid_address[gp][k] =
            grid_address[i][k] + bz_search_space[j][k] * mesh[k];
          bz_address_double[k] = bz_grid_address[gp][k] * 2 + is_shift[k];
        }
        bzgp = grg_get_double_grid_index(bz_address_double, bzmesh, is_shift);
        bz_map[bzgp] = gp;
        if (j != min_index) {
          boundary_num_gp++;
        }
      }
    }
  }

  return boundary_num_gp + total_num_gp;
}

static long get_bz_grid_addresses(long bz_grid_address[][3],
                                  long bz_map[][2],
                                  KPTCONST long grid_address[][3],
                                  const long mesh[3],
                                  KPTCONST double rec_lattice[3][3],
                                  const long is_shift[3])
{
  double tolerance, min_distance;
  double q_vector[3], distance[KPT_NUM_BZ_SEARCH_SPACE];
  long i, j, k, min_index, num_gp, multi;

  tolerance = get_tolerance_for_BZ_reduction(rec_lattice, mesh);
  num_gp = 0;

  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    multi = 0;

    for (j = 0; j < KPT_NUM_BZ_SEARCH_SPACE; j++) {
      for (k = 0; k < 3; k++) {
        q_vector[k] =
          ((grid_address[i][k] + bz_search_space[j][k] * mesh[k]) * 2 +
           is_shift[k]) / ((double)mesh[k]) / 2;
      }
      multiply_matrix_vector_d3(q_vector, rec_lattice, q_vector);
      distance[j] = norm_squared_d3(q_vector);
    }
    min_distance = distance[0];
    min_index = 0;
    for (j = 1; j < KPT_NUM_BZ_SEARCH_SPACE; j++) {
      if (distance[j] < min_distance) {
        min_distance = distance[j];
        min_index = j;
      }
    }

    for (j = 0; j < KPT_NUM_BZ_SEARCH_SPACE; j++) {
      if (distance[j] < min_distance + tolerance) {
        for (k = 0; k < 3; k++) {
          bz_grid_address[num_gp][k] =
            grid_address[i][k] + bz_search_space[j][k] * mesh[k];
        }
        num_gp++;
        multi++;
      }
    }
    bz_map[i][0] = multi;
    bz_map[i][1] = num_gp - multi;
  }

  return num_gp;
}

static double get_tolerance_for_BZ_reduction(KPTCONST double rec_lattice[3][3],
                                             const long mesh[3])
{
  long i, j;
  double tolerance;
  double length[3];

  for (i = 0; i < 3; i++) {
    length[i] = 0;
    for (j = 0; j < 3; j++) {
      length[i] += rec_lattice[j][i] * rec_lattice[j][i];
    }
    length[i] /= mesh[i] * mesh[i];
  }
  tolerance = length[0];
  for (i = 1; i < 3; i++) {
    if (tolerance < length[i]) {
      tolerance = length[i];
    }
  }
  tolerance *= 0.01;

  return tolerance;
}

static long get_num_ir(long ir_mapping_table[], const long mesh[3])
{
  long i, num_ir;

  num_ir = 0;

#pragma omp parallel for reduction(+:num_ir)
  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    if (ir_mapping_table[i] == i) {
      num_ir++;
    }
  }

#ifdef _OPENMP
  for (i = 0; i < mesh[0] * mesh[1] * mesh[2]; i++) {
    ir_mapping_table[i] = ir_mapping_table[ir_mapping_table[i]];
  }
#endif

  return num_ir;
}

static long check_mesh_symmetry(const long mesh[3],
                                const long is_shift[3],
                                const MatLONG *rot_reciprocal)
{
  long i, j, k, sum;
  long eq[3];

  eq[0] = 0; /* a=b */
  eq[1] = 0; /* b=c */
  eq[2] = 0; /* c=a */

  /* Check 3 and 6 fold rotations and non-convensional choice of unit cells */
  for (i = 0; i < rot_reciprocal->size; i++) {
    sum = 0;
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        sum += labs(rot_reciprocal->mat[i][j][k]);
      }
    }
    if (sum > 3) {
      return 0;
    }
  }

  for (i = 0; i < rot_reciprocal->size; i++) {
    if (rot_reciprocal->mat[i][0][0] == 0 &&
        rot_reciprocal->mat[i][1][0] == 1 &&
        rot_reciprocal->mat[i][2][0] == 0) {eq[0] = 1;}
    if (rot_reciprocal->mat[i][0][0] == 0 &&
        rot_reciprocal->mat[i][1][0] == 1 &&
        rot_reciprocal->mat[i][2][0] == 0) {eq[1] = 1;}
    if (rot_reciprocal->mat[i][0][0] == 0 &&
        rot_reciprocal->mat[i][1][0] == 0 &&
        rot_reciprocal->mat[i][2][0] == 1) {eq[2] = 1;}
  }


  return (((eq[0] && mesh[0] == mesh[1] && is_shift[0] == is_shift[1]) || (!eq[0])) &&
          ((eq[1] && mesh[1] == mesh[2] && is_shift[1] == is_shift[2]) || (!eq[1])) &&
          ((eq[2] && mesh[2] == mesh[0] && is_shift[2] == is_shift[0]) || (!eq[2])));
}


static long Nint(const double a)
{
  if (a < 0.0)
    return (long) (a - 0.5);
  else
    return (long) (a + 0.5);
}

static double Dabs(const double a)
{
  if (a < 0.0)
    return -a;
  else
    return a;
}

static void transpose_matrix_l3(long a[3][3], KPTCONST long b[3][3])
{
  long c[3][3];
  c[0][0] = b[0][0];
  c[0][1] = b[1][0];
  c[0][2] = b[2][0];
  c[1][0] = b[0][1];
  c[1][1] = b[1][1];
  c[1][2] = b[2][1];
  c[2][0] = b[0][2];
  c[2][1] = b[1][2];
  c[2][2] = b[2][2];
  kpt_copy_matrix_l3(a, c);
}

static void multiply_matrix_l3(long m[3][3],
                               KPTCONST long a[3][3],
                               KPTCONST long b[3][3])
{
  long i, j;                   /* a_ij */
  long c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
        a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  kpt_copy_matrix_l3(m, c);
}

static long check_identity_matrix_l3(KPTCONST long a[3][3],
                                     KPTCONST long b[3][3])
{
  if ( a[0][0] - b[0][0] ||
       a[0][1] - b[0][1] ||
       a[0][2] - b[0][2] ||
       a[1][0] - b[1][0] ||
       a[1][1] - b[1][1] ||
       a[1][2] - b[1][2] ||
       a[2][0] - b[2][0] ||
       a[2][1] - b[2][1] ||
       a[2][2] - b[2][2]) {
    return 0;
  }
  else {
    return 1;
  }
}

static void multiply_matrix_vector_ld3(double v[3],
                                       KPTCONST long a[3][3],
                                       const double b[3])
{
  long i;
  double c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

static void multiply_matrix_vector_l3(long v[3],
                                      KPTCONST long a[3][3],
                                      const long b[3])
{
  long i;
  long c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

static void multiply_matrix_vector_d3(double v[3],
                                      KPTCONST double a[3][3],
                                      const double b[3])
{
  long i;
  double c[3];
  for (i = 0; i < 3; i++)
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  for (i = 0; i < 3; i++)
    v[i] = c[i];
}

static double norm_squared_d3(const double a[3])
{
  return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
