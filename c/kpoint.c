/* Copyright (C) 2008 Atsushi Togo */
/* All rights reserved. */

/* This file is part of spglib. */

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
#include "mathfunc.h"
#include "kpoint.h"
#include "kgrid.h"

#ifdef KPTWARNING
#include <stdio.h>
#define warning_print(...) fprintf(stderr,__VA_ARGS__)
#else
#define warning_print(...)
#endif

static MatINT *get_point_group_reciprocal(const MatINT * rotations,
                                          const int is_time_reversal);
static MatINT *get_point_group_reciprocal_with_q(const MatINT * rot_reciprocal,
                                                 const double symprec,
                                                 const long num_q,
                                                 SPGCONST double qpoints[][3]);
static long get_dense_ir_reciprocal_mesh(int grid_address[][3],
                                         long ir_mapping_table[],
                                         const int mesh[3],
                                         const int is_shift[3],
                                         const MatINT *rot_reciprocal);
static long get_dense_ir_reciprocal_mesh_normal(int grid_address[][3],
                                                long ir_mapping_table[],
                                                const int mesh[3],
                                                const int is_shift[3],
                                                const MatINT *rot_reciprocal);
static long get_dense_ir_reciprocal_mesh_distortion(int grid_address[][3],
                                                    long ir_mapping_table[],
                                                    const int mesh[3],
                                                    const int is_shift[3],
                                                    const MatINT *rot_reciprocal);
static long get_dense_num_ir(long ir_mapping_table[], const int mesh[3]);
static int check_mesh_symmetry(const int mesh[3],
                               const int is_shift[3],
                               const MatINT *rot_reciprocal);



long kpt_get_dense_irreducible_reciprocal_mesh(int grid_address[][3],
                                               long ir_mapping_table[],
                                               const int mesh[3],
                                               const int is_shift[3],
                                               const MatINT *rot_reciprocal)
{
  long num_ir;

  num_ir = get_dense_ir_reciprocal_mesh(grid_address,
                                        ir_mapping_table,
                                        mesh,
                                        is_shift,
                                        rot_reciprocal);

  return num_ir;
}

MatINT *kpt_get_point_group_reciprocal(const MatINT * rotations,
                                       const int is_time_reversal)
{
  return get_point_group_reciprocal(rotations, is_time_reversal);
}

MatINT *kpt_get_point_group_reciprocal_with_q(const MatINT * rot_reciprocal,
                                              const double symprec,
                                              const long num_q,
                                              SPGCONST double qpoints[][3])
{
  return get_point_group_reciprocal_with_q(rot_reciprocal,
                                           symprec,
                                           num_q,
                                           qpoints);
}

/* Return NULL if failed */
static MatINT *get_point_group_reciprocal(const MatINT * rotations,
                                          const int is_time_reversal)
{
  int i, j, num_rot;
  MatINT *rot_reciprocal, *rot_return;
  int *unique_rot;
  SPGCONST int inversion[3][3] = {
    {-1, 0, 0 },
    { 0,-1, 0 },
    { 0, 0,-1 }
  };

  rot_reciprocal = NULL;
  rot_return = NULL;
  unique_rot = NULL;

  if (is_time_reversal) {
    if ((rot_reciprocal = mat_alloc_MatINT(rotations->size * 2)) == NULL) {
      return NULL;
    }
  } else {
    if ((rot_reciprocal = mat_alloc_MatINT(rotations->size)) == NULL) {
      return NULL;
    }
  }

  if ((unique_rot = (int*)malloc(sizeof(int) * rot_reciprocal->size)) == NULL) {
    warning_print("spglib: Memory of unique_rot could not be allocated.");
    mat_free_MatINT(rot_reciprocal);
    rot_reciprocal = NULL;
    return NULL;
  }

  for (i = 0; i < rot_reciprocal->size; i++) {
    unique_rot[i] = -1;
  }

  for (i = 0; i < rotations->size; i++) {
    mat_transpose_matrix_i3(rot_reciprocal->mat[i], rotations->mat[i]);

    if (is_time_reversal) {
      mat_multiply_matrix_i3(rot_reciprocal->mat[rotations->size+i],
                             inversion,
                             rot_reciprocal->mat[i]);
    }
  }

  num_rot = 0;
  for (i = 0; i < rot_reciprocal->size; i++) {
    for (j = 0; j < num_rot; j++) {
      if (mat_check_identity_matrix_i3(rot_reciprocal->mat[unique_rot[j]],
                                       rot_reciprocal->mat[i])) {
        goto escape;
      }
    }
    unique_rot[num_rot] = i;
    num_rot++;
  escape:
    ;
  }

  if ((rot_return = mat_alloc_MatINT(num_rot)) != NULL) {
    for (i = 0; i < num_rot; i++) {
      mat_copy_matrix_i3(rot_return->mat[i], rot_reciprocal->mat[unique_rot[i]]);
    }
  }

  free(unique_rot);
  unique_rot = NULL;
  mat_free_MatINT(rot_reciprocal);
  rot_reciprocal = NULL;

  return rot_return;
}

/* Return NULL if failed */
static MatINT *get_point_group_reciprocal_with_q(const MatINT * rot_reciprocal,
                                                 const double symprec,
                                                 const long num_q,
                                                 SPGCONST double qpoints[][3])
{
  int i, j, k, l, is_all_ok, num_rot;
  int *ir_rot;
  double q_rot[3], diff[3];
  MatINT * rot_reciprocal_q;

  ir_rot = NULL;
  rot_reciprocal_q = NULL;
  is_all_ok = 0;
  num_rot = 0;

  if ((ir_rot = (int*)malloc(sizeof(int) * rot_reciprocal->size)) == NULL) {
    warning_print("spglib: Memory of ir_rot could not be allocated.");
    return NULL;
  }

  for (i = 0; i < rot_reciprocal->size; i++) {
    ir_rot[i] = -1;
  }
  for (i = 0; i < rot_reciprocal->size; i++) {
    for (j = 0; j < num_q; j++) {
      is_all_ok = 0;
      mat_multiply_matrix_vector_id3(q_rot,
                                     rot_reciprocal->mat[i],
                                     qpoints[j]);

      for (k = 0; k < num_q; k++) {
        for (l = 0; l < 3; l++) {
          diff[l] = q_rot[l] - qpoints[k][l];
          diff[l] -= mat_Nint(diff[l]);
        }

        if (mat_Dabs(diff[0]) < symprec &&
            mat_Dabs(diff[1]) < symprec &&
            mat_Dabs(diff[2]) < symprec) {
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

  if ((rot_reciprocal_q = mat_alloc_MatINT(num_rot)) != NULL) {
    for (i = 0; i < num_rot; i++) {
      mat_copy_matrix_i3(rot_reciprocal_q->mat[i],
                         rot_reciprocal->mat[ir_rot[i]]);
    }
  }

  free(ir_rot);
  ir_rot = NULL;

  return rot_reciprocal_q;
}

static long get_dense_ir_reciprocal_mesh(int grid_address[][3],
                                         long ir_mapping_table[],
                                         const int mesh[3],
                                         const int is_shift[3],
                                         const MatINT *rot_reciprocal)
{
  if (check_mesh_symmetry(mesh, is_shift, rot_reciprocal)) {
    return get_dense_ir_reciprocal_mesh_normal(grid_address,
                                               ir_mapping_table,
                                               mesh,
                                               is_shift,
                                               rot_reciprocal);
  } else {
    return get_dense_ir_reciprocal_mesh_distortion(grid_address,
                                                   ir_mapping_table,
                                                   mesh,
                                                   is_shift,
                                                   rot_reciprocal);
  }
}

static long get_dense_ir_reciprocal_mesh_normal(int grid_address[][3],
                                                long ir_mapping_table[],
                                                const int mesh[3],
                                                const int is_shift[3],
                                                const MatINT *rot_reciprocal)
{
  /* In the following loop, mesh is doubled. */
  /* Even and odd mesh numbers correspond to */
  /* is_shift[i] are 0 or 1, respectively. */
  /* is_shift = [0,0,0] gives Gamma center mesh. */
  /* grid: reducible grid points */
  /* ir_mapping_table: the mapping from each point to ir-point. */

  long i, grid_point_rot;
  int j;
  int address_double[3], address_double_rot[3];

  kgd_get_all_grid_addresses(grid_address, mesh);

#pragma omp parallel for private(j, grid_point_rot, address_double, address_double_rot)
  for (i = 0; i < mesh[0] * mesh[1] * (long)(mesh[2]); i++) {
    kgd_get_grid_address_double_mesh(address_double,
                                     grid_address[i],
                                     mesh,
                                     is_shift);
    ir_mapping_table[i] = i;
    for (j = 0; j < rot_reciprocal->size; j++) {
      mat_multiply_matrix_vector_i3(address_double_rot,
                                    rot_reciprocal->mat[j],
                                    address_double);
      grid_point_rot = kgd_get_dense_grid_point_double_mesh(address_double_rot, mesh);
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

  return get_dense_num_ir(ir_mapping_table, mesh);
}

static long
get_dense_ir_reciprocal_mesh_distortion(int grid_address[][3],
                                        long ir_mapping_table[],
                                        const int mesh[3],
                                        const int is_shift[3],
                                        const MatINT *rot_reciprocal)
{
  long i, grid_point_rot;
  int j, k, indivisible;
  int address_double[3], address_double_rot[3];
  long long_address_double[3], long_address_double_rot[3], divisor[3];

  /* divisor, long_address_double, and long_address_double_rot have */
  /* long integer type to treat dense mesh. */

  kgd_get_all_grid_addresses(grid_address, mesh);

  for (j = 0; j < 3; j++) {
    divisor[j] = mesh[(j + 1) % 3] * mesh[(j + 2) % 3];
  }

#pragma omp parallel for private(j, k, grid_point_rot, address_double, address_double_rot, long_address_double, long_address_double_rot)
  for (i = 0; i < mesh[0] * mesh[1] * (long)(mesh[2]); i++) {
    kgd_get_grid_address_double_mesh(address_double,
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
        kgd_get_dense_grid_point_double_mesh(address_double_rot, mesh);
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

  return get_dense_num_ir(ir_mapping_table, mesh);
}

static long get_dense_num_ir(long ir_mapping_table[], const int mesh[3])
{
  long i, num_ir;

  num_ir = 0;

#pragma omp parallel for reduction(+:num_ir)
  for (i = 0; i < mesh[0] * mesh[1] * (long)(mesh[2]); i++) {
    if (ir_mapping_table[i] == i) {
      num_ir++;
    }
  }

#ifdef _OPENMP
  for (i = 0; i < mesh[0] * mesh[1] * (long)(mesh[2]); i++) {
    ir_mapping_table[i] = ir_mapping_table[ir_mapping_table[i]];
  }
#endif

  return num_ir;
}

static int check_mesh_symmetry(const int mesh[3],
                               const int is_shift[3],
                               const MatINT *rot_reciprocal)
{
  int i, j, k, sum;
  int eq[3];

  eq[0] = 0; /* a=b */
  eq[1] = 0; /* b=c */
  eq[2] = 0; /* c=a */

  /* Check 3 and 6 fold rotations and non-convensional choice of unit cells */
  for (i = 0; i < rot_reciprocal->size; i++) {
    sum = 0;
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        sum += abs(rot_reciprocal->mat[i][j][k]);
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
