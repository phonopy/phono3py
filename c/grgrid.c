/* Copyright (C) 2020 Atsushi Togo */
/* All rights reserved. */

/* This file is part of kspclib. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the kspclib project nor the names of its */
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
#include <stddef.h>
#include <assert.h>
#include "grgrid.h"
#include "snf3x3.h"

#define IDENTITY_TOL 1e-5

static void reduce_grid_address(long address[3], const long D_diag[3]);
static void reduce_double_grid_address(long address_double[3],
                                       const long D_diag[3]);
static long get_double_grid_index(const long address_double[3],
                                  const long D_diag[3],
                                  const long PS[3]);
static long get_grid_index_from_address(const long address[3],
                                        const long D_diag[3]);
static void get_all_grid_addresses(long grid_address[][3],
                                   const long D_diag[3]);
static void get_grid_address_from_index(long address[3],
                                        const long grid_index,
                                        const long D_diag[3]);
static void get_grid_address(long address[3],
                             const long address_double[3],
                             const long PS[3]);
static void get_double_grid_address(long address_double[3],
                                    const long address[3],
                                    const long PS[3]);
static long rotate_grid_index(const long grid_index,
                              MATCONST long rotation[3][3],
                              const long D_diag[3],
                              const long PS[3]);
static void get_ir_grid_map(long ir_grid_indices[],
                            MATCONST long (*rotations)[3][3],
                            const long num_rot,
                            const long D_diag[3],
                            const long PS[3]);
static long mat_get_determinant_l3(MATCONST long a[3][3]);
static double mat_get_determinant_d3(MATCONST double a[3][3]);
static void mat_cast_matrix_3l_to_3d(double m[3][3], MATCONST long a[3][3]);
static void mat_cast_matrix_3d_to_3l(long m[3][3], MATCONST double a[3][3]);
static long mat_get_similar_matrix_ld3(double m[3][3],
                                       MATCONST long a[3][3],
                                       MATCONST double b[3][3],
                                       const double precision);
static long mat_check_identity_matrix_l3(MATCONST long a[3][3],
                                         MATCONST long b[3][3]);
static long mat_check_identity_matrix_ld3(MATCONST long a[3][3],
                                          MATCONST double b[3][3],
                                          const double symprec);
static long mat_inverse_matrix_d3(double m[3][3],
                                  MATCONST double a[3][3],
                                  const double precision);
static void mat_transpose_matrix_l3(long a[3][3], MATCONST long b[3][3]);
static void mat_multiply_matrix_vector_l3(long v[3],
                                          MATCONST long a[3][3],
                                          const long b[3]);
static void mat_multiply_matrix_l3(long m[3][3],
                                   MATCONST long a[3][3],
                                   MATCONST long b[3][3]);
static void mat_multiply_matrix_ld3(double m[3][3],
                                    MATCONST long a[3][3],
                                    MATCONST double b[3][3]);
static void mat_multiply_matrix_d3(double m[3][3],
                                   MATCONST double a[3][3],
                                   MATCONST double b[3][3]);
static void mat_copy_matrix_l3(long a[3][3], MATCONST long b[3][3]);
static void mat_copy_matrix_d3(double a[3][3], MATCONST double b[3][3]);
static void mat_copy_vector_l3(long a[3], const long b[3]);
static long mat_modulo_l(const long a, const long b);
static long mat_Nint(const double a);
static double mat_Dabs(const double a);


long grg_get_snf3x3(long D_diag[3],
                    long P[3][3],
                    long Q[3][3],
                    MATCONST long A[3][3])
{
  long i, j, succeeded;
  long D[3][3];

  succeeded = 0;

  if (mat_get_determinant_l3(A) == 0) {
    goto err;
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      D[i][j] = A[i][j];
    }
  }

  succeeded = snf3x3(D, P, Q);
  for (i = 0; i < 3; i++) {
    D_diag[i] = D[i][i];
  }

err:
  return succeeded;
}

/*----------------------------------------*/
/* Transform rotations by D(Q^-1)RQ(D^-1) */
/*----------------------------------------*/
/* transformed_rots : D(Q^-1)RQ(D^-1) */
/* rotations : [num_rot][3][3] */
/*    Defined as q' = Rq where q is in the reciprocal primitive basis */
/*    vectors. */
/* num_rot : Number of rotations */
long grg_transform_rotations(long (*transformed_rots)[3][3],
                             MATCONST long (*rotations)[3][3],
                             const long num_rot,
                             const long D_diag[3],
                             MATCONST long Q[3][3])
{
  long i, j, k;
  double r[3][3], Q_double[3][3];

  /* Compute D(Q^-1)RQ(D^-1) by three steps */
  /* It is assumed that |det(Q)|=1 and Q^-1 has relatively small round-off */
  /* error, and we want to divide by D carefully. */
  /* 1. Compute (Q^-1)RQ */
  /* 2. Compute D(Q^-1)RQ */
  /* 3. Compute D(Q^-1)RQ(D^-1) */
  mat_cast_matrix_3l_to_3d(Q_double, Q);
  for (i = 0; i < num_rot; i++) {
    mat_get_similar_matrix_ld3(r, rotations[i], Q_double, 0);
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        r[j][k] *= D_diag[j];
        r[j][k] /= D_diag[k];
      }
    }
    mat_cast_matrix_3d_to_3l(transformed_rots[i], r);
    if (!mat_check_identity_matrix_ld3(transformed_rots[i], r, IDENTITY_TOL)) {
      return 0;
    }
  }

  return 1;
}

/* -------------------------------*/
/* Get all address in single grid */
/* -------------------------------*/
/* address : Single grid address. */
/* D_diag : Diagnal elements of D. */
void grg_get_all_grid_addresses(long grid_address[][3], const long D_diag[3])
{
  get_all_grid_addresses(grid_address, D_diag);
}

/* -------------------------------------------------------*/
/* Get address in double grid from address in single grid */
/* -------------------------------------------------------*/
/* address_double : Double grid address. */
/* address : Single grid address. */
/* D_diag : Diagnal elements of D. */
/* PS : Shifts transformed by P. s_i is 0 or 1. */
void grg_get_double_grid_address(long address_double[3],
                                 const long address[3],
                                 const long D_diag[3],
                                 const long PS[3])
{
  get_double_grid_address(address_double, address, PS);
  reduce_double_grid_address(address_double, D_diag);
}

/* -------------------------------------------------------*/
/* Get address in single grid from address in double grid */
/* -------------------------------------------------------*/
/* address : Single grid address. */
/* address_double : Double grid address. */
/* D_diag : Diagnal elements of D. */
/* PS : Shifts transformed by P. s_i is 0 or 1. */
void grg_get_grid_address(long address[3],
                          const long address_double[3],
                          const long D_diag[3],
                          const long PS[3])
{
  get_grid_address(address, address_double, PS);
  reduce_grid_address(address, D_diag);
}

/* -------------------------------------------------*/
/* Get grid point index from address in double grid */
/* -------------------------------------------------*/
/* address_double : Double grid address. */
/* D_diag : Diagnal elements of D. */
/* PS : Shifts transformed by P. s_i is 0 or 1. */
long grg_get_double_grid_index(const long address_double[3],
                               const long D_diag[3],
                               const long PS[3])
{
  return get_double_grid_index(address_double, D_diag, PS);
}

/* -------------------------------------------------*/
/* Get grid point index from address in single grid */
/* -------------------------------------------------*/
/* address : Single grid address. */
/* D_diag : Diagnal elements of D. */
long grg_get_grid_index(const long address[3], const long D_diag[3])
{
  long red_adrs[3];

  mat_copy_vector_l3(red_adrs, address);
  reduce_grid_address(red_adrs, D_diag);
  return get_grid_index_from_address(red_adrs, D_diag);
}

/* ---------------------------------------*/
/* Get grid address from grid point index */
/* ---------------------------------------*/
/* address : Single grid address. */
/* D_diag : Diagnal elements of D. */
void grg_get_grid_address_from_index(long address[3],
                                     const long grid_index,
                                     const long D_diag[3])
{
  get_grid_address_from_index(address, grid_index, D_diag);
}


/* ---------------------------*/
/* Rotate grid point by index */
/* ---------------------------*/
long grg_rotate_grid_index(const long grid_index,
                           MATCONST long rotation[3][3],
                           const long D_diag[3],
                           const long PS[3])
{
  return rotate_grid_index(grid_index, rotation, D_diag, PS);
}

/* -----------------------------*/
/* Find irreducible grid points */
/* -----------------------------*/
void grg_get_ir_grid_map(long ir_grid_indices[],
                         MATCONST long (*rotations)[3][3],
                         const long num_rot,
                         const long D_diag[3],
                         const long PS[3])
{
  get_ir_grid_map(ir_grid_indices,
                  rotations,
                  num_rot,
                  D_diag,
                  PS);
}

/* Extract unique rotation matrices and transpose them. */
/* When is_time_reversal == 1, inverse of the extracted matrices are */
/* included. */
/* Return 0 if failed */
long grg_get_reciprocal_point_group(long rec_rotations[48][3][3],
                                    MATCONST long (*rotations)[3][3],
                                    const long num_rot,
                                    const long is_time_reversal)
{
  long i, j, num_rot_ret, inv_exist;
  MATCONST long inversion[3][3] = {
    {-1, 0, 0 },
    { 0,-1, 0 },
    { 0, 0,-1 }
  };

  num_rot_ret = 0;
  for (i = 0; i < num_rot; i++) {
    for (j = 0; j < num_rot_ret; j++) {
      if (mat_check_identity_matrix_l3(rotations[i], rec_rotations[j])) {
        goto escape;
      }
    }
    if (num_rot_ret == 48) {
      goto err;
    }
    mat_copy_matrix_l3(rec_rotations[num_rot_ret], rotations[i]);
    num_rot_ret++;
  escape:
    ;
  }

  inv_exist = 0;
  if (is_time_reversal) {
    for (i = 0; i < num_rot_ret; i++) {
      if (mat_check_identity_matrix_l3(inversion, rec_rotations[i])) {
        inv_exist = 1;
        break;
      }
    }

    if (!inv_exist) {
      if (num_rot_ret > 24) {
        goto err;
      }

      for (i = 0; i < num_rot_ret; i++) {
        mat_multiply_matrix_l3(rec_rotations[num_rot_ret + i],
                               inversion, rec_rotations[i]);
      }
      num_rot_ret *= 2;
    }
  }

  for (i = 0; i < num_rot_ret; i++) {
    mat_transpose_matrix_l3(rec_rotations[i], rec_rotations[i]);
  }

  return num_rot_ret;
err:
  return 0;
}


static void reduce_grid_address(long address[3], const long D_diag[3])
{
  long i;

  for (i = 0; i < 3; i++) {
    address[i] = mat_modulo_l(address[i], D_diag[i]);
  }
}

static void reduce_double_grid_address(long address_double[3],
                                       const long D_diag[3])
{
  long i;

  for (i = 0; i < 3; i++) {
    address_double[i] = mat_modulo_l(address_double[i], 2 * D_diag[i]);
  }
}

static long get_double_grid_index(const long address_double[3],
                                  const long D_diag[3],
                                  const long PS[3])
{
  long address[3];

  grg_get_grid_address(address,
                       address_double,
                       D_diag,
                       PS);
  return get_grid_index_from_address(address, D_diag);
}

/* Here address elements have to be zero or positive. */
/* Therefore reduction to interval [0, D_diag[i]) has to be */
/* done outside of this function. */
/* See kgrid.h about GRID_ORDER_XYZ information. */
static long get_grid_index_from_address(const long address[3],
                                        const long D_diag[3])
{
#ifndef GRID_ORDER_XYZ
  return (address[2] * D_diag[0] * D_diag[1]
          + address[1] * D_diag[0] + address[0]);
#else
  return (address[0] * D_diag[1] * D_diag[2]
          + address[1] * D_diag[2] + address[2]);
#endif
}

static void get_all_grid_addresses(long grid_address[][3],
                                   const long D_diag[3])
{
  long i, j, k, grid_index;
  long address[3];

  for (i = 0; i < D_diag[0]; i++) {
    address[0] = i;
    for (j = 0; j < D_diag[1]; j++) {
      address[1] = j;
      for (k = 0; k < D_diag[2]; k++) {
        address[2] = k;
        grid_index = get_grid_index_from_address(address, D_diag);
        mat_copy_vector_l3(grid_address[grid_index], address);
      }
    }
  }
}

/* See grg_get_grid_address_from_index */
static void get_grid_address_from_index(long address[3],
                                        const long grid_index,
                                        const long D_diag[3])
{
  long nn;

#ifndef GRID_ORDER_XYZ
  nn = D_diag[0] * D_diag[1];
  address[0] = grid_index % D_diag[0];
  address[2] = grid_index / nn;
  address[1] = (grid_index - address[2] * nn) / D_diag[0];
#else
  nn = D_diag[1] * D_diag[2];
  address[2] = grid_index % D_diag[2];
  address[0] = grid_index / nn;
  address[1] = (grid_index - address[0] * nn) / D_diag[2];
#endif
}

/* Usually address has to be reduced to [0, D_diag[i]) */
/* by calling reduce_grid_address after this operation. */
static void get_grid_address(long address[3],
                             const long address_double[3],
                             const long PS[3])
{
  long i;

  for (i = 0; i < 3; i++) {
    address[i] = (address_double[i] - PS[i]) / 2;
  }
}

/* Usually address_double has to be reduced to [0, 2*D_diag[i]) */
/* by calling reduce_double_grid_address after this operation. */
static void get_double_grid_address(long address_double[3],
                                    const long address[3],
                                    const long PS[3])
{
  long i;

  for (i = 0; i < 3; i++) {
    address_double[i] = address[i] * 2 + PS[i];
  }
}

static long rotate_grid_index(const long grid_index,
                              MATCONST long rotation[3][3],
                              const long D_diag[3],
                              const long PS[3])
{
  long adrs[3], dadrs[3], dadrs_rot[3];

  get_grid_address_from_index(adrs, grid_index, D_diag);
  get_double_grid_address(dadrs, adrs, PS);
  mat_multiply_matrix_vector_l3(dadrs_rot, rotation, dadrs);
  return get_double_grid_index(dadrs_rot, D_diag, PS);
}

static void get_ir_grid_map(long ir_grid_indices[],
                            MATCONST long (*rotations)[3][3],
                            const long num_rot,
                            const long D_diag[3],
                            const long PS[3])
{
  long gp, num_gp, r_gp;
  long i;

  num_gp = D_diag[0] * D_diag[1] * D_diag[2];

  for (gp = 0; gp < num_gp; gp++) {
    ir_grid_indices[gp] = num_gp;
  }

  /* Do not simply multithreaded this for-loop. */
  /* This algorithm contains race condition in different gp's. */
  for (gp = 0; gp < num_gp; gp++) {
    for (i = 0; i < num_rot; i++) {
      r_gp = rotate_grid_index(gp, rotations[i], D_diag, PS);
      if (r_gp < gp) {
        ir_grid_indices[gp] = ir_grid_indices[r_gp];
        break;
      }
    }
    if (ir_grid_indices[gp] == num_gp) {
      ir_grid_indices[gp] = gp;
    }
  }

}

static long mat_get_determinant_l3(MATCONST long a[3][3])
{
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
    + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2])
    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

static double mat_get_determinant_d3(MATCONST double a[3][3])
{
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
    + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2])
    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

static void mat_cast_matrix_3l_to_3d(double m[3][3], MATCONST long a[3][3])
{
  m[0][0] = a[0][0];
  m[0][1] = a[0][1];
  m[0][2] = a[0][2];
  m[1][0] = a[1][0];
  m[1][1] = a[1][1];
  m[1][2] = a[1][2];
  m[2][0] = a[2][0];
  m[2][1] = a[2][1];
  m[2][2] = a[2][2];
}

static void mat_cast_matrix_3d_to_3l(long m[3][3], MATCONST double a[3][3])
{
  m[0][0] = mat_Nint(a[0][0]);
  m[0][1] = mat_Nint(a[0][1]);
  m[0][2] = mat_Nint(a[0][2]);
  m[1][0] = mat_Nint(a[1][0]);
  m[1][1] = mat_Nint(a[1][1]);
  m[1][2] = mat_Nint(a[1][2]);
  m[2][0] = mat_Nint(a[2][0]);
  m[2][1] = mat_Nint(a[2][1]);
  m[2][2] = mat_Nint(a[2][2]);
}

static long mat_get_similar_matrix_ld3(double m[3][3],
                                       MATCONST long a[3][3],
                                       MATCONST double b[3][3],
                                       const double precision)
{
  double c[3][3];
  if (!mat_inverse_matrix_d3(c, b, precision)) {
    warning_print("No similar matrix due to 0 determinant.\n");
    return 0;
  }
  mat_multiply_matrix_ld3(m, a, b);
  mat_multiply_matrix_d3(m, c, m);
  return 1;
}

static long mat_check_identity_matrix_l3(MATCONST long a[3][3],
                                         MATCONST long b[3][3])
{
  if (a[0][0] - b[0][0] ||
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

static long mat_check_identity_matrix_ld3(MATCONST long a[3][3],
                                          MATCONST double b[3][3],
                                          const double symprec)
{
  if (mat_Dabs(a[0][0] - b[0][0]) > symprec ||
      mat_Dabs(a[0][1] - b[0][1]) > symprec ||
      mat_Dabs(a[0][2] - b[0][2]) > symprec ||
      mat_Dabs(a[1][0] - b[1][0]) > symprec ||
      mat_Dabs(a[1][1] - b[1][1]) > symprec ||
      mat_Dabs(a[1][2] - b[1][2]) > symprec ||
      mat_Dabs(a[2][0] - b[2][0]) > symprec ||
      mat_Dabs(a[2][1] - b[2][1]) > symprec ||
      mat_Dabs(a[2][2] - b[2][2]) > symprec) {
    return 0;
  }
  else {
    return 1;
  }
}

static long mat_inverse_matrix_d3(double m[3][3],
                                  MATCONST double a[3][3],
                                  const double precision)
{
  double det;
  double c[3][3];
  det = mat_get_determinant_d3(a);
  if (mat_Dabs(det) < precision) {
    warning_print("No inverse matrix (det=%f)\n", det);
    return 0;
  }

  c[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
  c[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
  c[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
  c[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) / det;
  c[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / det;
  c[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) / det;
  c[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  c[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  c[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  mat_copy_matrix_d3(m, c);
  return 1;
}

static void mat_transpose_matrix_l3(long a[3][3], MATCONST long b[3][3])
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
  mat_copy_matrix_l3(a, c);
}

static void mat_multiply_matrix_vector_l3(long v[3],
                                          MATCONST long a[3][3],
                                          const long b[3])
{
  long i;
  long c[3];
  for (i = 0; i < 3; i++) {
    c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
  }
  for (i = 0; i < 3; i++) {
    v[i] = c[i];
  }
}

static void mat_multiply_matrix_l3(long m[3][3],
                                   MATCONST long a[3][3],
                                   MATCONST long b[3][3])
{
  long i, j;                   /* a_ij */
  long c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
        a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  mat_copy_matrix_l3(m, c);
}

static void mat_multiply_matrix_ld3(double m[3][3],
                                    MATCONST long a[3][3],
                                    MATCONST double b[3][3])
{
  long i, j;                   /* a_ij */
  double c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
        a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  mat_copy_matrix_d3(m, c);
}

static void mat_multiply_matrix_d3(double m[3][3],
                                   MATCONST double a[3][3],
                                   MATCONST double b[3][3])
{
  long i, j;                   /* a_ij */
  double c[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      c[i][j] =
        a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  mat_copy_matrix_d3(m, c);
}

static void mat_copy_matrix_l3(long a[3][3], MATCONST long b[3][3])
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

static void mat_copy_matrix_d3(double a[3][3], MATCONST double b[3][3])
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

static void mat_copy_vector_l3(long a[3], const long b[3])
{
  a[0] = b[0];
  a[1] = b[1];
  a[2] = b[2];
}

static long mat_modulo_l(const long a, const long b)
{
  long c;
  c = a % b;
  if (c < 0) {
    c += b;
  }
  return c;
}

static long mat_Nint(const double a)
{
  if (a < 0.0)
    return (long) (a - 0.5);
  else
    return (long) (a + 0.5);
}

static double mat_Dabs(const double a)
{
  if (a < 0.0)
    return -a;
  else
    return a;
}
