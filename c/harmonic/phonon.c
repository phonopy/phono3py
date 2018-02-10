/* Copyright (C) 2015 Atsushi Togo */
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

#include <math.h>
#include <string.h>
#include <dynmat.h>
#include <phonon.h>
#include <lapack_wrapper.h>

static int collect_undone_grid_points(int *undone,
                                      char *phonon_done,
                                      const int num_grid_points,
                                      const int *grid_points);
static void get_undone_phonons(double *frequencies,
                               lapack_complex_double *eigenvectors,
                               const int *undone_grid_points,
                               const int num_undone_grid_points,
                               const int *grid_address,
                               const int *mesh,
                               const double *fc2,
                               PHPYCONST double(*svecs_fc2)[27][3],
                               const int *multi_fc2,
                               const int num_patom,
                               const int num_satom,
                               const double *masses_fc2,
                               const int *p2s_fc2,
                               const int *s2p_fc2,
                               const double unit_conversion_factor,
                               const double *born,
                               const double *dielectric,
                               const double *reciprocal_lattice,
                               const double *q_direction,
                               const double nac_factor,
                               const char uplo);
static int get_phonons(lapack_complex_double *eigvecs,
                       double *freqs,
                       const double q[3],
                       const double *fc2,
                       const double *masses,
                       const int *p2s,
                       const int *s2p,
                       const int *multi,
                       const int num_patom,
                       const int num_satom,
                       PHPYCONST double(*svecs)[27][3],
                       const double *born,
                       const double *dielectric,
                       const double *reciprocal_lattice,
                       const double *q_direction,
                       const double nac_factor,
                       const double unit_conversion_factor,
                       const char uplo);
static void get_dynamical_matrix(lapack_complex_double *dynmat,
                                 const double q[3],
                                 const double *fc2,
                                 const double *masses,
                                 const int *p2s,
                                 const int *s2p,
                                 const int *multi,
                                 const int num_patom,
                                 const int num_satom,
                                 PHPYCONST double(*svecs)[27][3],
                                 const double *born,
                                 const double *dielectric,
                                 const double *reciprocal_lattice,
                                 const double *q_direction,
                                 const double nac_factor);
static double * get_charge_sum(const int num_patom,
                               const int num_satom,
                               const double q[3],
                               const double *born, /* Wang NAC unless NULL */
                               const double *dielectric,
                               const double *reciprocal_lattice,
                               const double *q_direction,
                               const double nac_factor);

void get_phonons_at_gridpoints(double *frequencies,
                               lapack_complex_double *eigenvectors,
                               char *phonon_done,
                               const int num_phonons,
                               const int *grid_points,
                               const int num_grid_points,
                               const int *grid_address,
                               const int *mesh,
                               const double *fc2,
                               PHPYCONST double(*svecs_fc2)[27][3],
                               const int *multi_fc2,
                               const int num_patom,
                               const int num_satom,
                               const double *masses_fc2,
                               const int *p2s_fc2,
                               const int *s2p_fc2,
                               const double unit_conversion_factor,
                               const double *born,
                               const double *dielectric,
                               const double *reciprocal_lattice,
                               const double *q_direction,
                               const double nac_factor,
                               const char uplo)
{
  int num_undone;
  int *undone;

  undone = (int*)malloc(sizeof(int) * num_phonons);
  num_undone = collect_undone_grid_points(undone,
                                          phonon_done,
                                          num_grid_points,
                                          grid_points);

  get_undone_phonons(frequencies,
                     eigenvectors,
                     undone,
                     num_undone,
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

  free(undone);
}

static int collect_undone_grid_points(int *undone,
                                      char *phonon_done,
                                      const int num_grid_points,
                                      const int *grid_points)
{
  int i, gp, num_undone;

  num_undone = 0;
  for (i = 0; i < num_grid_points; i++) {
    gp = grid_points[i];
    if (phonon_done[gp] == 0) {
      undone[num_undone] = gp;
      num_undone++;
      phonon_done[gp] = 1;
    }
  }

  return num_undone;
}

static void get_undone_phonons(double *frequencies,
                               lapack_complex_double *eigenvectors,
                               const int *undone_grid_points,
                               const int num_undone_grid_points,
                               const int *grid_address,
                               const int *mesh,
                               const double *fc2,
                               PHPYCONST double(*svecs_fc2)[27][3],
                               const int *multi_fc2,
                               const int num_patom,
                               const int num_satom,
                               const double *masses_fc2,
                               const int *p2s_fc2,
                               const int *s2p_fc2,
                               const double unit_conversion_factor,
                               const double *born,
                               const double *dielectric,
                               const double *reciprocal_lattice,
                               const double *q_direction,
                               const double nac_factor,
                               const char uplo)
{
  int i, j, gp, num_band;
  double q[3];

  num_band = num_patom * 3;

/* To avoid multithreaded BLAS in OpenMP loop */
#ifndef MULTITHREADED_BLAS
#pragma omp parallel for private(j, q, gp)
#endif
  for (i = 0; i < num_undone_grid_points; i++) {
    gp = undone_grid_points[i];
    for (j = 0; j < 3; j++) {
      q[j] = ((double)grid_address[gp * 3 + j]) / mesh[j];
    }

    if (gp == 0) {
      get_phonons(eigenvectors + num_band * num_band * gp,
                  frequencies + num_band * gp,
                  q,
                  fc2,
                  masses_fc2,
                  p2s_fc2,
                  s2p_fc2,
                  multi_fc2,
                  num_patom,
                  num_satom,
                  svecs_fc2,
                  born,
                  dielectric,
                  reciprocal_lattice,
                  q_direction,
                  nac_factor,
                  unit_conversion_factor,
                  uplo);
    } else {
      get_phonons(eigenvectors + num_band * num_band * gp,
                  frequencies + num_band * gp,
                  q,
                  fc2,
                  masses_fc2,
                  p2s_fc2,
                  s2p_fc2,
                  multi_fc2,
                  num_patom,
                  num_satom,
                  svecs_fc2,
                  born,
                  dielectric,
                  reciprocal_lattice,
                  NULL,
                  nac_factor,
                  unit_conversion_factor,
                  uplo);
    }
  }
}

static int get_phonons(lapack_complex_double *eigvecs,
                       double *freqs,
                       const double q[3],
                       const double *fc2,
                       const double *masses,
                       const int *p2s,
                       const int *s2p,
                       const int *multi,
                       const int num_patom,
                       const int num_satom,
                       PHPYCONST double(*svecs)[27][3],
                       const double *born,
                       const double *dielectric,
                       const double *reciprocal_lattice,
                       const double *q_direction,
                       const double nac_factor,
                       const double unit_conversion_factor,
                       const char uplo)
{
  int i, num_band, info;

  num_band = num_patom * 3;

  /* Store dynamical matrix in eigvecs array. */
  get_dynamical_matrix(eigvecs,
                       q,
                       fc2,
                       masses,
                       p2s,
                       s2p,
                       multi,
                       num_patom,
                       num_satom,
                       svecs,
                       born,
                       dielectric,
                       reciprocal_lattice,
                       q_direction,
                       nac_factor);

  /* Store eigenvalues in freqs array. */
  /* Eigenvectors are overwritten on eigvecs array. */
  info = phonopy_zheev(freqs, eigvecs, num_band, uplo);

  /* Sqrt of eigenvalues are re-stored in freqs array.*/
  for (i = 0; i < num_band; i++) {
    freqs[i] = sqrt(fabs(freqs[i])) *
      ((freqs[i] > 0) - (freqs[i] < 0)) * unit_conversion_factor;
  }

  return info;
}

static void get_dynamical_matrix(lapack_complex_double *dynmat,
                                 const double q[3],
                                 const double *fc2,
                                 const double *masses,
                                 const int *p2s,
                                 const int *s2p,
                                 const int *multi,
                                 const int num_patom,
                                 const int num_satom,
                                 PHPYCONST double(*svecs)[27][3],
                                 const double *born, /* Wang NAC unless NULL */
                                 const double *dielectric,
                                 const double *reciprocal_lattice,
                                 const double *q_direction,
                                 const double nac_factor)
{
  double *charge_sum;

  charge_sum = NULL;

  if (born) {
    charge_sum = get_charge_sum(num_patom,
                                num_satom,
                                q,
                                born,
                                dielectric,
                                reciprocal_lattice,
                                q_direction,
                                nac_factor);
  } else {
    charge_sum = NULL;
  }

  dym_get_dynamical_matrix_at_q((double*)dynmat,
                                num_patom,
                                num_satom,
                                fc2,
                                q,
                                svecs,
                                multi,
                                masses,
                                s2p,
                                p2s,
                                charge_sum,
                                0);
  if (charge_sum) {
    free(charge_sum);
    charge_sum = NULL;
  }
}

static double * get_charge_sum(const int num_patom,
                               const int num_satom,
                               const double q[3],
                               const double *born,
                               const double *dielectric,
                               const double *reciprocal_lattice,
                               const double *q_direction,
                               const double nac_factor)
{
  int i, j;
  double inv_dielectric_factor, dielectric_factor, tmp_val;
  double q_cart[3];
  double *charge_sum;

  if (fabs(q[0]) < 1e-10 && fabs(q[1]) < 1e-10 && fabs(q[2]) < 1e-10 &&
      (!q_direction)) {
    charge_sum = NULL;
  } else {
    charge_sum = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
    if (q_direction) {
      for (i = 0; i < 3; i++) {
        q_cart[i] = 0.0;
        for (j = 0; j < 3; j++) {
          q_cart[i] += reciprocal_lattice[i * 3 + j] * q_direction[j];
        }
      }
    } else {
      for (i = 0; i < 3; i++) {
        q_cart[i] = 0.0;
        for (j = 0; j < 3; j++) {
          q_cart[i] += reciprocal_lattice[i * 3 + j] * q[j];
        }
      }
    }

    inv_dielectric_factor = 0.0;
    for (i = 0; i < 3; i++) {
      tmp_val = 0.0;
      for (j = 0; j < 3; j++) {
        tmp_val += dielectric[i * 3 + j] * q_cart[j];
      }
      inv_dielectric_factor += tmp_val * q_cart[i];
    }
    /* N = num_satom / num_patom = number of prim-cell in supercell */
    /* N is used for Wang's method. */
    dielectric_factor = nac_factor /
      inv_dielectric_factor / num_satom * num_patom;
    dym_get_charge_sum(charge_sum,
                       num_patom,
                       dielectric_factor,
                       q_cart,
                       born);
  }

  return charge_sum;
}
