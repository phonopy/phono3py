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
#include <stddef.h>
#include <dynmat.h>
#include <phonon.h>
#include <lapack_wrapper.h>

static size_t collect_undone_grid_points(size_t *undone,
                                         char *phonon_done,
                                         const size_t num_grid_points,
                                         const size_t *grid_points);
static void get_undone_phonons(double *frequencies,
                               lapack_complex_double *eigenvectors,
                               const size_t *undone_grid_points,
                               const size_t num_undone_grid_points,
                               PHPYCONST int (*grid_address)[3],
                               const int mesh[3],
                               const double *fc2,
                               PHPYCONST double(*svecs_fc2)[27][3],
                               const int *multi_fc2,
                               const size_t num_patom,
                               const size_t num_satom,
                               const double *masses_fc2,
                               const int *p2s_fc2,
                               const int *s2p_fc2,
                               const double unit_conversion_factor,
                               PHPYCONST double (*born)[3][3],
                               PHPYCONST double dielectric[3][3],
                               PHPYCONST double reciprocal_lattice[3][3],
                               const double *q_direction,
                               const double nac_factor,
                               const char uplo);
static void get_gonze_undone_phonons(double *frequencies,
                                     lapack_complex_double *eigenvectors,
                                     const size_t *undone_grid_points,
                                     const size_t num_undone_grid_points,
                                     PHPYCONST int (*grid_address)[3],
                                     const int mesh[3],
                                     const double *fc2,
                                     PHPYCONST double(*svecs_fc2)[27][3],
                                     const int *multi_fc2,
                                     PHPYCONST double (*positions)[3],
                                     const size_t num_patom,
                                     const size_t num_satom,
                                     const double *masses_fc2,
                                     const int *p2s_fc2,
                                     const int *s2p_fc2,
                                     const double unit_conversion_factor,
                                     PHPYCONST double (*born)[3][3],
                                     PHPYCONST double dielectric[3][3],
                                     PHPYCONST double reciprocal_lattice[3][3],
                                     const double *q_direction,
                                     const double nac_factor,
                                     const double *dd_q0,
                                     PHPYCONST double(*G_list)[3],
                                     const size_t num_G_points,
                                     const double lambda,
                                     const char uplo);
static int get_phonons(lapack_complex_double *eigvecs,
                       double *freqs,
                       const double q[3],
                       const double *fc2,
                       const double *masses,
                       const int *p2s,
                       const int *s2p,
                       const int *multi,
                       const size_t num_patom,
                       const size_t num_satom,
                       PHPYCONST double(*svecs)[27][3],
                       const int is_nac,
                       PHPYCONST double (*born)[3][3],
                       PHPYCONST double dielectric[3][3],
                       PHPYCONST double reciprocal_lattice[3][3],
                       const double *q_direction,
                       const double nac_factor,
                       const double unit_conversion_factor,
                       const char uplo);
static int get_gonze_phonons(lapack_complex_double *eigvecs,
                             double *freqs,
                             const double q[3],
                             const double *fc2,
                             const double *masses,
                             const int *p2s,
                             const int *s2p,
                             const int *multi,
                             PHPYCONST double (*positions)[3],
                             const size_t num_patom,
                             const size_t num_satom,
                             PHPYCONST double(*svecs)[27][3],
                             const int is_nac,
                             PHPYCONST double (*born)[3][3],
                             PHPYCONST double dielectric[3][3],
                             PHPYCONST double reciprocal_lattice[3][3],
                             const double *q_direction,
                             const double nac_factor,
                             const double *dd_q0,
                             PHPYCONST double(*G_list)[3],
                             const size_t num_G_points,
                             const double lambda,
                             const double unit_conversion_factor,
                             const char uplo);
static void
get_dynamical_matrix(lapack_complex_double *dynmat,
                     const double q[3],
                     const double *fc2,
                     const double *masses,
                     const int *p2s,
                     const int *s2p,
                     const int *multi,
                     const size_t num_patom,
                     const size_t num_satom,
                     PHPYCONST double(*svecs)[27][3],
                     const int is_nac,
                     PHPYCONST double (*born)[3][3], /* Wang NAC unless NULL */
                     PHPYCONST double dielectric[3][3],
                     PHPYCONST double reciprocal_lattice[3][3],
                     const double *q_direction,
                     const double nac_factor);
static void get_charge_sum(double (*charge_sum)[3][3],
                           const size_t num_patom,
                           const size_t num_satom,
                           const double q[3],
                           PHPYCONST double (*born)[3][3],
                           PHPYCONST double dielectric[3][3],
                           PHPYCONST double reciprocal_lattice[3][3],
                           const double *q_direction,
                           const double nac_factor);
static int needs_nac(PHPYCONST double (*born)[3][3],
                     PHPYCONST int (*grid_address)[3],
                     const size_t gp,
                     const double *q_direction);

void
phn_get_phonons_at_gridpoints(double *frequencies,
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
                              const size_t num_patom,
                              const size_t num_satom,
                              const double *masses_fc2,
                              const int *p2s_fc2,
                              const int *s2p_fc2,
                              const double unit_conversion_factor,
                              PHPYCONST double (*born)[3][3],
                              PHPYCONST double dielectric[3][3],
                              PHPYCONST double reciprocal_lattice[3][3],
                              const double *q_direction, /* must be pointer */
                              const double nac_factor,
                              const char uplo)
{
  size_t num_undone;
  size_t *undone;

  undone = (size_t*)malloc(sizeof(size_t) * num_phonons);
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
  undone = NULL;
}

void
phn_get_gonze_phonons_at_gridpoints(double *frequencies,
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
                                    PHPYCONST double (*positions)[3],
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
  size_t num_undone;
  size_t *undone;

  undone = (size_t*)malloc(sizeof(size_t) * num_phonons);
  num_undone = collect_undone_grid_points(undone,
                                          phonon_done,
                                          num_grid_points,
                                          grid_points);

  get_gonze_undone_phonons(frequencies,
                           eigenvectors,
                           undone,
                           num_undone,
                           grid_address,
                           mesh,
                           fc2,
                           svecs_fc2,
                           multi_fc2,
                           positions,
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

  free(undone);
  undone = NULL;
}

static size_t collect_undone_grid_points(size_t *undone,
                                         char *phonon_done,
                                         const size_t num_grid_points,
                                         const size_t *grid_points)
{
  size_t i, gp, num_undone;

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
                               const size_t *undone_grid_points,
                               const size_t num_undone_grid_points,
                               PHPYCONST int (*grid_address)[3],
                               const int mesh[3],
                               const double *fc2,
                               PHPYCONST double(*svecs_fc2)[27][3],
                               const int *multi_fc2,
                               const size_t num_patom,
                               const size_t num_satom,
                               const double *masses_fc2,
                               const int *p2s_fc2,
                               const int *s2p_fc2,
                               const double unit_conversion_factor,
                               PHPYCONST double (*born)[3][3],
                               PHPYCONST double dielectric[3][3],
                               PHPYCONST double reciprocal_lattice[3][3],
                               const double *q_direction,
                               const double nac_factor,
                               const char uplo)
{
  size_t i, j, gp, num_band;
  int is_nac;
  double q[3];

  num_band = num_patom * 3;

/* To avoid multithreaded BLAS in OpenMP loop */
#ifndef MULTITHREADED_BLAS
#pragma omp parallel for private(j, q, gp, is_nac)
#endif
  for (i = 0; i < num_undone_grid_points; i++) {
    gp = undone_grid_points[i];
    for (j = 0; j < 3; j++) {
      q[j] = ((double)grid_address[gp][j]) / mesh[j];
    }

    is_nac = needs_nac(born, grid_address, gp, q_direction);
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
                is_nac,
                born,
                dielectric,
                reciprocal_lattice,
                q_direction,
                nac_factor,
                unit_conversion_factor,
                uplo);
  }
}

static void get_gonze_undone_phonons(double *frequencies,
                                     lapack_complex_double *eigenvectors,
                                     const size_t *undone_grid_points,
                                     const size_t num_undone_grid_points,
                                     PHPYCONST int (*grid_address)[3],
                                     const int mesh[3],
                                     const double *fc2,
                                     PHPYCONST double(*svecs_fc2)[27][3],
                                     const int *multi_fc2,
                                     PHPYCONST double (*positions)[3],
                                     const size_t num_patom,
                                     const size_t num_satom,
                                     const double *masses_fc2,
                                     const int *p2s_fc2,
                                     const int *s2p_fc2,
                                     const double unit_conversion_factor,
                                     PHPYCONST double (*born)[3][3],
                                     PHPYCONST double dielectric[3][3],
                                     PHPYCONST double reciprocal_lattice[3][3],
                                     const double *q_direction,
                                     const double nac_factor,
                                     const double *dd_q0,
                                     PHPYCONST double(*G_list)[3],
                                     const size_t num_G_points,
                                     const double lambda,
                                     const char uplo)
{
  size_t i, j, gp, num_band;
  int is_nac;
  double q[3];

  num_band = num_patom * 3;

/* To avoid multithreaded BLAS in OpenMP loop */
#ifndef MULTITHREADED_BLAS
#pragma omp parallel for private(j, q, gp)
#endif
  for (i = 0; i < num_undone_grid_points; i++) {
    gp = undone_grid_points[i];
    for (j = 0; j < 3; j++) {
      q[j] = ((double)grid_address[gp][j]) / mesh[j];
    }

    is_nac = needs_nac(born, grid_address, gp, q_direction);
    get_gonze_phonons(eigenvectors + num_band * num_band * gp,
                      frequencies + num_band * gp,
                      q,
                      fc2,
                      masses_fc2,
                      p2s_fc2,
                      s2p_fc2,
                      multi_fc2,
                      positions,
                      num_patom,
                      num_satom,
                      svecs_fc2,
                      is_nac,
                      born,
                      dielectric,
                      reciprocal_lattice,
                      q_direction,
                      nac_factor,
                      dd_q0,
                      G_list,
                      num_G_points,
                      lambda,
                      unit_conversion_factor,
                      uplo);
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
                       const size_t num_patom,
                       const size_t num_satom,
                       PHPYCONST double(*svecs)[27][3],
                       const int is_nac,
                       PHPYCONST double (*born)[3][3],
                       PHPYCONST double dielectric[3][3],
                       PHPYCONST double reciprocal_lattice[3][3],
                       const double *q_direction,
                       const double nac_factor,
                       const double unit_conversion_factor,
                       const char uplo)
{
  size_t i, num_band;
  int info;

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
                       is_nac,
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

static int get_gonze_phonons(lapack_complex_double *eigvecs,
                             double *freqs,
                             const double q[3],
                             const double *fc2,
                             const double *masses,
                             const int *p2s,
                             const int *s2p,
                             const int *multi,
                             PHPYCONST double (*positions)[3],
                             const size_t num_patom,
                             const size_t num_satom,
                             PHPYCONST double(*svecs)[27][3],
                             const int is_nac,
                             PHPYCONST double (*born)[3][3],
                             PHPYCONST double dielectric[3][3],
                             PHPYCONST double reciprocal_lattice[3][3],
                             const double *q_direction,
                             const double nac_factor,
                             const double *dd_q0,
                             PHPYCONST double(*G_list)[3],
                             const size_t num_G_points,
                             const double lambda,
                             const double unit_conversion_factor,
                             const char uplo)
{
  size_t i, j, k, l, adrs, num_band;
  int info;
  double mm;
  double q_cart[3];
  double *q_dir_cart;
  lapack_complex_double *dd;

  dd = NULL;
  q_dir_cart = NULL;
  num_band = num_patom * 3;

  dym_get_dynamical_matrix_at_q((double*)eigvecs,
                                num_patom,
                                num_satom,
                                fc2,
                                q,
                                svecs,
                                multi,
                                masses,
                                s2p,
                                p2s,
                                NULL,
                                0);

  dd = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_band * num_band);
  for (i = 0; i < 3; i++) {
    q_cart[i] = 0;
    for (j = 0; j < 3; j++) {
      q_cart[i] += reciprocal_lattice[i][j] * q[j];
    }
  }

  if (q_direction) {
    q_dir_cart = (double*)malloc(sizeof(double) * 3);
    for (i = 0; i < 3; i++) {
      q_dir_cart[i] = 0;
      for (j = 0; j < 3; j++) {
        q_dir_cart[i] += reciprocal_lattice[i][j] * q_direction[j];
      }
    }
  }

  dym_get_dipole_dipole((double*)dd,
                        dd_q0,
                        G_list,
                        num_G_points,
                        num_patom,
                        q_cart,
                        q_dir_cart,
                        born,
                        dielectric,
                        positions,
                        nac_factor,
                        lambda,
                        1e-5);

  if (q_direction) {
    free(q_dir_cart);
    q_dir_cart = NULL;
  }

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      mm = sqrt(masses[i] * masses[j]);
      for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
          adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
          eigvecs[adrs] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs[adrs]) +
            lapack_complex_double_real(dd[adrs]) / mm,
            lapack_complex_double_imag(eigvecs[adrs]) +
            lapack_complex_double_imag(dd[adrs]) / mm);
        }
      }
    }
  }

  free(dd);
  dd = NULL;

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

static void
get_dynamical_matrix(lapack_complex_double *dynmat,
                     const double q[3],
                     const double *fc2,
                     const double *masses,
                     const int *p2s,
                     const int *s2p,
                     const int *multi,
                     const size_t num_patom,
                     const size_t num_satom,
                     PHPYCONST double(*svecs)[27][3],
                     const int is_nac,
                     PHPYCONST double (*born)[3][3], /* Wang NAC unless NULL */
                     PHPYCONST double dielectric[3][3],
                     PHPYCONST double reciprocal_lattice[3][3],
                     const double *q_direction,
                     const double nac_factor)
{
  double (*charge_sum)[3][3];

  charge_sum = NULL;

  if (is_nac) {
    charge_sum = (double(*)[3][3])
      malloc(sizeof(double[3][3]) * num_patom * num_patom * 9);
    get_charge_sum(charge_sum,
                   num_patom,
                   num_satom,
                   q,
                   born,
                   dielectric,
                   reciprocal_lattice,
                   q_direction,
                   nac_factor);
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
  if (is_nac) {
    free(charge_sum);
    charge_sum = NULL;
  }
}

static void get_charge_sum(double (*charge_sum)[3][3],
                           const size_t num_patom,
                           const size_t num_satom,
                           const double q[3],
                           PHPYCONST double (*born)[3][3],
                           PHPYCONST double dielectric[3][3],
                           PHPYCONST double reciprocal_lattice[3][3],
                           const double *q_direction,
                           const double nac_factor)
{
  size_t i, j;
  double inv_dielectric_factor, dielectric_factor, tmp_val;
  double q_cart[3];

  if (q_direction) {
    for (i = 0; i < 3; i++) {
      q_cart[i] = 0.0;
      for (j = 0; j < 3; j++) {
        q_cart[i] += reciprocal_lattice[i][j] * q_direction[j];
      }
    }
  } else {
    for (i = 0; i < 3; i++) {
      q_cart[i] = 0.0;
      for (j = 0; j < 3; j++) {
        q_cart[i] += reciprocal_lattice[i][j] * q[j];
      }
    }
  }

  inv_dielectric_factor = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_val = 0.0;
    for (j = 0; j < 3; j++) {
      tmp_val += dielectric[i][j] * q_cart[j];
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

static int needs_nac(PHPYCONST double (*born)[3][3],
                     PHPYCONST int (*grid_address)[3],
                     const size_t gp,
                     const double *q_direction)
{
  int is_nac;

  if (born) {
    if (grid_address[gp][0] == 0 &&
        grid_address[gp][1] == 0 &&
        grid_address[gp][2] == 0 &&
        q_direction == NULL) {
      is_nac = 0;
    } else {
      is_nac = 1;
    }
  } else {
    is_nac = 0;
  }

  return is_nac;
}
