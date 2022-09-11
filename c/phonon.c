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

#include "phonon.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "dynmat.h"
#include "lapack_wrapper.h"

static long collect_undone_grid_points(long *undone, char *phonon_done,
                                       const long num_grid_points,
                                       const long *grid_points);
static void get_undone_phonons(
    double *frequencies, lapack_complex_double *eigenvectors,
    const long *undone_grid_points, const long num_undone_grid_points,
    const long (*grid_address)[3], const double QDinv[3][3], const double *fc2,
    const double (*svecs_fc2)[3], const long (*multi_fc2)[2],
    const long num_patom, const long num_satom, const double *masses_fc2,
    const long *p2s_fc2, const long *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, const double nac_factor, const char uplo);
static void get_gonze_undone_phonons(
    double *frequencies, lapack_complex_double *eigenvectors,
    const long *undone_grid_points, const long num_undone_grid_points,
    const long (*grid_address)[3], const double QDinv[3][3], const double *fc2,
    const double (*svecs_fc2)[3], const long (*multi_fc2)[2],
    const double (*positions)[3], const long num_patom, const long num_satom,
    const double *masses_fc2, const long *p2s_fc2, const long *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, const double nac_factor,
    const double (*dd_q0)[2], const double (*G_list)[3],
    const long num_G_points, const double lambda, const char uplo);
static void get_phonons(lapack_complex_double *eigvecs, const double q[3],
                        const double *fc2, const double *masses,
                        const long *p2s, const long *s2p,
                        const long (*multi)[2], const long num_patom,
                        const long num_satom, const double (*svecs)[3],
                        const long is_nac, const double (*born)[3][3],
                        const double dielectric[3][3],
                        const double reciprocal_lattice[3][3],
                        const double *q_direction, const double nac_factor,
                        const double unit_conversion_factor);
static void get_gonze_phonons(
    lapack_complex_double *eigvecs, const double q[3], const double *fc2,
    const double *masses, const long *p2s, const long *s2p,
    const long (*multi)[2], const double (*positions)[3], const long num_patom,
    const long num_satom, const double (*svecs)[3], const long is_nac,
    const double (*born)[3][3], const double dielectric[3][3],
    const double reciprocal_lattice[3][3], const double *q_direction,
    const double nac_factor, const double (*dd_q0)[2],
    const double (*G_list)[3], const long num_G_points, const double lambda);
static void get_dynamical_matrix(
    lapack_complex_double *dynmat, const double q[3], const double *fc2,
    const double *masses, const long *p2s, const long *s2p,
    const long (*multi)[2], const long num_patom, const long num_satom,
    const double (*svecs)[3], const long is_nac,
    const double (*born)[3][3], /* Wang NAC unless NULL */
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, const double nac_factor);
static void get_charge_sum(double (*charge_sum)[3][3], const long num_patom,
                           const long num_satom, const double q[3],
                           const double (*born)[3][3],
                           const double dielectric[3][3],
                           const double reciprocal_lattice[3][3],
                           const double *q_direction, const double nac_factor);
static long needs_nac(const double (*born)[3][3], const long (*grid_address)[3],
                      const long gp, const double *q_direction);

void phn_get_phonons_at_gridpoints(
    double *frequencies, lapack_complex_double *eigenvectors, char *phonon_done,
    const long num_phonons, const long *grid_points, const long num_grid_points,
    const long (*grid_address)[3], const double QDinv[3][3], const double *fc2,
    const double (*svecs_fc2)[3], const long (*multi_fc2)[2],
    const long num_patom, const long num_satom, const double *masses_fc2,
    const long *p2s_fc2, const long *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, /* must be pointer */
    const double nac_factor, const char uplo) {
    long num_undone;
    long *undone;

    undone = (long *)malloc(sizeof(long) * num_phonons);
    num_undone = collect_undone_grid_points(undone, phonon_done,
                                            num_grid_points, grid_points);

    get_undone_phonons(frequencies, eigenvectors, undone, num_undone,
                       grid_address, QDinv, fc2, svecs_fc2, multi_fc2,
                       num_patom, num_satom, masses_fc2, p2s_fc2, s2p_fc2,
                       unit_conversion_factor, born, dielectric,
                       reciprocal_lattice, q_direction, nac_factor, uplo);

    free(undone);
    undone = NULL;
}

void phn_get_gonze_phonons_at_gridpoints(
    double *frequencies, lapack_complex_double *eigenvectors, char *phonon_done,
    const long num_phonons, const long *grid_points, const long num_grid_points,
    const long (*grid_address)[3], const double QDinv[3][3], const double *fc2,
    const double (*svecs_fc2)[3], const long (*multi_fc2)[2],
    const double (*positions)[3], const long num_patom, const long num_satom,
    const double *masses_fc2, const long *p2s_fc2, const long *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, /* pointer */
    const double nac_factor, const double (*dd_q0)[2],
    const double (*G_list)[3], const long num_G_points, const double lambda,
    const char uplo) {
    long num_undone;
    long *undone;

    undone = (long *)malloc(sizeof(long) * num_phonons);
    num_undone = collect_undone_grid_points(undone, phonon_done,
                                            num_grid_points, grid_points);

    get_gonze_undone_phonons(
        frequencies, eigenvectors, undone, num_undone, grid_address, QDinv, fc2,
        svecs_fc2, multi_fc2, positions, num_patom, num_satom, masses_fc2,
        p2s_fc2, s2p_fc2, unit_conversion_factor, born, dielectric,
        reciprocal_lattice, q_direction, nac_factor, dd_q0, G_list,
        num_G_points, lambda, uplo);

    free(undone);
    undone = NULL;
}

static long collect_undone_grid_points(long *undone, char *phonon_done,
                                       const long num_grid_points,
                                       const long *grid_points) {
    long i, gp, num_undone;

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

static void get_undone_phonons(
    double *frequencies, lapack_complex_double *eigenvectors,
    const long *undone_grid_points, const long num_undone_grid_points,
    const long (*grid_address)[3], const double QDinv[3][3], const double *fc2,
    const double (*svecs_fc2)[3], const long (*multi_fc2)[2],
    const long num_patom, const long num_satom, const double *masses_fc2,
    const long *p2s_fc2, const long *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, const double nac_factor, const char uplo) {
    long i, j, gp, num_band;
    long is_nac, info;
    double q[3];
    double *freqs_tmp;

    num_band = num_patom * 3;

#ifdef _OPENMP
#pragma omp parallel for private(j, q, gp, is_nac)
#endif
    for (i = 0; i < num_undone_grid_points; i++) {
        gp = undone_grid_points[i];
        for (j = 0; j < 3; j++) {
            q[j] = QDinv[j][0] * grid_address[gp][0] +
                   QDinv[j][1] * grid_address[gp][1] +
                   QDinv[j][2] * grid_address[gp][2];
        }

        is_nac = needs_nac(born, grid_address, gp, q_direction);
        get_phonons(eigenvectors + num_band * num_band * gp, q, fc2, masses_fc2,
                    p2s_fc2, s2p_fc2, multi_fc2, num_patom, num_satom,
                    svecs_fc2, is_nac, born, dielectric, reciprocal_lattice,
                    q_direction, nac_factor, unit_conversion_factor);
    }

/* To avoid multithreaded BLAS in OpenMP loop */
#ifndef MULTITHREADED_BLAS
#ifdef _OPENMP
#pragma omp parallel for private(j, gp, freqs_tmp, info)
#endif
#endif
    for (i = 0; i < num_undone_grid_points; i++) {
        gp = undone_grid_points[i];
        freqs_tmp = frequencies + num_band * gp;
        /* Store eigenvalues in freqs array. */
        /* Eigenvectors are overwritten on eigvecs array. */
        info = phonopy_zheev(freqs_tmp, eigenvectors + num_band * num_band * gp,
                             num_band, uplo);

        /* Sqrt of eigenvalues are re-stored in freqs array.*/
        for (j = 0; j < num_band; j++) {
            freqs_tmp[j] = sqrt(fabs(freqs_tmp[j])) *
                           ((freqs_tmp[j] > 0) - (freqs_tmp[j] < 0)) *
                           unit_conversion_factor;
        }
    }
}

static void get_gonze_undone_phonons(
    double *frequencies, lapack_complex_double *eigenvectors,
    const long *undone_grid_points, const long num_undone_grid_points,
    const long (*grid_address)[3], const double QDinv[3][3], const double *fc2,
    const double (*svecs_fc2)[3], const long (*multi_fc2)[2],
    const double (*positions)[3], const long num_patom, const long num_satom,
    const double *masses_fc2, const long *p2s_fc2, const long *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, const double nac_factor,
    const double (*dd_q0)[2], const double (*G_list)[3],
    const long num_G_points, const double lambda, const char uplo) {
    long i, j, gp, num_band;
    long is_nac, info;
    double q[3];
    double *freqs_tmp;

    num_band = num_patom * 3;

#ifdef _OPENMP
#pragma omp parallel for private(j, q, gp, is_nac)
#endif
    for (i = 0; i < num_undone_grid_points; i++) {
        gp = undone_grid_points[i];
        for (j = 0; j < 3; j++) {
            q[j] = QDinv[j][0] * grid_address[gp][0] +
                   QDinv[j][1] * grid_address[gp][1] +
                   QDinv[j][2] * grid_address[gp][2];
        }
        is_nac = needs_nac(born, grid_address, gp, q_direction);
        get_gonze_phonons(eigenvectors + num_band * num_band * gp, q, fc2,
                          masses_fc2, p2s_fc2, s2p_fc2, multi_fc2, positions,
                          num_patom, num_satom, svecs_fc2, is_nac, born,
                          dielectric, reciprocal_lattice, q_direction,
                          nac_factor, dd_q0, G_list, num_G_points, lambda);
    }

/* To avoid multithreaded BLAS in OpenMP loop */
#ifndef MULTITHREADED_BLAS
#ifdef _OPENMP
#pragma omp parallel for private(j, gp, freqs_tmp, info)
#endif
#endif
    for (i = 0; i < num_undone_grid_points; i++) {
        gp = undone_grid_points[i];
        /* Store eigenvalues in freqs array. */
        /* Eigenvectors are overwritten on eigvecs array. */
        freqs_tmp = frequencies + num_band * gp;
        info = phonopy_zheev(freqs_tmp, eigenvectors + num_band * num_band * gp,
                             num_band, uplo);

        /* Sqrt of eigenvalues are re-stored in freqs array.*/
        for (j = 0; j < num_band; j++) {
            freqs_tmp[j] = sqrt(fabs(freqs_tmp[j])) *
                           ((freqs_tmp[j] > 0) - (freqs_tmp[j] < 0)) *
                           unit_conversion_factor;
        }
    }
}

static void get_phonons(lapack_complex_double *eigvecs, const double q[3],
                        const double *fc2, const double *masses,
                        const long *p2s, const long *s2p,
                        const long (*multi)[2], const long num_patom,
                        const long num_satom, const double (*svecs)[3],
                        const long is_nac, const double (*born)[3][3],
                        const double dielectric[3][3],
                        const double reciprocal_lattice[3][3],
                        const double *q_direction, const double nac_factor,
                        const double unit_conversion_factor) {
    /* Store dynamical matrix in eigvecs array. */
    get_dynamical_matrix(eigvecs, q, fc2, masses, p2s, s2p, multi, num_patom,
                         num_satom, svecs, is_nac, born, dielectric,
                         reciprocal_lattice, q_direction, nac_factor);
}

static void get_gonze_phonons(
    lapack_complex_double *eigvecs, const double q[3], const double *fc2,
    const double *masses, const long *p2s, const long *s2p,
    const long (*multi)[2], const double (*positions)[3], const long num_patom,
    const long num_satom, const double (*svecs)[3], const long is_nac,
    const double (*born)[3][3], const double dielectric[3][3],
    const double reciprocal_lattice[3][3], const double *q_direction,
    const double nac_factor, const double (*dd_q0)[2],
    const double (*G_list)[3], const long num_G_points, const double lambda) {
    long i, j, k, l, adrs, num_band;
    double mm;
    double q_cart[3];
    double *q_dir_cart;
    lapack_complex_double *dd;

    dd = NULL;
    q_dir_cart = NULL;
    num_band = num_patom * 3;

    dym_get_dynamical_matrix_at_q((double(*)[2])eigvecs, num_patom, num_satom,
                                  fc2, q, svecs, multi, masses, s2p, p2s, NULL,
                                  0);

    dd = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) *
                                         num_band * num_band);
    for (i = 0; i < 3; i++) {
        q_cart[i] = 0;
        for (j = 0; j < 3; j++) {
            q_cart[i] += reciprocal_lattice[i][j] * q[j];
        }
    }

    if (q_direction) {
        q_dir_cart = (double *)malloc(sizeof(double) * 3);
        for (i = 0; i < 3; i++) {
            q_dir_cart[i] = 0;
            for (j = 0; j < 3; j++) {
                q_dir_cart[i] += reciprocal_lattice[i][j] * q_direction[j];
            }
        }
    }

    dym_get_recip_dipole_dipole((double(*)[2])dd, dd_q0, G_list, num_G_points,
                                num_patom, q_cart, q_dir_cart, born, dielectric,
                                positions, nac_factor, lambda, 1e-5, 0);

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
}

static void get_dynamical_matrix(
    lapack_complex_double *dynmat, const double q[3], const double *fc2,
    const double *masses, const long *p2s, const long *s2p,
    const long (*multi)[2], const long num_patom, const long num_satom,
    const double (*svecs)[3], const long is_nac,
    const double (*born)[3][3], /* Wang NAC unless NULL */
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, const double nac_factor) {
    double(*charge_sum)[3][3];

    charge_sum = NULL;

    if (is_nac) {
        charge_sum = (double(*)[3][3])malloc(sizeof(double[3][3]) * num_patom *
                                             num_patom * 9);
        get_charge_sum(charge_sum, num_patom, num_satom, q, born, dielectric,
                       reciprocal_lattice, q_direction, nac_factor);
    }

    dym_get_dynamical_matrix_at_q((double(*)[2])dynmat, num_patom, num_satom,
                                  fc2, q, svecs, multi, masses, s2p, p2s,
                                  charge_sum, 0);
    if (is_nac) {
        free(charge_sum);
        charge_sum = NULL;
    }
}

static void get_charge_sum(double (*charge_sum)[3][3], const long num_patom,
                           const long num_satom, const double q[3],
                           const double (*born)[3][3],
                           const double dielectric[3][3],
                           const double reciprocal_lattice[3][3],
                           const double *q_direction, const double nac_factor) {
    long i, j;
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
    dielectric_factor =
        nac_factor / inv_dielectric_factor / num_satom * num_patom;
    dym_get_charge_sum(charge_sum, num_patom, dielectric_factor, q_cart, born);
}

static long needs_nac(const double (*born)[3][3], const long (*grid_address)[3],
                      const long gp, const double *q_direction) {
    long is_nac;

    if (born) {
        if (grid_address[gp][0] == 0 && grid_address[gp][1] == 0 &&
            grid_address[gp][2] == 0 && q_direction == NULL) {
            is_nac = 0;
        } else {
            is_nac = 1;
        }
    } else {
        is_nac = 0;
    }

    return is_nac;
}
