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

#include "reciprocal_to_normal.h"

#include <stdint.h>

#ifdef MULTITHREADED_BLAS
#if defined(MKL_BLAS) || defined(SCIPY_MKL_H)
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lapack_wrapper.h"

static void get_fc3_e0_e1_e2(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t *band_indices,
    const int64_t num_band0, const int64_t num_band,
    const double cutoff_frequency, const int64_t openmp_per_triplets);
static void get_fc3_e0(lapack_complex_double *fc3_e0,
                       const lapack_complex_double *fc3_reciprocal,
                       const lapack_complex_double *e0,
                       const int64_t band_index_0, const int64_t num_band);
static double get_fc3_sum(const lapack_complex_double *e1,
                          const lapack_complex_double *e2,
                          const lapack_complex_double *fc3_reciprocal,
                          const int64_t num_band);
static double get_fc3_sum_blas_like(const lapack_complex_double *e0,
                                    const lapack_complex_double *e1,
                                    const lapack_complex_double *e2,
                                    const lapack_complex_double *fc3_reciprocal,
                                    const int64_t num_band);
// Testing efficiency of BLAS
#ifdef MULTITHREADED_BLAS
static double get_fc3_sum_blas(const lapack_complex_double *e0,
                               const lapack_complex_double *e1,
                               const lapack_complex_double *e2,
                               const lapack_complex_double *fc3_reciprocal,
                               const int64_t num_band);
#endif

void reciprocal_to_normal_squared(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *eigvecs0,
    const lapack_complex_double *eigvecs1,
    const lapack_complex_double *eigvecs2, const double *masses,
    const int64_t *band_indices, const int64_t num_band0,
    const int64_t num_band, const double cutoff_frequency,
    const int64_t openmp_per_triplets) {
    int64_t i, j, ij, num_atom;
    double *inv_sqrt_masses;
    lapack_complex_double *e0, *e1, *e2;

    /* Inverse sqrt mass is multiplied with eigenvectors to reduce
     * number of */
    /* operations in get_fc3_sum. Three eigenvector matrices are
     * looped by */
    /* first loop leveraging contiguous memory layout of [e0,
     * e1, e2].
     */
    num_atom = num_band / 3;
    inv_sqrt_masses = (double *)malloc(sizeof(double) * num_band);
    for (i = 0; i < num_atom; i++) {
        for (j = 0; j < 3; j++) {
            inv_sqrt_masses[i * 3 + j] = 1.0 / sqrt(masses[i]);
        }
    }

    /* Transpose eigenvectors for the better data alignment in memory. */
    /* Memory space for three eigenvector matrices is allocated at once */
    /* to make it contiguous. */
    e0 = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) * 3 *
                                         num_band * num_band);
    e1 = e0 + num_band * num_band;
    e2 = e1 + num_band * num_band;

#ifdef _OPENMP
#pragma omp parallel for private(i, j) if (!openmp_per_triplets)
#endif
    for (ij = 0; ij < num_band * num_band; ij++) {
        i = ij / num_band;
        j = ij % num_band;
        e0[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs0[j * num_band + i]) *
                inv_sqrt_masses[j],
            lapack_complex_double_imag(eigvecs0[j * num_band + i]) *
                inv_sqrt_masses[j]);
        e1[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs1[j * num_band + i]) *
                inv_sqrt_masses[j],
            lapack_complex_double_imag(eigvecs1[j * num_band + i]) *
                inv_sqrt_masses[j]);
        e2[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs2[j * num_band + i]) *
                inv_sqrt_masses[j],
            lapack_complex_double_imag(eigvecs2[j * num_band + i]) *
                inv_sqrt_masses[j]);
    }

    free(inv_sqrt_masses);
    inv_sqrt_masses = NULL;

    get_fc3_e0_e1_e2(fc3_normal_squared, g_pos, num_g_pos, fc3_reciprocal,
                     freqs0, freqs1, freqs2, e0, e1, e2, band_indices,
                     num_band0, num_band, cutoff_frequency,
                     openmp_per_triplets);

    free(e0);
    e0 = NULL;
    e1 = NULL;
    e2 = NULL;
}

// This is less efficient than the one without multithreading
// but can be called when unit cell is large.
static void get_fc3_e0_e1_e2(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t *band_indices,
    const int64_t num_band0, const int64_t num_band,
    const double cutoff_frequency, const int64_t openmp_per_triplets) {
    int64_t i;
    lapack_complex_double *fc3_e0, zero;

    zero = lapack_make_complex_double(0, 0);
    fc3_e0 = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) *
                                             num_band0 * num_band * num_band);
    for (i = 0; i < num_band0 * num_band * num_band; i++) {
        fc3_e0[i] = zero;
    }

#ifdef _OPENMP
#pragma omp parallel for if (!openmp_per_triplets)
#endif
    for (i = 0; i < num_band0; i++) {
        get_fc3_e0(fc3_e0 + i * num_band * num_band, fc3_reciprocal, e0,
                   band_indices[i], num_band);
    }

#ifdef _OPENMP
#pragma omp parallel for if (!openmp_per_triplets)
#endif
    for (i = 0; i < num_g_pos; i++) {
        if (freqs0[band_indices[g_pos[i][0]]] > cutoff_frequency &&
            freqs1[g_pos[i][1]] > cutoff_frequency &&
            freqs2[g_pos[i][2]] > cutoff_frequency) {
            fc3_normal_squared[g_pos[i][3]] =
                get_fc3_sum(
                    e1 + g_pos[i][1] * num_band, e2 + g_pos[i][2] * num_band,
                    fc3_e0 + g_pos[i][0] * num_band * num_band, num_band) /
                (freqs0[band_indices[g_pos[i][0]]] * freqs1[g_pos[i][1]] *
                 freqs2[g_pos[i][2]]);
        } else {
            fc3_normal_squared[g_pos[i][3]] = 0;
        }
    }

    free(fc3_e0);
    fc3_e0 = NULL;
}

static void get_fc3_e0(lapack_complex_double *fc3_e0,
                       const lapack_complex_double *fc3_reciprocal,
                       const lapack_complex_double *e0,
                       const int64_t band_index_0, const int64_t num_band) {
    int64_t j, k;
    lapack_complex_double fc3_elem;

    for (j = 0; j < num_band; j++) {
        for (k = 0; k < num_band * num_band; k++) {
            fc3_elem =
                phonoc_complex_prod(fc3_reciprocal[j * num_band * num_band + k],
                                    e0[band_index_0 * num_band + j]);
            fc3_e0[k] = lapack_make_complex_double(
                lapack_complex_double_real(fc3_e0[k]) +
                    lapack_complex_double_real(fc3_elem),
                lapack_complex_double_imag(fc3_e0[k]) +
                    lapack_complex_double_imag(fc3_elem));
        }
    }
}

static double get_fc3_sum(const lapack_complex_double *e1,
                          const lapack_complex_double *e2,
                          const lapack_complex_double *fc3_e0,
                          const int64_t num_band) {
    int64_t i, j;
    double sum_real, sum_imag;
    lapack_complex_double *fc3_e0_e1, fc3_elem;

    fc3_e0_e1 = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) *
                                                num_band);
    for (i = 0; i < num_band; i++) {
        fc3_e0_e1[i] = lapack_make_complex_double(0, 0);
    }

    for (i = 0; i < num_band; i++) {
        for (j = 0; j < num_band; j++) {
            fc3_elem = phonoc_complex_prod(fc3_e0[i * num_band + j], e1[i]);
            fc3_e0_e1[j] = lapack_make_complex_double(
                lapack_complex_double_real(fc3_e0_e1[j]) +
                    lapack_complex_double_real(fc3_elem),
                lapack_complex_double_imag(fc3_e0_e1[j]) +
                    lapack_complex_double_imag(fc3_elem));
        }
    }
    sum_real = 0;
    sum_imag = 0;

    for (i = 0; i < num_band; i++) {
        fc3_elem = phonoc_complex_prod(fc3_e0_e1[i], e2[i]);
        sum_real += lapack_complex_double_real(fc3_elem);
        sum_imag += lapack_complex_double_imag(fc3_elem);
    }

    free(fc3_e0_e1);
    fc3_e0_e1 = NULL;

    return (sum_real * sum_real + sum_imag * sum_imag);
}

static double get_fc3_sum_blas_like(const lapack_complex_double *e0,
                                    const lapack_complex_double *e1,
                                    const lapack_complex_double *e2,
                                    const lapack_complex_double *fc3_reciprocal,
                                    const int64_t num_band) {
    int64_t i, j;
    double sum_real, sum_imag, retval_real, retval_imag;
    lapack_complex_double *e_12, fc3_e_12, fc3_e_012;

    e_12 = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) *
                                           num_band * num_band);

    for (i = 0; i < num_band; i++) {
        memcpy(e_12 + i * num_band, e2,
               sizeof(lapack_complex_double) * num_band);
        for (j = 0; j < num_band; j++) {
            e_12[i * num_band + j] =
                phonoc_complex_prod(e1[i], e_12[i * num_band + j]);
        }
    }

    retval_real = 0;
    retval_imag = 0;
    for (i = 0; i < num_band; i++) {
        sum_real = 0;
        sum_imag = 0;
        for (j = 0; j < num_band * num_band; j++) {
            fc3_e_12 = phonoc_complex_prod(
                fc3_reciprocal[i * num_band * num_band + j], e_12[j]);
            sum_real += lapack_complex_double_real(fc3_e_12);
            sum_imag += lapack_complex_double_imag(fc3_e_12);
        }
        fc3_e_012 = phonoc_complex_prod(
            e0[i], lapack_make_complex_double(sum_real, sum_imag));
        retval_real += lapack_complex_double_real(fc3_e_012);
        retval_imag += lapack_complex_double_imag(fc3_e_012);
    }

    free(e_12);
    e_12 = NULL;

    return retval_real * retval_real + retval_imag * retval_imag;
}

#ifdef MULTITHREADED_BLAS
static double get_fc3_sum_blas(const lapack_complex_double *e0,
                               const lapack_complex_double *e1,
                               const lapack_complex_double *e2,
                               const lapack_complex_double *fc3_reciprocal,
                               const int64_t num_band) {
    int64_t i;
    lapack_complex_double *fc3_e12, *e_12, zero, one, retval;

    e_12 = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) *
                                           num_band * num_band);
    fc3_e12 = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) *
                                              num_band);
    zero = lapack_make_complex_double(0, 0);
    one = lapack_make_complex_double(1, 0);

    for (i = 0; i < num_band; i++) {
        cblas_zcopy(num_band, e2, 1, e_12 + i * num_band, 1);
        cblas_zscal(num_band, e1 + i, e_12 + i * num_band, 1);
    }

    cblas_zgemv(CblasRowMajor, CblasNoTrans, num_band, num_band * num_band,
                &one, fc3_reciprocal, num_band * num_band, e_12, 1, &zero,
                fc3_e12, 1);
    cblas_zdotu_sub(num_band, e0, 1, fc3_e12, 1, &retval);

    free(e_12);
    e_12 = NULL;
    free(fc3_e12);
    fc3_e12 = NULL;

    return lapack_complex_double_real(retval) *
               lapack_complex_double_real(retval) +
           lapack_complex_double_imag(retval) *
               lapack_complex_double_imag(retval);
}
#endif
