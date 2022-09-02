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

#include <math.h>
#include <stdlib.h>

#include "lapack_wrapper.h"

#ifdef MEASURE_R2N
#include <time.h>
#include <unistd.h>
#endif

static double get_fc3_sum(const lapack_complex_double *e0,
                          const lapack_complex_double *e1,
                          const lapack_complex_double *e2,
                          const lapack_complex_double *fc3_reciprocal,
                          const long num_band);

void reciprocal_to_normal_squared(
    double *fc3_normal_squared, const long (*g_pos)[4], const long num_g_pos,
    const lapack_complex_double *fc3_reciprocal, const double *freqs0,
    const double *freqs1, const double *freqs2,
    const lapack_complex_double *eigvecs0,
    const lapack_complex_double *eigvecs1,
    const lapack_complex_double *eigvecs2, const double *masses,
    const long *band_indices, const long num_band,
    const double cutoff_frequency, const long openmp_at_bands) {
    long i, j, num_atom;
    double *inv_sqrt_masses;
    lapack_complex_double *e0, *e1, *e2;

    /* Inverse sqrt mass is multipled with eigenvectors to reduce number
     * of */
    /* operations in get_fc3_sum. Three eigenvector matrices are looped
     * by */
    /* first loop leveraging contiguous memory layout of [e0, e1, e2].
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
#pragma omp parallel for private(j) if (openmp_at_bands)
#endif
    for (i = 0; i < num_band; i++) {
        for (j = 0; j < num_band; j++) {
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
    }

    free(inv_sqrt_masses);
    inv_sqrt_masses = NULL;

#ifdef MEASURE_R2N
    double loopTotalCPUTime, loopTotalWallTime;
    time_t loopStartWallTime;
    clock_t loopStartCPUTime;
#endif

#ifdef MEASURE_R2N
    loopStartWallTime = time(NULL);
    loopStartCPUTime = clock();
#endif

#ifdef _OPENMP
#pragma omp parallel for if (openmp_at_bands)
#endif
    for (i = 0; i < num_g_pos; i++) {
        if (freqs0[band_indices[g_pos[i][0]]] > cutoff_frequency &&
            freqs1[g_pos[i][1]] > cutoff_frequency &&
            freqs2[g_pos[i][2]] > cutoff_frequency) {
            fc3_normal_squared[g_pos[i][3]] =
                get_fc3_sum(e0 + band_indices[g_pos[i][0]] * num_band,
                            e1 + g_pos[i][1] * num_band,
                            e2 + g_pos[i][2] * num_band, fc3_reciprocal,
                            num_band) /
                (freqs0[band_indices[g_pos[i][0]]] * freqs1[g_pos[i][1]] *
                 freqs2[g_pos[i][2]]);
        } else {
            fc3_normal_squared[g_pos[i][3]] = 0;
        }
    }

#ifdef MEASURE_R2N
    loopTotalCPUTime = (double)(clock() - loopStartCPUTime) / CLOCKS_PER_SEC;
    loopTotalWallTime = difftime(time(NULL), loopStartWallTime);
    printf("  %1.3fs (%1.3fs CPU)\n", loopTotalWallTime, loopTotalCPUTime);
#endif

    free(e0);
    e0 = NULL;
    e1 = NULL;
    e2 = NULL;
}

static double get_fc3_sum(const lapack_complex_double *e0,
                          const lapack_complex_double *e1,
                          const lapack_complex_double *e2,
                          const lapack_complex_double *fc3_reciprocal,
                          const long num_band) {
    long i, j, k;
    double sum_real, sum_imag;
    lapack_complex_double e_012, e_012_fc3, *e_12_cache;
    const lapack_complex_double *fc3_i;

    e_12_cache = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) *
                                                 num_band * num_band);

    sum_real = 0;
    sum_imag = 0;

    for (i = 0; i < num_band; i++) {
        for (j = 0; j < num_band; j++) {
            e_12_cache[i * num_band + j] = phonoc_complex_prod(e1[i], e2[j]);
        }
    }

    for (i = 0; i < num_band; i++) {
        fc3_i = fc3_reciprocal + i * num_band * num_band;
        for (j = 0; j < num_band; j++) {
            for (k = 0; k < num_band; k++) {
                e_012 =
                    phonoc_complex_prod(e0[i], e_12_cache[j * num_band + k]);
                e_012_fc3 = phonoc_complex_prod(e_012, fc3_i[j * num_band + k]);
                sum_real += lapack_complex_double_real(e_012_fc3);
                sum_imag += lapack_complex_double_imag(e_012_fc3);
            }
        }
    }

    free(e_12_cache);
    e_12_cache = NULL;

    return (sum_real * sum_real + sum_imag * sum_imag);
}
