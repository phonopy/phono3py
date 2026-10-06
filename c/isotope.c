/* SPDX-License-Identifier: BSD-3-Clause */

#include "isotope.h"

#include <stdint.h>
#include <stdlib.h>

#include "funcs.h"
#include "lapack_wrapper.h"
#include "phonoc_const.h"

void iso_get_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const int64_t num_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0, const double sigma,
    const double cutoff_frequency) {
    int64_t i, j, k, l, m, gp;
    double *e0_r, *e0_i, e1_r, e1_i, a, b, f, *f0, dist, sum_g, sum_g_k;

    e0_r = (double *)malloc(sizeof(double) * num_band * num_band0);
    e0_i = (double *)malloc(sizeof(double) * num_band * num_band0);
    f0 = (double *)malloc(sizeof(double) * num_band0);

    for (i = 0; i < num_band0; i++) {
        f0[i] = frequencies[grid_point * num_band + band_indices[i]];
        for (j = 0; j < num_band; j++) {
            e0_r[i * num_band + j] = lapack_complex_double_real(
                eigenvectors[grid_point * num_band * num_band + j * num_band +
                             band_indices[i]]);
            e0_i[i * num_band + j] = lapack_complex_double_imag(
                eigenvectors[grid_point * num_band * num_band + j * num_band +
                             band_indices[i]]);
        }
    }

    for (i = 0; i < num_band0; i++) {
        gamma[i] = 0;
    }

    for (i = 0; i < num_band0; i++) { /* band index0 */
        if (f0[i] < cutoff_frequency) {
            continue;
        }
        sum_g = 0;
#ifdef _OPENMP
#pragma omp parallel for private(gp, k, l, m, f, e1_r, e1_i, a, b, dist, \
                                     sum_g_k) reduction(+ : sum_g)
#endif
        for (j = 0; j < num_grid_points; j++) {
            gp = ir_grid_points[j];
            sum_g_k = 0;
            for (k = 0; k < num_band; k++) { /* band index */
                f = frequencies[gp * num_band + k];
                if (f < cutoff_frequency) {
                    continue;
                }
                dist = funcs_gaussian(f - f0[i], sigma);
                for (l = 0; l < num_band / 3; l++) { /* elements */
                    a = 0;
                    b = 0;
                    for (m = 0; m < 3; m++) {
                        e1_r = lapack_complex_double_real(
                            eigenvectors[gp * num_band * num_band +
                                         (l * 3 + m) * num_band + k]);
                        e1_i = lapack_complex_double_imag(
                            eigenvectors[gp * num_band * num_band +
                                         (l * 3 + m) * num_band + k]);
                        a += (e0_r[i * num_band + l * 3 + m] * e1_r +
                              e0_i[i * num_band + l * 3 + m] * e1_i);
                        b += (e0_i[i * num_band + l * 3 + m] * e1_r -
                              e0_r[i * num_band + l * 3 + m] * e1_i);
                    }
                    sum_g_k += (a * a + b * b) * mass_variances[l] * dist;
                }
            }
            sum_g += sum_g_k * weights[gp];
        }
        gamma[i] = sum_g;
    }

    for (i = 0; i < num_band0; i++) {
        /* Frequency unit to ang-freq: *(2pi)**2/(2pi) */
        /* Ang-freq to freq unit (for lifetime): /2pi */
        /* gamma = 1/2t */
        gamma[i] *= M_2PI / 4 * f0[i] * f0[i] / 2;
    }

    free(f0);
    f0 = NULL;
    free(e0_r);
    e0_r = NULL;
    free(e0_i);
    e0_i = NULL;
}

void iso_get_thm_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const int64_t num_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0,
    const double *integration_weights, const double cutoff_frequency) {
    int64_t i, j, k, l, m, gp;
    double *e0_r, *e0_i, *f0, *gamma_ij;
    double e1_r, e1_i, a, b, f, dist, sum_g_k;

    e0_r = (double *)malloc(sizeof(double) * num_band * num_band0);
    e0_i = (double *)malloc(sizeof(double) * num_band * num_band0);
    f0 = (double *)malloc(sizeof(double) * num_band0);

    for (i = 0; i < num_band0; i++) {
        f0[i] = frequencies[grid_point * num_band + band_indices[i]];
        for (j = 0; j < num_band; j++) {
            e0_r[i * num_band + j] = lapack_complex_double_real(
                eigenvectors[grid_point * num_band * num_band + j * num_band +
                             band_indices[i]]);
            e0_i[i * num_band + j] = lapack_complex_double_imag(
                eigenvectors[grid_point * num_band * num_band + j * num_band +
                             band_indices[i]]);
        }
    }

    gamma_ij = (double *)malloc(sizeof(double) * num_grid_points * num_band0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_grid_points * num_band0; i++) {
        gamma_ij[i] = 0;
    }

#ifdef _OPENMP
#pragma omp parallel for private(j, k, l, m, f, gp, e1_r, e1_i, a, b, dist, \
                                     sum_g_k)
#endif
    for (i = 0; i < num_grid_points; i++) {
        gp = ir_grid_points[i];
        for (j = 0; j < num_band0; j++) { /* band index0 */
            if (f0[j] < cutoff_frequency) {
                continue;
            }
            sum_g_k = 0;
            for (k = 0; k < num_band; k++) { /* band index */
                f = frequencies[gp * num_band + k];
                if (f < cutoff_frequency) {
                    continue;
                }
                dist = integration_weights[gp * num_band0 * num_band +
                                           j * num_band + k];
                for (l = 0; l < num_band / 3; l++) { /* elements */
                    a = 0;
                    b = 0;
                    for (m = 0; m < 3; m++) {
                        e1_r = lapack_complex_double_real(
                            eigenvectors[gp * num_band * num_band +
                                         (l * 3 + m) * num_band + k]);
                        e1_i = lapack_complex_double_imag(
                            eigenvectors[gp * num_band * num_band +
                                         (l * 3 + m) * num_band + k]);
                        a += (e0_r[j * num_band + l * 3 + m] * e1_r +
                              e0_i[j * num_band + l * 3 + m] * e1_i);
                        b += (e0_i[j * num_band + l * 3 + m] * e1_r -
                              e0_r[j * num_band + l * 3 + m] * e1_i);
                    }
                    sum_g_k += (a * a + b * b) * mass_variances[l] * dist;
                }
            }
            gamma_ij[i * num_band0 + j] = sum_g_k * weights[gp];
        }
    }

    for (i = 0; i < num_band0; i++) {
        gamma[i] = 0;
    }

    for (i = 0; i < num_grid_points; i++) {
        for (j = 0; j < num_band0; j++) {
            gamma[j] += gamma_ij[i * num_band0 + j];
        }
    }

    for (i = 0; i < num_band0; i++) {
        /* Frequency unit to ang-freq: *(2pi)**2/(2pi) */
        /* Ang-freq to freq unit (for lifetime): /2pi */
        /* gamma = 1/2t */
        gamma[i] *= M_2PI / 4 * f0[i] * f0[i] / 2;
    }

    free(gamma_ij);
    gamma_ij = NULL;
    free(f0);
    f0 = NULL;
    free(e0_r);
    e0_r = NULL;
    free(e0_i);
    e0_i = NULL;
}
