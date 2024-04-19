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

#include "imag_self_energy_with_g.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "lagrid.h"
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "triplet.h"

static long set_g_pos_frequency_point(long (*g_pos)[4], const long num_band0,
                                      const long num_band, const char *g_zero);
static void detailed_imag_self_energy_at_triplet(
    double *detailed_imag_self_energy, double *imag_self_energy,
    const long num_band0, const long num_band, const double *fc3_normal_squared,
    const double *frequencies, const long triplet[3], const double *g1,
    const double *g2_3, const char *g_zero, const double *temperatures,
    const long num_temps, const double cutoff_frequency);
static double collect_detailed_imag_self_energy(
    double *imag_self_energy, const long num_band,
    const double *fc3_normal_squared, const double *n1, const double *n2,
    const double *g1, const double *g2_3, const char *g_zero);
static double collect_detailed_imag_self_energy_0K(
    double *imag_self_energy, const long num_band,
    const double *fc3_normal_squared, const double *n1, const double *n2,
    const double *g, const char *g_zero);
static void set_occupations(double *n1, double *n2, const long num_band,
                            const double temperature, const long triplet[3],
                            const double *frequencies,
                            const double cutoff_frequency);

void ise_get_imag_self_energy_with_g(
    double *imag_self_energy, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const double *g, const char *g_zero,
    const double temperature, const double cutoff_frequency,
    const long num_frequency_points, const long frequency_point_index) {
    long i, j, num_triplets, num_band0, num_band, num_band_prod;
    long num_g_pos, g_index_dims, g_index_shift;
    long(*g_pos)[4];
    double *ise;
    long at_a_frequency_point;

    g_pos = NULL;
    ise = NULL;

    num_triplets = fc3_normal_squared->dims[0];
    num_band0 = fc3_normal_squared->dims[1];
    num_band = fc3_normal_squared->dims[2];
    num_band_prod = num_band0 * num_band * num_band;
    ise = (double *)malloc(sizeof(double) * num_triplets * num_band0);

    /**
     * g has the shape of
     *  (num_triplets, num_band0 or num_frequency_points, num_band, num_band)
     *
     * g_index_dims: Stride for 1st dim
     * g_index_shift: Fixed index shift for 2nd dim
     */
    if (frequency_point_index < 0) {
        /* frequency_points == frequencies at bands */
        at_a_frequency_point = 0;
        g_index_dims = num_band_prod;
        g_index_shift = 0;
    } else {
        /* At an arbitrary frequency point. */
        at_a_frequency_point = 1;
        g_index_dims = num_frequency_points * num_band * num_band;
        g_index_shift = frequency_point_index * num_band * num_band;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(guided) private(num_g_pos, g_pos)
#endif
    for (i = 0; i < num_triplets; i++) {
        /**
         * g_pos contains the indices of g that are known non-zeros in series.
         *
         * ise_set_g_pos works for frquency points as bands.
         * set_g_pos_frequency_point works for frequency sampling mode.
         */
        g_pos = (long(*)[4])malloc(sizeof(long[4]) * num_band_prod);
        if (at_a_frequency_point) {
            num_g_pos = set_g_pos_frequency_point(
                g_pos, num_band0, num_band,
                g_zero + i * g_index_dims + g_index_shift);
        } else {
            num_g_pos = ise_set_g_pos(g_pos, num_band0, num_band,
                                      g_zero + i * g_index_dims);
        }

        ise_imag_self_energy_at_triplet(
            ise + i * num_band0, num_band0, num_band,
            fc3_normal_squared->data + i * num_band_prod, frequencies,
            triplets[i], triplet_weights[i],
            g + i * g_index_dims + g_index_shift,
            g + (i + num_triplets) * g_index_dims + g_index_shift, g_pos,
            num_g_pos, &temperature, 1, cutoff_frequency, 0,
            at_a_frequency_point);

        free(g_pos);
        g_pos = NULL;
    }

    for (i = 0; i < num_band0; i++) {
        imag_self_energy[i] = 0;
    }

    for (i = 0; i < num_triplets; i++) {
        for (j = 0; j < num_band0; j++) {
            imag_self_energy[j] += ise[i * num_band0 + j];
        }
    }

    free(ise);
    ise = NULL;
}

void ise_get_detailed_imag_self_energy_with_g(
    double *detailed_imag_self_energy, double *imag_self_energy_N,
    double *imag_self_energy_U, const Darray *fc3_normal_squared,
    const double *frequencies, const long (*triplets)[3],
    const long *triplet_weights, const long (*bz_grid_addresses)[3],
    const double *g, const char *g_zero, const double temperature,
    const double cutoff_frequency) {
    double *ise;
    long i, j, num_triplets, num_band0, num_band, num_band_prod;
    long *is_N;
    double ise_tmp, N, U;

    ise = NULL;
    is_N = NULL;

    num_triplets = fc3_normal_squared->dims[0];
    num_band0 = fc3_normal_squared->dims[1];
    num_band = fc3_normal_squared->dims[2];
    num_band_prod = num_band0 * num_band * num_band;
    ise = (double *)malloc(sizeof(double) * num_triplets * num_band0);

    /* detailed_imag_self_energy has the same shape as fc3_normal_squared. */

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_triplets; i++) {
        detailed_imag_self_energy_at_triplet(
            detailed_imag_self_energy + i * num_band_prod, ise + i * num_band0,
            num_band0, num_band, fc3_normal_squared->data + i * num_band_prod,
            frequencies, triplets[i], g + i * num_band_prod,
            g + (i + num_triplets) * num_band_prod, g_zero + i * num_band_prod,
            &temperature, 1, cutoff_frequency);
    }

    is_N = (long *)malloc(sizeof(long) * num_triplets);
    for (i = 0; i < num_triplets; i++) {
        is_N[i] = tpl_is_N(triplets[i], bz_grid_addresses);
    }

    for (i = 0; i < num_band0; i++) {
        N = 0;
        U = 0;
        /* #ifdef _OPENMP */
        /* #pragma omp parallel for private(ise_tmp) reduction(+:N,U) */
        /* #endif */
        for (j = 0; j < num_triplets; j++) {
            ise_tmp = ise[j * num_band0 + i] * triplet_weights[j];
            if (is_N[j]) {
                N += ise_tmp;
            } else {
                U += ise_tmp;
            }
        }
        imag_self_energy_N[i] = N;
        imag_self_energy_U[i] = U;
    }

    free(is_N);
    is_N = NULL;
    free(ise);
    ise = NULL;
}

void ise_imag_self_energy_at_triplet(
    double *imag_self_energy, const long num_band0, const long num_band,
    const double *fc3_normal_squared, const double *frequencies,
    const long triplet[3], const long triplet_weight, const double *g1,
    const double *g2_3, const long (*g_pos)[4], const long num_g_pos,
    const double *temperatures, const long num_temps,
    const double cutoff_frequency, const long openmp_per_triplets,
    const long at_a_frequency_point) {
    long i, j;
    double *n1, *n2, *ise_at_g_pos;
    long g_pos_3;

    n1 = (double *)malloc(sizeof(double) * num_temps * num_band);
    n2 = (double *)malloc(sizeof(double) * num_temps * num_band);
    ise_at_g_pos = (double *)malloc(sizeof(double) * num_g_pos * num_temps);

    for (i = 0; i < num_temps; i++) {
        set_occupations(n1 + i * num_band, n2 + i * num_band, num_band,
                        temperatures[i], triplet, frequencies,
                        cutoff_frequency);
    }

#ifdef _OPENMP
#pragma omp parallel for private(j, g_pos_3) if (!openmp_per_triplets)
#endif
    for (i = 0; i < num_g_pos; i++) {
        if (at_a_frequency_point) {
            /** At a frequency point:
             * g_pos[i][3] is for Phi with the shape of (band0, band, band).
             * g_pos_3 is for g at-freq-point with the shape of (bands, bands).
             */
            g_pos_3 = g_pos[i][3] % (num_band * num_band);
        } else {
            g_pos_3 = g_pos[i][3];
        }
        for (j = 0; j < num_temps; j++) {
            if (n1[j * num_band + g_pos[i][1]] < 0 ||
                n2[j * num_band + g_pos[i][2]] < 0) {
                ise_at_g_pos[i * num_temps + j] = 0;
            } else {
                if (temperatures[j] > 0) {
                    ise_at_g_pos[i * num_temps + j] =
                        ((n1[j * num_band + g_pos[i][1]] +
                          n2[j * num_band + g_pos[i][2]] + 1) *
                             g1[g_pos_3] +
                         (n1[j * num_band + g_pos[i][1]] -
                          n2[j * num_band + g_pos[i][2]]) *
                             g2_3[g_pos_3]) *
                        fc3_normal_squared[g_pos[i][3]] * triplet_weight;
                } else {
                    ise_at_g_pos[i * num_temps + j] =
                        g1[g_pos_3] * fc3_normal_squared[g_pos[i][3]] *
                        triplet_weight;
                }
            }
        }
    }

    for (i = 0; i < num_temps * num_band0; i++) {
        imag_self_energy[i] = 0;
    }
    for (i = 0; i < num_g_pos; i++) {
        for (j = 0; j < num_temps; j++) {
            imag_self_energy[j * num_band0 + g_pos[i][0]] +=
                ise_at_g_pos[i * num_temps + j];
        }
    }

    free(ise_at_g_pos);
    ise_at_g_pos = NULL;
    free(n1);
    n1 = NULL;
    free(n2);
    n2 = NULL;
}

long ise_set_g_pos(long (*g_pos)[4], const long num_band0, const long num_band,
                   const char *g_zero) {
    long num_g_pos, j, k, l, jkl;

    num_g_pos = 0;
    jkl = 0;
    for (j = 0; j < num_band0; j++) {
        for (k = 0; k < num_band; k++) {
            for (l = 0; l < num_band; l++) {
                if (!g_zero[jkl]) {
                    g_pos[num_g_pos][0] = j;
                    g_pos[num_g_pos][1] = k;
                    g_pos[num_g_pos][2] = l;
                    g_pos[num_g_pos][3] = jkl;
                    num_g_pos++;
                }
                jkl++;
            }
        }
    }
    return num_g_pos;
}

/**
 * @brief
 *
 * @param g_pos
 * @param num_band0
 * @param num_band
 * @param g_zero
 * @return long
 */
static long set_g_pos_frequency_point(long (*g_pos)[4], const long num_band0,
                                      const long num_band, const char *g_zero) {
    long num_g_pos, j, k, l, kl, jkl;

    num_g_pos = 0;
    jkl = 0;
    for (j = 0; j < num_band0; j++) {
        kl = 0;
        for (k = 0; k < num_band; k++) {
            for (l = 0; l < num_band; l++) {
                if (!g_zero[kl]) {
                    g_pos[num_g_pos][0] = j;
                    g_pos[num_g_pos][1] = k;
                    g_pos[num_g_pos][2] = l;
                    g_pos[num_g_pos][3] = jkl;
                    num_g_pos++;
                }
                jkl++;
                kl++;
            }
        }
    }
    return num_g_pos;
}

static void detailed_imag_self_energy_at_triplet(
    double *detailed_imag_self_energy, double *imag_self_energy,
    const long num_band0, const long num_band, const double *fc3_normal_squared,
    const double *frequencies, const long triplet[3], const double *g1,
    const double *g2_3, const char *g_zero, const double *temperatures,
    const long num_temps, const double cutoff_frequency) {
    long i, j, adrs_shift;
    double *n1, *n2;

    n1 = NULL;
    n2 = NULL;

    n1 = (double *)malloc(sizeof(double) * num_band);
    n2 = (double *)malloc(sizeof(double) * num_band);

    for (i = 0; i < num_temps; i++) {
        set_occupations(n1, n2, num_band, temperatures[i], triplet, frequencies,
                        cutoff_frequency);

        for (j = 0; j < num_band0; j++) {
            adrs_shift = j * num_band * num_band;
            if (temperatures[i] > 0) {
                imag_self_energy[i * num_band0 + j] =
                    collect_detailed_imag_self_energy(
                        detailed_imag_self_energy + adrs_shift, num_band,
                        fc3_normal_squared + adrs_shift, n1, n2,
                        g1 + adrs_shift, g2_3 + adrs_shift,
                        g_zero + adrs_shift);
            } else {
                imag_self_energy[i * num_band0 + j] =
                    collect_detailed_imag_self_energy_0K(
                        detailed_imag_self_energy + adrs_shift, num_band,
                        fc3_normal_squared + adrs_shift, n1, n2,
                        g1 + adrs_shift, g_zero + adrs_shift);
            }
        }
    }

    free(n1);
    n1 = NULL;
    free(n2);
    n2 = NULL;
}

static double collect_detailed_imag_self_energy(
    double *imag_self_energy, const long num_band,
    const double *fc3_normal_squared, const double *n1, const double *n2,
    const double *g1, const double *g2_3, const char *g_zero) {
    long ij, i, j;
    double sum_g;

    sum_g = 0;
    for (ij = 0; ij < num_band * num_band; ij++) {
        imag_self_energy[ij] = 0;
        if (g_zero[ij]) {
            continue;
        }
        i = ij / num_band;
        j = ij % num_band;
        if (n1[i] < 0 || n2[j] < 0) {
            continue;
        }
        imag_self_energy[ij] =
            (((n1[i] + n2[j] + 1) * g1[ij] + (n1[i] - n2[j]) * g2_3[ij]) *
             fc3_normal_squared[ij]);
        sum_g += imag_self_energy[ij];
    }

    return sum_g;
}

static double collect_detailed_imag_self_energy_0K(
    double *imag_self_energy, const long num_band,
    const double *fc3_normal_squared, const double *n1, const double *n2,
    const double *g1, const char *g_zero) {
    long ij, i, j;
    double sum_g;

    sum_g = 0;
    for (ij = 0; ij < num_band * num_band; ij++) {
        imag_self_energy[ij] = 0;
        if (g_zero[ij]) {
            continue;
        }
        i = ij / num_band;
        j = ij % num_band;
        if (n1[i] < 0 || n2[j] < 0) {
            continue;
        }
        imag_self_energy[ij] = g1[ij] * fc3_normal_squared[ij];
        sum_g += imag_self_energy[ij];
    }

    return sum_g;
}

static void set_occupations(double *n1, double *n2, const long num_band,
                            const double temperature, const long triplet[3],
                            const double *frequencies,
                            const double cutoff_frequency) {
    long j;
    double f1, f2;

    for (j = 0; j < num_band; j++) {
        f1 = frequencies[triplet[1] * num_band + j];
        f2 = frequencies[triplet[2] * num_band + j];
        if (f1 > cutoff_frequency) {
            n1[j] = phonoc_bose_einstein(f1, temperature);
        } else {
            n1[j] = -1;
        }
        if (f2 > cutoff_frequency) {
            n2[j] = phonoc_bose_einstein(f2, temperature);
        } else {
            n2[j] = -1;
        }
    }
}
