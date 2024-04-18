/* Copyright (C) 2017 Atsushi Togo */
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

#include "pp_collision.h"

#include <stdio.h>
#include <stdlib.h>

#include "imag_self_energy_with_g.h"
#include "interaction.h"
#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "real_to_reciprocal.h"
#include "triplet.h"
#include "triplet_iw.h"

static void get_collision(
    double *ise, const long num_band0, const long num_band,
    const long num_temps, const double *temperatures, const double *g,
    const char *g_zero, const double *frequencies,
    const lapack_complex_double *eigenvectors, const long triplet[3],
    const long triplet_weight, const ConstBZGrid *bzgrid, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const double *masses, const long *band_indices, const long symmetrize_fc3_q,
    const double cutoff_frequency, const long openmp_per_triplets);
static void finalize_ise(double *imag_self_energy, const double *ise,
                         const long (*bz_grid_address)[3],
                         const long (*triplets)[3], const long num_triplets,
                         const long num_temps, const long num_band0,
                         const long is_NU);

void ppc_get_pp_collision(
    double *imag_self_energy,
    const long relative_grid_address[24][4][3], /* thm */
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const ConstBZGrid *bzgrid, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const double *masses, const Larray *band_indices,
    const Darray *temperatures, const long is_NU, const long symmetrize_fc3_q,
    const double cutoff_frequency, const long openmp_per_triplets) {
    long i;
    long num_band, num_band0, num_band_prod, num_temps;
    double *ise, *freqs_at_gp, *g;
    char *g_zero;
    long tp_relative_grid_address[2][24][4][3];

    ise = NULL;
    freqs_at_gp = NULL;
    g = NULL;
    g_zero = NULL;

    num_band0 = band_indices->dims[0];
    num_band = atom_triplets->multi_dims[1] * 3;
    num_band_prod = num_band0 * num_band * num_band;
    num_temps = temperatures->dims[0];
    ise =
        (double *)malloc(sizeof(double) * num_triplets * num_temps * num_band0);
    freqs_at_gp = (double *)malloc(sizeof(double) * num_band0);
    for (i = 0; i < num_band0; i++) {
        freqs_at_gp[i] =
            frequencies[triplets[0][0] * num_band + band_indices->data[i]];
    }

    tpl_set_relative_grid_address(tp_relative_grid_address,
                                  relative_grid_address, 2);
#ifdef _OPENMP
#pragma omp parallel for schedule(guided) private( \
        g, g_zero) if (openmp_per_triplets)
#endif
    for (i = 0; i < num_triplets; i++) {
        g = (double *)malloc(sizeof(double) * 2 * num_band_prod);
        g_zero = (char *)malloc(sizeof(char) * num_band_prod);
        tpi_get_integration_weight(g, g_zero, freqs_at_gp, /* used as f0 */
                                   num_band0, tp_relative_grid_address,
                                   triplets[i], 1, bzgrid,
                                   frequencies,           /* used as f1 */
                                   num_band, frequencies, /* used as f2 */
                                   num_band, 2, openmp_per_triplets);

        get_collision(ise + i * num_temps * num_band0, num_band0, num_band,
                      num_temps, temperatures->data, g, g_zero, frequencies,
                      eigenvectors, triplets[i], triplet_weights[i], bzgrid,
                      fc3, is_compact_fc3, atom_triplets, masses,
                      band_indices->data, symmetrize_fc3_q, cutoff_frequency,
                      openmp_per_triplets);

        free(g_zero);
        g_zero = NULL;
        free(g);
        g = NULL;
    }

    finalize_ise(imag_self_energy, ise, bzgrid->addresses, triplets,
                 num_triplets, num_temps, num_band0, is_NU);

    free(freqs_at_gp);
    freqs_at_gp = NULL;
    free(ise);
    ise = NULL;
}

void ppc_get_pp_collision_with_sigma(
    double *imag_self_energy, const double sigma, const double sigma_cutoff,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const ConstBZGrid *bzgrid, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const double *masses, const Larray *band_indices,
    const Darray *temperatures, const long is_NU, const long symmetrize_fc3_q,
    const double cutoff_frequency, const long openmp_per_triplets) {
    long i;
    long num_band, num_band0, num_band_prod, num_temps;
    long const_adrs_shift;
    double cutoff;
    double *ise, *freqs_at_gp, *g;
    char *g_zero;

    ise = NULL;
    freqs_at_gp = NULL;
    g = NULL;
    g_zero = NULL;

    num_band0 = band_indices->dims[0];
    num_band = atom_triplets->multi_dims[1] * 3;
    num_band_prod = num_band0 * num_band * num_band;
    num_temps = temperatures->dims[0];
    const_adrs_shift = num_band_prod;

    ise =
        (double *)malloc(sizeof(double) * num_triplets * num_temps * num_band0);
    freqs_at_gp = (double *)malloc(sizeof(double) * num_band0);
    for (i = 0; i < num_band0; i++) {
        freqs_at_gp[i] =
            frequencies[triplets[0][0] * num_band + band_indices->data[i]];
    }

    cutoff = sigma * sigma_cutoff;

#ifdef _OPENMP
#pragma omp parallel for schedule(guided) private( \
        g, g_zero) if (openmp_per_triplets)
#endif
    for (i = 0; i < num_triplets; i++) {
        g = (double *)malloc(sizeof(double) * 2 * num_band_prod);
        g_zero = (char *)malloc(sizeof(char) * num_band_prod);
        tpi_get_integration_weight_with_sigma(
            g, g_zero, sigma, cutoff, freqs_at_gp, num_band0, triplets[i],
            const_adrs_shift, frequencies, num_band, 2, 1);

        get_collision(ise + i * num_temps * num_band0, num_band0, num_band,
                      num_temps, temperatures->data, g, g_zero, frequencies,
                      eigenvectors, triplets[i], triplet_weights[i], bzgrid,
                      fc3, is_compact_fc3, atom_triplets, masses,
                      band_indices->data, symmetrize_fc3_q, cutoff_frequency,
                      openmp_per_triplets);

        free(g_zero);
        g_zero = NULL;
        free(g);
        g = NULL;
    }

    finalize_ise(imag_self_energy, ise, bzgrid->addresses, triplets,
                 num_triplets, num_temps, num_band0, is_NU);

    free(freqs_at_gp);
    freqs_at_gp = NULL;
    free(ise);
    ise = NULL;
}

static void get_collision(
    double *ise, const long num_band0, const long num_band,
    const long num_temps, const double *temperatures, const double *g,
    const char *g_zero, const double *frequencies,
    const lapack_complex_double *eigenvectors, const long triplet[3],
    const long triplet_weight, const ConstBZGrid *bzgrid, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const double *masses, const long *band_indices, const long symmetrize_fc3_q,
    const double cutoff_frequency, const long openmp_per_triplets) {
    long i;
    long num_band_prod, num_g_pos;
    double *fc3_normal_squared;
    long(*g_pos)[4];

    fc3_normal_squared = NULL;
    g_pos = NULL;

    num_band_prod = num_band0 * num_band * num_band;
    fc3_normal_squared = (double *)malloc(sizeof(double) * num_band_prod);
    g_pos = (long(*)[4])malloc(sizeof(long[4]) * num_band_prod);

    for (i = 0; i < num_band_prod; i++) {
        fc3_normal_squared[i] = 0;
    }

    num_g_pos = ise_set_g_pos(g_pos, num_band0, num_band, g_zero);

    itr_get_interaction_at_triplet(
        fc3_normal_squared, num_band0, num_band, g_pos, num_g_pos, frequencies,
        eigenvectors, triplet, bzgrid, fc3, is_compact_fc3, atom_triplets,
        masses, band_indices, symmetrize_fc3_q, cutoff_frequency, 0, 0,
        openmp_per_triplets);

    ise_imag_self_energy_at_triplet(
        ise, num_band0, num_band, fc3_normal_squared, frequencies, triplet,
        triplet_weight, g, g + num_band_prod, g_pos, num_g_pos, temperatures,
        num_temps, cutoff_frequency, openmp_per_triplets, 0);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
    free(g_pos);
    g_pos = NULL;
}

static void finalize_ise(double *imag_self_energy, const double *ise,
                         const long (*bz_grid_addresses)[3],
                         const long (*triplets)[3], const long num_triplets,
                         const long num_temps, const long num_band0,
                         const long is_NU) {
    long i, j, k;
    long is_N;

    if (is_NU) {
        for (i = 0; i < 2 * num_temps * num_band0; i++) {
            imag_self_energy[i] = 0;
        }
        for (i = 0; i < num_triplets; i++) {
            is_N = tpl_is_N(triplets[i], bz_grid_addresses);
            for (j = 0; j < num_temps; j++) {
                for (k = 0; k < num_band0; k++) {
                    if (is_N) {
                        imag_self_energy[j * num_band0 + k] +=
                            ise[i * num_temps * num_band0 + j * num_band0 + k];
                    } else {
                        imag_self_energy[num_temps * num_band0 + j * num_band0 +
                                         k] +=
                            ise[i * num_temps * num_band0 + j * num_band0 + k];
                    }
                }
            }
        }
    } else {
        for (i = 0; i < num_temps * num_band0; i++) {
            imag_self_energy[i] = 0;
        }
        for (i = 0; i < num_triplets; i++) {
            for (j = 0; j < num_temps; j++) {
                for (k = 0; k < num_band0; k++) {
                    imag_self_energy[j * num_band0 + k] +=
                        ise[i * num_temps * num_band0 + j * num_band0 + k];
                }
            }
        }
    }
}
