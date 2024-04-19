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

#include "interaction.h"

#include <stdio.h>
#include <stdlib.h>

#include "bzgrid.h"
#include "imag_self_energy_with_g.h"
#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "real_to_reciprocal.h"
#include "reciprocal_to_normal.h"

static const long index_exchange[6][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0},
                                          {2, 1, 0}, {0, 2, 1}, {1, 0, 2}};
static void real_to_normal(double *fc3_normal_squared, const long (*g_pos)[4],
                           const long num_g_pos, const double *freqs0,
                           const double *freqs1, const double *freqs2,
                           const lapack_complex_double *eigvecs0,
                           const lapack_complex_double *eigvecs1,
                           const lapack_complex_double *eigvecs2,
                           const double *fc3, const long is_compact_fc3,
                           const double q_vecs[3][3], /* q0, q1, q2 */
                           const AtomTriplets *atom_triplets,
                           const double *masses, const long *band_indices,
                           const long num_band, const double cutoff_frequency,
                           const long triplet_index, const long num_triplets,
                           const long openmp_per_triplets);
static void real_to_normal_sym_q(
    double *fc3_normal_squared, const long (*g_pos)[4], const long num_g_pos,
    double *const freqs[3], lapack_complex_double *const eigvecs[3],
    const double *fc3, const long is_compact_fc3,
    const double q_vecs[3][3], /* q0, q1, q2 */
    const AtomTriplets *atom_triplets, const double *masses,
    const long *band_indices, const long num_band0, const long num_band,
    const double cutoff_frequency, const long triplet_index,
    const long num_triplets, const long openmp_per_triplets);

/* fc3_normal_squared[num_triplets, num_band0, num_band, num_band] */
void itr_get_interaction(
    Darray *fc3_normal_squared, const char *g_zero, const Darray *frequencies,
    const lapack_complex_double *eigenvectors, const long (*triplets)[3],
    const long num_triplets, const ConstBZGrid *bzgrid, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const double *masses, const long *band_indices, const long symmetrize_fc3_q,
    const double cutoff_frequency, const long openmp_per_triplets) {
    long(*g_pos)[4];
    long i;
    long num_band, num_band0, num_band_prod, num_g_pos;

    g_pos = NULL;

    num_band0 = fc3_normal_squared->dims[1];
    num_band = frequencies->dims[1];
    num_band_prod = num_band0 * num_band * num_band;

#ifdef _OPENMP
#pragma omp parallel for schedule(guided) private( \
        num_g_pos, g_pos) if (openmp_per_triplets)
#endif
    for (i = 0; i < num_triplets; i++) {
        g_pos = (long(*)[4])malloc(sizeof(long[4]) * num_band_prod);
        num_g_pos = ise_set_g_pos(g_pos, num_band0, num_band,
                                  g_zero + i * num_band_prod);

        itr_get_interaction_at_triplet(
            fc3_normal_squared->data + i * num_band_prod, num_band0, num_band,
            g_pos, num_g_pos, frequencies->data, eigenvectors, triplets[i],
            bzgrid, fc3, is_compact_fc3, atom_triplets, masses, band_indices,
            symmetrize_fc3_q, cutoff_frequency, i, num_triplets,
            openmp_per_triplets);

        free(g_pos);
        g_pos = NULL;
    }
}

void itr_get_interaction_at_triplet(
    double *fc3_normal_squared, const long num_band0, const long num_band,
    const long (*g_pos)[4], const long num_g_pos, const double *frequencies,
    const lapack_complex_double *eigenvectors, const long triplet[3],
    const ConstBZGrid *bzgrid, const double *fc3, const long is_compact_fc3,
    const AtomTriplets *atom_triplets, const double *masses,
    const long *band_indices, const long symmetrize_fc3_q,
    const double cutoff_frequency,
    const long triplet_index, /* only for print */
    const long num_triplets,  /* only for print */
    const long openmp_per_triplets) {
    long j, k;
    double *freqs[3];
    lapack_complex_double *eigvecs[3];
    double q_vecs[3][3];

    for (j = 0; j < 3; j++) {
        for (k = 0; k < 3; k++) {
            q_vecs[j][k] =
                ((double)bzgrid->addresses[triplet[j]][k]) / bzgrid->D_diag[k];
        }
        bzg_multiply_matrix_vector_ld3(q_vecs[j], bzgrid->Q, q_vecs[j]);
    }

    if (symmetrize_fc3_q) {
        for (j = 0; j < 3; j++) {
            freqs[j] = (double *)malloc(sizeof(double) * num_band);
            eigvecs[j] = (lapack_complex_double *)malloc(
                sizeof(lapack_complex_double) * num_band * num_band);
            for (k = 0; k < num_band; k++) {
                freqs[j][k] = frequencies[triplet[j] * num_band + k];
            }
            for (k = 0; k < num_band * num_band; k++) {
                eigvecs[j][k] =
                    eigenvectors[triplet[j] * num_band * num_band + k];
            }
        }
        real_to_normal_sym_q(
            fc3_normal_squared, g_pos, num_g_pos, freqs, eigvecs, fc3,
            is_compact_fc3, q_vecs, /* q0, q1, q2 */
            atom_triplets, masses, band_indices, num_band0, num_band,
            cutoff_frequency, triplet_index, num_triplets, openmp_per_triplets);
        for (j = 0; j < 3; j++) {
            free(freqs[j]);
            freqs[j] = NULL;
            free(eigvecs[j]);
            eigvecs[j] = NULL;
        }
    } else {
        real_to_normal(fc3_normal_squared, g_pos, num_g_pos,
                       frequencies + triplet[0] * num_band,
                       frequencies + triplet[1] * num_band,
                       frequencies + triplet[2] * num_band,
                       eigenvectors + triplet[0] * num_band * num_band,
                       eigenvectors + triplet[1] * num_band * num_band,
                       eigenvectors + triplet[2] * num_band * num_band, fc3,
                       is_compact_fc3, q_vecs, /* q0, q1, q2 */
                       atom_triplets, masses, band_indices, num_band,
                       cutoff_frequency, triplet_index, num_triplets,
                       openmp_per_triplets);
    }
}

static void real_to_normal(double *fc3_normal_squared, const long (*g_pos)[4],
                           const long num_g_pos, const double *freqs0,
                           const double *freqs1, const double *freqs2,
                           const lapack_complex_double *eigvecs0,
                           const lapack_complex_double *eigvecs1,
                           const lapack_complex_double *eigvecs2,
                           const double *fc3, const long is_compact_fc3,
                           const double q_vecs[3][3], /* q0, q1, q2 */
                           const AtomTriplets *atom_triplets,
                           const double *masses, const long *band_indices,
                           const long num_band, const double cutoff_frequency,
                           const long triplet_index, const long num_triplets,
                           const long openmp_per_triplets) {
    lapack_complex_double *fc3_reciprocal;
    lapack_complex_double comp_zero;
    long i;

    comp_zero = lapack_make_complex_double(0, 0);
    fc3_reciprocal = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_band * num_band * num_band);
    for (i = 0; i < num_band * num_band * num_band; i++) {
        fc3_reciprocal[i] = comp_zero;
    }
    r2r_real_to_reciprocal(fc3_reciprocal, q_vecs, fc3, is_compact_fc3,
                           atom_triplets, openmp_per_triplets);

#ifdef MEASURE_R2N
    if ((!openmp_per_triplets) && num_triplets > 0) {
        printf("At triplet %d/%d (# of bands=%d):\n", triplet_index,
               num_triplets, num_band0);
    }
#endif
    reciprocal_to_normal_squared(
        fc3_normal_squared, g_pos, num_g_pos, fc3_reciprocal, freqs0, freqs1,
        freqs2, eigvecs0, eigvecs1, eigvecs2, masses, band_indices, num_band,
        cutoff_frequency, openmp_per_triplets);

    free(fc3_reciprocal);
    fc3_reciprocal = NULL;
}

static void real_to_normal_sym_q(
    double *fc3_normal_squared, const long (*g_pos)[4], const long num_g_pos,
    double *const freqs[3], lapack_complex_double *const eigvecs[3],
    const double *fc3, const long is_compact_fc3,
    const double q_vecs[3][3], /* q0, q1, q2 */
    const AtomTriplets *atom_triplets, const double *masses,
    const long *band_indices, const long num_band0, const long num_band,
    const double cutoff_frequency, const long triplet_index,
    const long num_triplets, const long openmp_per_triplets) {
    long i, j, k, l;
    long band_ex[3];
    double q_vecs_ex[3][3];
    double *fc3_normal_squared_ex;

    fc3_normal_squared_ex =
        (double *)malloc(sizeof(double) * num_band * num_band * num_band);

    for (i = 0; i < num_band0 * num_band * num_band; i++) {
        fc3_normal_squared[i] = 0;
    }

    for (i = 0; i < 6; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                q_vecs_ex[j][k] = q_vecs[index_exchange[i][j]][k];
            }
        }
        real_to_normal(
            fc3_normal_squared_ex, g_pos, num_g_pos,
            freqs[index_exchange[i][0]], freqs[index_exchange[i][1]],
            freqs[index_exchange[i][2]], eigvecs[index_exchange[i][0]],
            eigvecs[index_exchange[i][1]], eigvecs[index_exchange[i][2]], fc3,
            is_compact_fc3, q_vecs_ex, /* q0, q1, q2 */
            atom_triplets, masses, band_indices, num_band, cutoff_frequency,
            triplet_index, num_triplets, openmp_per_triplets);
        for (j = 0; j < num_band0; j++) {
            for (k = 0; k < num_band; k++) {
                for (l = 0; l < num_band; l++) {
                    band_ex[0] = band_indices[j];
                    band_ex[1] = k;
                    band_ex[2] = l;
                    fc3_normal_squared[j * num_band * num_band + k * num_band +
                                       l] +=
                        fc3_normal_squared_ex[band_ex[index_exchange[i][0]] *
                                                  num_band * num_band +
                                              band_ex[index_exchange[i][1]] *
                                                  num_band +
                                              band_ex[index_exchange[i][2]]] /
                        6;
                }
            }
        }
    }

    free(fc3_normal_squared_ex);
}
