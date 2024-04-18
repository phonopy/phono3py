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

#include "real_to_reciprocal.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "phonoc_const.h"

static void real_to_reciprocal_legacy(
    lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *pre_phase_factors,
    const lapack_complex_double *phase_factor1,
    const lapack_complex_double *phase_factor2, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const long openmp_per_triplets);
static void real_to_reciprocal_r0_average(
    lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *pre_phase_factors,
    const lapack_complex_double *phase_factor0,
    const lapack_complex_double *phase_factor1,
    const lapack_complex_double *phase_factor2, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const long openmp_per_triplets);
static void real_to_reciprocal_elements(
    lapack_complex_double *fc3_rec_elem,
    const lapack_complex_double *phase_factor1,
    const lapack_complex_double *phase_factor2, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const long pi0, const long pi1, const long pi2, const long leg_index);
static lapack_complex_double get_phase_factor(const double q[3],
                                              const double (*svecs)[3],
                                              const long multi[2]);
static lapack_complex_double get_pre_phase_factor(
    const long i_patom, const double q_vecs[3][3],
    const AtomTriplets *atom_triplets);
static lapack_complex_double sum_lapack_complex_double(lapack_complex_double a,
                                                       lapack_complex_double b);

/* fc3_reciprocal[num_patom, num_patom, num_patom, 3, 3, 3] */
void r2r_real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
                            const double q_vecs[3][3], const double *fc3,
                            const long is_compact_fc3,
                            const AtomTriplets *atom_triplets,
                            const long openmp_per_triplets) {
    long i, j, num_band, num_patom, num_satom, adrs_vec;
    lapack_complex_double *pre_phase_factors, *phase_factor0, *phase_factor1,
        *phase_factor2;

    num_patom = atom_triplets->multi_dims[1];
    num_satom = atom_triplets->multi_dims[0];

    pre_phase_factors = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_patom);
    for (i = 0; i < num_patom; i++) {
        pre_phase_factors[i] = get_pre_phase_factor(i, q_vecs, atom_triplets);
    }

    phase_factor0 = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_patom * num_satom);
    phase_factor1 = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_patom * num_satom);
    phase_factor2 = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_patom * num_satom);
    for (i = 0; i < num_patom; i++) {
        for (j = 0; j < num_satom; j++) {
            adrs_vec = j * atom_triplets->multi_dims[1] + i;
            phase_factor0[i * num_satom + j] =
                get_phase_factor(q_vecs[0], atom_triplets->svecs,
                                 atom_triplets->multiplicity[adrs_vec]);
            phase_factor1[i * num_satom + j] =
                get_phase_factor(q_vecs[1], atom_triplets->svecs,
                                 atom_triplets->multiplicity[adrs_vec]);
            phase_factor2[i * num_satom + j] =
                get_phase_factor(q_vecs[2], atom_triplets->svecs,
                                 atom_triplets->multiplicity[adrs_vec]);
        }
    }

    if (atom_triplets->make_r0_average) {
        real_to_reciprocal_r0_average(fc3_reciprocal, pre_phase_factors,
                                      phase_factor0, phase_factor1,
                                      phase_factor2, fc3, is_compact_fc3,
                                      atom_triplets, openmp_per_triplets);
        num_band = atom_triplets->multi_dims[1] * 3;
        for (i = 0; i < num_band * num_band * num_band; i++) {
            fc3_reciprocal[i] = lapack_make_complex_double(
                lapack_complex_double_real(fc3_reciprocal[i]) / 3,
                lapack_complex_double_imag(fc3_reciprocal[i]) / 3);
        }
    } else {
        real_to_reciprocal_legacy(
            fc3_reciprocal, pre_phase_factors, phase_factor1, phase_factor2,
            fc3, is_compact_fc3, atom_triplets, openmp_per_triplets);
    }

    free(pre_phase_factors);
    pre_phase_factors = NULL;
    free(phase_factor0);
    phase_factor1 = NULL;
    free(phase_factor1);
    phase_factor1 = NULL;
    free(phase_factor2);
    phase_factor2 = NULL;
}

static void real_to_reciprocal_legacy(
    lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *pre_phase_factors,
    const lapack_complex_double *phase_factor1,
    const lapack_complex_double *phase_factor2, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const long openmp_per_triplets) {
    long i, j, k, l, m, n, ijk;
    long num_patom, num_satom, num_band;
    lapack_complex_double fc3_rec_elem[27];

    num_patom = atom_triplets->multi_dims[1];
    num_satom = atom_triplets->multi_dims[0];
    num_band = num_patom * 3;

#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, l, m, n, \
                                     fc3_rec_elem) if (!openmp_per_triplets)
#endif
    for (ijk = 0; ijk < num_patom * num_patom * num_patom; ijk++) {
        i = ijk / (num_patom * num_patom);
        j = (ijk - (i * num_patom * num_patom)) / num_patom;
        k = ijk % num_patom;

        real_to_reciprocal_elements(fc3_rec_elem, phase_factor1 + i * num_satom,
                                    phase_factor2 + i * num_satom, fc3,
                                    is_compact_fc3, atom_triplets, i, j, k, 0);
        for (l = 0; l < 3; l++) {
            for (m = 0; m < 3; m++) {
                for (n = 0; n < 3; n++) {
                    fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                   (j * 3 + m) * num_band + k * 3 + n] =
                        phonoc_complex_prod(fc3_rec_elem[l * 9 + m * 3 + n],
                                            pre_phase_factors[i]);
                }
            }
        }
    }
}

// Summations are performed with respect to three different lattice reference
// point for the index of real space fc3 when make_r0_average=True. For cubic
// case, these three are roughly equivalent but small difference comes from the
// q-points in triplets used for summation implemented in
// real_to_reciprocal_elements().
// --sym-fc3q makes them almost equivalent.
static void real_to_reciprocal_r0_average(
    lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *pre_phase_factors,
    const lapack_complex_double *phase_factor0,
    const lapack_complex_double *phase_factor1,
    const lapack_complex_double *phase_factor2, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const long openmp_per_triplets) {
    long i, j, k, l, m, n, ijk;
    long num_patom, num_satom, num_band;
    lapack_complex_double fc3_rec_elem[27], fc3_rec;

    num_patom = atom_triplets->multi_dims[1];
    num_satom = atom_triplets->multi_dims[0];
    num_band = num_patom * 3;

#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, l, m, n, fc3_rec_elem, \
                                     fc3_rec) if (!openmp_per_triplets)
#endif
    for (ijk = 0; ijk < num_patom * num_patom * num_patom; ijk++) {
        i = ijk / (num_patom * num_patom);
        j = (ijk - (i * num_patom * num_patom)) / num_patom;
        k = ijk % num_patom;

        real_to_reciprocal_elements(fc3_rec_elem, phase_factor1 + i * num_satom,
                                    phase_factor2 + i * num_satom, fc3,
                                    is_compact_fc3, atom_triplets, i, j, k, 1);
        for (l = 0; l < 3; l++) {
            for (m = 0; m < 3; m++) {
                for (n = 0; n < 3; n++) {
                    fc3_rec = phonoc_complex_prod(
                        fc3_rec_elem[l * 9 + m * 3 + n], pre_phase_factors[i]);
                    fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                   (j * 3 + m) * num_band + k * 3 + n] =
                        sum_lapack_complex_double(
                            fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                           (j * 3 + m) * num_band + k * 3 + n],
                            fc3_rec);
                }
            }
        }

        // fc3_rec is stored in a way swapping jm <-> il.
        real_to_reciprocal_elements(fc3_rec_elem, phase_factor0 + j * num_satom,
                                    phase_factor2 + j * num_satom, fc3,
                                    is_compact_fc3, atom_triplets, j, i, k, 2);
        for (l = 0; l < 3; l++) {
            for (m = 0; m < 3; m++) {
                for (n = 0; n < 3; n++) {
                    fc3_rec = phonoc_complex_prod(
                        fc3_rec_elem[m * 9 + l * 3 + n], pre_phase_factors[j]);
                    fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                   (j * 3 + m) * num_band + k * 3 + n] =
                        sum_lapack_complex_double(
                            fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                           (j * 3 + m) * num_band + k * 3 + n],
                            fc3_rec);
                }
            }
        }

        // fc3_rec is stored in a way swapping kn <-> il.
        real_to_reciprocal_elements(fc3_rec_elem, phase_factor1 + k * num_satom,
                                    phase_factor0 + k * num_satom, fc3,
                                    is_compact_fc3, atom_triplets, k, j, i, 3);
        for (l = 0; l < 3; l++) {
            for (m = 0; m < 3; m++) {
                for (n = 0; n < 3; n++) {
                    fc3_rec = phonoc_complex_prod(
                        fc3_rec_elem[n * 9 + m * 3 + l], pre_phase_factors[k]);
                    fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                   (j * 3 + m) * num_band + k * 3 + n] =
                        sum_lapack_complex_double(
                            fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                           (j * 3 + m) * num_band + k * 3 + n],
                            fc3_rec);
                }
            }
        }
    }
}

static void real_to_reciprocal_elements(
    lapack_complex_double *fc3_rec_elem,
    const lapack_complex_double *phase_factor1,
    const lapack_complex_double *phase_factor2, const double *fc3,
    const long is_compact_fc3, const AtomTriplets *atom_triplets,
    const long pi0, const long pi1, const long pi2, const long leg_index) {
    long i, j, k, l;
    long num_satom, adrs_shift;
    lapack_complex_double phase_factor;
    double fc3_rec_real[27], fc3_rec_imag[27];

    num_satom = atom_triplets->multi_dims[0];

    for (i = 0; i < 27; i++) {
        fc3_rec_real[i] = 0;
        fc3_rec_imag[i] = 0;
    }

    if (is_compact_fc3) {
        i = pi0;
    } else {
        i = atom_triplets->p2s_map[pi0];
    }

    for (j = 0; j < num_satom; j++) {
        if (atom_triplets->s2p_map[j] != atom_triplets->p2s_map[pi1]) {
            continue;
        }

        for (k = 0; k < num_satom; k++) {
            if (atom_triplets->s2p_map[k] != atom_triplets->p2s_map[pi2]) {
                continue;
            }
            if (leg_index > 1) {
                if (atom_triplets->all_shortest[pi0 * num_satom * num_satom +
                                                j * num_satom + k]) {
                    continue;
                }
            }
            adrs_shift =
                i * 27 * num_satom * num_satom + j * 27 * num_satom + k * 27;
            phase_factor =
                phonoc_complex_prod(phase_factor1[j], phase_factor2[k]);

            if ((leg_index == 1) &&
                (atom_triplets->all_shortest[pi0 * num_satom * num_satom +
                                             j * num_satom + k])) {
                for (l = 0; l < 27; l++) {
                    fc3_rec_real[l] +=
                        lapack_complex_double_real(phase_factor) *
                        fc3[adrs_shift + l] * 3;
                    fc3_rec_imag[l] +=
                        lapack_complex_double_imag(phase_factor) *
                        fc3[adrs_shift + l] * 3;
                }
            } else {
                for (l = 0; l < 27; l++) {
                    fc3_rec_real[l] +=
                        lapack_complex_double_real(phase_factor) *
                        fc3[adrs_shift + l];
                    fc3_rec_imag[l] +=
                        lapack_complex_double_imag(phase_factor) *
                        fc3[adrs_shift + l];
                }
            }
        }
    }

    for (i = 0; i < 27; i++) {
        fc3_rec_elem[i] =
            lapack_make_complex_double(fc3_rec_real[i], fc3_rec_imag[i]);
    }
}

// This function doesn't need to think about position +
// lattice-translation because q+q'+q''=G.
static lapack_complex_double get_pre_phase_factor(
    const long i_patom, const double q_vecs[3][3],
    const AtomTriplets *atom_triplets) {
    long j, svecs_adrs;
    double pre_phase;
    lapack_complex_double pre_phase_factor;

    svecs_adrs = atom_triplets->p2s_map[i_patom] * atom_triplets->multi_dims[1];
    pre_phase = 0;
    for (j = 0; j < 3; j++) {
        pre_phase +=
            atom_triplets
                ->svecs[atom_triplets->multiplicity[svecs_adrs][1]][j] *
            (q_vecs[0][j] + q_vecs[1][j] + q_vecs[2][j]);
    }
    pre_phase *= M_2PI;
    pre_phase_factor =
        lapack_make_complex_double(cos(pre_phase), sin(pre_phase));
    return pre_phase_factor;
}

static lapack_complex_double get_phase_factor(const double q[3],
                                              const double (*svecs)[3],
                                              const long multi[2]) {
    long i, j;
    double sum_real, sum_imag, phase;

    sum_real = 0;
    sum_imag = 0;
    for (i = 0; i < multi[0]; i++) {
        phase = 0;
        for (j = 0; j < 3; j++) {
            phase += q[j] * svecs[multi[1] + i][j];
        }
        phase *= M_2PI;
        sum_real += cos(phase);
        sum_imag += sin(phase);
    }
    sum_real /= multi[0];
    sum_imag /= multi[0];

    return lapack_make_complex_double(sum_real, sum_imag);
}

static lapack_complex_double sum_lapack_complex_double(
    lapack_complex_double a, lapack_complex_double b) {
    double v_real, v_imag;
    v_real = lapack_complex_double_real(a) + lapack_complex_double_real(b);
    v_imag = lapack_complex_double_imag(a) + lapack_complex_double_imag(b);
    return lapack_make_complex_double(v_real, v_imag);
}
