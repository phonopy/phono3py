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

static void real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
                               const double q_vecs[3][3], const double *fc3,
                               const long is_compact_fc3,
                               const AtomTriplets *atom_triplets,
                               const long openmp_at_bands);
static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
                                        const double q1[3], const double q2[3],
                                        const double *fc3,
                                        const long is_compact_fc3,
                                        const AtomTriplets *atom_triplets,
                                        const long pi0, const long pi1,
                                        const long pi2);
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
                            const long openmp_at_bands) {
    real_to_reciprocal(fc3_reciprocal, q_vecs, fc3, is_compact_fc3,
                       atom_triplets, openmp_at_bands);
}

static void real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
                               const double q_vecs[3][3], const double *fc3,
                               const long is_compact_fc3,
                               const AtomTriplets *atom_triplets,
                               const long openmp_at_bands) {
    long i, j, k, l, m, n, ijk;
    long num_patom, num_band;
    lapack_complex_double pre_phase_factor, fc3_rec_elem[27], fc3_rec;

    num_patom = atom_triplets->multi_dims[1];
    num_band = num_patom * 3;

#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, l, m, n, fc3_rec_elem, fc3_rec, \
                                     pre_phase_factor) if (openmp_at_bands)
#endif
    for (ijk = 0; ijk < num_patom * num_patom * num_patom; ijk++) {
        i = ijk / (num_patom * num_patom);
        j = (ijk - (i * num_patom * num_patom)) / num_patom;
        k = ijk % num_patom;
        pre_phase_factor = get_pre_phase_factor(i, q_vecs, atom_triplets);
        real_to_reciprocal_elements(fc3_rec_elem, q_vecs[1], q_vecs[2], fc3,
                                    is_compact_fc3, atom_triplets, i, j, k);
        for (l = 0; l < 3; l++) {
            for (m = 0; m < 3; m++) {
                for (n = 0; n < 3; n++) {
                    fc3_rec = phonoc_complex_prod(
                        fc3_rec_elem[l * 9 + m * 3 + n], pre_phase_factor);
                    fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                   (j * 3 + m) * num_band + k * 3 + n] =
                        sum_lapack_complex_double(
                            fc3_reciprocal[(i * 3 + l) * num_band * num_band +
                                           (j * 3 + m) * num_band + k * 3 + n],
                            fc3_rec);
                }
            }
        }

        if (atom_triplets->make_r0_average) {
            real_to_reciprocal_elements(fc3_rec_elem, q_vecs[0], q_vecs[2], fc3,
                                        is_compact_fc3, atom_triplets, i, j, k);
            for (l = 0; l < 3; l++) {
                for (m = 0; m < 3; m++) {
                    for (n = 0; n < 3; n++) {
                        // fc3_rec is stored in a way swapping jm <-> il.
                        fc3_rec = phonoc_complex_prod(
                            fc3_rec_elem[l * 9 + m * 3 + n], pre_phase_factor);
                        fc3_reciprocal[(j * 3 + m) * num_band * num_band +
                                       (i * 3 + l) * num_band + k * 3 + n] =
                            sum_lapack_complex_double(
                                fc3_reciprocal[(j * 3 + m) * num_band *
                                                   num_band +
                                               (i * 3 + l) * num_band + k * 3 +
                                               n],
                                fc3_rec);
                    }
                }
            }

            real_to_reciprocal_elements(fc3_rec_elem, q_vecs[1], q_vecs[0], fc3,
                                        is_compact_fc3, atom_triplets, i, j, k);
            for (l = 0; l < 3; l++) {
                for (m = 0; m < 3; m++) {
                    for (n = 0; n < 3; n++) {
                        // fc3_rec is stored in a way swapping kn <-> il.
                        fc3_rec = phonoc_complex_prod(
                            fc3_rec_elem[l * 9 + m * 3 + n], pre_phase_factor);
                        fc3_reciprocal[(k * 3 + n) * num_band * num_band +
                                       (j * 3 + m) * num_band + i * 3 + l] =
                            sum_lapack_complex_double(
                                fc3_reciprocal[(k * 3 + n) * num_band *
                                                   num_band +
                                               (j * 3 + m) * num_band + i * 3 +
                                               l],
                                fc3_rec);
                    }
                }
            }
        }
    }
}

static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
                                        const double q1[3], const double q2[3],
                                        const double *fc3,
                                        const long is_compact_fc3,
                                        const AtomTriplets *atom_triplets,
                                        const long pi0, const long pi1,
                                        const long pi2) {
    long i, j, k, l;
    long num_satom, adrs_shift, adrs_vec1, adrs_vec2;
    lapack_complex_double phase_factor, phase_factor1, phase_factor2;
    double fc3_rec_real[27], fc3_rec_imag[27];

    for (i = 0; i < 27; i++) {
        fc3_rec_real[i] = 0;
        fc3_rec_imag[i] = 0;
    }

    num_satom = atom_triplets->multi_dims[0];

    if (is_compact_fc3) {
        i = pi0;
    } else {
        i = atom_triplets->p2s_map[pi0];
    }

    for (j = 0; j < num_satom; j++) {
        if (atom_triplets->s2p_map[j] != atom_triplets->p2s_map[pi1]) {
            continue;
        }

        adrs_vec1 = j * atom_triplets->multi_dims[1] + pi0;
        phase_factor1 = get_phase_factor(
            q1, atom_triplets->svecs, atom_triplets->multiplicity[adrs_vec1]);
        for (k = 0; k < num_satom; k++) {
            if (atom_triplets->s2p_map[k] != atom_triplets->p2s_map[pi2]) {
                continue;
            }
            adrs_vec2 = k * atom_triplets->multi_dims[1] + pi0;
            phase_factor2 =
                get_phase_factor(q2, atom_triplets->svecs,
                                 atom_triplets->multiplicity[adrs_vec2]);
            adrs_shift =
                i * 27 * num_satom * num_satom + j * 27 * num_satom + k * 27;
            phase_factor = phonoc_complex_prod(phase_factor1, phase_factor2);
            for (l = 0; l < 27; l++) {
                fc3_rec_real[l] += lapack_complex_double_real(phase_factor) *
                                   fc3[adrs_shift + l];
                fc3_rec_imag[l] += lapack_complex_double_imag(phase_factor) *
                                   fc3[adrs_shift + l];
            }
        }
    }

    for (i = 0; i < 27; i++) {
        fc3_rec_elem[i] =
            lapack_make_complex_double(fc3_rec_real[i], fc3_rec_imag[i]);
    }
}

// This function has no need to think about phase 2pi
static lapack_complex_double get_pre_phase_factor(
    const long i_patom, const double q_vecs[3][3],
    const AtomTriplets *atom_triplets) {
    long i, j, svecs_adrs;
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
