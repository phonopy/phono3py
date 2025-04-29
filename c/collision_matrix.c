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

#include "collision_matrix.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "funcs.h"
#include "phonoc_array.h"

static void get_collision_matrix(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t num_gp, const int64_t *map_q, const int64_t *rot_grid_points,
    const int64_t num_ir_gp, const int64_t num_rot,
    const double *rotations_cartesian, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency);
static void get_collision_matrix_at_gp(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t *map_q, const int64_t *rot_grid_points,
    const int64_t num_ir_gp, const int64_t num_rot,
    const double *rotations_cartesian, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency, const int64_t *gp2tp_map, const int64_t i);
static void get_reducible_collision_matrix(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t num_gp, const int64_t *map_q, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency);
static void get_reducible_collision_matrix_at_gp(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t num_gp, const int64_t *map_q, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency, const int64_t *gp2tp_map, const int64_t i);
static int64_t get_inv_sinh(double *inv_sinh, const int64_t gp,
                            const double temperature_THz,
                            const double *frequencies, const int64_t triplet[3],
                            const int64_t *triplets_map, const int64_t *map_q,
                            const int64_t num_band,
                            const double cutoff_frequency);
static int64_t *create_gp2tp_map(const int64_t *triplets_map,
                                 const int64_t num_gp);

void col_get_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q,
    const int64_t *rot_grid_points, const double *rotations_cartesian,
    const double *g, const int64_t num_ir_gp, const int64_t num_gp,
    const int64_t num_rot, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency) {
    int64_t num_triplets, num_band0, num_band;

    num_triplets = fc3_normal_squared->dims[0];
    num_band0 = fc3_normal_squared->dims[1];
    num_band = fc3_normal_squared->dims[2];

    get_collision_matrix(
        collision_matrix, fc3_normal_squared->data, num_band0, num_band,
        frequencies, triplets, triplets_map, num_gp, map_q, rot_grid_points,
        num_ir_gp, num_rot, rotations_cartesian,
        g + 2 * num_triplets * num_band0 * num_band * num_band, temperature_THz,
        unit_conversion_factor, cutoff_frequency);
}

void col_get_reducible_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q, const double *g,
    const int64_t num_gp, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency) {
    int64_t num_triplets, num_band, num_band0;

    num_triplets = fc3_normal_squared->dims[0];
    num_band0 = fc3_normal_squared->dims[1];
    num_band = fc3_normal_squared->dims[2];

    get_reducible_collision_matrix(
        collision_matrix, fc3_normal_squared->data, num_band0, num_band,
        frequencies, triplets, triplets_map, num_gp, map_q,
        g + 2 * num_triplets * num_band0 * num_band * num_band, temperature_THz,
        unit_conversion_factor, cutoff_frequency);
}

static void get_collision_matrix(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t num_gp, const int64_t *map_q, const int64_t *rot_grid_points,
    const int64_t num_ir_gp, const int64_t num_rot,
    const double *rotations_cartesian, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency) {
    int64_t i;
    int64_t *gp2tp_map;

    gp2tp_map = create_gp2tp_map(triplets_map, num_gp);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_ir_gp; i++) {
        get_collision_matrix_at_gp(
            collision_matrix, fc3_normal_squared, num_band0, num_band,
            frequencies, triplets, triplets_map, map_q, rot_grid_points,
            num_ir_gp, num_rot, rotations_cartesian, g, temperature_THz,
            unit_conversion_factor, cutoff_frequency, gp2tp_map, i);
    }

    free(gp2tp_map);
    gp2tp_map = NULL;
}

static void get_collision_matrix_at_gp(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t *map_q, const int64_t *rot_grid_points,
    const int64_t num_ir_gp, const int64_t num_rot,
    const double *rotations_cartesian, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency, const int64_t *gp2tp_map, const int64_t i) {
    int64_t j, k, l, m, n, ti, r_gp, swapped;

    double collision;
    double *inv_sinh;

    inv_sinh = (double *)malloc(sizeof(double) * num_band);
    for (j = 0; j < num_rot; j++) {
        r_gp = rot_grid_points[i * num_rot + j];
        ti = gp2tp_map[triplets_map[r_gp]];
        swapped = get_inv_sinh(inv_sinh, r_gp, temperature_THz, frequencies,
                               triplets[ti], triplets_map, map_q, num_band,
                               cutoff_frequency);

        for (k = 0; k < num_band0; k++) {
            for (l = 0; l < num_band; l++) {
                collision = 0;
                for (m = 0; m < num_band; m++) {
                    if (swapped) {
                        collision +=
                            fc3_normal_squared[ti * num_band0 * num_band *
                                                   num_band +
                                               k * num_band * num_band +
                                               m * num_band + l] *
                            g[ti * num_band0 * num_band * num_band +
                              k * num_band * num_band + m * num_band + l] *
                            inv_sinh[m] * unit_conversion_factor;
                    } else {
                        collision +=
                            fc3_normal_squared[ti * num_band0 * num_band *
                                                   num_band +
                                               k * num_band * num_band +
                                               l * num_band + m] *
                            g[ti * num_band0 * num_band * num_band +
                              k * num_band * num_band + l * num_band + m] *
                            inv_sinh[m] * unit_conversion_factor;
                    }
                }
                for (m = 0; m < 3; m++) {
                    for (n = 0; n < 3; n++) {
                        collision_matrix[k * 3 * num_ir_gp * num_band * 3 +
                                         m * num_ir_gp * num_band * 3 +
                                         i * num_band * 3 + l * 3 + n] +=
                            collision * rotations_cartesian[j * 9 + m * 3 + n];
                    }
                }
            }
        }
    }
    free(inv_sinh);
    inv_sinh = NULL;
}

static void get_reducible_collision_matrix(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t num_gp, const int64_t *map_q, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency) {
    int64_t i;
    int64_t *gp2tp_map;

    gp2tp_map = create_gp2tp_map(triplets_map, num_gp);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_gp; i++) {
        get_reducible_collision_matrix_at_gp(
            collision_matrix, fc3_normal_squared, num_band0, num_band,
            frequencies, triplets, triplets_map, num_gp, map_q, g,
            temperature_THz, unit_conversion_factor, cutoff_frequency,
            gp2tp_map, i);
    }

    free(gp2tp_map);
    gp2tp_map = NULL;
}

static void get_reducible_collision_matrix_at_gp(
    double *collision_matrix, const double *fc3_normal_squared,
    const int64_t num_band0, const int64_t num_band, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplets_map,
    const int64_t num_gp, const int64_t *map_q, const double *g,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency, const int64_t *gp2tp_map, const int64_t i) {
    int64_t j, k, l, ti, swapped;
    double collision;
    double *inv_sinh;

    inv_sinh = (double *)malloc(sizeof(double) * num_band);
    ti = gp2tp_map[triplets_map[i]];
    swapped =
        get_inv_sinh(inv_sinh, i, temperature_THz, frequencies, triplets[ti],
                     triplets_map, map_q, num_band, cutoff_frequency);

    for (j = 0; j < num_band0; j++) {
        for (k = 0; k < num_band; k++) {
            collision = 0;
            for (l = 0; l < num_band; l++) {
                if (swapped) {
                    collision += fc3_normal_squared[ti * num_band0 * num_band *
                                                        num_band +
                                                    j * num_band * num_band +
                                                    l * num_band + k] *
                                 g[ti * num_band0 * num_band * num_band +
                                   j * num_band * num_band + l * num_band + k] *
                                 inv_sinh[l] * unit_conversion_factor;
                } else {
                    collision += fc3_normal_squared[ti * num_band0 * num_band *
                                                        num_band +
                                                    j * num_band * num_band +
                                                    k * num_band + l] *
                                 g[ti * num_band0 * num_band * num_band +
                                   j * num_band * num_band + k * num_band + l] *
                                 inv_sinh[l] * unit_conversion_factor;
                }
            }
            collision_matrix[j * num_gp * num_band + i * num_band + k] +=
                collision;
        }
    }

    free(inv_sinh);
    inv_sinh = NULL;
}

static int64_t get_inv_sinh(double *inv_sinh, const int64_t gp,
                            const double temperature_THz,
                            const double *frequencies, const int64_t triplet[3],
                            const int64_t *triplets_map, const int64_t *map_q,
                            const int64_t num_band,
                            const double cutoff_frequency) {
    int64_t i, gp2, swapped;
    double f;

    /* This assumes the algorithm of get_ir_triplets_at_q_perm_q1q2, */
    /* where defined triplets_map[gp] == triplets_map[map_q[gp]]. */
    /* If triplets_map[map_q[gp]] != map_q[gp], q1 and q2 are permuted. */
    if (triplets_map[gp] == map_q[gp]) {
        gp2 = triplet[2];
        swapped = 0;
    } else {
        gp2 = triplet[1];
        swapped = 1;
    }

    for (i = 0; i < num_band; i++) {
        f = frequencies[gp2 * num_band + i];
        if (f > cutoff_frequency) {
            inv_sinh[i] = funcs_inv_sinh_occupation(f, temperature_THz);
        } else {
            inv_sinh[i] = 0;
        }
    }

    return swapped;
}

/* Symmetrically independent triplets are indexed. */
/* Inverse definition of ir_grid_points in get_BZ_triplets_at_q */
/* in triplet_grid.c. */
static int64_t *create_gp2tp_map(const int64_t *triplets_map,
                                 const int64_t num_gp) {
    int64_t i, num_ir;
    int64_t *gp2tp_map;

    gp2tp_map = (int64_t *)malloc(sizeof(int64_t) * num_gp);
    num_ir = 0;
    for (i = 0; i < num_gp; i++) {
        if (triplets_map[i] == i) {
            gp2tp_map[i] = num_ir;
            num_ir++;
        } else { /* This should not be used. */
            gp2tp_map[i] = -1;
        }
    }

    return gp2tp_map;
}
