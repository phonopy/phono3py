/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* These codes were originally parts of spglib, but only develped */
/* and used for phono3py. Therefore these were moved from spglib to */
/* phono3py. This file is part of phonopy. */

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

#include "triplet_grid.h"

#include <stddef.h>
#include <stdlib.h>

#include "bzgrid.h"
#include "grgrid.h"
#include "lagrid.h"
#include "triplet.h"

static long get_ir_triplets_at_q(long *map_triplets, long *map_q,
                                 const long grid_point, const long D_diag[3],
                                 const RotMats *rot_reciprocal,
                                 const long swappable);
static long get_ir_triplets_at_q_perm_q1q2(long *map_triplets,
                                           const long *map_q,
                                           const long grid_point,
                                           const long D_diag[3]);
static long get_ir_triplets_at_q_noperm(long *map_triplets, const long *map_q,
                                        const long grid_point,
                                        const long D_diag[3]);
static long get_BZ_triplets_at_q(long (*triplets)[3], const long grid_point,
                                 const ConstBZGrid *bzgrid,
                                 const long *map_triplets);
static void get_BZ_triplets_at_q_type1(long (*triplets)[3],
                                       const long grid_point,
                                       const ConstBZGrid *bzgrid,
                                       const long *ir_q1_gps,
                                       const long num_ir);
static void get_BZ_triplets_at_q_type2(long (*triplets)[3],
                                       const long grid_point,
                                       const ConstBZGrid *bzgrid,
                                       const long *ir_q1_gps,
                                       const long num_ir);
static double get_squared_distance(const long G[3], const double LQD_inv[3][3]);
static void get_LQD_inv(double LQD_inv[3][3], const ConstBZGrid *bzgrid);
static RotMats *get_reciprocal_point_group_with_q(const RotMats *rot_reciprocal,
                                                  const long D_diag[3],
                                                  const long grid_point);
static RotMats *get_reciprocal_point_group(const long (*rec_rotations_in)[3][3],
                                           const long num_rot,
                                           const long is_time_reversal,
                                           const long is_transpose);

long tpk_get_ir_triplets_at_q(long *map_triplets, long *map_q,
                              const long grid_point, const long D_diag[3],
                              const long is_time_reversal,
                              const long (*rec_rotations_in)[3][3],
                              const long num_rot, const long swappable) {
    long num_ir;
    RotMats *rotations;

    rotations = get_reciprocal_point_group(rec_rotations_in, num_rot,
                                           is_time_reversal, 0);
    if (rotations == NULL) {
        return 0;
    }

    num_ir = get_ir_triplets_at_q(map_triplets, map_q, grid_point, D_diag,
                                  rotations, swappable);
    bzg_free_RotMats(rotations);
    rotations = NULL;

    return num_ir;
}

long tpk_get_BZ_triplets_at_q(long (*triplets)[3], const long grid_point,
                              const ConstBZGrid *bzgrid,
                              const long *map_triplets) {
    return get_BZ_triplets_at_q(triplets, grid_point, bzgrid, map_triplets);
}

static long get_ir_triplets_at_q(long *map_triplets, long *map_q,
                                 const long grid_point, const long D_diag[3],
                                 const RotMats *rot_reciprocal,
                                 const long swappable) {
    long i, num_ir_q, num_ir_triplets;
    long PS[3];
    RotMats *rot_reciprocal_q;

    rot_reciprocal_q = NULL;

    for (i = 0; i < 3; i++) {
        PS[i] = 0;
    }

    /* Search irreducible q-points (map_q) with a stabilizer. */
    rot_reciprocal_q =
        get_reciprocal_point_group_with_q(rot_reciprocal, D_diag, grid_point);

    grg_get_ir_grid_map(map_q, rot_reciprocal_q->mat, rot_reciprocal_q->size,
                        D_diag, PS);
    num_ir_q = 0;
    for (i = 0; i < D_diag[0] * D_diag[1] * D_diag[2]; i++) {
        if (map_q[i] == i) {
            num_ir_q++;
        }
    }

    if (swappable) {
        num_ir_triplets = get_ir_triplets_at_q_perm_q1q2(map_triplets, map_q,
                                                         grid_point, D_diag);
    } else {
        num_ir_triplets = get_ir_triplets_at_q_noperm(map_triplets, map_q,
                                                      grid_point, D_diag);
    }

    bzg_free_RotMats(rot_reciprocal_q);
    rot_reciprocal_q = NULL;

    return num_ir_triplets;
}

static long get_ir_triplets_at_q_perm_q1q2(long *map_triplets,
                                           const long *map_q,
                                           const long grid_point,
                                           const long D_diag[3]) {
    long j, num_grid, num_ir_triplets, gp1, gp2;
    long adrs0[3], adrs1[3], adrs2[3];

    num_ir_triplets = 0;
    num_grid = D_diag[0] * D_diag[1] * D_diag[2];
    grg_get_grid_address_from_index(adrs0, grid_point, D_diag);

    // #ifdef _OPENMP
    // #pragma omp parallel for private(j, gp2, adrs1, adrs2)
    // #endif
    for (gp1 = 0; gp1 < num_grid; gp1++) {
        if (map_q[gp1] == gp1) {
            grg_get_grid_address_from_index(adrs1, gp1, D_diag);
            for (j = 0; j < 3; j++) {
                adrs2[j] = -adrs0[j] - adrs1[j];
            }
            /* If map_q[gp2] is smaller than current gp1, map_q[gp2] should */
            /* equal to a previous gp1 for which map_triplets is already */
            /* filled. So the counter is not incremented. */
            gp2 = grg_get_grid_index(adrs2, D_diag);
            if (map_q[gp2] < gp1) {
                map_triplets[gp1] = map_q[gp2];
            } else {
                map_triplets[gp1] = gp1;
                num_ir_triplets++;
            }
        }
    }

/* Fill unfilled elements of map_triplets. */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (gp1 = 0; gp1 < num_grid; gp1++) {
        if (map_q[gp1] != gp1) {
            /* map_q[gp1] is one of ir-gp1, so it is already filled. */
            map_triplets[gp1] = map_triplets[map_q[gp1]];
        }
    }

    return num_ir_triplets;
}

static long get_ir_triplets_at_q_noperm(long *map_triplets, const long *map_q,
                                        const long grid_point,
                                        const long D_diag[3]) {
    long gp1, num_grid, num_ir_triplets;

    num_ir_triplets = 0;
    num_grid = D_diag[0] * D_diag[1] * D_diag[2];

    for (gp1 = 0; gp1 < num_grid; gp1++) {
        if (map_q[gp1] == gp1) {
            map_triplets[gp1] = gp1;
            num_ir_triplets++;
        } else {
            map_triplets[gp1] = map_triplets[map_q[gp1]];
        }
    }

    return num_ir_triplets;
}

static long get_BZ_triplets_at_q(long (*triplets)[3], const long grid_point,
                                 const ConstBZGrid *bzgrid,
                                 const long *map_triplets) {
    long gp1, num_ir;
    long *ir_q1_gps;

    ir_q1_gps = NULL;
    num_ir = 0;

    if ((ir_q1_gps = (long *)malloc(sizeof(long) * bzgrid->size)) == NULL) {
        warning_print("Memory could not be allocated.");
        goto ret;
    }

    for (gp1 = 0; gp1 < bzgrid->size; gp1++) {
        if (map_triplets[gp1] == gp1) {
            ir_q1_gps[num_ir] = gp1;
            num_ir++;
        }
    }

    if (bzgrid->type == 1) {
        get_BZ_triplets_at_q_type1(triplets, grid_point, bzgrid, ir_q1_gps,
                                   num_ir);
    } else {
        get_BZ_triplets_at_q_type2(triplets, grid_point, bzgrid, ir_q1_gps,
                                   num_ir);
    }

    free(ir_q1_gps);
    ir_q1_gps = NULL;

ret:
    return num_ir;
}

static void get_BZ_triplets_at_q_type1(long (*triplets)[3],
                                       const long grid_point,
                                       const ConstBZGrid *bzgrid,
                                       const long *ir_q1_gps,
                                       const long num_ir) {
    long i, j, gp2, num_gp, num_bzgp, bz0, bz1, bz2;
    long bzgp[3], G[3];
    long bz_adrs0[3], bz_adrs1[3], bz_adrs2[3];
    const long *gp_map;
    const long(*bz_adrs)[3];
    double d2, min_d2, tolerance;
    double LQD_inv[3][3];

    gp_map = bzgrid->gp_map;
    bz_adrs = bzgrid->addresses;
    get_LQD_inv(LQD_inv, bzgrid);
    /* This tolerance is used to be consistent to BZ reduction in bzgrid. */
    tolerance = bzg_get_tolerance_for_BZ_reduction((BZGrid *)bzgrid);

    for (i = 0; i < 3; i++) {
        bz_adrs0[i] = bz_adrs[grid_point][i];
    }
    num_gp = bzgrid->D_diag[0] * bzgrid->D_diag[1] * bzgrid->D_diag[2];
    num_bzgp = num_gp * 8;

#ifdef _OPENMP
#pragma omp parallel for private(j, gp2, bzgp, G, bz_adrs1, bz_adrs2, d2, \
                                 min_d2, bz0, bz1, bz2)
#endif
    for (i = 0; i < num_ir; i++) {
        for (j = 0; j < 3; j++) {
            bz_adrs1[j] = bz_adrs[ir_q1_gps[i]][j];
            bz_adrs2[j] = -bz_adrs0[j] - bz_adrs1[j];
        }
        gp2 = grg_get_grid_index(bz_adrs2, bzgrid->D_diag);
        /* Negative value is the signal to initialize min_d2 later. */
        min_d2 = -1;
        for (bz0 = 0; bz0 < gp_map[num_bzgp + grid_point + 1] -
                                gp_map[num_bzgp + grid_point] + 1;
             bz0++) {
            if (bz0 == 0) {
                bzgp[0] = grid_point;
            } else {
                bzgp[0] = num_gp + gp_map[num_bzgp + grid_point] + bz0 - 1;
            }
            for (bz1 = 0; bz1 < gp_map[num_bzgp + ir_q1_gps[i] + 1] -
                                    gp_map[num_bzgp + ir_q1_gps[i]] + 1;
                 bz1++) {
                if (bz1 == 0) {
                    bzgp[1] = ir_q1_gps[i];
                } else {
                    bzgp[1] =
                        num_gp + gp_map[num_bzgp + ir_q1_gps[i]] + bz1 - 1;
                }
                for (bz2 = 0; bz2 < gp_map[num_bzgp + gp2 + 1] -
                                        gp_map[num_bzgp + gp2] + 1;
                     bz2++) {
                    if (bz2 == 0) {
                        bzgp[2] = gp2;
                    } else {
                        bzgp[2] = num_gp + gp_map[num_bzgp + gp2] + bz2 - 1;
                    }
                    for (j = 0; j < 3; j++) {
                        G[j] = bz_adrs[bzgp[0]][j] + bz_adrs[bzgp[1]][j] +
                               bz_adrs[bzgp[2]][j];
                    }
                    if (G[0] == 0 && G[1] == 0 && G[2] == 0) {
                        for (j = 0; j < 3; j++) {
                            triplets[i][j] = bzgp[j];
                        }
                        goto found;
                    }
                    d2 = get_squared_distance(G, LQD_inv);
                    if (d2 < min_d2 - tolerance || min_d2 < 0) {
                        min_d2 = d2;
                        for (j = 0; j < 3; j++) {
                            triplets[i][j] = bzgp[j];
                        }
                    }
                }
            }
        }
    found:;
    }
}

static void get_BZ_triplets_at_q_type2(long (*triplets)[3],
                                       const long grid_point,
                                       const ConstBZGrid *bzgrid,
                                       const long *ir_q1_gps,
                                       const long num_ir) {
    long i, j, gp0, gp2;
    long bzgp[3], G[3];
    long bz_adrs0[3], bz_adrs1[3], bz_adrs2[3];
    const long *gp_map;
    const long(*bz_adrs)[3];
    double d2, min_d2, tolerance;
    double LQD_inv[3][3];

    gp_map = bzgrid->gp_map;
    bz_adrs = bzgrid->addresses;
    get_LQD_inv(LQD_inv, bzgrid);
    /* This tolerance is used to be consistent to BZ reduction in bzgrid. */
    tolerance = bzg_get_tolerance_for_BZ_reduction((BZGrid *)bzgrid);

    for (i = 0; i < 3; i++) {
        bz_adrs0[i] = bz_adrs[grid_point][i];
    }
    gp0 = grg_get_grid_index(bz_adrs0, bzgrid->D_diag);

#ifdef _OPENMP
#pragma omp parallel for private(j, gp2, bzgp, G, bz_adrs1, bz_adrs2, d2, \
                                 min_d2)
#endif
    for (i = 0; i < num_ir; i++) {
        for (j = 0; j < 3; j++) {
            bz_adrs1[j] = bz_adrs[gp_map[ir_q1_gps[i]]][j];
            bz_adrs2[j] = -bz_adrs0[j] - bz_adrs1[j];
        }
        gp2 = grg_get_grid_index(bz_adrs2, bzgrid->D_diag);
        /* Negative value is the signal to initialize min_d2 later. */
        min_d2 = -1;
        for (bzgp[0] = gp_map[gp0]; bzgp[0] < gp_map[gp0 + 1]; bzgp[0]++) {
            for (bzgp[1] = gp_map[ir_q1_gps[i]];
                 bzgp[1] < gp_map[ir_q1_gps[i] + 1]; bzgp[1]++) {
                for (bzgp[2] = gp_map[gp2]; bzgp[2] < gp_map[gp2 + 1];
                     bzgp[2]++) {
                    for (j = 0; j < 3; j++) {
                        G[j] = bz_adrs[bzgp[0]][j] + bz_adrs[bzgp[1]][j] +
                               bz_adrs[bzgp[2]][j];
                    }
                    if (G[0] == 0 && G[1] == 0 && G[2] == 0) {
                        for (j = 0; j < 3; j++) {
                            triplets[i][j] = bzgp[j];
                        }
                        goto found;
                    }
                    d2 = get_squared_distance(G, LQD_inv);
                    if (d2 < min_d2 - tolerance || min_d2 < 0) {
                        min_d2 = d2;
                        for (j = 0; j < 3; j++) {
                            triplets[i][j] = bzgp[j];
                        }
                    }
                }
            }
        }
    found:;
    }
}

static double get_squared_distance(const long G[3],
                                   const double LQD_inv[3][3]) {
    double d, d2;
    long i;

    d2 = 0;
    for (i = 0; i < 3; i++) {
        d = LQD_inv[i][0] * G[0] + LQD_inv[i][1] * G[1] + LQD_inv[i][2] * G[2];
        d2 += d * d;
    }

    return d2;
}

static void get_LQD_inv(double LQD_inv[3][3], const ConstBZGrid *bzgrid) {
    long i, j, k;

    /* LQD^-1 */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                LQD_inv[i][k] =
                    bzgrid->reclat[i][j] * bzgrid->Q[j][k] / bzgrid->D_diag[k];
            }
        }
    }
}

/* Return NULL if failed */
static RotMats *get_reciprocal_point_group_with_q(const RotMats *rot_reciprocal,
                                                  const long D_diag[3],
                                                  const long grid_point) {
    long i, num_rot, gp_rot;
    long *ir_rot;
    long adrs[3], adrs_rot[3];
    RotMats *rot_reciprocal_q;

    ir_rot = NULL;
    rot_reciprocal_q = NULL;
    num_rot = 0;

    grg_get_grid_address_from_index(adrs, grid_point, D_diag);

    if ((ir_rot = (long *)malloc(sizeof(long) * rot_reciprocal->size)) ==
        NULL) {
        warning_print("Memory of ir_rot could not be allocated.");
        return NULL;
    }

    for (i = 0; i < rot_reciprocal->size; i++) {
        ir_rot[i] = -1;
    }
    for (i = 0; i < rot_reciprocal->size; i++) {
        lagmat_multiply_matrix_vector_l3(adrs_rot, rot_reciprocal->mat[i],
                                         adrs);
        gp_rot = grg_get_grid_index(adrs_rot, D_diag);

        if (gp_rot == grid_point) {
            ir_rot[num_rot] = i;
            num_rot++;
        }
    }

    if ((rot_reciprocal_q = bzg_alloc_RotMats(num_rot)) != NULL) {
        for (i = 0; i < num_rot; i++) {
            lagmat_copy_matrix_l3(rot_reciprocal_q->mat[i],
                                  rot_reciprocal->mat[ir_rot[i]]);
        }
    }

    free(ir_rot);
    ir_rot = NULL;

    return rot_reciprocal_q;
}

static RotMats *get_reciprocal_point_group(const long (*rec_rotations_in)[3][3],
                                           const long num_rot,
                                           const long is_time_reversal,
                                           const long is_transpose) {
    long i, num_rot_out;
    long rec_rotations_out[48][3][3];
    RotMats *rec_rotations;

    num_rot_out =
        grg_get_reciprocal_point_group(rec_rotations_out, rec_rotations_in,
                                       num_rot, is_time_reversal, is_transpose);
    if (num_rot_out == 0) {
        return NULL;
    }

    rec_rotations = bzg_alloc_RotMats(num_rot_out);
    for (i = 0; i < num_rot_out; i++) {
        lagmat_copy_matrix_l3(rec_rotations->mat[i], rec_rotations_out[i]);
    }

    return rec_rotations;
}
