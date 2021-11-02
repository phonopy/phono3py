/* Copyright (C) 2008 Atsushi Togo */
/* All rights reserved. */

/* This file was originally part of spglib. */

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

#include "bzgrid.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "grgrid.h"
#include "lagrid.h"

#define BZG_NUM_BZ_SEARCH_SPACE 125
#define GRID_TOLERANCE_FACTOR 0.01
static long bz_search_space[BZG_NUM_BZ_SEARCH_SPACE][3] = {
    {0, 0, 0},   {0, 0, 1},   {0, 0, 2},   {0, 0, -2},   {0, 0, -1},
    {0, 1, 0},   {0, 1, 1},   {0, 1, 2},   {0, 1, -2},   {0, 1, -1},
    {0, 2, 0},   {0, 2, 1},   {0, 2, 2},   {0, 2, -2},   {0, 2, -1},
    {0, -2, 0},  {0, -2, 1},  {0, -2, 2},  {0, -2, -2},  {0, -2, -1},
    {0, -1, 0},  {0, -1, 1},  {0, -1, 2},  {0, -1, -2},  {0, -1, -1},
    {1, 0, 0},   {1, 0, 1},   {1, 0, 2},   {1, 0, -2},   {1, 0, -1},
    {1, 1, 0},   {1, 1, 1},   {1, 1, 2},   {1, 1, -2},   {1, 1, -1},
    {1, 2, 0},   {1, 2, 1},   {1, 2, 2},   {1, 2, -2},   {1, 2, -1},
    {1, -2, 0},  {1, -2, 1},  {1, -2, 2},  {1, -2, -2},  {1, -2, -1},
    {1, -1, 0},  {1, -1, 1},  {1, -1, 2},  {1, -1, -2},  {1, -1, -1},
    {2, 0, 0},   {2, 0, 1},   {2, 0, 2},   {2, 0, -2},   {2, 0, -1},
    {2, 1, 0},   {2, 1, 1},   {2, 1, 2},   {2, 1, -2},   {2, 1, -1},
    {2, 2, 0},   {2, 2, 1},   {2, 2, 2},   {2, 2, -2},   {2, 2, -1},
    {2, -2, 0},  {2, -2, 1},  {2, -2, 2},  {2, -2, -2},  {2, -2, -1},
    {2, -1, 0},  {2, -1, 1},  {2, -1, 2},  {2, -1, -2},  {2, -1, -1},
    {-2, 0, 0},  {-2, 0, 1},  {-2, 0, 2},  {-2, 0, -2},  {-2, 0, -1},
    {-2, 1, 0},  {-2, 1, 1},  {-2, 1, 2},  {-2, 1, -2},  {-2, 1, -1},
    {-2, 2, 0},  {-2, 2, 1},  {-2, 2, 2},  {-2, 2, -2},  {-2, 2, -1},
    {-2, -2, 0}, {-2, -2, 1}, {-2, -2, 2}, {-2, -2, -2}, {-2, -2, -1},
    {-2, -1, 0}, {-2, -1, 1}, {-2, -1, 2}, {-2, -1, -2}, {-2, -1, -1},
    {-1, 0, 0},  {-1, 0, 1},  {-1, 0, 2},  {-1, 0, -2},  {-1, 0, -1},
    {-1, 1, 0},  {-1, 1, 1},  {-1, 1, 2},  {-1, 1, -2},  {-1, 1, -1},
    {-1, 2, 0},  {-1, 2, 1},  {-1, 2, 2},  {-1, 2, -2},  {-1, 2, -1},
    {-1, -2, 0}, {-1, -2, 1}, {-1, -2, 2}, {-1, -2, -2}, {-1, -2, -1},
    {-1, -1, 0}, {-1, -1, 1}, {-1, -1, 2}, {-1, -1, -2}, {-1, -1, -1}};

static void get_bz_grid_addresses_type1(BZGrid *bzgrid, const long Qinv[3][3]);
static void get_bz_grid_addresses_type2(BZGrid *bzgrid, const long Qinv[3][3]);
static void set_bz_address(long address[3], const long bz_index,
                           const long grid_address[3], const long D_diag[3],
                           const long nint[3], const long Qinv[3][3]);
static double get_bz_distances(long nint[3], double distances[],
                               const BZGrid *bzgrid, const long grid_address[3],
                               const double tolerance);
static void multiply_matrix_vector_d3(double v[3], const double a[3][3],
                                      const double b[3]);
static long get_inverse_unimodular_matrix_l3(long m[3][3], const long a[3][3]);
static double norm_squared_d3(const double a[3]);

long bzg_rotate_grid_index(const long bz_grid_index, const long rotation[3][3],
                           const ConstBZGrid *bzgrid) {
    long i, gp, num_bzgp, num_grgp;
    long dadrs[3], dadrs_rot[3], adrs_rot[3];

    grg_get_double_grid_address(dadrs, bzgrid->addresses[bz_grid_index],
                                bzgrid->PS);
    lagmat_multiply_matrix_vector_l3(dadrs_rot, rotation, dadrs);
    grg_get_grid_address(adrs_rot, dadrs_rot, bzgrid->PS);
    gp = grg_get_grid_index(adrs_rot, bzgrid->D_diag);

    if (bzgrid->type == 1) {
        if (bzgrid->addresses[gp][0] == adrs_rot[0] &&
            bzgrid->addresses[gp][1] == adrs_rot[1] &&
            bzgrid->addresses[gp][2] == adrs_rot[2]) {
            return gp;
        }
        num_grgp = bzgrid->D_diag[0] * bzgrid->D_diag[1] * bzgrid->D_diag[2];
        num_bzgp = num_grgp * 8;
        for (i = bzgrid->gp_map[num_bzgp + gp] + num_grgp;
             i < bzgrid->gp_map[num_bzgp + gp + 1] + num_grgp; i++) {
            if (bzgrid->addresses[i][0] == adrs_rot[0] &&
                bzgrid->addresses[i][1] == adrs_rot[1] &&
                bzgrid->addresses[i][2] == adrs_rot[2]) {
                return i;
            }
        }
    } else {
        for (i = bzgrid->gp_map[gp]; i < bzgrid->gp_map[gp + 1]; i++) {
            if (bzgrid->addresses[i][0] == adrs_rot[0] &&
                bzgrid->addresses[i][1] == adrs_rot[1] &&
                bzgrid->addresses[i][2] == adrs_rot[2]) {
                return i;
            }
        }
    }

    /* This should not happen, but possible when bzgrid is ill-defined. */
    return bzgrid->gp_map[gp];
}

long bzg_get_bz_grid_addresses(BZGrid *bzgrid) {
    long det;
    long Qinv[3][3];

    det = get_inverse_unimodular_matrix_l3(Qinv, bzgrid->Q);
    if (det == 0) {
        return 0;
    }

    if (bzgrid->type == 1) {
        get_bz_grid_addresses_type1(bzgrid, Qinv);
    } else {
        get_bz_grid_addresses_type2(bzgrid, Qinv);
    }

    return 1;
}

/* Note: Tolerance in squared distance. */
double bzg_get_tolerance_for_BZ_reduction(const BZGrid *bzgrid) {
    long i, j;
    double tolerance;
    double length[3];
    double reclatQ[3][3];

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            reclatQ[i][j] = bzgrid->reclat[i][0] * bzgrid->Q[0][j] +
                            bzgrid->reclat[i][1] * bzgrid->Q[1][j] +
                            bzgrid->reclat[i][2] * bzgrid->Q[2][j];
        }
    }

    for (i = 0; i < 3; i++) {
        length[i] = 0;
        for (j = 0; j < 3; j++) {
            length[i] += reclatQ[j][i] * reclatQ[j][i];
        }
        length[i] /= bzgrid->D_diag[i] * bzgrid->D_diag[i];
    }
    tolerance = length[0];
    for (i = 1; i < 3; i++) {
        if (tolerance < length[i]) {
            tolerance = length[i];
        }
    }
    tolerance *= GRID_TOLERANCE_FACTOR;

    return tolerance;
}

RotMats *bzg_alloc_RotMats(const long size) {
    RotMats *rotmats;

    rotmats = NULL;

    if ((rotmats = (RotMats *)malloc(sizeof(RotMats))) == NULL) {
        warning_print("Memory could not be allocated.");
        return NULL;
    }

    rotmats->size = size;
    if (size > 0) {
        if ((rotmats->mat = (long(*)[3][3])malloc(sizeof(long[3][3]) * size)) ==
            NULL) {
            warning_print("Memory could not be allocated ");
            warning_print("(RotMats, line %d, %s).\n", __LINE__, __FILE__);
            free(rotmats);
            rotmats = NULL;
            return NULL;
        }
    }
    return rotmats;
}

void bzg_free_RotMats(RotMats *rotmats) {
    if (rotmats->size > 0) {
        free(rotmats->mat);
        rotmats->mat = NULL;
    }
    free(rotmats);
}

void bzg_multiply_matrix_vector_ld3(double v[3], const long a[3][3],
                                    const double b[3]) {
    long i;
    double c[3];
    for (i = 0; i < 3; i++) {
        c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
    }
    for (i = 0; i < 3; i++) {
        v[i] = c[i];
    }
}

static void get_bz_grid_addresses_type1(BZGrid *bzgrid, const long Qinv[3][3]) {
    double tolerance, min_distance;
    double distances[BZG_NUM_BZ_SEARCH_SPACE];
    long bzmesh[3], bz_address_double[3], nint[3], gr_adrs[3];
    long i, j, k, boundary_num_gp, total_num_gp, bzgp, gp, num_bzmesh;
    long count, id_shift;

    tolerance = bzg_get_tolerance_for_BZ_reduction(bzgrid);
    for (j = 0; j < 3; j++) {
        bzmesh[j] = bzgrid->D_diag[j] * 2;
    }

    num_bzmesh = bzmesh[0] * bzmesh[1] * bzmesh[2];
    for (i = 0; i < num_bzmesh; i++) {
        bzgrid->gp_map[i] = num_bzmesh;
    }

    boundary_num_gp = 0;
    total_num_gp = bzgrid->D_diag[0] * bzgrid->D_diag[1] * bzgrid->D_diag[2];

    /* Multithreading doesn't work for this loop since gp calculated */
    /* with boundary_num_gp is unstable to store bz_grid_address. */
    bzgrid->gp_map[num_bzmesh] = 0;
    id_shift = 0;
    for (i = 0; i < total_num_gp; i++) {
        grg_get_grid_address_from_index(gr_adrs, i, bzgrid->D_diag);
        min_distance =
            get_bz_distances(nint, distances, bzgrid, gr_adrs, tolerance);
        count = 0;
        for (j = 0; j < BZG_NUM_BZ_SEARCH_SPACE; j++) {
            if (distances[j] < min_distance + tolerance) {
                if (count == 0) {
                    gp = i;
                } else {
                    gp = boundary_num_gp + total_num_gp;
                    boundary_num_gp++;
                }
                count++;
                set_bz_address(bzgrid->addresses[gp], j, gr_adrs,
                               bzgrid->D_diag, nint, Qinv);
                for (k = 0; k < 3; k++) {
                    bz_address_double[k] =
                        bzgrid->addresses[gp][k] * 2 + bzgrid->PS[k];
                }
                bzgp = grg_get_double_grid_index(bz_address_double, bzmesh,
                                                 bzgrid->PS);
                bzgrid->gp_map[bzgp] = gp;
                bzgrid->bzg2grg[gp] = i;
            }
        }
        /* This is used in get_BZ_triplets_at_q_type1. */
        /* The first one among those found is treated specially, so */
        /* excluded from the gp_map address shift by -1. */
        id_shift += count - 1;
        bzgrid->gp_map[num_bzmesh + i + 1] = id_shift;
    }
    bzgrid->size = boundary_num_gp + total_num_gp;
}

static void get_bz_grid_addresses_type2(BZGrid *bzgrid, const long Qinv[3][3]) {
    double tolerance, min_distance;
    double distances[BZG_NUM_BZ_SEARCH_SPACE];
    long nint[3], gr_adrs[3];
    long i, j, num_gp;

    tolerance = bzg_get_tolerance_for_BZ_reduction(bzgrid);
    num_gp = 0;
    /* The first element of gp_map is always 0. */
    bzgrid->gp_map[0] = 0;

    for (i = 0; i < bzgrid->D_diag[0] * bzgrid->D_diag[1] * bzgrid->D_diag[2];
         i++) {
        grg_get_grid_address_from_index(gr_adrs, i, bzgrid->D_diag);
        min_distance =
            get_bz_distances(nint, distances, bzgrid, gr_adrs, tolerance);
        for (j = 0; j < BZG_NUM_BZ_SEARCH_SPACE; j++) {
            if (distances[j] < min_distance + tolerance) {
                set_bz_address(bzgrid->addresses[num_gp], j, gr_adrs,
                               bzgrid->D_diag, nint, Qinv);
                bzgrid->bzg2grg[num_gp] = i;
                num_gp++;
            }
        }
        bzgrid->gp_map[i + 1] = num_gp;
    }

    bzgrid->size = num_gp;
}

static void set_bz_address(long address[3], const long bz_index,
                           const long grid_address[3], const long D_diag[3],
                           const long nint[3], const long Qinv[3][3]) {
    long i;
    long deltaG[3];

    for (i = 0; i < 3; i++) {
        deltaG[i] = bz_search_space[bz_index][i] - nint[i];
    }
    lagmat_multiply_matrix_vector_l3(deltaG, Qinv, deltaG);
    for (i = 0; i < 3; i++) {
        address[i] = grid_address[i] + deltaG[i] * D_diag[i];
    }
}

static double get_bz_distances(long nint[3], double distances[],
                               const BZGrid *bzgrid, const long grid_address[3],
                               const double tolerance) {
    long i, j;
    long dadrs[3];
    double min_distance;
    double q_vec[3], q_red[3];

    grg_get_double_grid_address(dadrs, grid_address, bzgrid->PS);

    for (i = 0; i < 3; i++) {
        q_red[i] = dadrs[i] / (2.0 * bzgrid->D_diag[i]);
    }
    bzg_multiply_matrix_vector_ld3(q_red, bzgrid->Q, q_red);
    for (i = 0; i < 3; i++) {
        nint[i] = lagmat_Nint(q_red[i]);
        q_red[i] -= nint[i];
    }

    for (i = 0; i < BZG_NUM_BZ_SEARCH_SPACE; i++) {
        for (j = 0; j < 3; j++) {
            q_vec[j] = q_red[j] + bz_search_space[i][j];
        }
        multiply_matrix_vector_d3(q_vec, bzgrid->reclat, q_vec);
        distances[i] = norm_squared_d3(q_vec);
    }

    /* Use of tolerance is important to select first one among similar
     * distances. Otherwise the choice of bz grid address among
     * those translationally equivalent can change by very tiny numerical
     * fluctuation. */
    min_distance = distances[0];
    for (i = 1; i < BZG_NUM_BZ_SEARCH_SPACE; i++) {
        if (distances[i] < min_distance - tolerance) {
            min_distance = distances[i];
        }
    }

    return min_distance;
}

static void multiply_matrix_vector_d3(double v[3], const double a[3][3],
                                      const double b[3]) {
    long i;
    double c[3];
    for (i = 0; i < 3; i++) {
        c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
    }
    for (i = 0; i < 3; i++) {
        v[i] = c[i];
    }
}

static long get_inverse_unimodular_matrix_l3(long m[3][3], const long a[3][3]) {
    long det;
    long c[3][3];

    det = lagmat_get_determinant_l3(a);
    if (labs(det) != 1) {
        return 0;
    }

    c[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
    c[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
    c[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
    c[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) / det;
    c[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / det;
    c[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) / det;
    c[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    c[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
    c[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
    lagmat_copy_matrix_l3(m, c);

    return det;
}

static double norm_squared_d3(const double a[3]) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
