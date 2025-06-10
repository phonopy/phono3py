/* Copyright (C) 2021 Atsushi Togo */
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

#include "gridsys.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bzgrid.h"
#include "grgrid.h"
#include "lagrid.h"
#include "niggli.h"
#include "recgrid.h"
#include "tetrahedron_method.h"
#include "triplet.h"
#include "triplet_iw.h"

#define GRIDSYS_NIGGLI_TOLERANCE 1e-5

void gridsys_get_all_grid_addresses(int64_t (*gr_grid_addresses)[3],
                                    const int64_t D_diag[3]) {
    grg_get_all_grid_addresses(gr_grid_addresses, D_diag);
}

void gridsys_get_double_grid_address(int64_t address_double[3],
                                     const int64_t address[3],
                                     const int64_t PS[3]) {
    grg_get_double_grid_address(address_double, address, PS);
}

void gridsys_get_grid_address_from_index(int64_t address[3],
                                         const int64_t grid_index,
                                         const int64_t D_diag[3]) {
    grg_get_grid_address_from_index(address, grid_index, D_diag);
}

int64_t gridsys_get_double_grid_index(const int64_t address_double[3],
                                      const int64_t D_diag[3],
                                      const int64_t PS[3]) {
    return grg_get_double_grid_index(address_double, D_diag, PS);
}

int64_t gridsys_get_grid_index_from_address(const int64_t address[3],
                                            const int64_t D_diag[3]) {
    return grg_get_grid_index(address, D_diag);
}

int64_t gridsys_rotate_grid_index(const int64_t grid_index,
                                  const int64_t rotation[3][3],
                                  const int64_t D_diag[3],
                                  const int64_t PS[3]) {
    return grg_rotate_grid_index(grid_index, rotation, D_diag, PS);
}

int64_t gridsys_get_reciprocal_point_group(int64_t rec_rotations[48][3][3],
                                           const int64_t (*rotations)[3][3],
                                           const int64_t num_rot,
                                           const int64_t is_time_reversal) {
    return grg_get_reciprocal_point_group(rec_rotations, rotations, num_rot,
                                          is_time_reversal, 1);
}

int64_t gridsys_get_snf3x3(int64_t D_diag[3], int64_t P[3][3], int64_t Q[3][3],
                           const int64_t A[3][3]) {
    return grg_get_snf3x3(D_diag, P, Q, A);
}

int64_t gridsys_transform_rotations(int64_t (*transformed_rots)[3][3],
                                    const int64_t (*rotations)[3][3],
                                    const int64_t num_rot,
                                    const int64_t D_diag[3],
                                    const int64_t Q[3][3]) {
    int64_t succeeded;
    succeeded = grg_transform_rotations(transformed_rots, rotations, num_rot,
                                        D_diag, Q);
    return succeeded;
}

void gridsys_get_ir_grid_map(int64_t *ir_grid_map,
                             const int64_t (*rotations)[3][3],
                             const int64_t num_rot, const int64_t D_diag[3],
                             const int64_t PS[3]) {
    grg_get_ir_grid_map(ir_grid_map, rotations, num_rot, D_diag, PS);
}

int64_t gridsys_get_bz_grid_addresses(
    int64_t (*bz_grid_addresses)[3], int64_t *bz_map, int64_t *bzg2grg,
    const int64_t D_diag[3], const int64_t Q[3][3], const int64_t PS[3],
    const double rec_lattice[3][3], const int64_t bz_grid_type) {
    RecgridBZGrid *bzgrid;
    int64_t i, j, size;
    int64_t inv_Mpr_int[3][3];
    double inv_Lr[3][3], inv_Mpr[3][3];
    double niggli_lattice[9];

    if ((bzgrid = (RecgridBZGrid *)malloc(sizeof(RecgridBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            niggli_lattice[i * 3 + j] = rec_lattice[i][j];
        }
    }
    if (!niggli_reduce(niggli_lattice, GRIDSYS_NIGGLI_TOLERANCE)) {
        return 0;
    }
    if (!lagmat_inverse_matrix_d3(inv_Lr, (double (*)[3])niggli_lattice,
                                  GRIDSYS_NIGGLI_TOLERANCE)) {
        return 0;
    }
    lagmat_multiply_matrix_d3(inv_Mpr, inv_Lr, rec_lattice);
    lagmat_cast_matrix_3d_to_3l(inv_Mpr_int, inv_Mpr);
    // printf("%ld %ld %ld\n", inv_Mpr_int[0][0], inv_Mpr_int[0][1],
    //        inv_Mpr_int[0][2]);
    // printf("%ld %ld %ld\n", inv_Mpr_int[1][0], inv_Mpr_int[1][1],
    //        inv_Mpr_int[1][2]);
    // printf("%ld %ld %ld\n", inv_Mpr_int[2][0], inv_Mpr_int[2][1],
    //        inv_Mpr_int[2][2]);

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->bzg2grg = bzg2grg;
    bzgrid->type = bz_grid_type;
    lagmat_multiply_matrix_l3(bzgrid->Q, inv_Mpr_int, Q);
    lagmat_copy_matrix_d3(bzgrid->reclat, (double (*)[3])niggli_lattice);
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = PS[i];
    }

    if (bzg_get_bz_grid_addresses(bzgrid)) {
        size = bzgrid->size;
    } else {
        size = 0;
    }

    free(bzgrid);
    bzgrid = NULL;

    return size;
}

int64_t gridsys_rotate_bz_grid_index(
    const int64_t bz_grid_index, const int64_t rotation[3][3],
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t D_diag[3], const int64_t PS[3], const int64_t bz_grid_type) {
    RecgridConstBZGrid *bzgrid;
    int64_t i, rot_bz_gp;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = PS[i];
    }

    rot_bz_gp = bzg_rotate_grid_index(bz_grid_index, rotation, bzgrid);

    free(bzgrid);
    bzgrid = NULL;

    return rot_bz_gp;
}

int64_t gridsys_get_triplets_at_q(int64_t *map_triplets, int64_t *map_q,
                                  const int64_t grid_point,
                                  const int64_t D_diag[3],
                                  const int64_t is_time_reversal,
                                  const int64_t num_rot,
                                  const int64_t (*rec_rotations)[3][3],
                                  const int64_t swappable) {
    return tpl_get_triplets_reciprocal_mesh_at_q(
        map_triplets, map_q, grid_point, D_diag, is_time_reversal, num_rot,
        rec_rotations, swappable);
}

int64_t gridsys_get_bz_triplets_at_q(
    int64_t (*ir_triplets)[3], const int64_t bz_grid_index,
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t *map_triplets, const int64_t num_map_triplets,
    const int64_t D_diag[3], const int64_t Q[3][3],
    const double reciprocal_lattice[3][3], const int64_t bz_grid_type) {
    RecgridConstBZGrid *bzgrid;
    int64_t i, j, num_ir;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = 0;
        for (j = 0; j < 3; j++) {
            bzgrid->reclat[i][j] = reciprocal_lattice[i][j];
            bzgrid->Q[i][j] = Q[i][j];
        }
    }
    bzgrid->size = num_map_triplets;

    num_ir = tpl_get_BZ_triplets_at_q(ir_triplets, bz_grid_index, bzgrid,
                                      map_triplets);
    free(bzgrid);
    bzgrid = NULL;

    return num_ir;
}

double gridsys_get_thm_integration_weight(const double omega,
                                          const double tetrahedra_omegas[24][4],
                                          const char function) {
    return thm_get_integration_weight(omega, tetrahedra_omegas, function);
}

void gridsys_get_thm_all_relative_grid_address(
    int64_t relative_grid_address[4][24][4][3]) {
    thm_get_all_relative_grid_address(relative_grid_address);
}

int64_t gridsys_get_thm_relative_grid_address(
    int64_t relative_grid_addresses[24][4][3], const double rec_lattice[3][3]) {
    return thm_get_relative_grid_address(relative_grid_addresses, rec_lattice);
}

/* relative_grid_addresses are given as P multiplied with those from */
/* dataset, i.e., */
/*     np.dot(relative_grid_addresses, P.T) */
int64_t gridsys_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const int64_t num_band0, const int64_t relative_grid_address[24][4][3],
    const int64_t D_diag[3], const int64_t (*triplets)[3],
    const int64_t num_triplets, const int64_t (*bz_grid_addresses)[3],
    const int64_t *bz_map, const int64_t bz_grid_type,
    const double *frequencies1, const int64_t num_band1,
    const double *frequencies2, const int64_t num_band2, const int64_t tp_type,
    const int64_t openmp_per_triplets) {
    RecgridConstBZGrid *bzgrid;
    int64_t i;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
    }

    tpl_get_integration_weight(iw, iw_zero, frequency_points, num_band0,
                               relative_grid_address, triplets, num_triplets,
                               bzgrid, frequencies1, num_band1, frequencies2,
                               num_band2, tp_type, openmp_per_triplets);
    free(bzgrid);
    bzgrid = NULL;

    return 1;
}

void gridsys_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const int64_t num_band0,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const double *frequencies, const int64_t num_band, const int64_t tp_type) {
    tpl_get_integration_weight_with_sigma(
        iw, iw_zero, sigma, sigma_cutoff, frequency_points, num_band0, triplets,
        num_triplets, frequencies, num_band, tp_type);
}
