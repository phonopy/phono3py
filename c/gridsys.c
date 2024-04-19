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

#include <stdio.h>
#include <stdlib.h>

#include "bzgrid.h"
#include "grgrid.h"
#include "lagrid.h"
#include "niggli.h"
#include "tetrahedron_method.h"
#include "triplet.h"
#include "triplet_iw.h"

#define GRIDSYS_NIGGLI_TOLERANCE 1e-5

void gridsys_get_all_grid_addresses(long (*gr_grid_addresses)[3],
                                    const long D_diag[3]) {
    grg_get_all_grid_addresses(gr_grid_addresses, D_diag);
}

void gridsys_get_double_grid_address(long address_double[3],
                                     const long address[3], const long PS[3]) {
    grg_get_double_grid_address(address_double, address, PS);
}

void gridsys_get_grid_address_from_index(long address[3], const long grid_index,
                                         const long D_diag[3]) {
    grg_get_grid_address_from_index(address, grid_index, D_diag);
}

long gridsys_get_double_grid_index(const long address_double[3],
                                   const long D_diag[3], const long PS[3]) {
    return grg_get_double_grid_index(address_double, D_diag, PS);
}

long gridsys_get_grid_index_from_address(const long address[3],
                                         const long D_diag[3]) {
    return grg_get_grid_index(address, D_diag);
}

long gridsys_rotate_grid_index(const long grid_index, const long rotation[3][3],
                               const long D_diag[3], const long PS[3]) {
    return grg_rotate_grid_index(grid_index, rotation, D_diag, PS);
}

long gridsys_get_reciprocal_point_group(long rec_rotations[48][3][3],
                                        const long (*rotations)[3][3],
                                        const long num_rot,
                                        const long is_time_reversal) {
    return grg_get_reciprocal_point_group(rec_rotations, rotations, num_rot,
                                          is_time_reversal, 1);
}

long gridsys_get_snf3x3(long D_diag[3], long P[3][3], long Q[3][3],
                        const long A[3][3]) {
    return grg_get_snf3x3(D_diag, P, Q, A);
}

long gridsys_transform_rotations(long (*transformed_rots)[3][3],
                                 const long (*rotations)[3][3],
                                 const long num_rot, const long D_diag[3],
                                 const long Q[3][3]) {
    long succeeded;
    succeeded = grg_transform_rotations(transformed_rots, rotations, num_rot,
                                        D_diag, Q);
    return succeeded;
}

void gridsys_get_ir_grid_map(long *ir_grid_map, const long (*rotations)[3][3],
                             const long num_rot, const long D_diag[3],
                             const long PS[3]) {
    grg_get_ir_grid_map(ir_grid_map, rotations, num_rot, D_diag, PS);
}

long gridsys_get_bz_grid_addresses(long (*bz_grid_addresses)[3], long *bz_map,
                                   long *bzg2grg, const long D_diag[3],
                                   const long Q[3][3], const long PS[3],
                                   const double rec_lattice[3][3],
                                   const long bz_grid_type) {
    BZGrid *bzgrid;
    long i, j, size;
    long inv_Mpr_int[3][3];
    double inv_Lr[3][3], inv_Mpr[3][3];
    double niggli_lattice[9];

    if ((bzgrid = (BZGrid *)malloc(sizeof(BZGrid))) == NULL) {
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
    if (!lagmat_inverse_matrix_d3(inv_Lr, (double(*)[3])niggli_lattice,
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
    lagmat_copy_matrix_d3(bzgrid->reclat, (double(*)[3])niggli_lattice);
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

long gridsys_rotate_bz_grid_index(const long bz_grid_index,
                                  const long rotation[3][3],
                                  const long (*bz_grid_addresses)[3],
                                  const long *bz_map, const long D_diag[3],
                                  const long PS[3], const long bz_grid_type) {
    ConstBZGrid *bzgrid;
    long i, rot_bz_gp;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
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

long gridsys_get_triplets_at_q(long *map_triplets, long *map_q,
                               const long grid_point, const long D_diag[3],
                               const long is_time_reversal, const long num_rot,
                               const long (*rec_rotations)[3][3],
                               const long swappable) {
    return tpl_get_triplets_reciprocal_mesh_at_q(
        map_triplets, map_q, grid_point, D_diag, is_time_reversal, num_rot,
        rec_rotations, swappable);
}

long gridsys_get_bz_triplets_at_q(long (*ir_triplets)[3],
                                  const long bz_grid_index,
                                  const long (*bz_grid_addresses)[3],
                                  const long *bz_map, const long *map_triplets,
                                  const long num_map_triplets,
                                  const long D_diag[3], const long Q[3][3],
                                  const long bz_grid_type) {
    ConstBZGrid *bzgrid;
    long i, j, num_ir;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
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
    long relative_grid_address[4][24][4][3]) {
    thm_get_all_relative_grid_address(relative_grid_address);
}

long gridsys_get_thm_relative_grid_address(
    long relative_grid_addresses[24][4][3], const double rec_lattice[3][3]) {
    return thm_get_relative_grid_address(relative_grid_addresses, rec_lattice);
}

/* relative_grid_addresses are given as P multipled with those from */
/* dataset, i.e., */
/*     np.dot(relative_grid_addresses, P.T) */
long gridsys_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long relative_grid_address[24][4][3],
    const long D_diag[3], const long (*triplets)[3], const long num_triplets,
    const long (*bz_grid_addresses)[3], const long *bz_map,
    const long bz_grid_type, const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets) {
    ConstBZGrid *bzgrid;
    long i;

    if ((bzgrid = (ConstBZGrid *)malloc(sizeof(ConstBZGrid))) == NULL) {
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
    const double *frequency_points, const long num_band0,
    const long (*triplets)[3], const long num_triplets,
    const double *frequencies, const long num_band, const long tp_type) {
    tpl_get_integration_weight_with_sigma(
        iw, iw_zero, sigma, sigma_cutoff, frequency_points, num_band0, triplets,
        num_triplets, frequencies, num_band, tp_type);
}
