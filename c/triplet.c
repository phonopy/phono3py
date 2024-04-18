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

#include "triplet.h"

#include "bzgrid.h"
#include "triplet_grid.h"
#include "triplet_iw.h"

long tpl_get_BZ_triplets_at_q(long (*triplets)[3], const long grid_point,
                              const ConstBZGrid *bzgrid,
                              const long *map_triplets) {
    return tpk_get_BZ_triplets_at_q(triplets, grid_point, bzgrid, map_triplets);
}

long tpl_get_triplets_reciprocal_mesh_at_q(
    long *map_triplets, long *map_q, const long grid_point, const long mesh[3],
    const long is_time_reversal, const long num_rot,
    const long (*rec_rotations)[3][3], const long swappable) {
    long num_ir;

    num_ir = tpk_get_ir_triplets_at_q(map_triplets, map_q, grid_point, mesh,
                                      is_time_reversal, rec_rotations, num_rot,
                                      swappable);
    return num_ir;
}

void tpl_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long relative_grid_address[24][4][3],
    const long (*triplets)[3], const long num_triplets,
    const ConstBZGrid *bzgrid, const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets) {
    long i, num_band_prod;
    long tp_relative_grid_address[2][24][4][3];

    tpl_set_relative_grid_address(tp_relative_grid_address,
                                  relative_grid_address, tp_type);
    num_band_prod = num_band0 * num_band1 * num_band2;

#ifdef _OPENMP
#pragma omp parallel for schedule(guided) if (openmp_per_triplets)
#endif
    for (i = 0; i < num_triplets; i++) {
        tpi_get_integration_weight(
            iw + i * num_band_prod, iw_zero + i * num_band_prod,
            frequency_points, /* f0 */
            num_band0, tp_relative_grid_address, triplets[i], num_triplets,
            bzgrid, frequencies1,    /* f1 */
            num_band1, frequencies2, /* f2 */
            num_band2, tp_type, openmp_per_triplets);
    }
}

void tpl_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const long num_band0,
    const long (*triplets)[3], const long num_triplets,
    const double *frequencies, const long num_band, const long tp_type) {
    long i, num_band_prod, const_adrs_shift;
    double cutoff;

    cutoff = sigma * sigma_cutoff;
    num_band_prod = num_band0 * num_band * num_band;
    const_adrs_shift = num_triplets * num_band0 * num_band * num_band;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_triplets; i++) {
        tpi_get_integration_weight_with_sigma(
            iw + i * num_band_prod, iw_zero + i * num_band_prod, sigma, cutoff,
            frequency_points, num_band0, triplets[i], const_adrs_shift,
            frequencies, num_band, tp_type, 0);
    }
}

long tpl_is_N(const long triplet[3], const long (*bz_grid_addresses)[3]) {
    long i, j, sum_q, is_N;

    is_N = 1;
    for (i = 0; i < 3; i++) {
        sum_q = 0;
        for (j = 0; j < 3; j++) { /* 1st, 2nd, 3rd triplet */
            sum_q += bz_grid_addresses[triplet[j]][i];
        }
        if (sum_q) {
            is_N = 0;
            break;
        }
    }
    return is_N;
}

void tpl_set_relative_grid_address(long tp_relative_grid_address[2][24][4][3],
                                   const long relative_grid_address[24][4][3],
                                   const long tp_type) {
    long i, j, k, l;
    long signs[2];

    signs[0] = 1;
    signs[1] = 1;
    if ((tp_type == 2) || (tp_type == 3)) {
        /* q1+q2+q3=G */
        /* To set q2+1, q3-1 is needed to keep G */
        signs[1] = -1;
    }
    /* tp_type == 4, q+k_i-k_f=G */

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 24; j++) {
            for (k = 0; k < 4; k++) {
                for (l = 0; l < 3; l++) {
                    tp_relative_grid_address[i][j][k][l] =
                        relative_grid_address[j][k][l] * signs[i];
                }
            }
        }
    }
}
