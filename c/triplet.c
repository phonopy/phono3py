/* SPDX-License-Identifier: BSD-3-Clause */

#include "triplet.h"

#include <stdint.h>

#include "recgrid.h"
#include "triplet_grid.h"
#include "triplet_iw.h"

int64_t tpl_get_BZ_triplets_at_q(int64_t (*triplets)[3],
                                 const int64_t grid_point,
                                 const RecgridConstBZGrid *bzgrid,
                                 const int64_t *map_triplets) {
    return tpk_get_BZ_triplets_at_q(triplets, grid_point, bzgrid, map_triplets);
}

int64_t tpl_get_triplets_reciprocal_mesh_at_q(
    int64_t *map_triplets, int64_t *map_q, const int64_t grid_point,
    const int64_t mesh[3], const int64_t is_time_reversal,
    const int64_t num_rot, const int64_t (*rec_rotations)[3][3],
    const int64_t swappable) {
    int64_t num_ir;

    num_ir = tpk_get_ir_triplets_at_q(map_triplets, map_q, grid_point, mesh,
                                      is_time_reversal, rec_rotations, num_rot,
                                      swappable);
    return num_ir;
}

void tpl_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const int64_t num_band0, const int64_t relative_grid_address[24][4][3],
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const RecgridConstBZGrid *bzgrid, const double *frequencies1,
    const int64_t num_band1, const double *frequencies2,
    const int64_t num_band2, const int64_t tp_type,
    const int64_t openmp_per_triplets) {
    int64_t i, num_band_prod;
    int64_t tp_relative_grid_address[2][24][4][3];

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
    const double *frequency_points, const int64_t num_band0,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const double *frequencies, const int64_t num_band, const int64_t tp_type) {
    int64_t i, num_band_prod, const_adrs_shift;
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

int64_t tpl_is_N(const int64_t triplet[3],
                 const int64_t (*bz_grid_addresses)[3]) {
    int64_t i, j, sum_q, is_N;

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

void tpl_set_relative_grid_address(
    int64_t tp_relative_grid_address[2][24][4][3],
    const int64_t relative_grid_address[24][4][3], const int64_t tp_type) {
    int64_t i, j, k, l;
    int64_t signs[2];

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
