/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __triplet_iw_H__
#define __triplet_iw_H__

#include <stdint.h>

#include "bzgrid.h"

void tpi_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const int64_t num_band0,
    const int64_t tp_relative_grid_address[2][24][4][3],
    const int64_t triplets[3], const int64_t num_triplets,
    const RecgridConstBZGrid *bzgrid, const double *frequencies1,
    const int64_t num_band1, const double *frequencies2,
    const int64_t num_band2, const int64_t tp_type,
    const int64_t openmp_per_triplets);
void tpi_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double cutoff,
    const double *frequency_points, const int64_t num_band0,
    const int64_t triplet[3], const int64_t const_adrs_shift,
    const double *frequencies, const int64_t num_band, const int64_t tp_type,
    const int64_t openmp_per_triplets);
void tpi_get_neighboring_grid_points(int64_t *neighboring_grid_points,
                                     const int64_t grid_point,
                                     const int64_t (*relative_grid_address)[3],
                                     const int64_t num_relative_grid_address,
                                     const RecgridConstBZGrid *bzgrid);

#endif
