/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __isotope_H__
#define __isotope_H__

#include <stdint.h>

#include "lapack_wrapper.h"

void iso_get_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const int64_t num_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0, const double sigma,
    const double cutoff_frequency);
void iso_get_thm_isotope_scattering_strength(
    double *gamma, const int64_t grid_point, const int64_t *ir_grid_points,
    const double *weights, const double *mass_variances,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const int64_t num_grid_points, const int64_t *band_indices,
    const int64_t num_band, const int64_t num_band0,
    const double *integration_weights, const double cutoff_frequency);
#endif
