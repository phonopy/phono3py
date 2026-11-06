/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __reciprocal_to_normal_H__
#define __reciprocal_to_normal_H__

#include <stdint.h>

#include "lapack_wrapper.h"

void reciprocal_to_normal_squared(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *eigvecs0,
    const lapack_complex_double *eigvecs1,
    const lapack_complex_double *eigvecs2, const double *masses,
    const int64_t *band_indices, const int64_t num_band0,
    const int64_t num_band, const double cutoff_frequency,
    const int64_t openmp_per_triplets);

#endif
