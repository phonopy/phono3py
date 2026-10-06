/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __real_self_energy_H__
#define __real_self_energy_H__

#include <stdint.h>

#include "phonoc_array.h"

#endif

void rse_get_real_self_energy_at_bands(
    double *real_self_energy, const Darray *fc3_normal_squared,
    const int64_t *band_indices, const double *frequencies,
    const int64_t (*triplets)[3], const int64_t *triplet_weights,
    const double epsilon, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency);
void rse_get_real_self_energy_at_frequency_point(
    double *real_self_energy, const double frequency_point,
    const Darray *fc3_normal_squared, const int64_t *band_indices,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const double epsilon,
    const double temperature_THz, const double unit_conversion_factor,
    const double cutoff_frequency);
