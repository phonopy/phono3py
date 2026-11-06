/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __imag_self_energy_with_g_H__
#define __imag_self_energy_with_g_H__

#include <stddef.h>
#include <stdint.h>

#include "phonoc_array.h"

void ise_get_imag_self_energy_with_g(
    double *imag_self_energy, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const double *g, const char *g_zero,
    const double temperature_THz, const double cutoff_frequency,
    const int64_t num_frequency_points, const int64_t frequency_point_index);
void ise_get_detailed_imag_self_energy_with_g(
    double *detailed_imag_self_energy, double *imag_self_energy_N,
    double *imag_self_energy_U, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplet_weights, const int64_t (*bz_grid_addresses)[3],
    const double *g, const char *g_zero, const double temperature_THz,
    const double cutoff_frequency);
void ise_imag_self_energy_at_triplet(
    double *imag_self_energy, const int64_t num_band0, const int64_t num_band,
    const double *fc3_normal_squared, const double *frequencies,
    const int64_t triplet[3], const int64_t triplet_weight, const double *g1,
    const double *g2_3, const int64_t (*g_pos)[4], const int64_t num_g_pos,
    const double *temperatures_THz, const int64_t num_temps,
    const double cutoff_frequency, const int64_t openmp_possible,
    const int64_t at_a_frequency_point);
int64_t ise_set_g_pos(int64_t (*g_pos)[4], const int64_t num_band0,
                      const int64_t num_band, const char *g_zero);

#endif
