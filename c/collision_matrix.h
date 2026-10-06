/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __collision_matrix_H__
#define __collision_matrix_H__

#include <stdint.h>

#include "phonoc_array.h"

void col_get_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q,
    const int64_t *rot_grid_points, const double *rotations_cartesian,
    const double *g, const int64_t num_ir_gp, const int64_t num_gp,
    const int64_t num_rot, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency);
void col_get_reducible_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q, const double *g,
    const int64_t num_gp, const double temperature_THz,
    const double unit_conversion_factor, const double cutoff_frequency);

#endif
