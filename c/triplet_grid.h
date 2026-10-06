/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __triplet_grid_H__
#define __triplet_grid_H__

#include <stdint.h>

#include "bzgrid.h"
#include "lagrid.h"

int64_t tpk_get_ir_triplets_at_q(int64_t *map_triplets, int64_t *map_q,
                                 const int64_t grid_point,
                                 const int64_t D_diag[3],
                                 const int64_t is_time_reversal,
                                 const int64_t (*rec_rotations_in)[3][3],
                                 const int64_t num_rot,
                                 const int64_t swappable);
int64_t tpk_get_BZ_triplets_at_q(int64_t (*triplets)[3],
                                 const int64_t grid_point,
                                 const RecgridConstBZGrid *bzgrid,
                                 const int64_t *map_triplets);

#endif
