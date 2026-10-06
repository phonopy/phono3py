/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __triplet_H__
#define __triplet_H__

#include <stddef.h>
#include <stdint.h>

#include "recgrid.h"

/* Irreducible triplets of k-points are searched under conservation of */
/* :math:``\mathbf{k}_1 + \mathbf{k}_2 + \mathbf{k}_3 = \mathbf{G}``. */
/* Memory spaces of grid_address[prod(mesh)][3], map_triplets[prod(mesh)] */
/* and map_q[prod(mesh)] are required. rotations are point-group- */
/* operations in real space for which duplicate operations are allowed */
/* in the input. */
int64_t tpl_get_triplets_reciprocal_mesh_at_q(
    int64_t *map_triplets, int64_t *map_q, const int64_t grid_point,
    const int64_t mesh[3], const int64_t is_time_reversal,
    const int64_t num_rot, const int64_t (*rec_rotations)[3][3],
    const int64_t swappable);
/* Irreducible grid-point-triplets in BZ are stored. */
/* triplets are recovered from grid_point and triplet_weights. */
/* BZ boundary is considered in this recovery. Therefore grid addresses */
/* are given not by grid_address, but by bz_grid_address. */
/* triplets[num_ir_triplets][3] = number of non-zero triplets weights*/
/* Number of ir-triplets is returned. */
int64_t tpl_get_BZ_triplets_at_q(int64_t (*triplets)[3],
                                 const int64_t grid_point,
                                 const RecgridConstBZGrid *bzgrid,
                                 const int64_t *map_triplets);
void tpl_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const int64_t num_band0, const int64_t relative_grid_address[24][4][3],
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const RecgridConstBZGrid *bzgrid, const double *frequencies1,
    const int64_t num_band1, const double *frequencies2,
    const int64_t num_band2, const int64_t tp_type,
    const int64_t openmp_per_triplets);
void tpl_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const int64_t num_band0,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const double *frequencies, const int64_t num_band, const int64_t tp_type);

int64_t tpl_is_N(const int64_t triplet[3],
                 const int64_t (*bz_grid_addresses)[3]);
void tpl_set_relative_grid_address(
    int64_t tp_relative_grid_address[2][24][4][3],
    const int64_t relative_grid_address[24][4][3], const int64_t tp_type);

#endif
