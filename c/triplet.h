/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* Some of these codes were originally parts of spglib, but only developed */
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
