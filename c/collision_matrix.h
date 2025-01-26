/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

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
    const int64_t num_rot, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency);
void col_get_reducible_collision_matrix(
    double *collision_matrix, const Darray *fc3_normal_squared,
    const double *frequencies, const int64_t (*triplets)[3],
    const int64_t *triplets_map, const int64_t *map_q, const double *g,
    const int64_t num_gp, const double temperature,
    const double unit_conversion_factor, const double cutoff_frequency);

#endif
