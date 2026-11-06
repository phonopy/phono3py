/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __real_to_reciprocal_H__
#define __real_to_reciprocal_H__

#include <stdint.h>

#include "lapack_wrapper.h"
#include "phonoc_array.h"

typedef struct {
    const double (*svecs)[3];
    int64_t multi_dims[2];
    const int64_t (*multiplicity)[2];
    const int64_t *p2s_map;
    const int64_t *s2p_map;
    int64_t make_r0_average;
    const char *all_shortest;
    const char *nonzero_indices;  // for compact fc3
} AtomTriplets;

void r2r_real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
                            const double q_vecs[3][3], const double *fc3,
                            const int64_t is_compact_fc3,
                            const AtomTriplets *atom_triplets,
                            const int64_t openmp_per_triplets);
#endif
