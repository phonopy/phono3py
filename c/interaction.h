/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __interaction_H__
#define __interaction_H__

#include <stdint.h>

#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "real_to_reciprocal.h"
#include "recgrid.h"

void itr_get_interaction(
    Darray *fc3_normal_squared, const char *g_zero, const Darray *frequencies,
    const lapack_complex_double *eigenvectors, const int64_t (*triplets)[3],
    const int64_t num_triplets, const RecgridConstBZGrid *bzgrid,
    const double *fc3, const int64_t is_compact_fc3,
    const AtomTriplets *atom_triplets, const double *masses,
    const int64_t *band_indices, const int64_t symmetrize_fc3_q,
    const double cutoff_frequency, const int64_t openmp_per_triplets);
void itr_get_interaction_at_triplet(
    double *fc3_normal_squared, const int64_t num_band0, const int64_t num_band,
    const int64_t (*g_pos)[4], const int64_t num_g_pos,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const int64_t triplet[3], const RecgridConstBZGrid *bzgrid,
    const double *fc3, const int64_t is_compact_fc3,
    const AtomTriplets *atom_triplets, const double *masses,
    const int64_t *band_indices, const int64_t symmetrize_fc3_q,
    const double cutoff_frequency,
    const int64_t triplet_index, /* only for print */
    const int64_t num_triplets,  /* only for print */
    const int64_t openmp_per_triplets);

#endif
