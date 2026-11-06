/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __pp_collision_H__
#define __pp_collision_H__

#include <stdint.h>

#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "real_to_reciprocal.h"
#include "recgrid.h"

void ppc_get_pp_collision(
    double *imag_self_energy,
    const int64_t relative_grid_address[24][4][3], /* thm */
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const int64_t *triplet_weights, const RecgridConstBZGrid *bzgrid,
    const double *fc3, const int64_t is_compact_fc3,
    const AtomTriplets *atom_triplets, const double *masses,
    const Larray *band_indices, const Darray *temperatures_THz,
    const int64_t is_NU, const int64_t symmetrize_fc3_q,
    const double cutoff_frequency, const int64_t openmp_per_triplets);
void ppc_get_pp_collision_with_sigma(
    double *imag_self_energy, const double sigma, const double sigma_cutoff,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const int64_t (*triplets)[3], const int64_t num_triplets,
    const int64_t *triplet_weights, const RecgridConstBZGrid *bzgrid,
    const double *fc3, const int64_t is_compact_fc3,
    const AtomTriplets *atom_triplets, const double *masses,
    const Larray *band_indices, const Darray *temperatures_THz,
    const int64_t is_NU, const int64_t symmetrize_fc3_q,
    const double cutoff_frequency, const int64_t openmp_per_triplets);

#endif
