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

#ifndef __pp_collision_H__
#define __pp_collision_H__

#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "real_to_reciprocal.h"
#include "recgrid.h"

void ppc_get_pp_collision(
    double *imag_self_energy,
    const long relative_grid_address[24][4][3], /* thm */
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const RecgridConstBZGrid *bzgrid,
    const double *fc3, const long is_compact_fc3,
    const AtomTriplets *atom_triplets, const double *masses,
    const Larray *band_indices, const Darray *temperatures, const long is_NU,
    const long symmetrize_fc3_q, const double cutoff_frequency,
    const long openmp_per_triplets);
void ppc_get_pp_collision_with_sigma(
    double *imag_self_energy, const double sigma, const double sigma_cutoff,
    const double *frequencies, const lapack_complex_double *eigenvectors,
    const long (*triplets)[3], const long num_triplets,
    const long *triplet_weights, const RecgridConstBZGrid *bzgrid,
    const double *fc3, const long is_compact_fc3,
    const AtomTriplets *atom_triplets, const double *masses,
    const Larray *band_indices, const Darray *temperatures, const long is_NU,
    const long symmetrize_fc3_q, const double cutoff_frequency,
    const long openmp_per_triplets);

#endif
