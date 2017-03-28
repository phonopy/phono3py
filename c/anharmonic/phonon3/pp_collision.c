/* Copyright (C) 2017 Atsushi Togo */
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

#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <phonoc_array.h>
#include <phonoc_const.h>
#include <phonoc_utils.h>
#include <phonon3_h/interaction.h>
#include <phonon3_h/real_to_reciprocal.h>
#include <phonon3_h/reciprocal_to_normal.h>

void get_pp_collision_with_g(double *imag_self_energy,
                             const double *g,
                             const char *g_zero,
                             const double *frequencies,
                             const lapack_complex_double *eigenvectors,
                             const Iarray *triplets,
                             const int *weights,
                             const int *grid_address,
                             const int *mesh,
                             const double *fc3,
                             const Darray *shortest_vectors,
                             const int *multiplicity,
                             const double *masses,
                             const int *p2s_map,
                             const int *s2p_map,
                             const Iarray *band_indices,
                             const double temperature,
                             const double cutoff_frequency)
{
  int i, num_band, num_band0, num_band_prod;

  num_band0 = band_indices->dims[0];
  num_band = shortest_vectors->dims[1] * 3;
  num_band_prod = num_band0 * num_band * num_band;

  if (triplets->dims[0] > num_band * num_band) {
#pragma omp parallel for schedule(guided)
    for (i = 0; i < triplets->dims[0]; i++) {
      get_interaction_at_triplet(
        fc3_normal_squared->data + i * num_band_prod,
        num_band0,
        num_band,
        g_zero + i * num_band_prod,
        frequencies,
        eigenvectors,
        triplets->data + i * 3,
        grid_address,
        mesh,
        fc3,
        shortest_vectors,
        multiplicity,
        masses,
        p2s_map,
        s2p_map,
        band_indices->data,
        symmetrize_fc3_q,
        cutoff_frequency,
        i,
        triplets->dims[0],
        0);
    }
  }
}
