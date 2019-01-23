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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <phonoc_array.h>
#include <phonoc_const.h>
#include <phonoc_utils.h>
#include <phonon3_h/real_to_reciprocal.h>
#include <lapack_wrapper.h>

static void
real_to_reciprocal_single_thread(lapack_complex_double *fc3_reciprocal,
                                 const double q[9],
                                 const double *fc3,
                                 const int is_compact_fc3,
                                 const double *shortest_vectors,
                                 const int svecs_dims[3],
                                 const int *multiplicity,
                                 const int *p2s_map,
                                 const int *s2p_map);
static void
real_to_reciprocal_openmp(lapack_complex_double *fc3_reciprocal,
                          const double q[9],
                          const double *fc3,
                          const int is_compact_fc3,
                          const double *shortest_vectors,
                          const int svecs_dims[3],
                          const int *multiplicity,
                          const int *p2s_map,
                          const int *s2p_map);
static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
                                        const double q[9],
                                        const double *fc3,
                                        const int is_compact_fc3,
                                        const double *shortest_vectors,
                                        const int svecs_dims[3],
                                        const int *multiplicity,
                                        const int *p2s,
                                        const int *s2p,
                                        const size_t pi0,
                                        const size_t pi1,
                                        const size_t pi2);
static lapack_complex_double get_phase_factor(const double q[],
                                              const int qi,
                                              const double *shortest_vectors,
                                              const int multi);
static lapack_complex_double
get_pre_phase_factor(const size_t i,
                     const double q[9],
                     const double *shortest_vectors,
                     const int svecs_dims[3],
                     const int *multiplicity,
                     const int *p2s_map);

/* fc3_reciprocal[num_patom, num_patom, num_patom, 3, 3, 3] */
void r2r_real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
                            const double q[9],
                            const double *fc3,
                            const int is_compact_fc3,
                            const double *shortest_vectors,
                            const int svecs_dims[3],
                            const int *multiplicity,
                            const int *p2s_map,
                            const int *s2p_map,
                            const int openmp_at_bands)
{
  if (openmp_at_bands) {
    real_to_reciprocal_openmp(fc3_reciprocal,
                              q,
                              fc3,
                              is_compact_fc3,
                              shortest_vectors,
                              svecs_dims,
                              multiplicity,
                              p2s_map,
                              s2p_map);
  } else {
    real_to_reciprocal_single_thread(fc3_reciprocal,
                                     q,
                                     fc3,
                                     is_compact_fc3,
                                     shortest_vectors,
                                     svecs_dims,
                                     multiplicity,
                                     p2s_map,
                                     s2p_map);
  }
}


static void
real_to_reciprocal_single_thread(lapack_complex_double *fc3_reciprocal,
                                 const double q[9],
                                 const double *fc3,
                                 const int is_compact_fc3,
                                 const double *shortest_vectors,
                                 const int svecs_dims[3],
                                 const int *multiplicity,
                                 const int *p2s_map,
                                 const int *s2p_map)
{
  size_t i, j, k;
  size_t num_patom, adrs_shift;
  lapack_complex_double pre_phase_factor;

  num_patom = svecs_dims[1];

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (k = 0; k < num_patom; k++) {
        real_to_reciprocal_elements(fc3_reciprocal +
                                    i * 27 * num_patom * num_patom +
                                    j * 27 * num_patom +
                                    k * 27,
                                    q,
                                    fc3,
                                    is_compact_fc3,
                                    shortest_vectors,
                                    svecs_dims,
                                    multiplicity,
                                    p2s_map,
                                    s2p_map,
                                    i, j, k);

      }
    }
    pre_phase_factor = get_pre_phase_factor(
      i, q, shortest_vectors, svecs_dims, multiplicity, p2s_map);
    adrs_shift = i * num_patom * num_patom * 27;
    for (j = 0; j < num_patom * num_patom * 27; j++) {
      fc3_reciprocal[adrs_shift + j] =
        phonoc_complex_prod(fc3_reciprocal[adrs_shift + j], pre_phase_factor);
    }
  }
}

static void
real_to_reciprocal_openmp(lapack_complex_double *fc3_reciprocal,
                          const double q[9],
                          const double *fc3,
                          const int is_compact_fc3,
                          const double *shortest_vectors,
                          const int svecs_dims[3],
                          const int *multiplicity,
                          const int *p2s_map,
                          const int *s2p_map)
{
  size_t i, j, k, jk;
  size_t num_patom, adrs_shift;
  lapack_complex_double pre_phase_factor;

  num_patom = svecs_dims[1];

  for (i = 0; i < num_patom; i++) {
#pragma omp parallel for private(j, k)
    for (jk = 0; jk < num_patom * num_patom; jk++) {
      j = jk / num_patom;
      k = jk % num_patom;
      real_to_reciprocal_elements(fc3_reciprocal +
                                  i * 27 * num_patom * num_patom +
                                  j * 27 * num_patom +
                                  k * 27,
                                  q,
                                  fc3,
                                  is_compact_fc3,
                                  shortest_vectors,
                                  svecs_dims,
                                  multiplicity,
                                  p2s_map,
                                  s2p_map,
                                  i, j, k);

    }
    pre_phase_factor = get_pre_phase_factor(
      i, q, shortest_vectors, svecs_dims, multiplicity, p2s_map);
    adrs_shift = i * num_patom * num_patom * 27;
#pragma omp parallel for
    for (j = 0; j < num_patom * num_patom * 27; j++) {
      fc3_reciprocal[adrs_shift + j] =
        phonoc_complex_prod(fc3_reciprocal[adrs_shift + j], pre_phase_factor);
    }
  }
}

static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
                                        const double q[9],
                                        const double *fc3,
                                        const int is_compact_fc3,
                                        const double *shortest_vectors,
                                        const int svecs_dims[3],
                                        const int *multiplicity,
                                        const int *p2s,
                                        const int *s2p,
                                        const size_t pi0,
                                        const size_t pi1,
                                        const size_t pi2)
{
  size_t i, j, k, l;
  size_t num_satom, adrs_shift, adrs_vec1, adrs_vec2 ;
  lapack_complex_double phase_factor, phase_factor1, phase_factor2;
  double fc3_rec_real[27], fc3_rec_imag[27];

  for (i = 0; i < 27; i++) {
    fc3_rec_real[i] = 0;
    fc3_rec_imag[i] = 0;
  }

  num_satom = svecs_dims[0];

  if (is_compact_fc3) {
    i = pi0;
  } else {
    i = p2s[pi0];
  }

  for (j = 0; j < num_satom; j++) {
    if (s2p[j] != p2s[pi1]) {
      continue;
    }

    adrs_vec1 = j * svecs_dims[1] + pi0;
    phase_factor1 = get_phase_factor(q,
                                     1,
                                     shortest_vectors +
                                     adrs_vec1 * svecs_dims[2] * 3,
                                     multiplicity[adrs_vec1]);
    for (k = 0; k < num_satom; k++) {
      if (s2p[k] != p2s[pi2]) {
        continue;
      }
      adrs_vec2 = k * svecs_dims[1] + pi0;
      phase_factor2 = get_phase_factor(q,
                                       2,
                                       shortest_vectors +
                                       adrs_vec2 * svecs_dims[2] * 3,
                                       multiplicity[adrs_vec2]);
      adrs_shift = i * 27 * num_satom * num_satom + j * 27 * num_satom + k * 27;
      phase_factor = phonoc_complex_prod(phase_factor1, phase_factor2);
      for (l = 0; l < 27; l++) {
        fc3_rec_real[l] +=
          lapack_complex_double_real(phase_factor) * fc3[adrs_shift + l];
        fc3_rec_imag[l] +=
          lapack_complex_double_imag(phase_factor) * fc3[adrs_shift + l];
      }
    }
  }

  for (i = 0; i < 27; i++) {
    fc3_rec_elem[i] =
      lapack_make_complex_double(fc3_rec_real[i], fc3_rec_imag[i]);
  }
}

static lapack_complex_double
get_pre_phase_factor(const size_t i,
                     const double q[9],
                     const double *shortest_vectors,
                     const int svecs_dims[3],
                     const int *multiplicity,
                     const int *p2s_map)
{
  int j;
  double pre_phase;
  lapack_complex_double pre_phase_factor;

  pre_phase = 0;
  for (j = 0; j < 3; j++) {
    pre_phase += shortest_vectors[
      p2s_map[i] * svecs_dims[1] * svecs_dims[2] * 3 + j] *
      (q[j] + q[3 + j] + q[6 + j]);
  }

  assert(multiplicity[p2s_map[i] * svecs_dims[1]] == 1);

  pre_phase *= M_2PI;
  pre_phase_factor = lapack_make_complex_double(cos(pre_phase),
                                                sin(pre_phase));
  return pre_phase_factor;
}

static lapack_complex_double get_phase_factor(const double q[],
                                              const int qi,
                                              const double *shortest_vectors,
                                              const int multi)
{
  int i, j;
  double sum_real, sum_imag, phase;

  sum_real = 0;
  sum_imag = 0;
  for (i = 0; i < multi; i++) {
    phase = 0;
    for (j = 0; j < 3; j++) {
      phase += q[qi * 3 + j] * shortest_vectors[i * 3 + j];
    }
    sum_real += cos(M_2PI * phase);
    sum_imag += sin(M_2PI * phase);
  }
  sum_real /= multi;
  sum_imag /= multi;

  return lapack_make_complex_double(sum_real, sum_imag);
}
