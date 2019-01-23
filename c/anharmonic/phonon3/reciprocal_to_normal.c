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

#include <stdlib.h>
#include <math.h>
#include <phonoc_utils.h>
#include <phonoc_const.h>
#include <phonoc_array.h>
#include <phonon3_h/reciprocal_to_normal.h>
#include <lapack_wrapper.h>

#ifdef MEASURE_R2N
#include <unistd.h>
#include <time.h>
#endif

static lapack_complex_double fc3_sum_in_reciprocal_to_normal
(const size_t bi0,
 const size_t bi1,
 const size_t bi2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const size_t num_atom);

static double get_fc3_sum
(const size_t j,
 const size_t k,
 const size_t bi,
 const double *freqs0,
 const double *freqs1,
 const double *freqs2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const size_t num_atom,
 const double cutoff_frequency);

void reciprocal_to_normal_squared
(double *fc3_normal_squared,
 PHPYCONST int (*g_pos)[4],
 const size_t num_g_pos,
 const lapack_complex_double *fc3_reciprocal,
 const double *freqs0,
 const double *freqs1,
 const double *freqs2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const double *masses,
 const int *band_indices,
 const size_t num_band0,
 const size_t num_band,
 const double cutoff_frequency,
 const int openmp_at_bands)
{
  size_t i, num_atom;

#ifdef MEASURE_R2N
  double loopTotalCPUTime, loopTotalWallTime;
  time_t loopStartWallTime;
  clock_t loopStartCPUTime;
#endif

  num_atom = num_band / 3;

#ifdef MEASURE_R2N
  loopStartWallTime = time(NULL);
  loopStartCPUTime = clock();
#endif

#pragma omp parallel for if (openmp_at_bands)
  for (i = 0; i < num_g_pos; i++) {
    if (freqs0[band_indices[g_pos[i][0]]] > cutoff_frequency) {
      fc3_normal_squared[g_pos[i][3]] = get_fc3_sum(g_pos[i][1],
                                                    g_pos[i][2],
                                                    band_indices[g_pos[i][0]],
                                                    freqs0,
                                                    freqs1,
                                                    freqs2,
                                                    eigvecs0,
                                                    eigvecs1,
                                                    eigvecs2,
                                                    fc3_reciprocal,
                                                    masses,
                                                    num_atom,
                                                    cutoff_frequency);
    }
  }

#ifdef MEASURE_R2N
      loopTotalCPUTime = (double)(clock() - loopStartCPUTime) / CLOCKS_PER_SEC;
      loopTotalWallTime = difftime(time(NULL), loopStartWallTime);
      printf("  %1.3fs (%1.3fs CPU)\n", loopTotalWallTime, loopTotalCPUTime);
#endif

}

static double get_fc3_sum
(const size_t j,
 const size_t k,
 const size_t bi,
 const double *freqs0,
 const double *freqs1,
 const double *freqs2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const size_t num_atom,
 const double cutoff_frequency)
{
  double fff, sum_real, sum_imag;
  lapack_complex_double fc3_sum;

  if (freqs1[j] > cutoff_frequency && freqs2[k] > cutoff_frequency) {
    fff = freqs0[bi] * freqs1[j] * freqs2[k];
    fc3_sum = fc3_sum_in_reciprocal_to_normal
      (bi, j, k,
       eigvecs0, eigvecs1, eigvecs2,
       fc3_reciprocal,
       masses,
       num_atom);
    sum_real = lapack_complex_double_real(fc3_sum);
    sum_imag = lapack_complex_double_imag(fc3_sum);
    return (sum_real * sum_real + sum_imag * sum_imag) / fff;
  } else {
    return 0;
  }
}

static lapack_complex_double fc3_sum_in_reciprocal_to_normal
(const size_t bi0,
 const size_t bi1,
 const size_t bi2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const size_t num_atom)
{
  size_t baseIndex, index_l, index_lm, i, j, k, l, m, n;
  double sum_real, sum_imag, mmm, mass_l, mass_lm;
  lapack_complex_double eig_prod, eig_prod1;

  sum_real = 0;
  sum_imag = 0;

  for (l = 0; l < num_atom; l++) {
    mass_l = masses[l];
    index_l = l * num_atom * num_atom * 27;

    for (m = 0; m < num_atom; m++) {
      mass_lm = mass_l * masses[m];
      index_lm = index_l + m * num_atom * 27;

      for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
          eig_prod1 = phonoc_complex_prod
            (eigvecs0[(l * 3 + i) * num_atom * 3 + bi0],
             eigvecs1[(m * 3 + j) * num_atom * 3 + bi1]);

          for (n = 0; n < num_atom; n++) {
            mmm = 1.0 / sqrt(mass_lm * masses[n]);
            baseIndex = index_lm + n * 27 + i * 9 + j * 3;

            for (k = 0; k < 3; k++) {
              eig_prod = phonoc_complex_prod
                (eig_prod1, eigvecs2[(n * 3 + k) * num_atom * 3 + bi2]);
              eig_prod = phonoc_complex_prod
                (eig_prod, fc3_reciprocal[baseIndex + k]);
              sum_real += lapack_complex_double_real(eig_prod) * mmm;
              sum_imag += lapack_complex_double_imag(eig_prod) * mmm;
            }
          }
        }
      }
    }
  }
  return lapack_make_complex_double(sum_real, sum_imag);
}
