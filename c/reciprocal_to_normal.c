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
#include "reciprocal_to_normal.h"
#include "lapack_wrapper.h"

#ifdef MEASURE_R2N
#include <unistd.h>
#include <time.h>
#endif

static double get_fc3_sum(const long bi0,
                          const long bi1,
                          const long bi2,
                          const double *freqs0,
                          const double *freqs1,
                          const double *freqs2,
                          const lapack_complex_double *e0,
                          const lapack_complex_double *e1,
                          const lapack_complex_double *e2,
                          const lapack_complex_double *fc3_reciprocal,
                          const long num_atom,
                          const long num_band,
                          const double cutoff_frequency);

static double fc3_sum_in_reciprocal_to_normal(const lapack_complex_double *e0,
                                              const lapack_complex_double *e1,
                                              const lapack_complex_double *e2,
                                              const lapack_complex_double *fc3_reciprocal,
                                              const long num_atom);

void reciprocal_to_normal_squared(double *fc3_normal_squared,
                                  const long (*g_pos)[4],
                                  const long num_g_pos,
                                  const lapack_complex_double *fc3_reciprocal,
                                  const double *freqs0,
                                  const double *freqs1,
                                  const double *freqs2,
                                  const lapack_complex_double *eigvecs0,
                                  const lapack_complex_double *eigvecs1,
                                  const lapack_complex_double *eigvecs2,
                                  const double *masses,
                                  const long *band_indices,
                                  const long num_band,
                                  const double cutoff_frequency,
                                  const long openmp_at_bands)
{
  long i, j, k, num_atom;
  double real, imag;
  double *inv_sqrt_masses;
  lapack_complex_double *e0, *e1, *e2;

  num_atom = num_band / 3;

  inv_sqrt_masses = (double *)malloc(sizeof(double) * num_atom);
  for (i = 0; i < num_atom; i++)
  {
    inv_sqrt_masses[i] = 1.0 / sqrt(masses[i]);
  }

  /* Transpose eigenvectors for the better data alignment in memory. */
  e0 = (lapack_complex_double *)
      malloc(sizeof(lapack_complex_double) * num_band * num_band);
  e1 = (lapack_complex_double *)
      malloc(sizeof(lapack_complex_double) * num_band * num_band);
  e2 = (lapack_complex_double *)
      malloc(sizeof(lapack_complex_double) * num_band * num_band);

  for (i = 0; i < num_band; i++)
  {
    for (j = 0; j < num_band; j++)
    {
      e0[j * num_band + i] = eigvecs0[i * num_band + j];
      e1[j * num_band + i] = eigvecs1[i * num_band + j];
      e2[j * num_band + i] = eigvecs2[i * num_band + j];
    }
  }

  for (i = 0; i < num_band; i++)
  {
    for (j = 0; j < num_atom; j++)
    {
      for (k = 0; k < 3; k++)
      {
        real = lapack_complex_double_real(e0[i * num_band + j * 3 + k]);
        imag = lapack_complex_double_imag(e0[i * num_band + j * 3 + k]);
        e0[i * num_band + j * 3 + k] = lapack_make_complex_double(real * inv_sqrt_masses[j], imag * inv_sqrt_masses[j]);
        real = lapack_complex_double_real(e1[i * num_band + j * 3 + k]);
        imag = lapack_complex_double_imag(e1[i * num_band + j * 3 + k]);
        e1[i * num_band + j * 3 + k] = lapack_make_complex_double(real * inv_sqrt_masses[j], imag * inv_sqrt_masses[j]);
        real = lapack_complex_double_real(e2[i * num_band + j * 3 + k]);
        imag = lapack_complex_double_imag(e2[i * num_band + j * 3 + k]);
        e2[i * num_band + j * 3 + k] = lapack_make_complex_double(real * inv_sqrt_masses[j], imag * inv_sqrt_masses[j]);
      }
    }
  }

#ifdef MEASURE_R2N
  double loopTotalCPUTime,
      loopTotalWallTime;
  time_t loopStartWallTime;
  clock_t loopStartCPUTime;
#endif

#ifdef MEASURE_R2N
  loopStartWallTime = time(NULL);
  loopStartCPUTime = clock();
#endif

#ifdef PHPYOPENMP
#pragma omp parallel for if (openmp_at_bands)
#endif
  for (i = 0; i < num_g_pos; i++)
  {
    if (freqs0[band_indices[g_pos[i][0]]] > cutoff_frequency)
    {
      fc3_normal_squared[g_pos[i][3]] = get_fc3_sum(band_indices[g_pos[i][0]],
                                                    g_pos[i][1],
                                                    g_pos[i][2],
                                                    freqs0,
                                                    freqs1,
                                                    freqs2,
                                                    e0,
                                                    e1,
                                                    e2,
                                                    fc3_reciprocal,
                                                    num_atom,
                                                    num_band,
                                                    cutoff_frequency);
    }
  }

#ifdef MEASURE_R2N
  loopTotalCPUTime = (double)(clock() - loopStartCPUTime) / CLOCKS_PER_SEC;
  loopTotalWallTime = difftime(time(NULL), loopStartWallTime);
  printf("  %1.3fs (%1.3fs CPU)\n", loopTotalWallTime, loopTotalCPUTime);
#endif

  free(inv_sqrt_masses);
  inv_sqrt_masses = NULL;
  free(e0);
  e0 = NULL;
  free(e1);
  e1 = NULL;
  free(e2);
  e2 = NULL;
}

static double get_fc3_sum(const long bi0,
                          const long bi1,
                          const long bi2,
                          const double *freqs0,
                          const double *freqs1,
                          const double *freqs2,
                          const lapack_complex_double *e0,
                          const lapack_complex_double *e1,
                          const lapack_complex_double *e2,
                          const lapack_complex_double *fc3_reciprocal,
                          const long num_atom,
                          const long num_band,
                          const double cutoff_frequency)
{
  double fff;

  if (freqs1[bi1] > cutoff_frequency && freqs2[bi2] > cutoff_frequency)
  {
    fff = freqs0[bi0] * freqs1[bi1] * freqs2[bi2];
    return fc3_sum_in_reciprocal_to_normal(e0 + bi0 * num_band,
                                           e1 + bi1 * num_band,
                                           e2 + bi2 * num_band,
                                           fc3_reciprocal,
                                           num_atom) /
           fff;
  }
  else
  {
    return 0;
  }
}

static double fc3_sum_in_reciprocal_to_normal(const lapack_complex_double *e0,
                                              const lapack_complex_double *e1,
                                              const lapack_complex_double *e2,
                                              const lapack_complex_double *fc3_reciprocal,
                                              const long num_atom)
{
  long index_l, index_lm, index_lmn, i, j, k, l, m, n;
  double sum_real, sum_imag;
  lapack_complex_double eig_prod;

  sum_real = 0;
  sum_imag = 0;

  for (l = 0; l < num_atom; l++)
  {
    index_l = l * num_atom * num_atom * 27;
    for (i = 0; i < 3; i++)
    {
      for (m = 0; m < num_atom; m++)
      {
        index_lm = index_l + m * num_atom * 27;
        for (j = 0; j < 3; j++)
        {
          eig_prod = phonoc_complex_prod(e0[l * 3 + i], e1[m * 3 + j]);
          for (n = 0; n < num_atom; n++)
          {
            index_lmn = index_lm + n * 27 + i * 9 + j * 3;
            for (k = 0; k < 3; k++)
            {
              eig_prod = phonoc_complex_prod(eig_prod, e2[n * 3 + k]);
              eig_prod = phonoc_complex_prod(eig_prod, fc3_reciprocal[index_lmn + k]);
              sum_real += lapack_complex_double_real(eig_prod);
              sum_imag += lapack_complex_double_imag(eig_prod);
            }
          }
        }
      }
    }
  }
  return (sum_real * sum_real + sum_imag * sum_imag);
}
