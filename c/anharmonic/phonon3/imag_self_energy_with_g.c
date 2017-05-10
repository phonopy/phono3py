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

#include <stdio.h>
#include <stdlib.h>
#include <phonoc_array.h>
#include <phonoc_utils.h>
#include <phonoc_const.h>
#include <phonon3_h/imag_self_energy_with_g.h>
#include <triplet_h/triplet.h>

static double
collect_detailed_imag_self_energy(double *imag_self_energy,
				  const int num_band,
				  const double *fc3_normal_squared,
				  const double *n1,
				  const double *n2,
				  const double *g1,
				  const double *g2_3,
				  const char *g_zero);
static double
collect_detailed_imag_self_energy_0K(double *imag_self_energy,
				     const int num_band,
				     const double *fc3_normal_squared,
				     const double *n1,
				     const double *n2,
				     const double *g,
				     const char *g_zero);
static void set_occupations(double *n1,
                            double *n2,
                            const int num_band,
                            const double temperature,
                            const int *triplets,
                            const double *frequencies,
                            const double cutoff_frequency);
static int set_g_pos(int (*g_pos)[4],
                     const int i_triplet,
                     const int num_band0,
                     const int num_band,
                     const int num_band_prod,
                     const char *g_zero);

void get_imag_self_energy_at_bands_with_g(double *imag_self_energy,
					  const Darray *fc3_normal_squared,
					  const double *frequencies,
					  const int *triplets,
					  const int *weights,
					  const double *g,
					  const char *g_zero,
					  const double temperature,
					  const double cutoff_frequency)
{
  int i, j, num_triplets, num_band0, num_band, num_band_prod, num_g_pos;
  int (*g_pos)[4];
  double *ise;

  g_pos = NULL;
  ise = NULL;

  num_triplets = fc3_normal_squared->dims[0];
  num_band0 = fc3_normal_squared->dims[1];
  num_band = fc3_normal_squared->dims[2];
  num_band_prod = num_band0 * num_band * num_band;
  ise = (double*)malloc(sizeof(double) * num_triplets * num_band0);

#pragma omp parallel for private(num_g_pos, j, g_pos)
  for (i = 0; i < num_triplets; i++) {
    g_pos = (int(*)[4])malloc(sizeof(int[4]) * num_band_prod);
    num_g_pos = set_g_pos(g_pos,
                          i,
                          num_band0,
                          num_band,
                          num_band_prod,
                          g_zero);

    imag_self_energy_at_triplet(ise + i * num_band0,
                                num_band0,
                                num_band,
                                fc3_normal_squared->data + i * num_band_prod,
                                frequencies,
                                triplets + i * 3,
                                weights[i],
                                g + i * num_band_prod,
                                g + (i + num_triplets) * num_band_prod,
                                g_pos,
                                num_g_pos,
                                &temperature,
                                1,
                                cutoff_frequency,
                                0);

    free(g_pos);
    g_pos = NULL;
  }

  for (i = 0; i < num_band0; i++) {
    imag_self_energy[i] = 0;
  }

  for (i = 0; i < num_triplets; i++) {
    for (j = 0; j < num_band0; j++) {
      imag_self_energy[j] += ise[i * num_band0 + j];
    }
  }

  free(ise);
  ise = NULL;
}

void get_detailed_imag_self_energy_at_bands_with_g
(double *detailed_imag_self_energy,
 double *imag_self_energy_N,
 double *imag_self_energy_U,
 const Darray *fc3_normal_squared,
 const double *frequencies,
 const int *triplets,
 const int *weights,
 const int *grid_address,
 const double *g,
 const char *g_zero,
 const double temperature,
 const double cutoff_frequency)
{
  double *ise;
  int i, j, num_triplets, num_band0, num_band, num_band_prod;
  int *is_N;
  double ise_tmp, N, U;

  ise = NULL;
  is_N = NULL;

  num_triplets = fc3_normal_squared->dims[0];
  num_band0 = fc3_normal_squared->dims[1];
  num_band = fc3_normal_squared->dims[2];
  num_band_prod = num_band0 * num_band * num_band;
  ise = (double*)malloc(sizeof(double) * num_triplets * num_band0);

  /* detailed_imag_self_energy has the same shape as fc3_normal_squared. */

#pragma omp parallel for
  for (i = 0; i < num_triplets; i++) {
    detailed_imag_self_energy_at_triplet
      (detailed_imag_self_energy + i * num_band_prod,
       ise + i * num_band0,
       num_band0,
       num_band,
       fc3_normal_squared->data + i * num_band_prod,
       frequencies,
       triplets + i * 3,
       g + i * num_band_prod,
       g + (i + num_triplets) * num_band_prod,
       g_zero + i * num_band_prod,
       &temperature,
       1,
       cutoff_frequency);
  }

  is_N = (int*)malloc(sizeof(int) * num_triplets);
  for (i = 0; i < num_triplets; i++) {
    is_N[i] = tpl_is_N(triplets + i * 3, grid_address);
  }

  for (i = 0; i < num_band0; i++) {
    N = 0;
    U = 0;
/* #pragma omp parallel for private(ise_tmp) reduction(+:N,U) */
    for (j = 0; j < num_triplets; j++) {
      ise_tmp = ise[j * num_band0 + i] * weights[j];
      if (is_N[j]) {
        N += ise_tmp;
      } else {
        U += ise_tmp;
      }
    }
    imag_self_energy_N[i] = N;
    imag_self_energy_U[i] = U;
  }

  free(is_N);
  is_N = NULL;
  free(ise);
  ise = NULL;
}

void imag_self_energy_at_triplet(double *imag_self_energy,
                                 const int num_band0,
                                 const int num_band,
                                 const double *fc3_normal_squared,
                                 const double *frequencies,
                                 const int *triplets,
                                 const int triplet_weight,
                                 const double *g1,
                                 const double *g2_3,
                                 PHPYCONST int (*g_pos)[4],
                                 const int num_g_pos,
                                 const double *temperatures,
                                 const int num_temps,
                                 const double cutoff_frequency,
                                 const int openmp_at_bands)
{
  int i, j;
  double *n1, *n2;

  n1 = (double*)malloc(sizeof(double) * num_temps * num_band);
  n2 = (double*)malloc(sizeof(double) * num_temps * num_band);
  for (i = 0; i < num_temps; i++) {
    set_occupations(n1 + i * num_band,
                    n2 + i * num_band,
                    num_band,
                    temperatures[i],
                    triplets,
                    frequencies,
                    cutoff_frequency);
  }

  for (i = 0; i < num_band0 * num_temps; i++) {
    imag_self_energy[i] = 0;
  }

#pragma omp parallel for private(j) if (openmp_at_bands)
  for (i = 0; i < num_g_pos; i++) {
    for (j = 0; j < num_temps; j++) {
      if (n1[j * num_band + g_pos[i][1]] < 0 ||
          n2[j * num_band + g_pos[i][2]] < 0) {
        ;
      } else {
        if (temperatures[j] > 0) {
          imag_self_energy[j * num_band0 + g_pos[i][0]] +=
            ((n1[j * num_band + g_pos[i][1]] +
              n2[j * num_band + g_pos[i][2]] + 1) * g1[g_pos[i][3]] +
             (n1[j * num_band + g_pos[i][1]] -
              n2[j * num_band + g_pos[i][2]]) * g2_3[g_pos[i][3]]) *
            fc3_normal_squared[g_pos[i][3]] * triplet_weight;
        } else {
          imag_self_energy[j * num_band0 + g_pos[i][0]] +=
            g1[g_pos[i][3]] * fc3_normal_squared[g_pos[i][3]] * triplet_weight;
        }
      }
    }
  }

  free(n1);
  n1 = NULL;
  free(n2);
  n2 = NULL;
}

void
detailed_imag_self_energy_at_triplet(double *detailed_imag_self_energy,
				     double *imag_self_energy,
				     const int num_band0,
				     const int num_band,
				     const double *fc3_normal_squared,
				     const double *frequencies,
				     const int *triplets,
				     const double *g1,
				     const double *g2_3,
				     const char *g_zero,
				     const double *temperatures,
                                     const int num_temps,
				     const double cutoff_frequency)
{
  int i, j, adrs_shift;
  double *n1, *n2;

  n1 = NULL;
  n2 = NULL;

  n1 = (double*)malloc(sizeof(double) * num_band);
  n2 = (double*)malloc(sizeof(double) * num_band);

  for (i = 0; i < num_temps; i++) {
    set_occupations(n1,
                    n2,
                    num_band,
                    temperatures[i],
                    triplets,
                    frequencies,
                    cutoff_frequency);
    
    for (j = 0; j < num_band0; j++) {
      adrs_shift = j * num_band * num_band;
      if (temperatures[i] > 0) {
        imag_self_energy[i * num_band0 + j] =
          collect_detailed_imag_self_energy
          (detailed_imag_self_energy + adrs_shift,
           num_band,
           fc3_normal_squared + adrs_shift,
           n1,
           n2,
           g1 + adrs_shift,
           g2_3 + adrs_shift,
           g_zero + adrs_shift);
      } else {
        imag_self_energy[i * num_band0 + j] =
          collect_detailed_imag_self_energy_0K
          (detailed_imag_self_energy + adrs_shift,
           num_band,
           fc3_normal_squared + adrs_shift,
           n1,
           n2,
           g1 + adrs_shift,
           g_zero + adrs_shift);
      }
    }
  }

  free(n1);
  n1 = NULL;
  free(n2);
  n2 = NULL;
}

static double
collect_detailed_imag_self_energy(double *imag_self_energy,
				  const int num_band,
				  const double *fc3_normal_squared,
				  const double *n1,
				  const double *n2,
				  const double *g1,
				  const double *g2_3,
				  const char *g_zero)
{
  int ij, i, j;
  double sum_g;

  sum_g = 0;
  for (ij = 0; ij < num_band * num_band; ij++) {
    imag_self_energy[ij] = 0;
    if (g_zero[ij]) {continue;}
    i = ij / num_band;
    j = ij % num_band;
    if (n1[i] < 0 || n2[j] < 0) {continue;}
    imag_self_energy[ij] = (((n1[i] + n2[j] + 1) * g1[ij] +
			     (n1[i] - n2[j]) * g2_3[ij]) *
			    fc3_normal_squared[ij]);
    sum_g += imag_self_energy[ij];
  }

  return sum_g;
}

static double
collect_detailed_imag_self_energy_0K(double *imag_self_energy,
				     const int num_band,
				     const double *fc3_normal_squared,
				     const double *n1,
				     const double *n2,
				     const double *g1,
				     const char *g_zero)
{
  int ij, i, j;
  double sum_g;

  sum_g = 0;
  for (ij = 0; ij < num_band * num_band; ij++) {
    imag_self_energy[ij] = 0;
    if (g_zero[ij]) {continue;}
    i = ij / num_band;
    j = ij % num_band;
    if (n1[i] < 0 || n2[j] < 0) {continue;}
    imag_self_energy[ij] = g1[ij] * fc3_normal_squared[ij];
    sum_g += imag_self_energy[ij];
  }

  return sum_g;
}

static void set_occupations(double *n1,
                            double *n2,
                            const int num_band,
                            const double temperature,
                            const int *triplets,
                            const double *frequencies,
                            const double cutoff_frequency)
{
  int j;
  double f1, f2;

  for (j = 0; j < num_band; j++) {
    f1 = frequencies[triplets[1] * num_band + j];
    f2 = frequencies[triplets[2] * num_band + j];
    if (f1 > cutoff_frequency) {
      n1[j] = bose_einstein(f1, temperature);
    } else {
      n1[j] = -1;
    }
    if (f2 > cutoff_frequency) {
      n2[j] = bose_einstein(f2, temperature);
    } else {
      n2[j] = -1;
    }
  }
}

static int set_g_pos(int (*g_pos)[4],
                     const int i_triplet,
                     const int num_band0,
                     const int num_band,
                     const int num_band_prod,
                     const char *g_zero)
{
  int num_g_pos, j, k, l, jkl;

  num_g_pos = 0;
  jkl = 0;
  for (j = 0; j < num_band0; j++) {
    for (k = 0; k < num_band; k++) {
      for (l = 0; l < num_band; l++) {
        if (!g_zero[jkl + i_triplet * num_band_prod]) {
          g_pos[num_g_pos][0] = j;
          g_pos[num_g_pos][1] = k;
          g_pos[num_g_pos][2] = l;
          g_pos[num_g_pos][3] = jkl;
          num_g_pos++;
        }
        jkl++;
      }
    }
  }
  return num_g_pos;
}
