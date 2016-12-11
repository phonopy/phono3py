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
#include <phonon3_h/imag_self_energy_with_g.h>

static void imag_self_energy_at_bands(double *imag_self_energy,
				      const Darray *fc3_normal_squared,
				      const double *frequencies,
				      const int *triplets,
				      const double *g,
				      const char *g_zero,
				      const double temperature,
				      const double cutoff_frequency);
static double sum_imag_self_energy_at_band(const int num_band,
					   const double *fc3_normal_squared,
					   const double *n1,
					   const double *n2,
					   const double *g1,
					   const double *g2_3,
					   const char *g_zero);
static double sum_imag_self_energy_at_band_0K(const int num_band,
					      const double *fc3_normal_squared,
					      const double *n1,
					      const double *n2,
					      const double *g,
					      const char *g_zero);
static void
detailed_imag_self_energy_at_bands(double *detailed_imag_self_energy,
				   double *imag_self_energy,
				   const Darray *fc3_normal_squared,
				   const double *frequencies,
				   const int *triplets,
				   const double *g,
				   const char *g_zero,
				   const double temperature,
				   const double unit_conversion_factor,
				   const double cutoff_frequency);
static double
collect_detailed_imag_self_energy(double *imag_self_energy,
				  const int num_band,
				  const double *fc3_normal_squared,
				  const double *n1,
				  const double *n2,
				  const double *g1,
				  const double *g2_3,
				  const char *g_zero,
				  const double unit_conversion_factor);
static double
collect_detailed_imag_self_energy_0K(double *imag_self_energy,
				     const int num_band,
				     const double *fc3_normal_squared,
				     const double *n1,
				     const double *n2,
				     const double *g,
				     const char *g_zero,
				     const double unit_conversion_factor);
static void sum_imag_self_energy_along_triplets
(double *imag_self_energy,
 const double *ise,
 const int *weights,
 const int num_band0,
 const int num_triplets,
 const double unit_conversion_factor);
static void sum_imag_self_energy_N_and_U_along_triplets
(double *imag_self_energy_N,
 double *imag_self_energy_U,
 const double *ise,
 const int *triplets,
 const int *weights,
 const int *grid_address,
 const int num_band0,
 const int num_triplets);
static void set_tmp_values(double *n1,
			   double *n2,
			   const int i,
			   const int num_band,
			   const double temperature,
			   const int *triplets,
			   const double *frequencies,
			   const double cutoff_frequency);

void get_imag_self_energy_at_bands_with_g(double *imag_self_energy,
					  const Darray *fc3_normal_squared,
					  const double *frequencies,
					  const int *triplets,
					  const int *weights,
					  const double *g,
					  const char *g_zero,
					  const double temperature,
					  const double unit_conversion_factor,
					  const double cutoff_frequency)
{
  int num_triplets, num_band0;
  double *ise;

  num_triplets = fc3_normal_squared->dims[0];
  num_band0 = fc3_normal_squared->dims[1];
  ise = (double*)malloc(sizeof(double) * num_triplets * num_band0);

  imag_self_energy_at_bands(ise,
			    fc3_normal_squared,
			    frequencies,
			    triplets,
			    g,
			    g_zero,
			    temperature,
			    cutoff_frequency);
  
  sum_imag_self_energy_along_triplets(imag_self_energy,
				      ise,
				      weights,
				      num_band0,
				      num_triplets,
				      unit_conversion_factor);

  free(ise);
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
 const double unit_conversion_factor,
 const double cutoff_frequency)
{
  double *ise;
  int num_triplets;
  int num_band0;

  num_triplets = fc3_normal_squared->dims[0];
  num_band0 = fc3_normal_squared->dims[1];
  ise = (double*)malloc(sizeof(double) * num_triplets * num_band0);

  detailed_imag_self_energy_at_bands(detailed_imag_self_energy,
				     ise,
				     fc3_normal_squared,
				     frequencies,
				     triplets,
				     g,
				     g_zero,
				     temperature,
				     unit_conversion_factor,
				     cutoff_frequency);

  sum_imag_self_energy_N_and_U_along_triplets
    (imag_self_energy_N,
     imag_self_energy_U,
     ise,
     triplets,
     weights,
     grid_address,
     num_band0,
     num_triplets);

  free(ise);
}

static void imag_self_energy_at_bands(double *imag_self_energy,
				      const Darray *fc3_normal_squared,
				      const double *frequencies,
				      const int *triplets,
				      const double *g,
				      const char *g_zero,
				      const double temperature,
				      const double cutoff_frequency)
{
  int i, j, num_triplets, num_band0, num_band, adrs_shift;
  double *n1, *n2;

  num_triplets = fc3_normal_squared->dims[0];
  num_band0 = fc3_normal_squared->dims[1];
  num_band = fc3_normal_squared->dims[2];

#pragma omp parallel for private(j, n1, n2, adrs_shift)
  for (i = 0; i < num_triplets; i++) {
    n1 = (double*)malloc(sizeof(double) * num_band);
    n2 = (double*)malloc(sizeof(double) * num_band);
    set_tmp_values(n1,
    		   n2,
    		   i,
    		   num_band,
    		   temperature,
    		   triplets,
    		   frequencies,
    		   cutoff_frequency);

    for (j = 0; j < num_band0; j++) {
      adrs_shift = i * num_band0 * num_band * num_band + j * num_band * num_band;
      if (temperature > 0) {
	imag_self_energy[i * num_band0 + j] =
	  sum_imag_self_energy_at_band
	  (num_band,
	   fc3_normal_squared->data + adrs_shift,
	   n1,
	   n2,
	   g + adrs_shift,
	   g + (i + num_triplets) * num_band0 * num_band * num_band +
	   j * num_band * num_band,
	   g_zero + adrs_shift);
      } else {
	imag_self_energy[i * num_band0 + j] =
	  sum_imag_self_energy_at_band_0K
	  (num_band,
	   fc3_normal_squared->data + adrs_shift,
	   n1,
	   n2,
	   g + adrs_shift,
	   g_zero + adrs_shift);
      }
    }
    free(n1);
    free(n2);
  }
}

static double sum_imag_self_energy_at_band(const int num_band,
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
    if (g_zero[ij]) {continue;}
    i = ij / num_band;
    j = ij % num_band;
    if (n1[i] < 0 || n2[j] < 0) {continue;}
    sum_g += ((n1[i] + n2[j] + 1) * g1[ij] +
	      (n1[i] - n2[j]) * g2_3[ij]) * fc3_normal_squared[ij];
  }
  return sum_g;
}

static double sum_imag_self_energy_at_band_0K(const int num_band,
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
    if (g_zero[ij]) {continue;}
    i = ij / num_band;
    j = ij % num_band;
    if (n1[i] < 0 || n2[j] < 0) {continue;}
    sum_g += g1[ij] * fc3_normal_squared[ij];
  }
  return sum_g;
}

static void
detailed_imag_self_energy_at_bands(double *detailed_imag_self_energy,
				   double *imag_self_energy,
				   const Darray *fc3_normal_squared,
				   const double *frequencies,
				   const int *triplets,
				   const double *g,
				   const char *g_zero,
				   const double temperature,
				   const double unit_conversion_factor,
				   const double cutoff_frequency)
{
  int i, j, num_triplets, num_band0, num_band, adrs_shift;
  double *n1, *n2;

  num_triplets = fc3_normal_squared->dims[0];
  num_band0 = fc3_normal_squared->dims[1];
  num_band = fc3_normal_squared->dims[2];

#pragma omp parallel for private(j, n1, n2, adrs_shift)
  for (i = 0; i < num_triplets; i++) {
    n1 = (double*)malloc(sizeof(double) * num_band);
    n2 = (double*)malloc(sizeof(double) * num_band);
    set_tmp_values(n1,
    		   n2,
    		   i,
    		   num_band,
    		   temperature,
    		   triplets,
    		   frequencies,
    		   cutoff_frequency);
    
    for (j = 0; j < num_band0; j++) {
      adrs_shift = i * num_band0 * num_band * num_band + j * num_band * num_band;
      if (temperature > 0) {
	imag_self_energy[i * num_band0 + j] =
	  collect_detailed_imag_self_energy
	  (detailed_imag_self_energy + adrs_shift,
	   num_band,
	   fc3_normal_squared->data + adrs_shift,
	   n1,
	   n2,
	   g + adrs_shift,
	   g + num_triplets * num_band0 * num_band * num_band + adrs_shift,
	   g_zero + adrs_shift,
	   unit_conversion_factor);
      } else {
	imag_self_energy[i * num_band0 + j] =
	  collect_detailed_imag_self_energy_0K
	  (detailed_imag_self_energy + adrs_shift,
	   num_band,
	   fc3_normal_squared->data + adrs_shift,
	   n1,
	   n2,
	   g + adrs_shift,
	   g_zero + adrs_shift,
	   unit_conversion_factor);
      }
    }
    free(n1);
    free(n2);
  }
}

static double
collect_detailed_imag_self_energy(double *imag_self_energy,
				  const int num_band,
				  const double *fc3_normal_squared,
				  const double *n1,
				  const double *n2,
				  const double *g1,
				  const double *g2_3,
				  const char *g_zero,
				  const double unit_conversion_factor)
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
			    fc3_normal_squared[ij] * unit_conversion_factor);
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
				     const char *g_zero,
				     const double unit_conversion_factor)
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
    imag_self_energy[ij] = (g1[ij] * fc3_normal_squared[ij] *
			    unit_conversion_factor);
    sum_g += imag_self_energy[ij];
  }

  return sum_g;
}

static void sum_imag_self_energy_along_triplets
(double *imag_self_energy,
 const double *ise,
 const int *weights,
 const int num_band0,
 const int num_triplets,
 const double unit_conversion_factor)
{
  int i, j;
  double sum_g;

  for (i = 0; i < num_band0; i++) {
    sum_g = 0;
    /* OpenMP wasn't good in performance with the test of Macbook pro 2 cores. */
/* #pragma omp parallel for reduction(+:sum_g) */
    for (j = 0; j < num_triplets; j++) {
      sum_g += ise[j * num_band0 + i] * weights[j];
    }
    imag_self_energy[i] = sum_g * unit_conversion_factor;
  }
}

static void sum_imag_self_energy_N_and_U_along_triplets
(double *imag_self_energy_N,
 double *imag_self_energy_U,
 const double *ise,
 const int *triplets,
 const int *weights,
 const int *grid_address,
 const int num_band0,
 const int num_triplets)
{
  int i, j, k, sum_q;
  char *is_N;
  double sum_g_N, sum_g_U, g;

  is_N = (char*)malloc(sizeof(char) * num_triplets);

  for (i = 0; i < num_triplets; i++) {
    is_N[i] = 1;
    for (j = 0; j < 3; j++) {
      sum_q = 0;
      for (k = 0; k < 3; k++) { /* 1st, 2nd, 3rd triplet */
	sum_q += grid_address[triplets[i * 3 + k] * 3 + j];
      }
      if (sum_q) {
	is_N[i] = 0;
	break;
      }
    }
  }

  for (i = 0; i < num_band0; i++) {
    sum_g_N = 0;
    sum_g_U = 0;
    for (j = 0; j < num_triplets; j++) {
      g = ise[j * num_band0 + i] * weights[j];
      if (is_N[j]) {
	sum_g_N += g;
      } else {
	sum_g_U += g;
      }
    }
    imag_self_energy_N[i] = sum_g_N;
    imag_self_energy_U[i] = sum_g_U;
  }

  free(is_N);
}

static void set_tmp_values(double *n1,
			   double *n2,
			   const int i,
			   const int num_band,
			   const double temperature,
			   const int *triplets,
			   const double *frequencies,
			   const double cutoff_frequency)
{
  int j, gp1, gp2;
  double f1, f2;

  gp1 = triplets[i * 3 + 1];
  gp2 = triplets[i * 3 + 2];
  for (j = 0; j < num_band; j++) {
    f1 = frequencies[gp1 * num_band + j];
    f2 = frequencies[gp2 * num_band + j];
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
