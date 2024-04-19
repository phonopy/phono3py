/* Copyright (C) 2016 Atsushi Togo */
/* All rights reserved. */

/* phono3py. This file is part of phonopy. */

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

#ifndef __triplet_iw_H__
#define __triplet_iw_H__

#include "bzgrid.h"

void tpi_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long tp_relative_grid_address[2][24][4][3],
    const long triplets[3], const long num_triplets, const ConstBZGrid *bzgrid,
    const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets);
void tpi_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double cutoff,
    const double *frequency_points, const long num_band0, const long triplet[3],
    const long const_adrs_shift, const double *frequencies, const long num_band,
    const long tp_type, const long openmp_per_triplets);
void tpi_get_neighboring_grid_points(long *neighboring_grid_points,
                                     const long grid_point,
                                     const long (*relative_grid_address)[3],
                                     const long num_relative_grid_address,
                                     const ConstBZGrid *bzgrid);

#endif
