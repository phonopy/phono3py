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

#ifndef __phonon_H__
#define __phonon_H__

#include <dynmat.h>
#include <lapack_wrapper.h>

void
phn_get_phonons_at_gridpoints(double *frequencies,
                              lapack_complex_double *eigenvectors,
                              char *phonon_done,
                              const size_t num_phonons,
                              const size_t *grid_points,
                              const size_t num_grid_points,
                              PHPYCONST int (*grid_address)[3],
                              const int mesh[3],
                              const double *fc2,
                              PHPYCONST double(*svecs_fc2)[27][3],
                              const int *multi_fc2,
                              const size_t num_patom,
                              const size_t num_satom,
                              const double *masses_fc2,
                              const int *p2s_fc2,
                              const int *s2p_fc2,
                              const double unit_conversion_factor,
                              PHPYCONST double (*born)[3][3],
                              PHPYCONST double dielectric[3][3],
                              PHPYCONST double reciprocal_lattice[3][3],
                              const double *q_direction, /* must be pointer */
                              const double nac_factor,
                              const char uplo);
void
phn_get_gonze_phonons_at_gridpoints(double *frequencies,
                                    lapack_complex_double *eigenvectors,
                                    char *phonon_done,
                                    const size_t num_phonons,
                                    const size_t *grid_points,
                                    const size_t num_grid_points,
                                    PHPYCONST int (*grid_address)[3],
                                    const int mesh[3],
                                    const double *fc2,
                                    PHPYCONST double(*svecs_fc2)[27][3],
                                    const int *multi_fc2,
                                    PHPYCONST double (*positions)[3],
                                    const size_t num_patom,
                                    const size_t num_satom,
                                    const double *masses_fc2,
                                    const int *p2s_fc2,
                                    const int *s2p_fc2,
                                    const double unit_conversion_factor,
                                    PHPYCONST double (*born)[3][3],
                                    PHPYCONST double dielectric[3][3],
                                    PHPYCONST double reciprocal_lattice[3][3],
                                    const double *q_direction, /* pointer */
                                    const double nac_factor,
                                    const double *dd_q0,
                                    PHPYCONST double(*G_list)[3],
                                    const size_t num_G_points,
                                    const double lambda,
                                    const char uplo);
#endif
