/* Copyright (C) 2021 Atsushi Togo */
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

#include "phononcalc.h"

#include "lapack_wrapper.h"
#include "phonon.h"

void phcalc_get_phonons_at_gridpoints(
    double *frequencies, _lapack_complex_double *eigenvectors,
    char *phonon_done, const long num_phonons, const long *grid_points,
    const long num_grid_points, const long (*grid_address)[3],
    const double QDinv[3][3], const double *fc2, const double (*svecs_fc2)[3],
    const long (*multi_fc2)[2], const double (*positions_fc2)[3],
    const long num_patom, const long num_satom, const double *masses_fc2,
    const long *p2s_fc2, const long *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, /* pointer */
    const double nac_factor, const double (*dd_q0)[2],
    const double (*G_list)[3], const long num_G_points, const double lambda,
    const char uplo) {
    if (!dd_q0) {
        phn_get_phonons_at_gridpoints(
            frequencies, (lapack_complex_double *)eigenvectors, phonon_done,
            num_phonons, grid_points, num_grid_points, grid_address, QDinv, fc2,
            svecs_fc2, multi_fc2, num_patom, num_satom, masses_fc2, p2s_fc2,
            s2p_fc2, unit_conversion_factor, born, dielectric,
            reciprocal_lattice, q_direction, nac_factor, uplo);
    } else {
        phn_get_gonze_phonons_at_gridpoints(
            frequencies, (lapack_complex_double *)eigenvectors, phonon_done,
            num_phonons, grid_points, num_grid_points, grid_address, QDinv, fc2,
            svecs_fc2, multi_fc2, positions_fc2, num_patom, num_satom,
            masses_fc2, p2s_fc2, s2p_fc2, unit_conversion_factor, born,
            dielectric, reciprocal_lattice, q_direction, nac_factor, dd_q0,
            G_list, num_G_points, lambda, uplo);
    }
}
