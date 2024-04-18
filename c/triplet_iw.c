/* Copyright (C) 2016 Atsushi Togo */
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

#include "triplet_iw.h"

#include <math.h>

#include "grgrid.h"
#include "phonoc_utils.h"
#include "tetrahedron_method.h"
#include "triplet.h"

static void set_freq_vertices(double freq_vertices[3][24][4],
                              const double *frequencies1,
                              const double *frequencies2,
                              const long vertices[2][24][4],
                              const long num_band1, const long num_band2,
                              const long b1, const long b2, const long tp_type);
static long set_g(double g[3], const double f0,
                  const double freq_vertices[3][24][4], const long max_i);
static void get_triplet_tetrahedra_vertices(
    long vertices[2][24][4], const long tp_relative_grid_address[2][24][4][3],
    const long triplet[3], const ConstBZGrid *bzgrid);
static void get_neighboring_grid_points_type1(
    long *neighboring_grid_points, const long grid_point,
    const long (*relative_grid_address)[3],
    const long num_relative_grid_address, const ConstBZGrid *bzgrid);
static void get_neighboring_grid_points_type2(
    long *neighboring_grid_points, const long grid_point,
    const long (*relative_grid_address)[3],
    const long num_relative_grid_address, const ConstBZGrid *bzgrid);

void tpi_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long tp_relative_grid_address[2][24][4][3],
    const long triplets[3], const long num_triplets, const ConstBZGrid *bzgrid,
    const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets) {
    long max_i, j, b1, b2, b12, num_band_prod, adrs_shift;
    long vertices[2][24][4];
    double g[3];
    double freq_vertices[3][24][4];

    get_triplet_tetrahedra_vertices(vertices, tp_relative_grid_address,
                                    triplets, bzgrid);

    num_band_prod = num_triplets * num_band0 * num_band1 * num_band2;

    /* tp_type: Type of integration weights stored */
    /* */
    /* g0 -> \delta(f0 - (-f1 + f2)) */
    /* g1 -> \delta(f0 - (f1 - f2)) */
    /* g2 -> \delta(f0 - (f1 + f2)) */
    /* */
    /* tp_type = 2: (g[2], g[0] - g[1]) mainly for ph-ph */
    /* tp_type = 3: (g[2], g[0] - g[1], g[0] + g[1] + g[2]) mainly for ph-ph */
    /* tp_type = 4: (g[0]) mainly for el-ph phonon decay, */
    /*              f0: ph, f1: el_i, f2: el_f */

    if ((tp_type == 2) || (tp_type == 3)) {
        max_i = 3;
    }
    if (tp_type == 4) {
        max_i = 1;
    }

#ifdef _OPENMP
#pragma omp parallel for private(j, b1, b2, adrs_shift, g, \
                                     freq_vertices) if (!openmp_per_triplets)
#endif
    for (b12 = 0; b12 < num_band1 * num_band2; b12++) {
        b1 = b12 / num_band2;
        b2 = b12 % num_band2;
        set_freq_vertices(freq_vertices, frequencies1, frequencies2, vertices,
                          num_band1, num_band2, b1, b2, tp_type);
        for (j = 0; j < num_band0; j++) {
            adrs_shift = j * num_band1 * num_band2 + b1 * num_band2 + b2;
            iw_zero[adrs_shift] =
                set_g(g, frequency_points[j], freq_vertices, max_i);
            if (tp_type == 2) {
                iw[adrs_shift] = g[2];
                adrs_shift += num_band_prod;
                iw[adrs_shift] = g[0] - g[1];
            }
            if (tp_type == 3) {
                iw[adrs_shift] = g[2];
                adrs_shift += num_band_prod;
                iw[adrs_shift] = g[0] - g[1];
                adrs_shift += num_band_prod;
                iw[adrs_shift] = g[0] + g[1] + g[2];
            }
            if (tp_type == 4) {
                iw[adrs_shift] = g[0];
            }
        }
    }
}

void tpi_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double cutoff,
    const double *frequency_points, const long num_band0, const long triplet[3],
    const long const_adrs_shift, const double *frequencies, const long num_band,
    const long tp_type, const long openmp_per_triplets) {
    long j, b12, b1, b2, adrs_shift;
    double f0, f1, f2, g0, g1, g2;

#ifdef _OPENMP
#pragma omp parallel for private(j, b1, b2, f0, f1, f2, g0, g1, g2, \
                                     adrs_shift) if (!openmp_per_triplets)
#endif
    for (b12 = 0; b12 < num_band * num_band; b12++) {
        b1 = b12 / num_band;
        b2 = b12 % num_band;
        f1 = frequencies[triplet[1] * num_band + b1];
        f2 = frequencies[triplet[2] * num_band + b2];
        for (j = 0; j < num_band0; j++) {
            f0 = frequency_points[j];
            adrs_shift = j * num_band * num_band + b1 * num_band + b2;

            if ((tp_type == 2) || (tp_type == 3)) {
                if (cutoff > 0 && fabs(f0 + f1 - f2) > cutoff &&
                    fabs(f0 - f1 + f2) > cutoff &&
                    fabs(f0 - f1 - f2) > cutoff) {
                    iw_zero[adrs_shift] = 1;
                    g0 = 0;
                    g1 = 0;
                    g2 = 0;
                } else {
                    iw_zero[adrs_shift] = 0;
                    g0 = phonoc_gaussian(f0 + f1 - f2, sigma);
                    g1 = phonoc_gaussian(f0 - f1 + f2, sigma);
                    g2 = phonoc_gaussian(f0 - f1 - f2, sigma);
                }
                if (tp_type == 2) {
                    iw[adrs_shift] = g2;
                    adrs_shift += const_adrs_shift;
                    iw[adrs_shift] = g0 - g1;
                }
                if (tp_type == 3) {
                    iw[adrs_shift] = g2;
                    adrs_shift += const_adrs_shift;
                    iw[adrs_shift] = g0 - g1;
                    adrs_shift += const_adrs_shift;
                    iw[adrs_shift] = g0 + g1 + g2;
                }
            }
            if (tp_type == 4) {
                if (cutoff > 0 && fabs(f0 + f1 - f2) > cutoff) {
                    iw_zero[adrs_shift] = 1;
                    iw[adrs_shift] = 0;
                } else {
                    iw_zero[adrs_shift] = 0;
                    iw[adrs_shift] = phonoc_gaussian(f0 + f1 - f2, sigma);
                }
            }
        }
    }
}

/**
 * @brief Return grid points of relative grid adddresses in BZ-grid
 *
 * @param neighboring_grid_points Grid points of relative grid addresses in
 * BZ-grid.
 * @param grid_point Grid point of interest.
 * @param relative_grid_address Relative grid address wrt grid point of
 * interest.
 * @param num_relative_grid_address Number of relative grid addresses.
 * @param bzgrid
 */
void tpi_get_neighboring_grid_points(long *neighboring_grid_points,
                                     const long grid_point,
                                     const long (*relative_grid_address)[3],
                                     const long num_relative_grid_address,
                                     const ConstBZGrid *bzgrid) {
    if (bzgrid->type == 1) {
        get_neighboring_grid_points_type1(neighboring_grid_points, grid_point,
                                          relative_grid_address,
                                          num_relative_grid_address, bzgrid);
    } else {
        get_neighboring_grid_points_type2(neighboring_grid_points, grid_point,
                                          relative_grid_address,
                                          num_relative_grid_address, bzgrid);
    }
}

static void set_freq_vertices(double freq_vertices[3][24][4],
                              const double *frequencies1,
                              const double *frequencies2,
                              const long vertices[2][24][4],
                              const long num_band1, const long num_band2,
                              const long b1, const long b2,
                              const long tp_type) {
    long i, j;
    double f1, f2;

    for (i = 0; i < 24; i++) {
        for (j = 0; j < 4; j++) {
            f1 = frequencies1[vertices[0][i][j] * num_band1 + b1];
            f2 = frequencies2[vertices[1][i][j] * num_band2 + b2];
            if ((tp_type == 2) || (tp_type == 3)) {
                if (f1 < 0) {
                    f1 = 0;
                }
                if (f2 < 0) {
                    f2 = 0;
                }
                freq_vertices[0][i][j] = -f1 + f2;
                freq_vertices[1][i][j] = f1 - f2;
                freq_vertices[2][i][j] = f1 + f2;
            } else {
                freq_vertices[0][i][j] = -f1 + f2;
            }
        }
    }
}

/* Integration weight g is calculated. */
/* iw_zero = 1 means g[0] to g[max_i - 1] are all zero. */
/* max_i depends on what we compute, e.g., ph-ph lifetime, */
/* ph-ph collision matrix, and el-ph relaxation time. */
/* iw_zero is definitely determined by in_tetrahedra in case that */
/* f0 is out of the tetrahedra. */
/* iw_zero=1 information can be used to omit to compute particles */
/* interaction strength that is often heaviest part in throughout */
/* calculation. */
static long set_g(double g[3], const double f0,
                  const double freq_vertices[3][24][4], const long max_i) {
    long i, iw_zero;

    iw_zero = 1;

    for (i = 0; i < max_i; i++) {
        if (thm_in_tetrahedra(f0, freq_vertices[i])) {
            g[i] = thm_get_integration_weight(f0, freq_vertices[i], 'I');
            iw_zero = 0;
        } else {
            g[i] = 0;
        }
    }

    return iw_zero;
}

static void get_triplet_tetrahedra_vertices(
    long vertices[2][24][4], const long tp_relative_grid_address[2][24][4][3],
    const long triplet[3], const ConstBZGrid *bzgrid) {
    long i, j;

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 24; j++) {
            tpi_get_neighboring_grid_points(vertices[i][j], triplet[i + 1],
                                            tp_relative_grid_address[i][j], 4,
                                            bzgrid);
        }
    }
}

static void get_neighboring_grid_points_type1(
    long *neighboring_grid_points, const long grid_point,
    const long (*relative_grid_address)[3],
    const long num_relative_grid_address, const ConstBZGrid *bzgrid) {
    long bzmesh[3], bz_address[3];
    long i, j, bz_gp, prod_bz_mesh;

    for (i = 0; i < 3; i++) {
        bzmesh[i] = bzgrid->D_diag[i] * 2;
    }
    prod_bz_mesh = bzmesh[0] * bzmesh[1] * bzmesh[2];
    for (i = 0; i < num_relative_grid_address; i++) {
        for (j = 0; j < 3; j++) {
            bz_address[j] =
                bzgrid->addresses[grid_point][j] + relative_grid_address[i][j];
        }
        bz_gp = bzgrid->gp_map[grg_get_grid_index(bz_address, bzmesh)];
        if (bz_gp == prod_bz_mesh) {
            neighboring_grid_points[i] =
                grg_get_grid_index(bz_address, bzgrid->D_diag);
        } else {
            neighboring_grid_points[i] = bz_gp;
        }
    }
}

static void get_neighboring_grid_points_type2(
    long *neighboring_grid_points, const long grid_point,
    const long (*relative_grid_address)[3],
    const long num_relative_grid_address, const ConstBZGrid *bzgrid) {
    long bz_address[3];
    long i, j, gp;

    for (i = 0; i < num_relative_grid_address; i++) {
        for (j = 0; j < 3; j++) {
            bz_address[j] =
                bzgrid->addresses[grid_point][j] + relative_grid_address[i][j];
        }
        gp = grg_get_grid_index(bz_address, bzgrid->D_diag);
        neighboring_grid_points[i] = bzgrid->gp_map[gp];
        if (bzgrid->gp_map[gp + 1] - bzgrid->gp_map[gp] > 1) {
            for (j = bzgrid->gp_map[gp]; j < bzgrid->gp_map[gp + 1]; j++) {
                if (bz_address[0] == bzgrid->addresses[j][0] &&
                    bz_address[1] == bzgrid->addresses[j][1] &&
                    bz_address[2] == bzgrid->addresses[j][2]) {
                    neighboring_grid_points[i] = j;
                    break;
                }
            }
        }
    }
}
