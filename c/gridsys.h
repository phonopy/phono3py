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

#ifndef __gridsys_H__
#define __gridsys_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Generalized regular (GR) grid

Integer grid matrix M_g is unimodular transformed to integer diagnonal matrix D
by D = P M_g Q, which can be achieved by Smith normal form like transformation.
P and Q used are integer unimodular matrices with determinant=1.

S in PS is doubled shift with respect to microzone basis vectors, i.e.,
half-grid shift along an axis corresponds to 1.

*/

/**
 * @brief Return all GR-grid addresses with respect to n_1, n_2, n_3
 *
 * @param gr_grid_addresses all GR-grid addresses
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @return void
 */
void gridsys_get_all_grid_addresses(long (*gr_grid_addresses)[3],
                                    const long D_diag[3]);

/**
 * @brief Return double grid address in GR-grid
 *
 * @param address_double double grid address, i.e., possibly with shift in
 * GR-grid
 * @param address single grid address in GR-grid
 * @param PS shift in GR-grid
 * @return void
 */
void gridsys_get_double_grid_address(long address_double[3],
                                     const long address[3], const long PS[3]);

/**
 * @brief Return single grid address in GR-grid with given grid point index.
 *
 * @param address Single grid address in GR-grid
 * @param grid_index Grid point index in GR-grid
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @return void
 */
void gridsys_get_grid_address_from_index(long address[3], const long grid_index,
                                         const long D_diag[3]);

/**
 * @brief
 *
 * @param address_double
 * @param D_diag
 * @param PS
 * @return long
 */
long gridsys_get_double_grid_index(const long address_double[3],
                                   const long D_diag[3], const long PS[3]);
long gridsys_get_grid_index_from_address(const long address[3],
                                         const long D_diag[3]);
long gridsys_rotate_grid_index(const long grid_index, const long rotation[3][3],
                               const long D_diag[3], const long PS[3]);
long gridsys_get_reciprocal_point_group(long rec_rotations[48][3][3],
                                        const long (*rotations)[3][3],
                                        const long num_rot,
                                        const long is_time_reversal);
long gridsys_get_snf3x3(long D_diag[3], long P[3][3], long Q[3][3],
                        const long A[3][3]);
long gridsys_transform_rotations(long (*transformed_rots)[3][3],
                                 const long (*rotations)[3][3],
                                 const long num_rot, const long D_diag[3],
                                 const long Q[3][3]);
double gridsys_get_thm_integration_weight(const double omega,
                                          const double tetrahedra_omegas[24][4],
                                          const char function);
void gridsys_get_thm_all_relative_grid_address(
    long relative_grid_address[4][24][4][3]);
long gridsys_get_thm_relative_grid_address(
    long relative_grid_addresses[24][4][3], const double rec_lattice[3][3]);
void gridsys_get_ir_grid_map(long *ir_grid_map, const long (*rotations)[3][3],
                             const long num_rot, const long D_diag[3],
                             const long PS[3]);
long gridsys_get_bz_grid_addresses(long (*bz_grid_addresses)[3], long *bz_map,
                                   long *bzg2grg, const long D_diag[3],
                                   const long Q[3][3], const long PS[3],
                                   const double rec_lattice[3][3],
                                   const long type);
long gridsys_get_triplets_at_q(long *map_triplets, long *map_q,
                               const long grid_point, const long D_diag[3],
                               const long is_time_reversal, const long num_rot,
                               const long (*rec_rotations)[3][3],
                               const long swappable);
long gridsys_get_BZ_triplets_at_q(long (*triplets)[3], const long grid_point,
                                  const long (*bz_grid_addresses)[3],
                                  const long *bz_map, const long *map_triplets,
                                  const long num_map_triplets,
                                  const long D_diag[3], const long Q[3][3],
                                  const long bz_grid_type);
long gridsys_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long relative_grid_address[24][4][3],
    const long D_diag[3], const long (*triplets)[3], const long num_triplets,
    const long (*bz_grid_addresses)[3], const long *bz_map,
    const long bz_grid_type, const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets, const long openmp_per_bands);
void gridsys_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const long num_band0,
    const long (*triplets)[3], const long num_triplets,
    const double *frequencies, const long num_band, const long tp_type);

#ifdef __cplusplus
}
#endif

#endif
