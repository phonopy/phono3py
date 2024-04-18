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
 * @param gr_grid_addresses All GR-grid addresses
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @return void
 */
void gridsys_get_all_grid_addresses(long (*gr_grid_addresses)[3],
                                    const long D_diag[3]);

/**
 * @brief Return double grid address in GR-grid
 *
 * @param address_double Double grid address, i.e., possibly with shift in
 * GR-grid
 * @param address Single grid address in GR-grid
 * @param PS Shift in GR-grid
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
 * @brief Return grid point index of double grid address in GR-grid
 *
 * @param address_double Double grid address, i.e., possibly with shift in
 * GR-grid
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @param PS Shift in GR-grid
 * @return long
 */
long gridsys_get_double_grid_index(const long address_double[3],
                                   const long D_diag[3], const long PS[3]);

/**
 * @brief Return grid point index of single grid address in GR-grid
 *
 * @param address Single grid address in GR-grid
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @return long
 */
long gridsys_get_grid_index_from_address(const long address[3],
                                         const long D_diag[3]);

/**
 * @brief Return grid point index of rotated address of given grid point index.
 *
 * @param grid_index Grid point index in GR-grid
 * @param rotation Transformed rotation in reciprocal space tilde-R^T
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @param PS Shift in GR-grid
 * @return long
 */
long gridsys_rotate_grid_index(const long grid_index, const long rotation[3][3],
                               const long D_diag[3], const long PS[3]);

/**
 * @brief Return {R^T} of crystallographic point group {R} with and without time
 * reversal symmetry.
 *
 * @param rec_rotations Rotations in reciprocal space {R^T}
 * @param rotations Rotations in direct space {R}
 * @param num_rot Number of given rotations |{R}|
 * @param is_time_reversal With (1) or without (0) time reversal symmetry
 * @return long
 */
long gridsys_get_reciprocal_point_group(long rec_rotations[48][3][3],
                                        const long (*rotations)[3][3],
                                        const long num_rot,
                                        const long is_time_reversal);

/**
 * @brief Return D, P, Q of Smith normal form of A.
 *
 * @param D_diag Diagonal elements of diagnoal matrix D
 * @param P Unimodular matrix P
 * @param Q Unimodular matrix Q
 * @param A Integer matrix
 * @return long
 */
long gridsys_get_snf3x3(long D_diag[3], long P[3][3], long Q[3][3],
                        const long A[3][3]);

/**
 * @brief Transform {R^T} to {R^T} with respect to transformed microzone basis
 * vectors in GR-grid
 *
 * @param transformed_rots Transformed rotation matrices in reciprocal space
 * {tilde-R^T}
 * @param rotations Original rotations matrices in reciprocal space {R^T}
 * @param num_rot Number of rotation matrices
 * @param D_diag Diagonal elements of diagnoal matrix D of Smith normal form
 * @param Q Unimodular matrix Q of Smith normal form
 * @return long
 */
long gridsys_transform_rotations(long (*transformed_rots)[3][3],
                                 const long (*rotations)[3][3],
                                 const long num_rot, const long D_diag[3],
                                 const long Q[3][3]);

/**
 * @brief Return mapping table from GR-grid points to GR-ir-grid points
 *
 * @param ir_grid_map Grid point index mapping to ir-grid point indices with
 * array size of prod(D_diag)
 * @param rotations Transformed rotation matrices in reciprocal space
 * @param num_rot Number of rotation matrices
 * @param D_diag Diagonal elements of diagnoal matrix D of Smith normal form
 * @param PS Shift in GR-grid
 */
void gridsys_get_ir_grid_map(long *ir_grid_map, const long (*rotations)[3][3],
                             const long num_rot, const long D_diag[3],
                             const long PS[3]);

/**
 * @brief Find shortest grid points from Gamma considering periodicity of
 * reciprocal lattice. See the details in docstring of BZGrid
 *
 * @param bz_grid_addresses Grid point addresses of shortest grid points
 * @param bz_map List of accumulated numbers of BZ grid points from the
 * first GR grid point to the last grid point. In type-II, [0, 1, 3, 4, ...]
 * means multiplicities of [1, 2, 1, ...], with len(bz_map)=product(D_diag) + 1.
 * @param bzg2grg Mapping table of bz_grid_addresses to gr_grid_addresses. In
 * type-II, len(bzg2grg) == len(bz_grid_addresses) <= (D_diag[0] + 1) *
 * (D_diag[1] + 1) * (D_diag[2] + 1).
 * @param D_diag Diagonal elements of diagnoal matrix D of Smith normal form
 * @param Q Unimodular matrix Q of Smith normal form
 * @param PS Shift in GR-grid
 * @param rec_lattice Reduced reciprocal basis vectors in column vectors
 * @param bz_grid_type Data structure type I (old and sparse) or II (new and
 * dense, recommended) of bz_map
 * @return long Number of bz_grid_addresses stored.
 */
long gridsys_get_bz_grid_addresses(long (*bz_grid_addresses)[3], long *bz_map,
                                   long *bzg2grg, const long D_diag[3],
                                   const long Q[3][3], const long PS[3],
                                   const double rec_lattice[3][3],
                                   const long bz_grid_type);

/**
 * @brief Return index of rotated bz grid point
 *
 * @param bz_grid_index BZ grid point index
 * @param rotation Transformed rotation in reciprocal space tilde-R^T
 * @param bz_grid_addresses BZ grid point adddresses
 * @param bz_map List of accumulated numbers of BZ grid points from the
 * first GR grid point to the last grid point. In type-II, [0, 1, 3, 4, ...]
 * means multiplicities of [1, 2, 1, ...], with len(bz_map)=product(D_diag) + 1.
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @param PS Shift in GR-grid
 * @param bz_grid_type Data structure type I (old and sparse) or II (new and
 * dense, recommended) of bz_map
 * @return long
 */
long gridsys_rotate_bz_grid_index(const long bz_grid_index,
                                  const long rotation[3][3],
                                  const long (*bz_grid_addresses)[3],
                                  const long *bz_map, const long D_diag[3],
                                  const long PS[3], const long bz_grid_type);

/**
 * @brief Find independent q' of (q, q', q'') with given q.
 *
 * @param map_triplets Mapping table from all grid points to grid points of
 * independent q' in GR-grid
 * @param map_q Mapping table from all grid points to grid point indices of
 * irreducible q-points under the stabilizer subgroup of q
 * @param grid_index Grid point index of q in GR-grid
 * @param D_diag Diagonal elements of diagnoal matrix D of Smith normal form
 * @param is_time_reversal With (1) or without (0) time reversal symmetry
 * @param num_rot Number of rotation matrices
 * @param rec_rotations Transformed rotation matrices in reciprocal space
 * @param swappable With (1) or without (0) permutation symmetry between q'
 * and q''
 * @return long Number of unique element of map_triplets
 */
long gridsys_get_triplets_at_q(long *map_triplets, long *map_q,
                               const long grid_index, const long D_diag[3],
                               const long is_time_reversal, const long num_rot,
                               const long (*rec_rotations)[3][3],
                               const long swappable);

/**
 * @brief Search grid point triplets considering BZ surface.
 *
 * @param ir_triplets Ir-triplets by grid point indices in BZ-grid
 * @param bz_grid_index Grid point of q in BZ-grid
 * @param bz_grid_addresses Grid point addresses of shortest grid points
 * @param bz_map  List of accumulated numbers of BZ grid points from the
 * first GR grid point to the last grid point. In type-II, [0, 1, 3, 4, ...]
 * means multiplicities of [1, 2, 1, ...], with len(bz_map)=product(D_diag) + 1.
 * @param map_triplets Mapping table from all grid points to grid points of
 * independent q'
 * @param num_map_triplets First dimension of map_triplets
 * @param D_diag Diagonal elements of diagnoal matrix D of Smith normal form
 * @param Q Unimodular matrix Q of Smith normal form
 * @param bz_grid_type Data structure type I (old and sparse) or II (new and
 * dense, recommended) of bz_map
 * @return long
 */
long gridsys_get_bz_triplets_at_q(long (*ir_triplets)[3],
                                  const long bz_grid_index,
                                  const long (*bz_grid_addresses)[3],
                                  const long *bz_map, const long *map_triplets,
                                  const long num_map_triplets,
                                  const long D_diag[3], const long Q[3][3],
                                  const long bz_grid_type);

/**
 * @brief Return integration weight of linear tetrahedron method
 *
 * @param omega A frequency point where integration weight is computed
 * @param tetrahedra_omegas Frequencies at vertices of 6 tetrahedra
 * @param function I (delta function) or J (theta function)
 * @return double
 */
double gridsys_get_thm_integration_weight(const double omega,
                                          const double tetrahedra_omegas[24][4],
                                          const char function);

/**
 * @brief Return predefined relative grid addresses of 4 different
 * main diagonals for linear tetrahedron method
 *
 * @param relative_grid_address predefined relative grid addresses for linear
 * tetrahedron method
 */
void gridsys_get_thm_all_relative_grid_address(
    long relative_grid_address[4][24][4][3]);

/**
 * @brief Return predefined relative grid addresses of main diagonal determined
 * from reciprocal basis vectors for linear tetrahedron method
 *
 * @param relative_grid_addresses predefined relative grid addresses of given
 * reciprocal basis vectors
 * @param rec_lattice Reciprocal basis vectors in column vectors
 * @return * long
 */
long gridsys_get_thm_relative_grid_address(
    long relative_grid_addresses[24][4][3], const double rec_lattice[3][3]);

long gridsys_get_integration_weight(
    double *iw, char *iw_zero, const double *frequency_points,
    const long num_band0, const long relative_grid_address[24][4][3],
    const long D_diag[3], const long (*triplets)[3], const long num_triplets,
    const long (*bz_grid_addresses)[3], const long *bz_map,
    const long bz_grid_type, const double *frequencies1, const long num_band1,
    const double *frequencies2, const long num_band2, const long tp_type,
    const long openmp_per_triplets);
void gridsys_get_integration_weight_with_sigma(
    double *iw, char *iw_zero, const double sigma, const double sigma_cutoff,
    const double *frequency_points, const long num_band0,
    const long (*triplets)[3], const long num_triplets,
    const double *frequencies, const long num_band, const long tp_type);

#ifdef __cplusplus
}
#endif

#endif
