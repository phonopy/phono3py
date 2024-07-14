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

#ifndef __recgrid_H__
#define __recgrid_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    long size;
    long (*mat)[3][3];
} RecgridMats;

/* Data structure of Brillouin zone grid
 *
 * size : long
 *     Number of grid points in Brillouin zone including its surface.
 * D_diag : long array
 *     Diagonal part of matrix D of SNF.
 *     shape=(3, )
 * PS : long array
 *     Matrix P of SNF multiplied by shift.
 *     shape=(3, )
 * gp_map : long array
 *     Type1 : Twice enlarged grid space along basis vectors.
 *             Grid index is recovered in the same way as regular grid.
 *             shape=(prod(mesh * 2), )
 *     Type2 : In the last index, multiplicity and array index of
 *             each address of the grid point are stored. Here,
 *             multiplicity means the number of translationally
 *             equivalent grid points in BZ.
 *             shape=(prod(mesh), 2) -> flattened.
 * addresses : long array
 *     Grid point addresses.
 *     shape=(size, 3)
 * reclat : double array
 *     Reciprocal basis vectors given as column vectors.
 *     shape=(3, 3)
 * type : long
 *     1 or 2. */
typedef struct {
    long size;
    long D_diag[3];
    long Q[3][3];
    long PS[3];
    long *gp_map;
    long *bzg2grg;
    long (*addresses)[3];
    double reclat[3][3];
    long type;
} RecgridBZGrid;

typedef struct {
    long size;
    long D_diag[3];
    long Q[3][3];
    long PS[3];
    const long *gp_map;
    const long *bzg2grg;
    const long (*addresses)[3];
    double reclat[3][3];
    long type;
} RecgridConstBZGrid;

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
void recgrid_get_all_grid_addresses(long (*gr_grid_addresses)[3],
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
void recgrid_get_double_grid_address(long address_double[3],
                                     const long address[3], const long PS[3]);

/**
 * @brief Return single grid address in GR-grid with given grid point index.
 *
 * @param address Single grid address in GR-grid
 * @param grid_index Grid point index in GR-grid
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @return void
 */
void recgrid_get_grid_address_from_index(long address[3], const long grid_index,
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
long recgrid_get_double_grid_index(const long address_double[3],
                                   const long D_diag[3], const long PS[3]);

/**
 * @brief Return grid point index of single grid address in GR-grid
 *
 * @param address Single grid address in GR-grid
 * @param D_diag Numbers of divisions along a, b, c directions of GR-grid
 * @return long
 */
long recgrid_get_grid_index_from_address(const long address[3],
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
long recgrid_rotate_grid_index(const long grid_index, const long rotation[3][3],
                               const long D_diag[3], const long PS[3]);

/**
 * @brief Return {R^T} of crystallographic point group {R} with and without time
 * reversal symmetry.
 *
 * @param rec_rotations Rotations in reciprocal space {R^T}
 * @param rotations Rotations in direct space {R}
 * @param num_rot Number of given rotations |{R}|
 * @param is_time_reversal With (1) or without (0) time reversal symmetry
 * @param is_transpose With (1) or without (0) transpose of rotation matrices
 * @return long
 */
long recgrid_get_reciprocal_point_group(long rec_rotations[48][3][3],
                                        const long (*rotations)[3][3],
                                        const long num_rot,
                                        const long is_time_reversal,
                                        const long is_transpose);
/**
 * @brief Return D, P, Q of Smith normal form of A.
 *
 * @param D_diag Diagonal elements of diagnoal matrix D
 * @param P Unimodular matrix P
 * @param Q Unimodular matrix Q
 * @param A Integer matrix
 * @return long
 */
long recgrid_get_snf3x3(long D_diag[3], long P[3][3], long Q[3][3],
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
long recgrid_transform_rotations(long (*transformed_rots)[3][3],
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
 * @return long Number of ir_grid_points.
 */
long recgrid_get_ir_grid_map(long *ir_grid_map, const long (*rotations)[3][3],
                             const long num_rot, const long D_diag[3],
                             const long PS[3]);

/**
 * @brief Find shortest grid points from Gamma considering periodicity of
 * reciprocal lattice. See the details in docstring of RecgridBZGrid
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
long recgrid_get_bz_grid_addresses(long (*bz_grid_addresses)[3], long *bz_map,
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
long recgrid_rotate_bz_grid_index(const long bz_grid_index,
                                  const long rotation[3][3],
                                  const long (*bz_grid_addresses)[3],
                                  const long *bz_map, const long D_diag[3],
                                  const long PS[3], const long bz_grid_type);
double recgrid_get_tolerance_for_BZ_reduction(const RecgridBZGrid *bzgrid);
RecgridMats *recgrid_alloc_RotMats(const long size);
void recgrid_free_RotMats(RecgridMats *rotmats);

#ifdef __cplusplus
}
#endif

#endif
