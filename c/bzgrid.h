/* Copyright (C) 2008 Atsushi Togo */
/* All rights reserved. */

/* This file is part of spglib. */

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

#ifndef __bzgrid_H__
#define __bzgrid_H__

typedef struct {
    long size;
    long (*mat)[3][3];
} RotMats;

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
} BZGrid;

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
} ConstBZGrid;

long bzg_rotate_grid_index(const long grid_index, const long rotation[3][3],
                           const ConstBZGrid *bzgrid);
long bzg_get_bz_grid_addresses(BZGrid *bzgrid);
double bzg_get_tolerance_for_BZ_reduction(const BZGrid *bzgrid);
RotMats *bzg_alloc_RotMats(const long size);
void bzg_free_RotMats(RotMats *rotmats);
void bzg_multiply_matrix_vector_ld3(double v[3], const long a[3][3],
                                    const double b[3]);

#endif
