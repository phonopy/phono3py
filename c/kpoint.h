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

#ifndef __kpoint_H__
#define __kpoint_H__

#ifndef KPTCONST
#define KPTCONST
#endif

typedef struct {
  long size;
  long (*mat)[3][3];
} MatLONG;

long kpt_get_irreducible_reciprocal_mesh(long grid_address[][3],
                                         long ir_mapping_table[],
                                         const long mesh[3],
                                         const long is_shift[3],
                                         const MatLONG *rot_reciprocal);
MatLONG *kpt_get_point_group_reciprocal(const MatLONG * rotations,
                                        const long is_time_reversal);
MatLONG *kpt_get_point_group_reciprocal_with_q(const MatLONG * rot_reciprocal,
                                               const double symprec,
                                               const long num_q,
                                               KPTCONST double qpoints[][3]);
long kpt_get_stabilized_reciprocal_mesh(long grid_address[][3],
                                        long ir_mapping_table[],
                                        const long mesh[3],
                                        const long is_shift[3],
                                        const long is_time_reversal,
                                        const MatLONG * rotations,
                                        const long num_q,
                                        KPTCONST double qpoints[][3]);
long kpt_relocate_BZ_grid_address(long bz_grid_address[][3],
                                  long bz_map[],
                                  KPTCONST long grid_address[][3],
                                  const long mesh[3],
                                  KPTCONST double rec_lattice[3][3],
                                  const long is_shift[3]);
void kpt_copy_matrix_l3(long a[3][3], KPTCONST long b[3][3]);
MatLONG * kpt_alloc_MatLONG(const long size);
void kpt_free_MatLONG(MatLONG * matlong);

#endif
