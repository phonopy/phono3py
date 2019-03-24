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

#ifndef __fc3_H__
#define __fc3_H__

void fc3_distribute_fc3(double *fc3,
                        const int target,
                        const int source,
                        const int *atom_mapping,
                        const size_t num_atom,
                        const double *rot_cart);
void fc3_rotate_delta_fc2(double (*fc3)[3][3][3],
                          PHPYCONST double (*delta_fc2s)[3][3],
                          const double *inv_U,
                          PHPYCONST double (*site_sym_cart)[3][3],
                          const int *rot_map_syms,
                          const size_t num_atom,
                          const size_t num_site_sym,
                          const size_t num_disp);
void fc3_set_permutation_symmetry_fc3(double *fc3, const size_t num_atom);
void fc3_set_permutation_symmetry_compact_fc3(double * fc3,
                                              const int p2s[],
                                              const int s2pp[],
                                              const int nsym_list[],
                                              const int perms[],
                                              const size_t n_satom,
                                              const size_t n_patom);
void fc3_transpose_compact_fc3(double * fc3,
                               const int p2s[],
                               const int s2pp[],
                               const int nsym_list[],
                               const int perms[],
                               const size_t n_satom,
                               const size_t n_patom,
                               const int t_type);

#endif
