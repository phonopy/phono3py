/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __fc3_H__
#define __fc3_H__
#include <stdint.h>

void fc3_distribute_fc3(double *fc3, const int64_t target, const int64_t source,
                        const int64_t *atom_mapping, const int64_t num_atom,
                        const double *rot_cart);
void fc3_rotate_delta_fc2(double (*fc3)[3][3][3],
                          const double (*delta_fc2s)[3][3], const double *inv_U,
                          const double (*site_sym_cart)[3][3],
                          const int64_t *rot_map_syms, const int64_t num_atom,
                          const int64_t num_site_sym, const int64_t num_disp);
void fc3_set_permutation_symmetry_fc3(double *fc3, const int64_t num_atom);
void fc3_set_permutation_symmetry_compact_fc3(double *fc3, const int64_t p2s[],
                                              const int64_t s2pp[],
                                              const int64_t nsym_list[],
                                              const int64_t perms[],
                                              const int64_t n_satom,
                                              const int64_t n_patom);
void fc3_transpose_compact_fc3(double *fc3, const int64_t p2s[],
                               const int64_t s2pp[], const int64_t nsym_list[],
                               const int64_t perms[], const int64_t n_satom,
                               const int64_t n_patom, const int64_t t_type);

#endif
