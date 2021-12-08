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

#include "fc3.h"

#include <stdlib.h>

static void rotate_delta_fc2s(double (*rot_delta_fc2s)[3][3], const long i_atom,
                              const long j_atom,
                              const double (*delta_fc2s)[3][3],
                              const double (*site_sym_cart)[3][3],
                              const long *rot_map_sym, const long num_atom,
                              const long num_site_sym, const long num_disp);
static void tensor2_rotation(double rot_tensor[3][3], const double tensor[3][3],
                             const double r[3][3]);
static void tensor3_rotation(double *rot_tensor, const double *tensor,
                             const double *rot_cartesian);
static double tensor3_rotation_elem(const double *tensor, const double *r,
                                    const long pos);
static void copy_permutation_symmetry_fc3_elem(double *fc3,
                                               const double fc3_elem[27],
                                               const long a, const long b,
                                               const long c,
                                               const long num_atom);
static void set_permutation_symmetry_fc3_elem(double *fc3_elem,
                                              const double *fc3, const long a,
                                              const long b, const long c,
                                              const long num_atom);
static void set_permutation_symmetry_compact_fc3(
    double *fc3, const long p2s[], const long s2pp[], const long nsym_list[],
    const long perms[], const long n_satom, const long n_patom);
static void transpose_compact_fc3_type01(double *fc3, const long p2s[],
                                         const long s2pp[],
                                         const long nsym_list[],
                                         const long perms[], const long n_satom,
                                         const long n_patom, const long t_type);
static void transpose_compact_fc3_type2(double *fc3, const long p2s[],
                                        const long s2pp[],
                                        const long nsym_list[],
                                        const long perms[], const long n_satom,
                                        const long n_patom);

void fc3_distribute_fc3(double *fc3, const long target, const long source,
                        const long *atom_mapping, const long num_atom,
                        const double *rot_cart) {
    long i, j, adrs_out, adrs_in;

    for (i = 0; i < num_atom; i++) {
        for (j = 0; j < num_atom; j++) {
            adrs_out = (num_atom * num_atom * target + num_atom * i + j) * 27;
            adrs_in = (num_atom * num_atom * source +
                       num_atom * atom_mapping[i] + atom_mapping[j]) *
                      27;
            tensor3_rotation(fc3 + adrs_out, fc3 + adrs_in, rot_cart);
        }
    }
}

void fc3_rotate_delta_fc2(double (*fc3)[3][3][3],
                          const double (*delta_fc2s)[3][3], const double *inv_U,
                          const double (*site_sym_cart)[3][3],
                          const long *rot_map_syms, const long num_atom,
                          const long num_site_sym, const long num_disp) {
    long i_atoms, i, j, k, l, m, n;
    double(*rot_delta_fc2s)[3][3];

    rot_delta_fc2s =
        (double(*)[3][3])malloc(sizeof(double[3][3]) * num_site_sym * num_disp);
    for (i_atoms = 0; i_atoms < num_atom * num_atom; i_atoms++) {
        i = i_atoms / num_atom;
        j = i_atoms % num_atom;
        rotate_delta_fc2s(rot_delta_fc2s, i, j, delta_fc2s, site_sym_cart,
                          rot_map_syms, num_atom, num_site_sym, num_disp);
        for (k = 0; k < 3; k++) {
            for (l = 0; l < 3; l++) {
                for (m = 0; m < 3; m++) {
                    fc3[i_atoms][k][l][m] = 0;
                    for (n = 0; n < num_site_sym * num_disp; n++) {
                        fc3[i_atoms][k][l][m] +=
                            inv_U[k * num_site_sym * num_disp + n] *
                            rot_delta_fc2s[n][l][m];
                    }
                }
            }
        }
    }

    free(rot_delta_fc2s);
    rot_delta_fc2s = NULL;
}

void fc3_set_permutation_symmetry_fc3(double *fc3, const long num_atom) {
    double fc3_elem[27];
    long i, j, k;

#ifdef _OPENMP
#pragma omp parallel for private(j, k, fc3_elem)
#endif
    for (i = 0; i < num_atom; i++) {
        for (j = i; j < num_atom; j++) {
            for (k = j; k < num_atom; k++) {
                set_permutation_symmetry_fc3_elem(fc3_elem, fc3, i, j, k,
                                                  num_atom);
                copy_permutation_symmetry_fc3_elem(fc3, fc3_elem, i, j, k,
                                                   num_atom);
            }
        }
    }
}

void fc3_set_permutation_symmetry_compact_fc3(
    double *fc3, const long p2s[], const long s2pp[], const long nsym_list[],
    const long perms[], const long n_satom, const long n_patom) {
    set_permutation_symmetry_compact_fc3(fc3, p2s, s2pp, nsym_list, perms,
                                         n_satom, n_patom);
}

void fc3_transpose_compact_fc3(double *fc3, const long p2s[], const long s2pp[],
                               const long nsym_list[], const long perms[],
                               const long n_satom, const long n_patom,
                               const long t_type) {
    /* Three types of index permutations                       */
    /*     t_type=0: dim[0] <-> dim[1]                         */
    /*     t_type=1: dim[0] <-> dim[2]                         */
    /*     t_type=2: dim[1] <-> dim[2]                         */
    if (t_type == 0 || t_type == 1) {
        transpose_compact_fc3_type01(fc3, p2s, s2pp, nsym_list, perms, n_satom,
                                     n_patom, t_type);
    } else {
        if (t_type == 2) {
            transpose_compact_fc3_type2(fc3, p2s, s2pp, nsym_list, perms,
                                        n_satom, n_patom);
        }
    }
}

static void rotate_delta_fc2s(double (*rot_delta_fc2s)[3][3], const long i_atom,
                              const long j_atom,
                              const double (*delta_fc2s)[3][3],
                              const double (*site_sym_cart)[3][3],
                              const long *rot_map_sym, const long num_atom,
                              const long num_site_sym, const long num_disp) {
    long i, j;

    for (i = 0; i < num_disp; i++) {
        for (j = 0; j < num_site_sym; j++) {
            tensor2_rotation(
                rot_delta_fc2s[i * num_site_sym + j],
                delta_fc2s[i * num_atom * num_atom +
                           rot_map_sym[j * num_atom + i_atom] * num_atom +
                           rot_map_sym[j * num_atom + j_atom]],
                site_sym_cart[j]);
        }
    }
}

static void tensor2_rotation(double rot_tensor[3][3], const double tensor[3][3],
                             const double r[3][3]) {
    long i, j, k, l;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            rot_tensor[i][j] = 0;
        }
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                for (l = 0; l < 3; l++) {
                    rot_tensor[i][j] += r[i][k] * r[j][l] * tensor[k][l];
                }
            }
        }
    }
}

static void tensor3_rotation(double *rot_tensor, const double *tensor,
                             const double *rot_cartesian) {
    long l;

    for (l = 0; l < 27; l++) {
        rot_tensor[l] = tensor3_rotation_elem(tensor, rot_cartesian, l);
    }
}

static double tensor3_rotation_elem(const double *tensor, const double *r,
                                    const long pos) {
    long i, j, k, l, m, n;
    double sum;

    l = pos / 9;
    m = (pos % 9) / 3;
    n = pos % 3;

    sum = 0.0;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                sum += r[l * 3 + i] * r[m * 3 + j] * r[n * 3 + k] *
                       tensor[i * 9 + j * 3 + k];
            }
        }
    }
    return sum;
}

static void copy_permutation_symmetry_fc3_elem(double *fc3,
                                               const double fc3_elem[27],
                                               const long a, const long b,
                                               const long c,
                                               const long num_atom) {
    long i, j, k;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                fc3[a * num_atom * num_atom * 27 + b * num_atom * 27 + c * 27 +
                    i * 9 + j * 3 + k] = fc3_elem[i * 9 + j * 3 + k];
                fc3[a * num_atom * num_atom * 27 + c * num_atom * 27 + b * 27 +
                    i * 9 + k * 3 + j] = fc3_elem[i * 9 + j * 3 + k];
                fc3[b * num_atom * num_atom * 27 + a * num_atom * 27 + c * 27 +
                    j * 9 + i * 3 + k] = fc3_elem[i * 9 + j * 3 + k];
                fc3[b * num_atom * num_atom * 27 + c * num_atom * 27 + a * 27 +
                    j * 9 + k * 3 + i] = fc3_elem[i * 9 + j * 3 + k];
                fc3[c * num_atom * num_atom * 27 + a * num_atom * 27 + b * 27 +
                    k * 9 + i * 3 + j] = fc3_elem[i * 9 + j * 3 + k];
                fc3[c * num_atom * num_atom * 27 + b * num_atom * 27 + a * 27 +
                    k * 9 + j * 3 + i] = fc3_elem[i * 9 + j * 3 + k];
            }
        }
    }
}

static void set_permutation_symmetry_fc3_elem(double *fc3_elem,
                                              const double *fc3, const long a,
                                              const long b, const long c,
                                              const long num_atom) {
    long i, j, k;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                fc3_elem[i * 9 + j * 3 + k] =
                    (fc3[a * num_atom * num_atom * 27 + b * num_atom * 27 +
                         c * 27 + i * 9 + j * 3 + k] +
                     fc3[a * num_atom * num_atom * 27 + c * num_atom * 27 +
                         b * 27 + i * 9 + k * 3 + j] +
                     fc3[b * num_atom * num_atom * 27 + a * num_atom * 27 +
                         c * 27 + j * 9 + i * 3 + k] +
                     fc3[b * num_atom * num_atom * 27 + c * num_atom * 27 +
                         a * 27 + j * 9 + k * 3 + i] +
                     fc3[c * num_atom * num_atom * 27 + a * num_atom * 27 +
                         b * 27 + k * 9 + i * 3 + j] +
                     fc3[c * num_atom * num_atom * 27 + b * num_atom * 27 +
                         a * 27 + k * 9 + j * 3 + i]) /
                    6;
            }
        }
    }
}

static void set_permutation_symmetry_compact_fc3(
    double *fc3, const long p2s[], const long s2pp[], const long nsym_list[],
    const long perms[], const long n_satom, const long n_patom) {
    /* fc3 shape=(n_patom, n_satom, n_satom, 3, 3, 3)          */
    /* 1D indexing:                                            */
    /*     i * n_satom * n_satom * 27 + j * n_satom * 27 +     */
    /*     k * 27 + l * 9 + m * 3 + n                          */
    long i, j, k, l, m, n, i_p, j_p, k_p;
    long done_any;
    long i_trans_j, k_trans_j, i_trans_k, j_trans_k;
    long adrs[6];
    double fc3_elem[3][3][3];
    char *done;

    done = NULL;
    done = (char *)malloc(sizeof(char) * n_patom * n_satom * n_satom);
    for (i = 0; i < n_patom * n_satom * n_satom; i++) {
        done[i] = 0;
    }

    for (i_p = 0; i_p < n_patom; i_p++) {
        i = p2s[i_p];
        for (j = 0; j < n_satom; j++) {
            j_p = s2pp[j];
            i_trans_j = perms[nsym_list[j] * n_satom + i];
            for (k = 0; k < n_satom; k++) {
                k_p = s2pp[k];
                k_trans_j = perms[nsym_list[j] * n_satom + k];
                i_trans_k = perms[nsym_list[k] * n_satom + i];
                j_trans_k = perms[nsym_list[k] * n_satom + j];

                /* ijk, ikj, jik, jki, kij, kji */
                adrs[0] = i_p * n_satom * n_satom + j * n_satom + k;
                adrs[1] = i_p * n_satom * n_satom + k * n_satom + j;
                adrs[2] =
                    j_p * n_satom * n_satom + i_trans_j * n_satom + k_trans_j;
                adrs[3] =
                    j_p * n_satom * n_satom + k_trans_j * n_satom + i_trans_j;
                adrs[4] =
                    k_p * n_satom * n_satom + i_trans_k * n_satom + j_trans_k;
                adrs[5] =
                    k_p * n_satom * n_satom + j_trans_k * n_satom + i_trans_k;

                done_any = 0;
                for (l = 0; l < 6; l++) {
                    if (done[adrs[l]]) {
                        done_any = 1;
                        break;
                    }
                }
                if (done_any) {
                    continue;
                }

                for (l = 0; l < 6; l++) {
                    done[adrs[l]] = 1;
                    adrs[l] *= 27;
                }

                for (l = 0; l < 3; l++) {
                    for (m = 0; m < 3; m++) {
                        for (n = 0; n < 3; n++) {
                            fc3_elem[l][m][n] =
                                fc3[adrs[0] + l * 9 + m * 3 + n];
                            fc3_elem[l][m][n] +=
                                fc3[adrs[1] + l * 9 + n * 3 + m];
                            fc3_elem[l][m][n] +=
                                fc3[adrs[2] + m * 9 + l * 3 + n];
                            fc3_elem[l][m][n] +=
                                fc3[adrs[3] + m * 9 + n * 3 + l];
                            fc3_elem[l][m][n] +=
                                fc3[adrs[4] + n * 9 + l * 3 + m];
                            fc3_elem[l][m][n] +=
                                fc3[adrs[5] + n * 9 + m * 3 + l];
                            fc3_elem[l][m][n] /= 6;
                        }
                    }
                }
                for (l = 0; l < 3; l++) {
                    for (m = 0; m < 3; m++) {
                        for (n = 0; n < 3; n++) {
                            fc3[adrs[0] + l * 9 + m * 3 + n] =
                                fc3_elem[l][m][n];
                            fc3[adrs[1] + l * 9 + n * 3 + m] =
                                fc3_elem[l][m][n];
                            fc3[adrs[2] + m * 9 + l * 3 + n] =
                                fc3_elem[l][m][n];
                            fc3[adrs[3] + m * 9 + n * 3 + l] =
                                fc3_elem[l][m][n];
                            fc3[adrs[4] + n * 9 + l * 3 + m] =
                                fc3_elem[l][m][n];
                            fc3[adrs[5] + n * 9 + m * 3 + l] =
                                fc3_elem[l][m][n];
                        }
                    }
                }
            }
        }
    }

    free(done);
    done = NULL;
}

static void transpose_compact_fc3_type01(double *fc3, const long p2s[],
                                         const long s2pp[],
                                         const long nsym_list[],
                                         const long perms[], const long n_satom,
                                         const long n_patom,
                                         const long t_type) {
    /* Three types of index permutations                       */
    /*     t_type=0: dim[0] <-> dim[1]                         */
    /*     t_type=1: dim[0] <-> dim[2]                         */
    /*     t_type=2: dim[1] <-> dim[2]                         */
    long i, j, k, l, m, n, i_p, j_p, i_trans, k_trans;
    long adrs, adrs_t;
    double fc3_elem[3][3][3];
    char *done;

    done = NULL;
    done = (char *)malloc(sizeof(char) * n_satom * n_patom);
    for (i = 0; i < n_satom * n_patom; i++) {
        done[i] = 0;
    }

    for (i_p = 0; i_p < n_patom; i_p++) {
        i = p2s[i_p];
        for (j = 0; j < n_satom; j++) {
            j_p = s2pp[j];
            if (!done[i_p * n_satom + j]) {
                /* (j, i) -- nsym_list[j] --> (j', i') */
                /* nsym_list[j] translates j to j' where j' is in */
                /* primitive cell. The same translation sends i to i' */
                /* where i' is not necessarily to be in primitive cell. */
                /* Thus, i' = perms[nsym_list[j] * n_satom + i] */
                i_trans = perms[nsym_list[j] * n_satom + i];
                done[i_p * n_satom + j] = 1;
                done[j_p * n_satom + i_trans] = 1;
                for (k = 0; k < n_satom; k++) {
                    k_trans = perms[nsym_list[j] * n_satom + k];

                    switch (t_type) {
                        case 0:
                            adrs = (i_p * n_satom * n_satom + j * n_satom + k) *
                                   27;
                            adrs_t = (j_p * n_satom * n_satom +
                                      i_trans * n_satom + k_trans) *
                                     27;
                            for (l = 0; l < 3; l++) {
                                for (m = 0; m < 3; m++) {
                                    for (n = 0; n < 3; n++) {
                                        fc3_elem[l][m][n] =
                                            fc3[adrs + l * 9 + m * 3 + n];
                                    }
                                }
                            }
                            if (adrs != adrs_t) {
                                for (l = 0; l < 3; l++) {
                                    for (m = 0; m < 3; m++) {
                                        for (n = 0; n < 3; n++) {
                                            fc3[adrs + l * 9 + m * 3 + n] =
                                                fc3[adrs_t + m * 9 + l * 3 + n];
                                        }
                                    }
                                }
                            }
                            for (l = 0; l < 3; l++) {
                                for (m = 0; m < 3; m++) {
                                    for (n = 0; n < 3; n++) {
                                        fc3[adrs_t + m * 9 + l * 3 + n] =
                                            fc3_elem[l][m][n];
                                    }
                                }
                            }
                            break;
                        case 1:
                            adrs = (i_p * n_satom * n_satom + k * n_satom + j) *
                                   27;
                            adrs_t = (j_p * n_satom * n_satom +
                                      k_trans * n_satom + i_trans) *
                                     27;
                            for (l = 0; l < 3; l++) {
                                for (m = 0; m < 3; m++) {
                                    for (n = 0; n < 3; n++) {
                                        fc3_elem[l][m][n] =
                                            fc3[adrs + l * 9 + m * 3 + n];
                                    }
                                }
                            }
                            if (adrs != adrs_t) {
                                for (l = 0; l < 3; l++) {
                                    for (m = 0; m < 3; m++) {
                                        for (n = 0; n < 3; n++) {
                                            fc3[adrs + l * 9 + m * 3 + n] =
                                                fc3[adrs_t + n * 9 + m * 3 + l];
                                        }
                                    }
                                }
                            }
                            for (l = 0; l < 3; l++) {
                                for (m = 0; m < 3; m++) {
                                    for (n = 0; n < 3; n++) {
                                        fc3[adrs_t + n * 9 + m * 3 + l] =
                                            fc3_elem[l][m][n];
                                    }
                                }
                            }
                            break;
                    } /* end switch */
                }
            }
        }
    }

    free(done);
    done = NULL;
}

static void transpose_compact_fc3_type2(double *fc3, const long p2s[],
                                        const long s2pp[],
                                        const long nsym_list[],
                                        const long perms[], const long n_satom,
                                        const long n_patom) {
    long j, k, l, m, n, i_p;
    long adrs, adrs_t;
    double fc3_elem[3][3][3];

    for (i_p = 0; i_p < n_patom; i_p++) {
        for (j = 0; j < n_satom; j++) {
            for (k = j; k < n_satom; k++) { /* k >= j */
                adrs = (i_p * n_satom * n_satom + j * n_satom + k) * 27;
                adrs_t = (i_p * n_satom * n_satom + k * n_satom + j) * 27;
                for (l = 0; l < 3; l++) {
                    for (m = 0; m < 3; m++) {
                        for (n = 0; n < 3; n++) {
                            fc3_elem[l][m][n] = fc3[adrs + l * 9 + m * 3 + n];
                        }
                    }
                }
                if (k != j) {
                    for (l = 0; l < 3; l++) {
                        for (m = 0; m < 3; m++) {
                            for (n = 0; n < 3; n++) {
                                fc3[adrs + l * 9 + m * 3 + n] =
                                    fc3[adrs_t + l * 9 + n * 3 + m];
                            }
                        }
                    }
                }
                for (l = 0; l < 3; l++) {
                    for (m = 0; m < 3; m++) {
                        for (n = 0; n < 3; n++) {
                            fc3[adrs_t + l * 9 + n * 3 + m] = fc3_elem[l][m][n];
                        }
                    }
                }
            }
        }
    }
}
