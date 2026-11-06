/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __phononcalc_H__
#define __phononcalc_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct {
    double re;
    double im;
} _lapack_complex_double;

void phcalc_get_phonons_at_gridpoints(
    double *frequencies, _lapack_complex_double *eigenvectors,
    char *phonon_done, const int64_t num_phonons, const int64_t *grid_points,
    const int64_t num_grid_points, const int64_t (*grid_address)[3],
    const double QDinv[3][3], const double *fc2, const double (*svecs_fc2)[3],
    const int64_t (*multi_fc2)[2], const double (*positions_fc2)[3],
    const int64_t num_patom, const int64_t num_satom, const double *masses_fc2,
    const int64_t *p2s_fc2, const int64_t *s2p_fc2,
    const double unit_conversion_factor, const double (*born)[3][3],
    const double dielectric[3][3], const double reciprocal_lattice[3][3],
    const double *q_direction, /* pointer */
    const double nac_factor, const double (*dd_q0)[2],
    const double (*G_list)[3], const int64_t num_G_points, const double lambda,
    const char uplo);

#ifdef __cplusplus
}
#endif

#endif
