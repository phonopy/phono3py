/* SPDX-License-Identifier: BSD-3-Clause */

#include "phononcalc.h"

#include <stdint.h>

#include "lapack_wrapper.h"
#include "phonon.h"

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
    const char uplo) {
    if (!dd_q0) {
        phn_get_phonons_at_gridpoints(
            frequencies, (lapack_complex_double *)eigenvectors, phonon_done,
            num_phonons, grid_points, num_grid_points, grid_address, QDinv, fc2,
            svecs_fc2, multi_fc2, num_patom, num_satom, masses_fc2, p2s_fc2,
            s2p_fc2, unit_conversion_factor, born, dielectric,
            reciprocal_lattice, q_direction, nac_factor, uplo);
    } else {
        phn_get_gonze_phonons_at_gridpoints(
            frequencies, (lapack_complex_double *)eigenvectors, phonon_done,
            num_phonons, grid_points, num_grid_points, grid_address, QDinv, fc2,
            svecs_fc2, multi_fc2, positions_fc2, num_patom, num_satom,
            masses_fc2, p2s_fc2, s2p_fc2, unit_conversion_factor, born,
            dielectric, reciprocal_lattice, q_direction, nac_factor, dd_q0,
            G_list, num_G_points, lambda, uplo);
    }
}
