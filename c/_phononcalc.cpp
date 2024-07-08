#include <math.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "phononcalc.h"

namespace nb = nanobind;

void py_get_phonons_at_gridpoints(
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_phonon_done, nb::ndarray<> py_grid_points,
    nb::ndarray<> py_grid_address, nb::ndarray<> py_QDinv, nb::ndarray<> py_fc2,
    nb::ndarray<> py_shortest_vectors_fc2, nb::ndarray<> py_multiplicity_fc2,
    nb::ndarray<> py_positions_fc2, nb::ndarray<> py_masses_fc2,
    nb::ndarray<> py_p2s_map_fc2, nb::ndarray<> py_s2p_map_fc2,
    double unit_conversion_factor, nb::ndarray<> py_born_effective_charge,
    nb::ndarray<> py_dielectric_constant, nb::ndarray<> py_reciprocal_lattice,
    nb::ndarray<> py_q_direction, double nac_factor, nb::ndarray<> py_dd_q0,
    nb::ndarray<> py_G_list, double lambda, long is_nac, long is_nac_q_zero,
    long use_GL_NAC, const char *uplo) {
    double(*born)[3][3];
    double(*dielectric)[3];
    double *q_dir;
    double *freqs;
    _lapack_complex_double *eigvecs;
    char *phonon_done;
    long *grid_points;
    long(*grid_address)[3];
    double(*QDinv)[3];
    double *fc2;
    double(*svecs_fc2)[3];
    long(*multi_fc2)[2];
    double(*positions_fc2)[3];
    double *masses_fc2;
    long *p2s_fc2;
    long *s2p_fc2;
    double(*rec_lat)[3];
    double(*dd_q0)[2];
    double(*G_list)[3];
    long num_patom, num_satom, num_phonons, num_grid_points, num_G_points;

    freqs = (double *)py_frequencies.data();
    eigvecs = (_lapack_complex_double *)py_eigenvectors.data();
    phonon_done = (char *)py_phonon_done.data();
    grid_points = (long *)py_grid_points.data();
    grid_address = (long(*)[3])py_grid_address.data();
    QDinv = (double(*)[3])py_QDinv.data();
    fc2 = (double *)py_fc2.data();
    svecs_fc2 = (double(*)[3])py_shortest_vectors_fc2.data();
    multi_fc2 = (long(*)[2])py_multiplicity_fc2.data();
    masses_fc2 = (double *)py_masses_fc2.data();
    p2s_fc2 = (long *)py_p2s_map_fc2.data();
    s2p_fc2 = (long *)py_s2p_map_fc2.data();
    rec_lat = (double(*)[3])py_reciprocal_lattice.data();
    num_patom = (long)py_multiplicity_fc2.shape(1);
    num_satom = (long)py_multiplicity_fc2.shape(0);
    num_phonons = (long)py_frequencies.shape(0);
    num_grid_points = (long)py_grid_points.shape(0);

    if (is_nac) {
        born = (double(*)[3][3])py_born_effective_charge.data();
        dielectric = (double(*)[3])py_dielectric_constant.data();
    } else {
        born = NULL;
        dielectric = NULL;
    }

    if (is_nac_q_zero) {
        q_dir = (double *)py_q_direction.data();
        if (fabs(q_dir[0]) < 1e-10 && fabs(q_dir[1]) < 1e-10 &&
            fabs(q_dir[2]) < 1e-10) {
            q_dir = NULL;
        }
    } else {
        q_dir = NULL;
    }

    if (use_GL_NAC) {
        dd_q0 = (double(*)[2])py_dd_q0.data();
        G_list = (double(*)[3])py_G_list.data();
        num_G_points = (long)py_G_list.shape(0);
        positions_fc2 = (double(*)[3])py_positions_fc2.data();
    } else {
        dd_q0 = NULL;
        G_list = NULL;
        num_G_points = 0;
        positions_fc2 = NULL;
    }

    phcalc_get_phonons_at_gridpoints(
        freqs, eigvecs, phonon_done, num_phonons, grid_points, num_grid_points,
        grid_address, QDinv, fc2, svecs_fc2, multi_fc2, positions_fc2,
        num_patom, num_satom, masses_fc2, p2s_fc2, s2p_fc2,
        unit_conversion_factor, born, dielectric, rec_lat, q_dir, nac_factor,
        dd_q0, G_list, num_G_points, lambda, uplo[0]);
}

NB_MODULE(_phononcalc, m) {
    m.def("phonons_at_gridpoints", py_get_phonons_at_gridpoints);
}
