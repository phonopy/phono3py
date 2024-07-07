#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "phono3py.h"
#include "phonoc_array.h"

namespace nb = nanobind;

static Larray *convert_to_larray(nb::ndarray<> npyary) {
    long i;
    Larray *ary;

    ary = (Larray *)malloc(sizeof(Larray));
    for (i = 0; i < npyary.ndim(); i++) {
        ary->dims[i] = npyary.shape(i);
    }
    ary->data = (long *)npyary.data();
    return ary;
}

static Darray *convert_to_darray(nb::ndarray<> npyary) {
    int i;
    Darray *ary;

    ary = (Darray *)malloc(sizeof(Darray));
    for (i = 0; i < npyary.ndim(); i++) {
        ary->dims[i] = npyary.shape(i);
    }
    ary->data = (double *)npyary.data();
    return ary;
}

void py_get_interaction(nb::ndarray<> py_fc3_normal_squared,
                        nb::ndarray<> py_g_zero, nb::ndarray<> py_frequencies,
                        nb::ndarray<> py_eigenvectors,
                        nb::ndarray<> py_triplets,
                        nb::ndarray<> py_bz_grid_addresses,
                        nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
                        nb::ndarray<> py_fc3, nb::ndarray<> py_svecs,
                        nb::ndarray<> py_multi, nb::ndarray<> py_masses,
                        nb::ndarray<> py_p2s_map, nb::ndarray<> py_s2p_map,
                        nb::ndarray<> py_band_indices, long symmetrize_fc3_q,
                        long make_r0_average, nb::ndarray<> py_all_shortest,
                        double cutoff_frequency, long openmp_per_triplets) {
    Darray *fc3_normal_squared;
    Darray *freqs;
    _lapack_complex_double *eigvecs;
    long(*triplets)[3];
    long num_triplets;
    char *g_zero;
    long(*bz_grid_addresses)[3];
    long *D_diag;
    long(*Q)[3];
    double *fc3;
    double(*svecs)[3];
    long(*multi)[2];
    double *masses;
    char *all_shortest;
    long *p2s;
    long *s2p;
    long *band_indices;
    long multi_dims[2];
    long i;
    long is_compact_fc3;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    freqs = convert_to_darray(py_frequencies);
    /* npy_cdouble and lapack_complex_double may not be compatible. */
    /* So eigenvectors should not be used in Python side */
    eigvecs = (_lapack_complex_double *)py_eigenvectors.data();
    triplets = (long(*)[3])py_triplets.data();
    num_triplets = (long)py_triplets.shape(0);
    g_zero = (char *)py_g_zero.data();
    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();
    D_diag = (long *)py_D_diag.data();
    Q = (long(*)[3])py_Q.data();
    fc3 = (double *)py_fc3.data();
    if (py_fc3.shape(0) == py_fc3.shape(1)) {
        is_compact_fc3 = 0;
    } else {
        is_compact_fc3 = 1;
    }
    svecs = (double(*)[3])py_svecs.data();
    for (i = 0; i < 2; i++) {
        multi_dims[i] = py_multi.shape(i);
    }
    multi = (long(*)[2])py_multi.data();
    masses = (double *)py_masses.data();
    p2s = (long *)py_p2s_map.data();
    s2p = (long *)py_s2p_map.data();
    band_indices = (long *)py_band_indices.data();
    all_shortest = (char *)py_all_shortest.data();

    ph3py_get_interaction(fc3_normal_squared, g_zero, freqs, eigvecs, triplets,
                          num_triplets, bz_grid_addresses, D_diag, Q, fc3,
                          is_compact_fc3, svecs, multi_dims, multi, masses, p2s,
                          s2p, band_indices, symmetrize_fc3_q, make_r0_average,
                          all_shortest, cutoff_frequency, openmp_per_triplets);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
    free(freqs);
    freqs = NULL;
}

NB_MODULE(_phonopy, m) { m.def("interaction", py_get_interaction); }
