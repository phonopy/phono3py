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

#include <Python.h>
#include <assert.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// #include "lapack_wrapper.h"
#include "phono3py.h"
#include "phonoc_array.h"

static PyObject *py_get_interaction(PyObject *self, PyObject *args);
static PyObject *py_get_pp_collision(PyObject *self, PyObject *args);
static PyObject *py_get_pp_collision_with_sigma(PyObject *self, PyObject *args);
static PyObject *py_get_imag_self_energy_with_g(PyObject *self, PyObject *args);
static PyObject *py_get_detailed_imag_self_energy_with_g(PyObject *self,
                                                         PyObject *args);
static PyObject *py_get_real_self_energy_at_bands(PyObject *self,
                                                  PyObject *args);
static PyObject *py_get_real_self_energy_at_frequency_point(PyObject *self,
                                                            PyObject *args);
static PyObject *py_get_collision_matrix(PyObject *self, PyObject *args);
static PyObject *py_get_reducible_collision_matrix(PyObject *self,
                                                   PyObject *args);
static PyObject *py_symmetrize_collision_matrix(PyObject *self, PyObject *args);
static PyObject *py_expand_collision_matrix(PyObject *self, PyObject *args);
static PyObject *py_distribute_fc3(PyObject *self, PyObject *args);
static PyObject *py_rotate_delta_fc2s(PyObject *self, PyObject *args);
static PyObject *py_get_isotope_strength(PyObject *self, PyObject *args);
static PyObject *py_get_thm_isotope_strength(PyObject *self, PyObject *args);
static PyObject *py_get_permutation_symmetry_fc3(PyObject *self,
                                                 PyObject *args);
static PyObject *py_get_permutation_symmetry_compact_fc3(PyObject *self,
                                                         PyObject *args);
static PyObject *py_transpose_compact_fc3(PyObject *self, PyObject *args);
static PyObject *py_get_thm_relative_grid_address(PyObject *self,
                                                  PyObject *args);
static PyObject *py_get_neighboring_grid_points(PyObject *self, PyObject *args);
static PyObject *py_get_thm_integration_weights_at_grid_points(PyObject *self,
                                                               PyObject *args);
static PyObject *py_tpl_get_triplets_reciprocal_mesh_at_q(PyObject *self,
                                                          PyObject *args);
static PyObject *py_tpl_get_BZ_triplets_at_q(PyObject *self, PyObject *args);
static PyObject *py_get_triplets_integration_weights(PyObject *self,
                                                     PyObject *args);
static PyObject *py_get_triplets_integration_weights_with_sigma(PyObject *self,
                                                                PyObject *args);
static PyObject *py_get_grid_index_from_address(PyObject *self, PyObject *args);
static PyObject *py_get_gr_grid_addresses(PyObject *self, PyObject *args);
static PyObject *py_get_reciprocal_rotations(PyObject *self, PyObject *args);
static PyObject *py_transform_rotations(PyObject *self, PyObject *args);
static PyObject *py_get_snf3x3(PyObject *self, PyObject *args);
static PyObject *py_get_ir_grid_map(PyObject *self, PyObject *args);
static PyObject *py_get_bz_grid_addresses(PyObject *self, PyObject *args);
static PyObject *py_rotate_bz_grid_addresses(PyObject *self, PyObject *args);
static PyObject *py_diagonalize_collision_matrix(PyObject *self,
                                                 PyObject *args);
static PyObject *py_pinv_from_eigensolution(PyObject *self, PyObject *args);
static PyObject *py_get_default_colmat_solver(PyObject *self, PyObject *args);
static PyObject *py_lapacke_pinv(PyObject *self, PyObject *args);
static PyObject *py_get_omp_max_threads(PyObject *self, PyObject *args);

static void show_colmat_info(const PyArrayObject *collision_matrix_py,
                             const long i_sigma, const long i_temp,
                             const long adrs_shift);
static Larray *convert_to_larray(PyArrayObject *npyary);
static Darray *convert_to_darray(PyArrayObject *npyary);

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state *)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef _phono3py_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {"interaction", (PyCFunction)py_get_interaction, METH_VARARGS,
     "Interaction of triplets"},
    {"pp_collision", (PyCFunction)py_get_pp_collision, METH_VARARGS,
     "Collision and ph-ph calculation"},
    {"pp_collision_with_sigma", (PyCFunction)py_get_pp_collision_with_sigma,
     METH_VARARGS, "Collision and ph-ph calculation for smearing method"},
    {"imag_self_energy_with_g", (PyCFunction)py_get_imag_self_energy_with_g,
     METH_VARARGS, "Imaginary part of self energy at frequency points with g"},
    {"detailed_imag_self_energy_with_g",
     (PyCFunction)py_get_detailed_imag_self_energy_with_g, METH_VARARGS,
     "Detailed contribution to imaginary part of self energy at frequency "
     "points with g"},
    {"real_self_energy_at_bands", (PyCFunction)py_get_real_self_energy_at_bands,
     METH_VARARGS, "Real part of self energy from third order force constants"},
    {"real_self_energy_at_frequency_point",
     (PyCFunction)py_get_real_self_energy_at_frequency_point, METH_VARARGS,
     "Real part of self energy from third order force constants at a frequency "
     "point"},
    {"collision_matrix", (PyCFunction)py_get_collision_matrix, METH_VARARGS,
     "Collision matrix with g"},
    {"reducible_collision_matrix",
     (PyCFunction)py_get_reducible_collision_matrix, METH_VARARGS,
     "Collision matrix with g for reducible grid points"},
    {"symmetrize_collision_matrix", (PyCFunction)py_symmetrize_collision_matrix,
     METH_VARARGS, "Symmetrize collision matrix"},
    {"expand_collision_matrix", (PyCFunction)py_expand_collision_matrix,
     METH_VARARGS, "Expand collision matrix"},
    {"distribute_fc3", (PyCFunction)py_distribute_fc3, METH_VARARGS,
     "Distribute least fc3 to full fc3"},
    {"rotate_delta_fc2s", (PyCFunction)py_rotate_delta_fc2s, METH_VARARGS,
     "Rotate delta fc2s"},
    {"isotope_strength", (PyCFunction)py_get_isotope_strength, METH_VARARGS,
     "Isotope scattering strength"},
    {"thm_isotope_strength", (PyCFunction)py_get_thm_isotope_strength,
     METH_VARARGS, "Isotope scattering strength for tetrahedron_method"},
    {"permutation_symmetry_fc3", (PyCFunction)py_get_permutation_symmetry_fc3,
     METH_VARARGS, "Set permutation symmetry for fc3"},
    {"permutation_symmetry_compact_fc3",
     (PyCFunction)py_get_permutation_symmetry_compact_fc3, METH_VARARGS,
     "Set permutation symmetry for compact-fc3"},
    {"transpose_compact_fc3", (PyCFunction)py_transpose_compact_fc3,
     METH_VARARGS, "Transpose compact fc3"},
    {"tetrahedra_relative_grid_address",
     (PyCFunction)py_get_thm_relative_grid_address, METH_VARARGS,
     "Relative grid addresses of vertices of 24 tetrahedra"},
    {"neighboring_grid_points", (PyCFunction)py_get_neighboring_grid_points,
     METH_VARARGS, "Neighboring grid points by relative grid addresses"},
    {"integration_weights_at_grid_points",
     (PyCFunction)py_get_thm_integration_weights_at_grid_points, METH_VARARGS,
     "Integration weights of tetrahedron method at grid points"},
    {"triplets_reciprocal_mesh_at_q",
     (PyCFunction)py_tpl_get_triplets_reciprocal_mesh_at_q, METH_VARARGS,
     "Triplets on reciprocal mesh points at a specific q-point"},
    {"BZ_triplets_at_q", (PyCFunction)py_tpl_get_BZ_triplets_at_q, METH_VARARGS,
     "Triplets in reciprocal primitive lattice are transformed to those in "
     "BZ."},
    {"triplets_integration_weights",
     (PyCFunction)py_get_triplets_integration_weights, METH_VARARGS,
     "Integration weights of tetrahedron method for triplets"},
    {"triplets_integration_weights_with_sigma",
     (PyCFunction)py_get_triplets_integration_weights_with_sigma, METH_VARARGS,
     "Integration weights of smearing method for triplets"},
    {"grid_index_from_address", (PyCFunction)py_get_grid_index_from_address,
     METH_VARARGS, "Grid index from grid address"},
    {"ir_grid_map", (PyCFunction)py_get_ir_grid_map, METH_VARARGS,
     "Reciprocal mesh points with ir grid mapping table"},
    {"gr_grid_addresses", (PyCFunction)py_get_gr_grid_addresses, METH_VARARGS,
     "Get generalized regular grid addresses"},
    {"reciprocal_rotations", (PyCFunction)py_get_reciprocal_rotations,
     METH_VARARGS, "Return rotation matrices in reciprocal space"},
    {"transform_rotations", (PyCFunction)py_transform_rotations, METH_VARARGS,
     "Transform rotations to those in generalized regular grid"},
    {"snf3x3", (PyCFunction)py_get_snf3x3, METH_VARARGS,
     "Get Smith formal form for 3x3 integer matrix"},
    {"bz_grid_addresses", (PyCFunction)py_get_bz_grid_addresses, METH_VARARGS,
     "Get grid addresses including Brillouin zone surface"},
    {"rotate_bz_grid_index", (PyCFunction)py_rotate_bz_grid_addresses,
     METH_VARARGS, "Rotate grid point considering Brillouin zone surface"},
    {"diagonalize_collision_matrix",
     (PyCFunction)py_diagonalize_collision_matrix, METH_VARARGS,
     "Diagonalize and optionally pseudo-inverse using Lapack dsyev(d)"},
    {"pinv_from_eigensolution", (PyCFunction)py_pinv_from_eigensolution,
     METH_VARARGS, "Pseudo-inverse from eigensolution"},
    {"default_colmat_solver", (PyCFunction)py_get_default_colmat_solver,
     METH_VARARGS, "Return default collison matrix solver by integer value"},
    {"lapacke_pinv", (PyCFunction)py_lapacke_pinv, METH_VARARGS,
     "Pseudo inversion using lapacke."},
    {"omp_max_threads", py_get_omp_max_threads, METH_VARARGS,
     "Return openmp max number of threads. Return 0 unless openmp is "
     "activated. "},
    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3

static int _phono3py_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _phono3py_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,       "_phono3py",       NULL,
    sizeof(struct module_state), _phono3py_methods, NULL,
    _phono3py_traverse,          _phono3py_clear,   NULL};

#define INITERROR return NULL

PyObject *PyInit__phono3py(void)
#else
#define INITERROR return

void init_phono3py(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_phono3py", _phono3py_methods);
#endif
    struct module_state *st;

    if (module == NULL) INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("_phono3py.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static PyObject *py_get_interaction(PyObject *self, PyObject *args) {
    PyArrayObject *py_fc3_normal_squared;
    PyArrayObject *py_g_zero;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_eigenvectors;
    PyArrayObject *py_triplets;
    PyArrayObject *py_bz_grid_addresses;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_Q;
    PyArrayObject *py_svecs;
    PyArrayObject *py_multi;
    PyArrayObject *py_fc3;
    PyArrayObject *py_masses;
    PyArrayObject *py_p2s_map;
    PyArrayObject *py_s2p_map;
    PyArrayObject *py_band_indices;
    PyArrayObject *py_all_shortest;
    double cutoff_frequency;
    long symmetrize_fc3_q;
    long make_r0_average;
    long openmp_per_triplets;

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

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOllOdl", &py_fc3_normal_squared,
                          &py_g_zero, &py_frequencies, &py_eigenvectors,
                          &py_triplets, &py_bz_grid_addresses, &py_D_diag,
                          &py_Q, &py_fc3, &py_svecs, &py_multi, &py_masses,
                          &py_p2s_map, &py_s2p_map, &py_band_indices,
                          &symmetrize_fc3_q, &make_r0_average, &py_all_shortest,
                          &cutoff_frequency, &openmp_per_triplets)) {
        return NULL;
    }

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    freqs = convert_to_darray(py_frequencies);
    /* npy_cdouble and lapack_complex_double may not be compatible. */
    /* So eigenvectors should not be used in Python side */
    eigvecs = (_lapack_complex_double *)PyArray_DATA(py_eigenvectors);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    num_triplets = (long)PyArray_DIMS(py_triplets)[0];
    g_zero = (char *)PyArray_DATA(py_g_zero);
    bz_grid_addresses = (long(*)[3])PyArray_DATA(py_bz_grid_addresses);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    Q = (long(*)[3])PyArray_DATA(py_Q);
    fc3 = (double *)PyArray_DATA(py_fc3);
    if (PyArray_DIMS(py_fc3)[0] == PyArray_DIMS(py_fc3)[1]) {
        is_compact_fc3 = 0;
    } else {
        is_compact_fc3 = 1;
    }
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    for (i = 0; i < 2; i++) {
        multi_dims[i] = PyArray_DIMS(py_multi)[i];
    }
    multi = (long(*)[2])PyArray_DATA(py_multi);
    masses = (double *)PyArray_DATA(py_masses);
    p2s = (long *)PyArray_DATA(py_p2s_map);
    s2p = (long *)PyArray_DATA(py_s2p_map);
    band_indices = (long *)PyArray_DATA(py_band_indices);
    all_shortest = (char *)PyArray_DATA(py_all_shortest);

    ph3py_get_interaction(fc3_normal_squared, g_zero, freqs, eigvecs, triplets,
                          num_triplets, bz_grid_addresses, D_diag, Q, fc3,
                          is_compact_fc3, svecs, multi_dims, multi, masses, p2s,
                          s2p, band_indices, symmetrize_fc3_q, make_r0_average,
                          all_shortest, cutoff_frequency, openmp_per_triplets);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
    free(freqs);
    freqs = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_pp_collision(PyObject *self, PyObject *args) {
    PyArrayObject *py_gamma;
    PyArrayObject *py_relative_grid_address;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_eigenvectors;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplet_weights;
    PyArrayObject *py_bz_grid_addresses;
    PyArrayObject *py_bz_map;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_Q;
    PyArrayObject *py_fc3;
    PyArrayObject *py_svecs;
    PyArrayObject *py_multi;
    PyArrayObject *py_masses;
    PyArrayObject *py_p2s_map;
    PyArrayObject *py_s2p_map;
    PyArrayObject *py_band_indices;
    PyArrayObject *py_temperatures;
    PyArrayObject *py_all_shortest;
    double cutoff_frequency;
    long is_NU;
    long symmetrize_fc3_q;
    long make_r0_average;
    long bz_grid_type;
    long openmp_per_triplets;

    double *gamma;
    long(*relative_grid_address)[4][3];
    double *frequencies;
    _lapack_complex_double *eigenvectors;
    long(*triplets)[3];
    long num_triplets;
    long *triplet_weights;
    long(*bz_grid_addresses)[3];
    long *bz_map;
    long *D_diag;
    long(*Q)[3];
    double *fc3;
    double(*svecs)[3];
    long(*multi)[2];
    double *masses;
    long *p2s;
    long *s2p;
    Larray *band_indices;
    Darray *temperatures;
    char *all_shortest;
    long multi_dims[2];
    long i;
    long is_compact_fc3;

    if (!PyArg_ParseTuple(
            args, "OOOOOOOOlOOOOOOOOOOlllOdl", &py_gamma,
            &py_relative_grid_address, &py_frequencies, &py_eigenvectors,
            &py_triplets, &py_triplet_weights, &py_bz_grid_addresses,
            &py_bz_map, &bz_grid_type, &py_D_diag, &py_Q, &py_fc3, &py_svecs,
            &py_multi, &py_masses, &py_p2s_map, &py_s2p_map, &py_band_indices,
            &py_temperatures, &is_NU, &symmetrize_fc3_q, &make_r0_average,
            &py_all_shortest, &cutoff_frequency, &openmp_per_triplets)) {
        return NULL;
    }

    gamma = (double *)PyArray_DATA(py_gamma);
    relative_grid_address =
        (long(*)[4][3])PyArray_DATA(py_relative_grid_address);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    eigenvectors = (_lapack_complex_double *)PyArray_DATA(py_eigenvectors);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    num_triplets = (long)PyArray_DIMS(py_triplets)[0];
    triplet_weights = (long *)PyArray_DATA(py_triplet_weights);
    bz_grid_addresses = (long(*)[3])PyArray_DATA(py_bz_grid_addresses);
    bz_map = (long *)PyArray_DATA(py_bz_map);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    Q = (long(*)[3])PyArray_DATA(py_Q);
    fc3 = (double *)PyArray_DATA(py_fc3);
    if (PyArray_DIMS(py_fc3)[0] == PyArray_DIMS(py_fc3)[1]) {
        is_compact_fc3 = 0;
    } else {
        is_compact_fc3 = 1;
    }
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    for (i = 0; i < 2; i++) {
        multi_dims[i] = PyArray_DIMS(py_multi)[i];
    }
    multi = (long(*)[2])PyArray_DATA(py_multi);
    masses = (double *)PyArray_DATA(py_masses);
    p2s = (long *)PyArray_DATA(py_p2s_map);
    s2p = (long *)PyArray_DATA(py_s2p_map);
    band_indices = convert_to_larray(py_band_indices);
    temperatures = convert_to_darray(py_temperatures);
    all_shortest = (char *)PyArray_DATA(py_all_shortest);

    ph3py_get_pp_collision(
        gamma, relative_grid_address, frequencies, eigenvectors, triplets,
        num_triplets, triplet_weights, bz_grid_addresses, bz_map, bz_grid_type,
        D_diag, Q, fc3, is_compact_fc3, svecs, multi_dims, multi, masses, p2s,
        s2p, band_indices, temperatures, is_NU, symmetrize_fc3_q,
        make_r0_average, all_shortest, cutoff_frequency, openmp_per_triplets);

    free(band_indices);
    band_indices = NULL;
    free(temperatures);
    temperatures = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_pp_collision_with_sigma(PyObject *self,
                                                PyObject *args) {
    PyArrayObject *py_gamma;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_eigenvectors;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplet_weights;
    PyArrayObject *py_bz_grid_addresses;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_Q;
    PyArrayObject *py_fc3;
    PyArrayObject *py_svecs;
    PyArrayObject *py_multi;
    PyArrayObject *py_masses;
    PyArrayObject *py_p2s_map;
    PyArrayObject *py_s2p_map;
    PyArrayObject *py_band_indices;
    PyArrayObject *py_temperatures;
    PyArrayObject *py_all_shortest;
    long is_NU;
    long symmetrize_fc3_q;
    double sigma;
    double sigma_cutoff;
    long make_r0_average;
    double cutoff_frequency;
    long openmp_per_triplets;

    double *gamma;
    double *frequencies;
    _lapack_complex_double *eigenvectors;
    long(*triplets)[3];
    long num_triplets;
    long *triplet_weights;
    long(*bz_grid_addresses)[3];
    long *D_diag;
    long(*Q)[3];
    double *fc3;
    double(*svecs)[3];
    long(*multi)[2];
    double *masses;
    long *p2s;
    long *s2p;
    Larray *band_indices;
    Darray *temperatures;
    char *all_shortest;
    long multi_dims[2];
    long i;
    long is_compact_fc3;

    if (!PyArg_ParseTuple(
            args, "OddOOOOOOOOOOOOOOOlllOdl", &py_gamma, &sigma, &sigma_cutoff,
            &py_frequencies, &py_eigenvectors, &py_triplets,
            &py_triplet_weights, &py_bz_grid_addresses, &py_D_diag, &py_Q,
            &py_fc3, &py_svecs, &py_multi, &py_masses, &py_p2s_map, &py_s2p_map,
            &py_band_indices, &py_temperatures, &is_NU, &symmetrize_fc3_q,
            &make_r0_average, &py_all_shortest, &cutoff_frequency,
            &openmp_per_triplets)) {
        return NULL;
    }

    gamma = (double *)PyArray_DATA(py_gamma);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    eigenvectors = (_lapack_complex_double *)PyArray_DATA(py_eigenvectors);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    num_triplets = (long)PyArray_DIMS(py_triplets)[0];
    triplet_weights = (long *)PyArray_DATA(py_triplet_weights);
    bz_grid_addresses = (long(*)[3])PyArray_DATA(py_bz_grid_addresses);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    Q = (long(*)[3])PyArray_DATA(py_Q);
    fc3 = (double *)PyArray_DATA(py_fc3);
    if (PyArray_DIMS(py_fc3)[0] == PyArray_DIMS(py_fc3)[1]) {
        is_compact_fc3 = 0;
    } else {
        is_compact_fc3 = 1;
    }
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    for (i = 0; i < 2; i++) {
        multi_dims[i] = PyArray_DIMS(py_multi)[i];
    }
    multi = (long(*)[2])PyArray_DATA(py_multi);
    masses = (double *)PyArray_DATA(py_masses);
    p2s = (long *)PyArray_DATA(py_p2s_map);
    s2p = (long *)PyArray_DATA(py_s2p_map);
    band_indices = convert_to_larray(py_band_indices);
    temperatures = convert_to_darray(py_temperatures);
    all_shortest = (char *)PyArray_DATA(py_all_shortest);

    ph3py_get_pp_collision_with_sigma(
        gamma, sigma, sigma_cutoff, frequencies, eigenvectors, triplets,
        num_triplets, triplet_weights, bz_grid_addresses, D_diag, Q, fc3,
        is_compact_fc3, svecs, multi_dims, multi, masses, p2s, s2p,
        band_indices, temperatures, is_NU, symmetrize_fc3_q, make_r0_average,
        all_shortest, cutoff_frequency, openmp_per_triplets);

    free(band_indices);
    band_indices = NULL;
    free(temperatures);
    temperatures = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_imag_self_energy_with_g(PyObject *self,
                                                PyObject *args) {
    PyArrayObject *py_gamma;
    PyArrayObject *py_fc3_normal_squared;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplet_weights;
    PyArrayObject *py_g;
    PyArrayObject *py_g_zero;
    double cutoff_frequency, temperature;
    long frequency_point_index;

    Darray *fc3_normal_squared;
    double *gamma;
    double *g;
    char *g_zero;
    double *frequencies;
    long(*triplets)[3];
    long *triplet_weights;
    long num_frequency_points;

    if (!PyArg_ParseTuple(args, "OOOOOdOOdl", &py_gamma, &py_fc3_normal_squared,
                          &py_triplets, &py_triplet_weights, &py_frequencies,
                          &temperature, &py_g, &py_g_zero, &cutoff_frequency,
                          &frequency_point_index)) {
        return NULL;
    }

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    gamma = (double *)PyArray_DATA(py_gamma);
    g = (double *)PyArray_DATA(py_g);
    g_zero = (char *)PyArray_DATA(py_g_zero);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    triplet_weights = (long *)PyArray_DATA(py_triplet_weights);
    num_frequency_points = (long)PyArray_DIMS(py_g)[2];

    ph3py_get_imag_self_energy_at_bands_with_g(
        gamma, fc3_normal_squared, frequencies, triplets, triplet_weights, g,
        g_zero, temperature, cutoff_frequency, num_frequency_points,
        frequency_point_index);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_detailed_imag_self_energy_with_g(PyObject *self,
                                                         PyObject *args) {
    PyArrayObject *py_gamma_detail;
    PyArrayObject *py_gamma_N;
    PyArrayObject *py_gamma_U;
    PyArrayObject *py_fc3_normal_squared;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplet_weights;
    PyArrayObject *py_bz_grid_addresses;
    PyArrayObject *py_g;
    PyArrayObject *py_g_zero;
    double cutoff_frequency, temperature;

    Darray *fc3_normal_squared;
    double *gamma_detail;
    double *gamma_N;
    double *gamma_U;
    double *g;
    char *g_zero;
    double *frequencies;
    long(*triplets)[3];
    long *triplet_weights;
    long(*bz_grid_addresses)[3];

    if (!PyArg_ParseTuple(args, "OOOOOOOOdOOd", &py_gamma_detail, &py_gamma_N,
                          &py_gamma_U, &py_fc3_normal_squared, &py_triplets,
                          &py_triplet_weights, &py_bz_grid_addresses,
                          &py_frequencies, &temperature, &py_g, &py_g_zero,
                          &cutoff_frequency)) {
        return NULL;
    }

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    gamma_detail = (double *)PyArray_DATA(py_gamma_detail);
    gamma_N = (double *)PyArray_DATA(py_gamma_N);
    gamma_U = (double *)PyArray_DATA(py_gamma_U);
    g = (double *)PyArray_DATA(py_g);
    g_zero = (char *)PyArray_DATA(py_g_zero);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    triplet_weights = (long *)PyArray_DATA(py_triplet_weights);
    bz_grid_addresses = (long(*)[3])PyArray_DATA(py_bz_grid_addresses);

    ph3py_get_detailed_imag_self_energy_at_bands_with_g(
        gamma_detail, gamma_N, gamma_U, fc3_normal_squared, frequencies,
        triplets, triplet_weights, bz_grid_addresses, g, g_zero, temperature,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_real_self_energy_at_bands(PyObject *self,
                                                  PyObject *args) {
    PyArrayObject *py_shift;
    PyArrayObject *py_fc3_normal_squared;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplet_weights;
    PyArrayObject *py_band_indices;
    double epsilon, unit_conversion_factor, cutoff_frequency, temperature;

    Darray *fc3_normal_squared;
    double *shift;
    double *frequencies;
    long *band_indices;
    long(*triplets)[3];
    long *triplet_weights;

    if (!PyArg_ParseTuple(args, "OOOOOOdddd", &py_shift, &py_fc3_normal_squared,
                          &py_triplets, &py_triplet_weights, &py_frequencies,
                          &py_band_indices, &temperature, &epsilon,
                          &unit_conversion_factor, &cutoff_frequency)) {
        return NULL;
    }

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    shift = (double *)PyArray_DATA(py_shift);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    band_indices = (long *)PyArray_DATA(py_band_indices);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    triplet_weights = (long *)PyArray_DATA(py_triplet_weights);

    ph3py_get_real_self_energy_at_bands(
        shift, fc3_normal_squared, band_indices, frequencies, triplets,
        triplet_weights, epsilon, temperature, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_real_self_energy_at_frequency_point(PyObject *self,
                                                            PyObject *args) {
    PyArrayObject *py_shift;
    PyArrayObject *py_fc3_normal_squared;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplet_weights;
    PyArrayObject *py_band_indices;
    double frequency_point, epsilon, unit_conversion_factor, cutoff_frequency;
    double temperature;

    Darray *fc3_normal_squared;
    double *shift;
    double *frequencies;
    long *band_indices;
    long(*triplets)[3];
    long *triplet_weights;

    if (!PyArg_ParseTuple(args, "OdOOOOOdddd", &py_shift, &frequency_point,
                          &py_fc3_normal_squared, &py_triplets,
                          &py_triplet_weights, &py_frequencies,
                          &py_band_indices, &temperature, &epsilon,
                          &unit_conversion_factor, &cutoff_frequency)) {
        return NULL;
    }

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    shift = (double *)PyArray_DATA(py_shift);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    band_indices = (long *)PyArray_DATA(py_band_indices);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    triplet_weights = (long *)PyArray_DATA(py_triplet_weights);

    ph3py_get_real_self_energy_at_frequency_point(
        shift, frequency_point, fc3_normal_squared, band_indices, frequencies,
        triplets, triplet_weights, epsilon, temperature, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_collision_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_collision_matrix;
    PyArrayObject *py_fc3_normal_squared;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplets_map;
    PyArrayObject *py_map_q;
    PyArrayObject *py_g;
    PyArrayObject *py_rotated_grid_points;
    PyArrayObject *py_rotations_cartesian;
    double temperature, unit_conversion_factor, cutoff_frequency;

    Darray *fc3_normal_squared;
    double *collision_matrix;
    double *g;
    double *frequencies;
    long(*triplets)[3];
    long *triplets_map;
    long *map_q;
    long *rotated_grid_points;
    long num_gp, num_ir_gp, num_rot;
    double *rotations_cartesian;

    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOddd", &py_collision_matrix, &py_fc3_normal_squared,
            &py_frequencies, &py_g, &py_triplets, &py_triplets_map, &py_map_q,
            &py_rotated_grid_points, &py_rotations_cartesian, &temperature,
            &unit_conversion_factor, &cutoff_frequency)) {
        return NULL;
    }

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    collision_matrix = (double *)PyArray_DATA(py_collision_matrix);
    g = (double *)PyArray_DATA(py_g);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    triplets_map = (long *)PyArray_DATA(py_triplets_map);
    num_gp = (long)PyArray_DIMS(py_triplets_map)[0];
    map_q = (long *)PyArray_DATA(py_map_q);
    rotated_grid_points = (long *)PyArray_DATA(py_rotated_grid_points);
    num_ir_gp = (long)PyArray_DIMS(py_rotated_grid_points)[0];
    num_rot = (long)PyArray_DIMS(py_rotated_grid_points)[1];
    rotations_cartesian = (double *)PyArray_DATA(py_rotations_cartesian);

    assert(num_rot == PyArray_DIMS(py_rotations_cartesian)[0]);
    assert(num_gp == PyArray_DIMS(py_frequencies)[0]);

    ph3py_get_collision_matrix(collision_matrix, fc3_normal_squared,
                               frequencies, triplets, triplets_map, map_q,
                               rotated_grid_points, rotations_cartesian, g,
                               num_ir_gp, num_gp, num_rot, temperature,
                               unit_conversion_factor, cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_get_reducible_collision_matrix(PyObject *self,
                                                   PyObject *args) {
    PyArrayObject *py_collision_matrix;
    PyArrayObject *py_fc3_normal_squared;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_triplets;
    PyArrayObject *py_triplets_map;
    PyArrayObject *py_map_q;
    PyArrayObject *py_g;
    double temperature, unit_conversion_factor, cutoff_frequency;

    Darray *fc3_normal_squared;
    double *collision_matrix;
    double *g;
    double *frequencies;
    long(*triplets)[3];
    long *triplets_map;
    long num_gp;
    long *map_q;

    if (!PyArg_ParseTuple(
            args, "OOOOOOOddd", &py_collision_matrix, &py_fc3_normal_squared,
            &py_frequencies, &py_g, &py_triplets, &py_triplets_map, &py_map_q,
            &temperature, &unit_conversion_factor, &cutoff_frequency)) {
        return NULL;
    }

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    collision_matrix = (double *)PyArray_DATA(py_collision_matrix);
    g = (double *)PyArray_DATA(py_g);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    triplets_map = (long *)PyArray_DATA(py_triplets_map);
    num_gp = (long)PyArray_DIMS(py_triplets_map)[0];
    map_q = (long *)PyArray_DATA(py_map_q);

    ph3py_get_reducible_collision_matrix(
        collision_matrix, fc3_normal_squared, frequencies, triplets,
        triplets_map, map_q, g, num_gp, temperature, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;

    Py_RETURN_NONE;
}

static PyObject *py_symmetrize_collision_matrix(PyObject *self,
                                                PyObject *args) {
    PyArrayObject *py_collision_matrix;

    double *collision_matrix;
    long num_band, num_grid_points, num_temp, num_sigma;
    long num_column;

    if (!PyArg_ParseTuple(args, "O", &py_collision_matrix)) {
        return NULL;
    }

    collision_matrix = (double *)PyArray_DATA(py_collision_matrix);
    num_sigma = (long)PyArray_DIMS(py_collision_matrix)[0];
    num_temp = (long)PyArray_DIMS(py_collision_matrix)[1];
    num_grid_points = (long)PyArray_DIMS(py_collision_matrix)[2];
    num_band = (long)PyArray_DIMS(py_collision_matrix)[3];

    if (PyArray_NDIM(py_collision_matrix) == 8) {
        num_column = num_grid_points * num_band * 3;
    } else {
        num_column = num_grid_points * num_band;
    }

    ph3py_symmetrize_collision_matrix(collision_matrix, num_column, num_temp,
                                      num_sigma);

    Py_RETURN_NONE;
}

static PyObject *py_expand_collision_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_collision_matrix;
    PyArrayObject *py_ir_grid_points;
    PyArrayObject *py_rot_grid_points;

    double *collision_matrix;
    long *rot_grid_points;
    long *ir_grid_points;
    long num_band, num_grid_points, num_temp, num_sigma, num_rot, num_ir_gp;

    if (!PyArg_ParseTuple(args, "OOO", &py_collision_matrix, &py_ir_grid_points,
                          &py_rot_grid_points)) {
        return NULL;
    }

    collision_matrix = (double *)PyArray_DATA(py_collision_matrix);
    rot_grid_points = (long *)PyArray_DATA(py_rot_grid_points);
    ir_grid_points = (long *)PyArray_DATA(py_ir_grid_points);
    num_sigma = (long)PyArray_DIMS(py_collision_matrix)[0];
    num_temp = (long)PyArray_DIMS(py_collision_matrix)[1];
    num_grid_points = (long)PyArray_DIMS(py_collision_matrix)[2];
    num_band = (long)PyArray_DIMS(py_collision_matrix)[3];
    num_rot = (long)PyArray_DIMS(py_rot_grid_points)[0];
    num_ir_gp = (long)PyArray_DIMS(py_ir_grid_points)[0];

    ph3py_expand_collision_matrix(collision_matrix, rot_grid_points,
                                  ir_grid_points, num_ir_gp, num_grid_points,
                                  num_rot, num_sigma, num_temp, num_band);

    Py_RETURN_NONE;
}

static PyObject *py_get_isotope_strength(PyObject *self, PyObject *args) {
    PyArrayObject *py_gamma;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_eigenvectors;
    PyArrayObject *py_band_indices;
    PyArrayObject *py_mass_variances;
    PyArrayObject *py_ir_grid_points;
    PyArrayObject *py_weights;

    long grid_point;
    double cutoff_frequency;
    double sigma;

    double *gamma;
    double *frequencies;
    long *ir_grid_points;
    double *weights;
    _lapack_complex_double *eigenvectors;
    long *band_indices;
    double *mass_variances;
    long num_band, num_band0, num_ir_grid_points;

    if (!PyArg_ParseTuple(args, "OlOOOOOOdd", &py_gamma, &grid_point,
                          &py_ir_grid_points, &py_weights, &py_mass_variances,
                          &py_frequencies, &py_eigenvectors, &py_band_indices,
                          &sigma, &cutoff_frequency)) {
        return NULL;
    }

    gamma = (double *)PyArray_DATA(py_gamma);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    eigenvectors = (_lapack_complex_double *)PyArray_DATA(py_eigenvectors);
    ir_grid_points = (long *)PyArray_DATA(py_ir_grid_points);
    weights = (double *)PyArray_DATA(py_weights);
    band_indices = (long *)PyArray_DATA(py_band_indices);
    mass_variances = (double *)PyArray_DATA(py_mass_variances);
    num_band = (long)PyArray_DIMS(py_frequencies)[1];
    num_band0 = (long)PyArray_DIMS(py_band_indices)[0];
    num_ir_grid_points = (long)PyArray_DIMS(py_ir_grid_points)[0];

    ph3py_get_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        eigenvectors, num_ir_grid_points, band_indices, num_band, num_band0,
        sigma, cutoff_frequency);

    Py_RETURN_NONE;
}

static PyObject *py_get_thm_isotope_strength(PyObject *self, PyObject *args) {
    PyArrayObject *py_gamma;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_eigenvectors;
    PyArrayObject *py_band_indices;
    PyArrayObject *py_mass_variances;
    PyArrayObject *py_ir_grid_points;
    PyArrayObject *py_weights;
    PyArrayObject *py_integration_weights;
    long grid_point;
    double cutoff_frequency;

    double *gamma;
    double *frequencies;
    long *ir_grid_points;
    double *weights;
    _lapack_complex_double *eigenvectors;
    long *band_indices;
    double *mass_variances;
    long num_band, num_band0, num_ir_grid_points;
    double *integration_weights;

    if (!PyArg_ParseTuple(args, "OlOOOOOOOd", &py_gamma, &grid_point,
                          &py_ir_grid_points, &py_weights, &py_mass_variances,
                          &py_frequencies, &py_eigenvectors, &py_band_indices,
                          &py_integration_weights, &cutoff_frequency)) {
        return NULL;
    }

    gamma = (double *)PyArray_DATA(py_gamma);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    ir_grid_points = (long *)PyArray_DATA(py_ir_grid_points);
    weights = (double *)PyArray_DATA(py_weights);
    eigenvectors = (_lapack_complex_double *)PyArray_DATA(py_eigenvectors);
    band_indices = (long *)PyArray_DATA(py_band_indices);
    mass_variances = (double *)PyArray_DATA(py_mass_variances);
    num_band = (long)PyArray_DIMS(py_frequencies)[1];
    num_band0 = (long)PyArray_DIMS(py_band_indices)[0];
    integration_weights = (double *)PyArray_DATA(py_integration_weights);
    num_ir_grid_points = (long)PyArray_DIMS(py_ir_grid_points)[0];

    ph3py_get_thm_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        eigenvectors, num_ir_grid_points, band_indices, num_band, num_band0,
        integration_weights, cutoff_frequency);

    Py_RETURN_NONE;
}

static PyObject *py_distribute_fc3(PyObject *self, PyObject *args) {
    PyArrayObject *force_constants_third;
    long target;
    long source;
    PyArrayObject *rotation_cart_inv;
    PyArrayObject *atom_mapping_py;

    double *fc3;
    double *rot_cart_inv;
    long *atom_mapping;
    long num_atom;

    if (!PyArg_ParseTuple(args, "OllOO", &force_constants_third, &target,
                          &source, &atom_mapping_py, &rotation_cart_inv)) {
        return NULL;
    }

    fc3 = (double *)PyArray_DATA(force_constants_third);
    rot_cart_inv = (double *)PyArray_DATA(rotation_cart_inv);
    atom_mapping = (long *)PyArray_DATA(atom_mapping_py);
    num_atom = (long)PyArray_DIMS(atom_mapping_py)[0];

    ph3py_distribute_fc3(fc3, target, source, atom_mapping, num_atom,
                         rot_cart_inv);

    Py_RETURN_NONE;
}

static PyObject *py_rotate_delta_fc2s(PyObject *self, PyObject *args) {
    PyArrayObject *py_fc3;
    PyArrayObject *py_delta_fc2s;
    PyArrayObject *py_inv_U;
    PyArrayObject *py_site_sym_cart;
    PyArrayObject *py_rot_map_syms;

    double(*fc3)[3][3][3];
    double(*delta_fc2s)[3][3];
    double *inv_U;
    double(*site_sym_cart)[3][3];
    long *rot_map_syms;
    long num_atom, num_disp, num_site_sym;

    if (!PyArg_ParseTuple(args, "OOOOO", &py_fc3, &py_delta_fc2s, &py_inv_U,
                          &py_site_sym_cart, &py_rot_map_syms)) {
        return NULL;
    }

    /* (num_atom, num_atom, 3, 3, 3) */
    fc3 = (double(*)[3][3][3])PyArray_DATA(py_fc3);
    /* (n_u1, num_atom, num_atom, 3, 3) */
    delta_fc2s = (double(*)[3][3])PyArray_DATA(py_delta_fc2s);
    /* (3, n_u1 * n_sym) */
    inv_U = (double *)PyArray_DATA(py_inv_U);
    /* (n_sym, 3, 3) */
    site_sym_cart = (double(*)[3][3])PyArray_DATA(py_site_sym_cart);
    /* (n_sym, natom) */
    rot_map_syms = (long *)PyArray_DATA(py_rot_map_syms);

    num_atom = (long)PyArray_DIMS(py_fc3)[0];
    num_disp = (long)PyArray_DIMS(py_delta_fc2s)[0];
    num_site_sym = (long)PyArray_DIMS(py_site_sym_cart)[0];

    ph3py_rotate_delta_fc2(fc3, delta_fc2s, inv_U, site_sym_cart, rot_map_syms,
                           num_atom, num_site_sym, num_disp);

    Py_RETURN_NONE;
}

static PyObject *py_get_permutation_symmetry_fc3(PyObject *self,
                                                 PyObject *args) {
    PyArrayObject *py_fc3;

    double *fc3;
    long num_atom;

    if (!PyArg_ParseTuple(args, "O", &py_fc3)) {
        return NULL;
    }

    fc3 = (double *)PyArray_DATA(py_fc3);
    num_atom = (long)PyArray_DIMS(py_fc3)[0];

    ph3py_get_permutation_symmetry_fc3(fc3, num_atom);

    Py_RETURN_NONE;
}

static PyObject *py_get_permutation_symmetry_compact_fc3(PyObject *self,
                                                         PyObject *args) {
    PyArrayObject *py_fc3;
    PyArrayObject *py_permutations;
    PyArrayObject *py_s2pp_map;
    PyArrayObject *py_p2s_map;
    PyArrayObject *py_nsym_list;

    double *fc3;
    long *s2pp;
    long *p2s;
    long *nsym_list;
    long *perms;
    long n_patom, n_satom;

    if (!PyArg_ParseTuple(args, "OOOOO", &py_fc3, &py_permutations,
                          &py_s2pp_map, &py_p2s_map, &py_nsym_list)) {
        return NULL;
    }

    fc3 = (double *)PyArray_DATA(py_fc3);
    perms = (long *)PyArray_DATA(py_permutations);
    s2pp = (long *)PyArray_DATA(py_s2pp_map);
    p2s = (long *)PyArray_DATA(py_p2s_map);
    nsym_list = (long *)PyArray_DATA(py_nsym_list);
    n_patom = (long)PyArray_DIMS(py_fc3)[0];
    n_satom = (long)PyArray_DIMS(py_fc3)[1];

    ph3py_get_permutation_symmetry_compact_fc3(fc3, p2s, s2pp, nsym_list, perms,
                                               n_satom, n_patom);

    Py_RETURN_NONE;
}

static PyObject *py_transpose_compact_fc3(PyObject *self, PyObject *args) {
    PyArrayObject *py_fc3;
    PyArrayObject *py_permutations;
    PyArrayObject *py_s2pp_map;
    PyArrayObject *py_p2s_map;
    PyArrayObject *py_nsym_list;
    long t_type;

    double *fc3;
    long *s2pp;
    long *p2s;
    long *nsym_list;
    long *perms;
    long n_patom, n_satom;

    if (!PyArg_ParseTuple(args, "OOOOOl", &py_fc3, &py_permutations,
                          &py_s2pp_map, &py_p2s_map, &py_nsym_list, &t_type)) {
        return NULL;
    }

    fc3 = (double *)PyArray_DATA(py_fc3);
    perms = (long *)PyArray_DATA(py_permutations);
    s2pp = (long *)PyArray_DATA(py_s2pp_map);
    p2s = (long *)PyArray_DATA(py_p2s_map);
    nsym_list = (long *)PyArray_DATA(py_nsym_list);
    n_patom = (long)PyArray_DIMS(py_fc3)[0];
    n_satom = (long)PyArray_DIMS(py_fc3)[1];

    ph3py_transpose_compact_fc3(fc3, p2s, s2pp, nsym_list, perms, n_satom,
                                n_patom, t_type);

    Py_RETURN_NONE;
}

static PyObject *py_get_thm_relative_grid_address(PyObject *self,
                                                  PyObject *args) {
    PyArrayObject *py_relative_grid_address;
    PyArrayObject *py_reciprocal_lattice_py;

    long(*relative_grid_address)[4][3];
    double(*reciprocal_lattice)[3];

    if (!PyArg_ParseTuple(args, "OO", &py_relative_grid_address,
                          &py_reciprocal_lattice_py)) {
        return NULL;
    }

    relative_grid_address =
        (long(*)[4][3])PyArray_DATA(py_relative_grid_address);
    reciprocal_lattice = (double(*)[3])PyArray_DATA(py_reciprocal_lattice_py);

    ph3py_get_relative_grid_address(relative_grid_address, reciprocal_lattice);

    Py_RETURN_NONE;
}

static PyObject *py_get_neighboring_grid_points(PyObject *self,
                                                PyObject *args) {
    PyArrayObject *py_relative_grid_points;
    PyArrayObject *py_grid_points;
    PyArrayObject *py_relative_grid_address;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_bz_grid_address;
    PyArrayObject *py_bz_map;
    long bz_grid_type;

    long *relative_grid_points;
    long *grid_points;
    long num_grid_points, num_relative_grid_address;
    long(*relative_grid_address)[3];
    long *D_diag;
    long(*bz_grid_address)[3];
    long *bz_map;

    if (!PyArg_ParseTuple(args, "OOOOOOl", &py_relative_grid_points,
                          &py_grid_points, &py_relative_grid_address,
                          &py_D_diag, &py_bz_grid_address, &py_bz_map,
                          &bz_grid_type)) {
        return NULL;
    }

    relative_grid_points = (long *)PyArray_DATA(py_relative_grid_points);
    grid_points = (long *)PyArray_DATA(py_grid_points);
    num_grid_points = (long)PyArray_DIMS(py_grid_points)[0];
    relative_grid_address = (long(*)[3])PyArray_DATA(py_relative_grid_address);
    num_relative_grid_address = (long)PyArray_DIMS(py_relative_grid_address)[0];
    D_diag = (long *)PyArray_DATA(py_D_diag);
    bz_grid_address = (long(*)[3])PyArray_DATA(py_bz_grid_address);
    bz_map = (long *)PyArray_DATA(py_bz_map);

    ph3py_get_neighboring_gird_points(
        relative_grid_points, grid_points, relative_grid_address, D_diag,
        bz_grid_address, bz_map, bz_grid_type, num_grid_points,
        num_relative_grid_address);

    Py_RETURN_NONE;
}

static PyObject *py_get_thm_integration_weights_at_grid_points(PyObject *self,
                                                               PyObject *args) {
    PyArrayObject *py_iw;
    PyArrayObject *py_frequency_points;
    PyArrayObject *py_relative_grid_address;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_grid_points;
    PyArrayObject *py_frequencies;
    PyArrayObject *py_bz_grid_address;
    PyArrayObject *py_gp2irgp_map;
    PyArrayObject *py_bz_map;
    long bz_grid_type;
    char *function;

    double *iw;
    double *frequency_points;
    long num_frequency_points, num_band, num_gp;
    long(*relative_grid_address)[4][3];
    long *D_diag;
    long *grid_points;
    long(*bz_grid_address)[3];
    long *bz_map;
    long *gp2irgp_map;
    double *frequencies;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOls", &py_iw, &py_frequency_points,
                          &py_relative_grid_address, &py_D_diag,
                          &py_grid_points, &py_frequencies, &py_bz_grid_address,
                          &py_bz_map, &py_gp2irgp_map, &bz_grid_type,
                          &function)) {
        return NULL;
    }

    iw = (double *)PyArray_DATA(py_iw);
    frequency_points = (double *)PyArray_DATA(py_frequency_points);
    num_frequency_points = (long)PyArray_DIMS(py_frequency_points)[0];
    relative_grid_address =
        (long(*)[4][3])PyArray_DATA(py_relative_grid_address);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    grid_points = (long *)PyArray_DATA(py_grid_points);
    num_gp = (long)PyArray_DIMS(py_grid_points)[0];
    bz_grid_address = (long(*)[3])PyArray_DATA(py_bz_grid_address);
    bz_map = (long *)PyArray_DATA(py_bz_map);
    gp2irgp_map = (long *)PyArray_DATA(py_gp2irgp_map);
    frequencies = (double *)PyArray_DATA(py_frequencies);
    num_band = (long)PyArray_DIMS(py_frequencies)[1];

    ph3py_get_thm_integration_weights_at_grid_points(
        iw, frequency_points, num_frequency_points, num_band, num_gp,
        relative_grid_address, D_diag, grid_points, bz_grid_address, bz_map,
        bz_grid_type, frequencies, gp2irgp_map, function[0]);

    Py_RETURN_NONE;
}

static PyObject *py_tpl_get_triplets_reciprocal_mesh_at_q(PyObject *self,
                                                          PyObject *args) {
    PyArrayObject *py_map_triplets;
    PyArrayObject *py_map_q;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_rotations;
    long fixed_grid_number;
    long is_time_reversal;
    long swappable;

    long *map_triplets;
    long *map_q;
    long *D_diag;
    long(*rot)[3][3];
    long num_rot;
    long num_ir;

    if (!PyArg_ParseTuple(args, "OOlOlOl", &py_map_triplets, &py_map_q,
                          &fixed_grid_number, &py_D_diag, &is_time_reversal,
                          &py_rotations, &swappable)) {
        return NULL;
    }

    map_triplets = (long *)PyArray_DATA(py_map_triplets);
    map_q = (long *)PyArray_DATA(py_map_q);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    rot = (long(*)[3][3])PyArray_DATA(py_rotations);
    num_rot = (long)PyArray_DIMS(py_rotations)[0];
    num_ir = ph3py_get_triplets_reciprocal_mesh_at_q(
        map_triplets, map_q, fixed_grid_number, D_diag, is_time_reversal,
        num_rot, rot, swappable);

    return PyLong_FromLong(num_ir);
}

static PyObject *py_tpl_get_BZ_triplets_at_q(PyObject *self, PyObject *args) {
    PyArrayObject *py_triplets;
    PyArrayObject *py_bz_grid_address;
    PyArrayObject *py_bz_map;
    PyArrayObject *py_map_triplets;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_Q;
    long grid_point;
    long bz_grid_type;

    long(*triplets)[3];
    long(*bz_grid_address)[3];
    long *bz_map;
    long *map_triplets;
    long num_map_triplets;
    long *D_diag;
    long(*Q)[3];
    long num_ir;

    if (!PyArg_ParseTuple(args, "OlOOOOOl", &py_triplets, &grid_point,
                          &py_bz_grid_address, &py_bz_map, &py_map_triplets,
                          &py_D_diag, &py_Q, &bz_grid_type)) {
        return NULL;
    }

    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    bz_grid_address = (long(*)[3])PyArray_DATA(py_bz_grid_address);
    bz_map = (long *)PyArray_DATA(py_bz_map);
    map_triplets = (long *)PyArray_DATA(py_map_triplets);
    num_map_triplets = (long)PyArray_DIMS(py_map_triplets)[0];
    D_diag = (long *)PyArray_DATA(py_D_diag);
    Q = (long(*)[3])PyArray_DATA(py_Q);

    num_ir = ph3py_get_BZ_triplets_at_q(triplets, grid_point, bz_grid_address,
                                        bz_map, map_triplets, num_map_triplets,
                                        D_diag, Q, bz_grid_type);

    return PyLong_FromLong(num_ir);
}

static PyObject *py_get_triplets_integration_weights(PyObject *self,
                                                     PyObject *args) {
    PyArrayObject *py_iw;
    PyArrayObject *py_iw_zero;
    PyArrayObject *py_frequency_points;
    PyArrayObject *py_relative_grid_address;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_triplets;
    PyArrayObject *py_frequencies1;
    PyArrayObject *py_frequencies2;
    PyArrayObject *py_bz_grid_addresses;
    PyArrayObject *py_bz_map;
    long bz_grid_type;
    long tp_type;

    double *iw;
    char *iw_zero;
    double *frequency_points;
    long(*relative_grid_address)[4][3];
    long *D_diag;
    long(*triplets)[3];
    long(*bz_grid_addresses)[3];
    long *bz_map;
    double *frequencies1, *frequencies2;
    long num_band0, num_band1, num_band2, num_triplets;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOll", &py_iw, &py_iw_zero,
                          &py_frequency_points, &py_relative_grid_address,
                          &py_D_diag, &py_triplets, &py_frequencies1,
                          &py_frequencies2, &py_bz_grid_addresses, &py_bz_map,
                          &bz_grid_type, &tp_type)) {
        return NULL;
    }

    iw = (double *)PyArray_DATA(py_iw);
    iw_zero = (char *)PyArray_DATA(py_iw_zero);
    frequency_points = (double *)PyArray_DATA(py_frequency_points);
    num_band0 = (long)PyArray_DIMS(py_frequency_points)[0];
    relative_grid_address =
        (long(*)[4][3])PyArray_DATA(py_relative_grid_address);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    num_triplets = (long)PyArray_DIMS(py_triplets)[0];
    bz_grid_addresses = (long(*)[3])PyArray_DATA(py_bz_grid_addresses);
    bz_map = (long *)PyArray_DATA(py_bz_map);
    frequencies1 = (double *)PyArray_DATA(py_frequencies1);
    frequencies2 = (double *)PyArray_DATA(py_frequencies2);
    num_band1 = (long)PyArray_DIMS(py_frequencies1)[1];
    num_band2 = (long)PyArray_DIMS(py_frequencies2)[1];

    ph3py_get_integration_weight(
        iw, iw_zero, frequency_points, num_band0, relative_grid_address, D_diag,
        triplets, num_triplets, bz_grid_addresses, bz_map, bz_grid_type,
        frequencies1, num_band1, frequencies2, num_band2, tp_type, 1);

    Py_RETURN_NONE;
}

static PyObject *py_get_triplets_integration_weights_with_sigma(
    PyObject *self, PyObject *args) {
    PyArrayObject *py_iw;
    PyArrayObject *py_iw_zero;
    PyArrayObject *py_frequency_points;
    PyArrayObject *py_triplets;
    PyArrayObject *py_frequencies;
    double sigma, sigma_cutoff;

    double *iw;
    char *iw_zero;
    double *frequency_points;
    long(*triplets)[3];
    double *frequencies;
    long num_band0, num_band, num_iw, num_triplets;

    if (!PyArg_ParseTuple(args, "OOOOOdd", &py_iw, &py_iw_zero,
                          &py_frequency_points, &py_triplets, &py_frequencies,
                          &sigma, &sigma_cutoff)) {
        return NULL;
    }

    iw = (double *)PyArray_DATA(py_iw);
    iw_zero = (char *)PyArray_DATA(py_iw_zero);
    frequency_points = (double *)PyArray_DATA(py_frequency_points);
    num_band0 = (long)PyArray_DIMS(py_frequency_points)[0];
    triplets = (long(*)[3])PyArray_DATA(py_triplets);
    num_triplets = (long)PyArray_DIMS(py_triplets)[0];
    frequencies = (double *)PyArray_DATA(py_frequencies);
    num_band = (long)PyArray_DIMS(py_frequencies)[1];
    num_iw = (long)PyArray_DIMS(py_iw)[0];

    ph3py_get_integration_weight_with_sigma(
        iw, iw_zero, sigma, sigma_cutoff, frequency_points, num_band0, triplets,
        num_triplets, frequencies, num_band, num_iw);

    Py_RETURN_NONE;
}

static PyObject *py_get_grid_index_from_address(PyObject *self,
                                                PyObject *args) {
    PyArrayObject *py_address;
    PyArrayObject *py_D_diag;

    long *address;
    long *D_diag;
    long gp;

    if (!PyArg_ParseTuple(args, "OO", &py_address, &py_D_diag)) {
        return NULL;
    }

    address = (long *)PyArray_DATA(py_address);
    D_diag = (long *)PyArray_DATA(py_D_diag);

    gp = ph3py_get_grid_index_from_address(address, D_diag);

    return PyLong_FromLong(gp);
}

static PyObject *py_get_gr_grid_addresses(PyObject *self, PyObject *args) {
    PyArrayObject *py_gr_grid_addresses;
    PyArrayObject *py_D_diag;

    long(*gr_grid_addresses)[3];
    long *D_diag;

    if (!PyArg_ParseTuple(args, "OO", &py_gr_grid_addresses, &py_D_diag)) {
        return NULL;
    }

    gr_grid_addresses = (long(*)[3])PyArray_DATA(py_gr_grid_addresses);
    D_diag = (long *)PyArray_DATA(py_D_diag);

    ph3py_get_gr_grid_addresses(gr_grid_addresses, D_diag);

    Py_RETURN_NONE;
}

static PyObject *py_get_reciprocal_rotations(PyObject *self, PyObject *args) {
    PyArrayObject *py_rec_rotations;
    PyArrayObject *py_rotations;
    long is_time_reversal;

    long(*rec_rotations)[3][3];
    long(*rotations)[3][3];
    long num_rot, num_rec_rot;

    if (!PyArg_ParseTuple(args, "OOl", &py_rec_rotations, &py_rotations,
                          &is_time_reversal)) {
        return NULL;
    }

    rec_rotations = (long(*)[3][3])PyArray_DATA(py_rec_rotations);
    rotations = (long(*)[3][3])PyArray_DATA(py_rotations);
    num_rot = (long)PyArray_DIMS(py_rotations)[0];

    num_rec_rot = ph3py_get_reciprocal_rotations(rec_rotations, rotations,
                                                 num_rot, is_time_reversal);

    return PyLong_FromLong(num_rec_rot);
}

static PyObject *py_transform_rotations(PyObject *self, PyObject *args) {
    PyArrayObject *py_transformed_rotations;
    PyArrayObject *py_rotations;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_Q;

    long(*transformed_rotations)[3][3];
    long(*rotations)[3][3];
    long *D_diag;
    long(*Q)[3];
    long num_rot, succeeded;

    if (!PyArg_ParseTuple(args, "OOOO", &py_transformed_rotations,
                          &py_rotations, &py_D_diag, &py_Q)) {
        return NULL;
    }

    transformed_rotations =
        (long(*)[3][3])PyArray_DATA(py_transformed_rotations);
    rotations = (long(*)[3][3])PyArray_DATA(py_rotations);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    Q = (long(*)[3])PyArray_DATA(py_Q);
    num_rot = (long)PyArray_DIMS(py_transformed_rotations)[0];

    succeeded = ph3py_transform_rotations(transformed_rotations, rotations,
                                          num_rot, D_diag, Q);
    if (succeeded) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject *py_get_snf3x3(PyObject *self, PyObject *args) {
    PyArrayObject *py_D_diag;
    PyArrayObject *py_P;
    PyArrayObject *py_Q;
    PyArrayObject *py_A;

    long *D_diag;
    long(*P)[3];
    long(*Q)[3];
    long(*A)[3];
    long succeeded;

    if (!PyArg_ParseTuple(args, "OOOO", &py_D_diag, &py_P, &py_Q, &py_A)) {
        return NULL;
    }

    D_diag = (long *)PyArray_DATA(py_D_diag);
    P = (long(*)[3])PyArray_DATA(py_P);
    Q = (long(*)[3])PyArray_DATA(py_Q);
    A = (long(*)[3])PyArray_DATA(py_A);

    succeeded = ph3py_get_snf3x3(D_diag, P, Q, A);
    if (succeeded) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject *py_get_ir_grid_map(PyObject *self, PyObject *args) {
    PyArrayObject *py_grid_mapping_table;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_is_shift;
    PyArrayObject *py_rotations;

    long *D_diag;
    long *is_shift;
    long(*rot)[3][3];
    long num_rot;

    long *grid_mapping_table;
    long num_ir;

    if (!PyArg_ParseTuple(args, "OOOO", &py_grid_mapping_table, &py_D_diag,
                          &py_is_shift, &py_rotations)) {
        return NULL;
    }

    D_diag = (long *)PyArray_DATA(py_D_diag);
    is_shift = (long *)PyArray_DATA(py_is_shift);
    rot = (long(*)[3][3])PyArray_DATA(py_rotations);
    num_rot = (long)PyArray_DIMS(py_rotations)[0];
    grid_mapping_table = (long *)PyArray_DATA(py_grid_mapping_table);

    num_ir = ph3py_get_ir_grid_map(grid_mapping_table, D_diag, is_shift, rot,
                                   num_rot);
    return PyLong_FromLong(num_ir);
}

static PyObject *py_get_bz_grid_addresses(PyObject *self, PyObject *args) {
    PyArrayObject *py_bz_grid_addresses;
    PyArrayObject *py_bz_map;
    PyArrayObject *py_bzg2grg;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_Q;
    PyArrayObject *py_PS;
    PyArrayObject *py_reciprocal_lattice;
    long type;

    long(*bz_grid_addresses)[3];
    long *bz_map;
    long *bzg2grg;
    long *D_diag;
    long(*Q)[3];
    long *PS;
    double(*reciprocal_lattice)[3];
    long num_total_gp;

    if (!PyArg_ParseTuple(args, "OOOOOOOl", &py_bz_grid_addresses, &py_bz_map,
                          &py_bzg2grg, &py_D_diag, &py_Q, &py_PS,
                          &py_reciprocal_lattice, &type)) {
        return NULL;
    }

    bz_grid_addresses = (long(*)[3])PyArray_DATA(py_bz_grid_addresses);
    bz_map = (long *)PyArray_DATA(py_bz_map);
    bzg2grg = (long *)PyArray_DATA(py_bzg2grg);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    Q = (long(*)[3])PyArray_DATA(py_Q);
    PS = (long *)PyArray_DATA(py_PS);
    reciprocal_lattice = (double(*)[3])PyArray_DATA(py_reciprocal_lattice);

    num_total_gp =
        ph3py_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg, D_diag,
                                    Q, PS, reciprocal_lattice, type);

    return PyLong_FromLong(num_total_gp);
}

static PyObject *py_rotate_bz_grid_addresses(PyObject *self, PyObject *args) {
    PyArrayObject *py_bz_grid_addresses;
    PyArrayObject *py_rotation;
    PyArrayObject *py_bz_map;
    PyArrayObject *py_D_diag;
    PyArrayObject *py_PS;
    long bz_grid_index;
    long type;

    long(*bz_grid_addresses)[3];
    long(*rotation)[3];
    long *bz_map;
    long *D_diag;
    long *PS;
    long ret_bz_gp;

    if (!PyArg_ParseTuple(args, "lOOOOOl", &bz_grid_index, &py_rotation,
                          &py_bz_grid_addresses, &py_bz_map, &py_D_diag, &py_PS,
                          &type)) {
        return NULL;
    }

    bz_grid_addresses = (long(*)[3])PyArray_DATA(py_bz_grid_addresses);
    rotation = (long(*)[3])PyArray_DATA(py_rotation);
    bz_map = (long *)PyArray_DATA(py_bz_map);
    D_diag = (long *)PyArray_DATA(py_D_diag);
    PS = (long *)PyArray_DATA(py_PS);

    ret_bz_gp = ph3py_rotate_bz_grid_index(
        bz_grid_index, rotation, bz_grid_addresses, bz_map, D_diag, PS, type);

    return PyLong_FromLong(ret_bz_gp);
}

static PyObject *py_diagonalize_collision_matrix(PyObject *self,
                                                 PyObject *args) {
    PyArrayObject *py_collision_matrix;
    PyArrayObject *py_eigenvalues;
    double cutoff;
    long i_sigma, i_temp, is_pinv, solver;

    double *collision_matrix;
    double *eigvals;
    long num_temp, num_grid_point, num_band;
    long num_column, adrs_shift;
    long info;

    if (!PyArg_ParseTuple(args, "OOlldll", &py_collision_matrix,
                          &py_eigenvalues, &i_sigma, &i_temp, &cutoff, &solver,
                          &is_pinv)) {
        return NULL;
    }

    collision_matrix = (double *)PyArray_DATA(py_collision_matrix);
    eigvals = (double *)PyArray_DATA(py_eigenvalues);

    if (PyArray_NDIM(py_collision_matrix) == 2) {
        num_temp = 1;
        num_column = (long)PyArray_DIM(py_collision_matrix, 1);
    } else {
        num_temp = (long)PyArray_DIM(py_collision_matrix, 1);
        num_grid_point = (long)PyArray_DIM(py_collision_matrix, 2);
        num_band = (long)PyArray_DIM(py_collision_matrix, 3);
        if (PyArray_NDIM(py_collision_matrix) == 8) {
            num_column = num_grid_point * num_band * 3;
        } else {
            num_column = num_grid_point * num_band;
        }
    }
    adrs_shift = (i_sigma * num_column * num_column * num_temp +
                  i_temp * num_column * num_column);

    /* show_colmat_info(py_collision_matrix, i_sigma, i_temp, adrs_shift); */

    info = ph3py_phonopy_dsyev(collision_matrix + adrs_shift, eigvals,
                               num_column, solver);
    if (is_pinv) {
        ph3py_pinv_from_eigensolution(collision_matrix + adrs_shift, eigvals,
                                      num_column, cutoff, 0);
    }

    return PyLong_FromLong(info);
}

static PyObject *py_pinv_from_eigensolution(PyObject *self, PyObject *args) {
    PyArrayObject *py_collision_matrix;
    PyArrayObject *py_eigenvalues;
    double cutoff;
    long i_sigma, i_temp, pinv_method;

    double *collision_matrix;
    double *eigvals;
    long num_temp, num_grid_point, num_band;
    long num_column, adrs_shift;

    if (!PyArg_ParseTuple(args, "OOlldl", &py_collision_matrix, &py_eigenvalues,
                          &i_sigma, &i_temp, &cutoff, &pinv_method)) {
        return NULL;
    }

    collision_matrix = (double *)PyArray_DATA(py_collision_matrix);
    eigvals = (double *)PyArray_DATA(py_eigenvalues);
    num_temp = (long)PyArray_DIMS(py_collision_matrix)[1];
    num_grid_point = (long)PyArray_DIMS(py_collision_matrix)[2];
    num_band = (long)PyArray_DIMS(py_collision_matrix)[3];

    if (PyArray_NDIM(py_collision_matrix) == 8) {
        num_column = num_grid_point * num_band * 3;
    } else {
        num_column = num_grid_point * num_band;
    }
    adrs_shift = (i_sigma * num_column * num_column * num_temp +
                  i_temp * num_column * num_column);

    /* show_colmat_info(py_collision_matrix, i_sigma, i_temp, adrs_shift); */

    ph3py_pinv_from_eigensolution(collision_matrix + adrs_shift, eigvals,
                                  num_column, cutoff, pinv_method);

    Py_RETURN_NONE;
}

static PyObject *py_get_default_colmat_solver(PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

#if defined(MKL_LAPACKE) || defined(SCIPY_MKL_H)
    return PyLong_FromLong((long)1);
#else
    return PyLong_FromLong((long)4);
#endif
}

static PyObject *py_lapacke_pinv(PyObject *self, PyObject *args) {
    PyArrayObject *data_in_py;
    PyArrayObject *data_out_py;
    double cutoff;

    int m;
    int n;
    double *data_in;
    double *data_out;
    int info;

    if (!PyArg_ParseTuple(args, "OOd", &data_out_py, &data_in_py, &cutoff)) {
        return NULL;
    }

    m = (long)PyArray_DIMS(data_in_py)[0];
    n = (long)PyArray_DIMS(data_in_py)[1];
    data_in = (double *)PyArray_DATA(data_in_py);
    data_out = (double *)PyArray_DATA(data_out_py);

    info = ph3py_phonopy_pinv(data_out, data_in, m, n, cutoff);

    return PyLong_FromLong((long)info);
}

static PyObject *py_get_omp_max_threads(PyObject *self, PyObject *args) {
    return PyLong_FromLong(ph3py_get_max_threads());
}

static void show_colmat_info(const PyArrayObject *py_collision_matrix,
                             const long i_sigma, const long i_temp,
                             const long adrs_shift) {
    long i;

    printf(" Array_shape:(");
    for (i = 0; i < PyArray_NDIM(py_collision_matrix); i++) {
        printf("%d", (int)PyArray_DIM(py_collision_matrix, i));
        if (i < PyArray_NDIM(py_collision_matrix) - 1) {
            printf(",");
        } else {
            printf("), ");
        }
    }
    printf("Data shift:%lu [%lu, %lu]\n", adrs_shift, i_sigma, i_temp);
}

/**
 * @brief Convert numpy "int_" array to phono3py long array structure.
 *
 * @param npyary
 * @return Larray*
 */
static Larray *convert_to_larray(PyArrayObject *npyary) {
    long i;
    Larray *ary;

    ary = (Larray *)malloc(sizeof(Larray));
    for (i = 0; i < PyArray_NDIM(npyary); i++) {
        ary->dims[i] = PyArray_DIMS(npyary)[i];
    }
    ary->data = (long *)PyArray_DATA(npyary);
    return ary;
}

/**
 * @brief Convert numpy "double" array to phono3py double array structure.
 *
 * @param npyary
 * @return Darray*
 * @note PyArray_NDIM receives non-const (PyArrayObject *).
 */
static Darray *convert_to_darray(PyArrayObject *npyary) {
    int i;
    Darray *ary;

    ary = (Darray *)malloc(sizeof(Darray));
    for (i = 0; i < PyArray_NDIM(npyary); i++) {
        ary->dims[i] = PyArray_DIMS(npyary)[i];
    }
    ary->data = (double *)PyArray_DATA(npyary);
    return ary;
}
