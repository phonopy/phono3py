/* Copyright (C) 2021 Atsushi Togo */
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
#include <numpy/arrayobject.h>

// #include "lapack_wrapper.h"
#include "phononcalc.h"

static PyObject *py_get_phonons_at_gridpoints(PyObject *self, PyObject *args);

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

static PyMethodDef _phononcalc_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {"phonons_at_gridpoints", py_get_phonons_at_gridpoints, METH_VARARGS,
     "Set phonons at grid points"},
    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3

static int _phononcalc_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _phononcalc_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,       "_phononcalc",       NULL,
    sizeof(struct module_state), _phononcalc_methods, NULL,
    _phononcalc_traverse,        _phononcalc_clear,   NULL};

#define INITERROR return NULL

PyObject *PyInit__phononcalc(void)
#else
#define INITERROR return

void init_phononcalc(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_phononcalc", _phononcalc_methods);
#endif
    struct module_state *st;

    if (module == NULL) INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("_phononcalc.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static PyObject *py_get_phonons_at_gridpoints(PyObject *self, PyObject *args) {
    PyArrayObject *py_frequencies;
    PyArrayObject *py_eigenvectors;
    PyArrayObject *py_phonon_done;
    PyArrayObject *py_grid_points;
    PyArrayObject *py_grid_address;
    PyArrayObject *py_QDinv;
    PyArrayObject *py_shortest_vectors_fc2;
    PyArrayObject *py_multiplicity_fc2;
    PyArrayObject *py_positions_fc2;
    PyArrayObject *py_fc2;
    PyArrayObject *py_masses_fc2;
    PyArrayObject *py_p2s_map_fc2;
    PyArrayObject *py_s2p_map_fc2;
    PyArrayObject *py_reciprocal_lattice;
    PyArrayObject *py_born_effective_charge;
    PyArrayObject *py_q_direction;
    PyArrayObject *py_dielectric_constant;
    PyArrayObject *py_dd_q0;
    PyArrayObject *py_G_list;
    double nac_factor;
    double unit_conversion_factor;
    double lambda;
    char *uplo;

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

    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOOOdOOOOdOOds", &py_frequencies, &py_eigenvectors,
            &py_phonon_done, &py_grid_points, &py_grid_address, &py_QDinv,
            &py_fc2, &py_shortest_vectors_fc2, &py_multiplicity_fc2,
            &py_positions_fc2, &py_masses_fc2, &py_p2s_map_fc2, &py_s2p_map_fc2,
            &unit_conversion_factor, &py_born_effective_charge,
            &py_dielectric_constant, &py_reciprocal_lattice, &py_q_direction,
            &nac_factor, &py_dd_q0, &py_G_list, &lambda, &uplo)) {
        return NULL;
    }

    freqs = (double *)PyArray_DATA(py_frequencies);
    eigvecs = (_lapack_complex_double *)PyArray_DATA(py_eigenvectors);
    phonon_done = (char *)PyArray_DATA(py_phonon_done);
    grid_points = (long *)PyArray_DATA(py_grid_points);
    grid_address = (long(*)[3])PyArray_DATA(py_grid_address);
    QDinv = (double(*)[3])PyArray_DATA(py_QDinv);
    fc2 = (double *)PyArray_DATA(py_fc2);
    svecs_fc2 = (double(*)[3])PyArray_DATA(py_shortest_vectors_fc2);
    multi_fc2 = (long(*)[2])PyArray_DATA(py_multiplicity_fc2);
    masses_fc2 = (double *)PyArray_DATA(py_masses_fc2);
    p2s_fc2 = (long *)PyArray_DATA(py_p2s_map_fc2);
    s2p_fc2 = (long *)PyArray_DATA(py_s2p_map_fc2);
    rec_lat = (double(*)[3])PyArray_DATA(py_reciprocal_lattice);
    num_patom = (long)PyArray_DIMS(py_multiplicity_fc2)[1];
    num_satom = (long)PyArray_DIMS(py_multiplicity_fc2)[0];
    num_phonons = (long)PyArray_DIMS(py_frequencies)[0];
    num_grid_points = (long)PyArray_DIMS(py_grid_points)[0];
    if ((PyObject *)py_born_effective_charge == Py_None) {
        born = NULL;
    } else {
        born = (double(*)[3][3])PyArray_DATA(py_born_effective_charge);
    }
    if ((PyObject *)py_dielectric_constant == Py_None) {
        dielectric = NULL;
    } else {
        dielectric = (double(*)[3])PyArray_DATA(py_dielectric_constant);
    }
    if ((PyObject *)py_q_direction == Py_None) {
        q_dir = NULL;
    } else {
        q_dir = (double *)PyArray_DATA(py_q_direction);
        if (fabs(q_dir[0]) < 1e-10 && fabs(q_dir[1]) < 1e-10 &&
            fabs(q_dir[2]) < 1e-10) {
            q_dir = NULL;
        }
    }
    if ((PyObject *)py_dd_q0 == Py_None) {
        dd_q0 = NULL;
    } else {
        dd_q0 = (double(*)[2])PyArray_DATA(py_dd_q0);
    }
    if ((PyObject *)py_G_list == Py_None) {
        G_list = NULL;
        num_G_points = 0;
    } else {
        G_list = (double(*)[3])PyArray_DATA(py_G_list);
        num_G_points = (long)PyArray_DIMS(py_G_list)[0];
    }
    if ((PyObject *)py_positions_fc2 == Py_None) {
        positions_fc2 = NULL;
    } else {
        positions_fc2 = (double(*)[3])PyArray_DATA(py_positions_fc2);
    }

    phcalc_get_phonons_at_gridpoints(
        freqs, eigvecs, phonon_done, num_phonons, grid_points, num_grid_points,
        grid_address, QDinv, fc2, svecs_fc2, multi_fc2, positions_fc2,
        num_patom, num_satom, masses_fc2, p2s_fc2, s2p_fc2,
        unit_conversion_factor, born, dielectric, rec_lat, q_dir, nac_factor,
        dd_q0, G_list, num_G_points, lambda, uplo[0]);

    Py_RETURN_NONE;
}
