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
#include <stdio.h>
#include <assert.h>
#include <numpy/arrayobject.h>
#include <lapack_wrapper.h>
#include <phonon.h>
#include <phonoc_array.h>

static PyObject * py_phonopy_pinv(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);

struct module_state {
  PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
  struct module_state *st = GETSTATE(m);
  PyErr_SetString(st->error, "something bad happened");
  return NULL;
}

static PyMethodDef _lapackepy_methods[] = {
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
  {"pinv", py_phonopy_pinv, METH_VARARGS, "Pseudo-inverse using Lapack dgesvd"},
  {"zheev", py_phonopy_zheev, METH_VARARGS, "Lapack zheev wrapper"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int _lapackepy_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(GETSTATE(m)->error);
  return 0;
}

static int _lapackepy_clear(PyObject *m) {
  Py_CLEAR(GETSTATE(m)->error);
  return 0;
}

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_lapackepy",
  NULL,
  sizeof(struct module_state),
  _lapackepy_methods,
  NULL,
  _lapackepy_traverse,
  _lapackepy_clear,
  NULL
};

#define INITERROR return NULL

PyObject *
PyInit__lapackepy(void)

#else
#define INITERROR return

  void
  init_lapackepy(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule("_lapackepy", _lapackepy_methods);
#endif
  struct module_state *st;

  if (module == NULL)
    INITERROR;
  st = GETSTATE(module);

  st->error = PyErr_NewException("_lapackepy.Error", NULL, NULL);
  if (st->error == NULL) {
    Py_DECREF(module);
    INITERROR;
  }

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}

static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args)
{
  PyArrayObject* dynamical_matrix;
  PyArrayObject* eigenvalues;

  int dimension;
  npy_cdouble *dynmat;
  double *eigvals;
  lapack_complex_double *a;
  int i, info;

  if (!PyArg_ParseTuple(args, "OO",
                        &dynamical_matrix,
                        &eigenvalues)) {
    return NULL;
  }

  dimension = (int)PyArray_DIMS(dynamical_matrix)[0];
  dynmat = (npy_cdouble*)PyArray_DATA(dynamical_matrix);
  eigvals = (double*)PyArray_DATA(eigenvalues);

  a = (lapack_complex_double*) malloc(sizeof(lapack_complex_double) *
                                      dimension * dimension);
  for (i = 0; i < dimension * dimension; i++) {
    a[i] = lapack_make_complex_double(dynmat[i].real, dynmat[i].imag);
  }

  info = phonopy_zheev(eigvals, a, dimension, 'L');

  for (i = 0; i < dimension * dimension; i++) {
    dynmat[i].real = lapack_complex_double_real(a[i]);
    dynmat[i].imag = lapack_complex_double_imag(a[i]);
  }

  free(a);

  return PyLong_FromLong((long) info);
}

static PyObject * py_phonopy_pinv(PyObject *self, PyObject *args)
{
  PyArrayObject* data_in_py;
  PyArrayObject* data_out_py;
  double cutoff;

  int m;
  int n;
  double *data_in;
  double *data_out;
  int info;

  if (!PyArg_ParseTuple(args, "OOd",
                        &data_out_py,
                        &data_in_py,
                        &cutoff)) {
    return NULL;
  }

  m = (int)PyArray_DIMS(data_in_py)[0];
  n = (int)PyArray_DIMS(data_in_py)[1];
  data_in = (double*)PyArray_DATA(data_in_py);
  data_out = (double*)PyArray_DATA(data_out_py);

  info = phonopy_pinv(data_out, data_in, m, n, cutoff);

  return PyLong_FromLong((long) info);
}
