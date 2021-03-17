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
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include "lapack_wrapper.h"
#include "phono3py.h"
#include "phonoc_array.h"


static PyObject * py_get_interaction(PyObject *self, PyObject *args);
static PyObject * py_get_pp_collision(PyObject *self, PyObject *args);
static PyObject *
py_get_pp_collision_with_sigma(PyObject *self, PyObject *args);
static PyObject *
py_get_imag_self_energy_with_g(PyObject *self, PyObject *args);
static PyObject *
py_get_detailed_imag_self_energy_with_g(PyObject *self, PyObject *args);
static PyObject * py_get_real_self_energy_at_bands(PyObject *self,
                                                   PyObject *args);
static PyObject * py_get_real_self_energy_at_frequency_point(PyObject *self,
                                                             PyObject *args);
static PyObject * py_get_collision_matrix(PyObject *self, PyObject *args);
static PyObject * py_get_reducible_collision_matrix(PyObject *self,
                                                    PyObject *args);
static PyObject * py_symmetrize_collision_matrix(PyObject *self,
                                                 PyObject *args);
static PyObject * py_expand_collision_matrix(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc3(PyObject *self, PyObject *args);
static PyObject * py_rotate_delta_fc2s(PyObject *self, PyObject *args);
static PyObject * py_get_isotope_strength(PyObject *self, PyObject *args);
static PyObject * py_get_thm_isotope_strength(PyObject *self, PyObject *args);
static PyObject *
py_set_permutation_symmetry_fc3(PyObject *self, PyObject *args);
static PyObject *
py_set_permutation_symmetry_compact_fc3(PyObject *self, PyObject *args);
static PyObject * py_set_permutation_symmetry_fc3(PyObject *self,
                                                  PyObject *args);
static PyObject * py_transpose_compact_fc3(PyObject *self, PyObject *args);
static PyObject * py_get_neighboring_gird_points(PyObject *self, PyObject *args);
static PyObject * py_set_integration_weights(PyObject *self, PyObject *args);
static PyObject *
py_tpl_get_triplets_reciprocal_mesh_at_q(PyObject *self, PyObject *args);
static PyObject * py_tpl_get_BZ_triplets_at_q(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights_with_sigma(PyObject *self, PyObject *args);
static PyObject *
py_diagonalize_collision_matrix(PyObject *self, PyObject *args);
static PyObject * py_pinv_from_eigensolution(PyObject *self, PyObject *args);
static PyObject * py_get_default_colmat_solver(PyObject *self, PyObject *args);

static void pinv_from_eigensolution(double *data,
                                    const double *eigvals,
                                    const size_t size,
                                    const double cutoff,
                                    const int pinv_method);
static void show_colmat_info(const PyArrayObject *collision_matrix_py,
                             const size_t i_sigma,
                             const size_t i_temp,
                             const size_t adrs_shift);
static Iarray* convert_to_iarray(const PyArrayObject* npyary);
static Darray* convert_to_darray(const PyArrayObject* npyary);


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

static PyMethodDef _phono3py_methods[] = {
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
 {"interaction",
   (PyCFunction)py_get_interaction,
   METH_VARARGS,
   "Interaction of triplets"},
  {"pp_collision",
   (PyCFunction)py_get_pp_collision,
   METH_VARARGS,
   "Collision and ph-ph calculation"},
  {"pp_collision_with_sigma",
   (PyCFunction)py_get_pp_collision_with_sigma,
   METH_VARARGS,
   "Collision and ph-ph calculation for smearing method"},
  {"imag_self_energy_with_g",
   (PyCFunction)py_get_imag_self_energy_with_g,
   METH_VARARGS,
   "Imaginary part of self energy at frequency points with g"},
  {"detailed_imag_self_energy_with_g",
   (PyCFunction)py_get_detailed_imag_self_energy_with_g,
   METH_VARARGS,
   "Detailed contribution to imaginary part of self energy at frequency points with g"},
  {"real_self_energy_at_bands",
   (PyCFunction)py_get_real_self_energy_at_bands,
   METH_VARARGS,
   "Real part of self energy from third order force constants"},
  {"real_self_energy_at_frequency_point",
   (PyCFunction)py_get_real_self_energy_at_frequency_point,
   METH_VARARGS,
   "Real part of self energy from third order force constants at a frequency point"},
  {"collision_matrix",
   (PyCFunction)py_get_collision_matrix,
   METH_VARARGS,
   "Collision matrix with g"},
  {"reducible_collision_matrix",
   (PyCFunction)py_get_reducible_collision_matrix,
   METH_VARARGS,
   "Collision matrix with g for reducible grid points"},
  {"symmetrize_collision_matrix",
   (PyCFunction)py_symmetrize_collision_matrix,
   METH_VARARGS,
   "Symmetrize collision matrix"},
  {"expand_collision_matrix",
   (PyCFunction)py_expand_collision_matrix,
   METH_VARARGS,
   "Expand collision matrix"},
  {"distribute_fc3",
   (PyCFunction)py_distribute_fc3,
   METH_VARARGS,
   "Distribute least fc3 to full fc3"},
  {"rotate_delta_fc2s",
   (PyCFunction)py_rotate_delta_fc2s,
   METH_VARARGS,
   "Rotate delta fc2s"},
  {"isotope_strength",
   (PyCFunction)py_get_isotope_strength,
   METH_VARARGS,
   "Isotope scattering strength"},
  {"thm_isotope_strength",
   (PyCFunction)py_get_thm_isotope_strength,
   METH_VARARGS,
   "Isotope scattering strength for tetrahedron_method"},
  {"permutation_symmetry_fc3",
   (PyCFunction)py_set_permutation_symmetry_fc3,
   METH_VARARGS,
   "Set permutation symmetry for fc3"},
  {"permutation_symmetry_compact_fc3",
   (PyCFunction)py_set_permutation_symmetry_compact_fc3,
   METH_VARARGS,
   "Set permutation symmetry for compact-fc3"},
  {"transpose_compact_fc3",
   (PyCFunction)py_transpose_compact_fc3,
   METH_VARARGS,
   "Transpose compact fc3"},
  {"neighboring_grid_points",
   (PyCFunction)py_get_neighboring_gird_points,
   METH_VARARGS,
   "Neighboring grid points by relative grid addresses"},
  {"integration_weights",
   (PyCFunction)py_set_integration_weights,
   METH_VARARGS,
   "Integration weights of tetrahedron method"},
  {"triplets_reciprocal_mesh_at_q",
   (PyCFunction)py_tpl_get_triplets_reciprocal_mesh_at_q,
   METH_VARARGS,
   "Triplets on reciprocal mesh points at a specific q-point"},
  {"BZ_triplets_at_q",
   (PyCFunction)py_tpl_get_BZ_triplets_at_q,
   METH_VARARGS,
   "Triplets in reciprocal primitive lattice are transformed to those in BZ."},
  {"triplets_integration_weights",
   (PyCFunction)py_set_triplets_integration_weights,
   METH_VARARGS,
   "Integration weights of tetrahedron method for triplets"},
  {"triplets_integration_weights_with_sigma",
   (PyCFunction)py_set_triplets_integration_weights_with_sigma,
   METH_VARARGS,
   "Integration weights of smearing method for triplets"},
  {"diagonalize_collision_matrix",
   (PyCFunction)py_diagonalize_collision_matrix,
   METH_VARARGS,
   "Diagonalize and optionally pseudo-inverse using Lapack dsyev(d)"},
  {"pinv_from_eigensolution",
   (PyCFunction)py_pinv_from_eigensolution,
   METH_VARARGS,
   "Pseudo-inverse from eigensolution"},
  {"default_colmat_solver",
   (PyCFunction)py_get_default_colmat_solver,
   METH_VARARGS,
   "Return default collison matrix solver by integer value"},
  {NULL, NULL, 0, NULL}
};

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
  PyModuleDef_HEAD_INIT,
  "_phono3py",
  NULL,
  sizeof(struct module_state),
  _phono3py_methods,
  NULL,
  _phono3py_traverse,
  _phono3py_clear,
  NULL
};

#define INITERROR return NULL

PyObject *
PyInit__phono3py(void)

#else
#define INITERROR return

  void
  init_phono3py(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule("_phono3py", _phono3py_methods);
#endif
  struct module_state *st;

  if (module == NULL)
    INITERROR;
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

static PyObject * py_get_interaction(PyObject *self, PyObject *args)
{
  PyArrayObject *py_fc3_normal_squared;
  PyArrayObject *py_g_zero;
  PyArrayObject *py_frequencies;
  PyArrayObject *py_eigenvectors;
  PyArrayObject *py_triplets;
  PyArrayObject *py_grid_address;
  PyArrayObject *py_mesh;
  PyArrayObject *py_shortest_vectors;
  PyArrayObject *py_multiplicities;
  PyArrayObject *py_fc3;
  PyArrayObject *py_masses;
  PyArrayObject *py_p2s_map;
  PyArrayObject *py_s2p_map;
  PyArrayObject *py_band_indices;
  double cutoff_frequency;
  int symmetrize_fc3_q;

  Darray *fc3_normal_squared;
  Darray *freqs;
  lapack_complex_double *eigvecs;
  size_t (*triplets)[3];
  npy_intp num_triplets;
  char* g_zero;
  int *grid_address;
  int *mesh;
  double *fc3;
  double *svecs;
  int *multi;
  double *masses;
  int *p2s;
  int *s2p;
  int *band_indices;
  int svecs_dims[3];
  int i;
  int is_compact_fc3;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOid",
                        &py_fc3_normal_squared,
                        &py_g_zero,
                        &py_frequencies,
                        &py_eigenvectors,
                        &py_triplets,
                        &py_grid_address,
                        &py_mesh,
                        &py_fc3,
                        &py_shortest_vectors,
                        &py_multiplicities,
                        &py_masses,
                        &py_p2s_map,
                        &py_s2p_map,
                        &py_band_indices,
                        &symmetrize_fc3_q,
                        &cutoff_frequency)) {
    return NULL;
  }


  fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
  freqs = convert_to_darray(py_frequencies);
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  eigvecs = (lapack_complex_double*)PyArray_DATA(py_eigenvectors);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  num_triplets = PyArray_DIMS(py_triplets)[0];
  g_zero = (char*)PyArray_DATA(py_g_zero);
  grid_address = (int*)PyArray_DATA(py_grid_address);
  mesh = (int*)PyArray_DATA(py_mesh);
  fc3 = (double*)PyArray_DATA(py_fc3);
  if (PyArray_DIMS(py_fc3)[0] == PyArray_DIMS(py_fc3)[1]) {
    is_compact_fc3 = 0;
  } else {
    is_compact_fc3 = 1;
  }
  svecs = (double*)PyArray_DATA(py_shortest_vectors);
  for (i = 0; i < 3; i++) {
    svecs_dims[i] = PyArray_DIMS(py_shortest_vectors)[i];
  }
  multi = (int*)PyArray_DATA(py_multiplicities);
  masses = (double*)PyArray_DATA(py_masses);
  p2s = (int*)PyArray_DATA(py_p2s_map);
  s2p = (int*)PyArray_DATA(py_s2p_map);
  band_indices = (int*)PyArray_DATA(py_band_indices);

  ph3py_get_interaction(fc3_normal_squared,
                        g_zero,
                        freqs,
                        eigvecs,
                        triplets,
                        num_triplets,
                        grid_address,
                        mesh,
                        fc3,
                        is_compact_fc3,
                        svecs,
                        svecs_dims,
                        multi,
                        masses,
                        p2s,
                        s2p,
                        band_indices,
                        symmetrize_fc3_q,
                        cutoff_frequency);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;
  free(freqs);
  freqs = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_get_pp_collision(PyObject *self, PyObject *args)
{
  PyArrayObject *py_gamma;
  PyArrayObject *py_relative_grid_address;
  PyArrayObject *py_frequencies;
  PyArrayObject *py_eigenvectors;
  PyArrayObject *py_triplets;
  PyArrayObject *py_triplet_weights;
  PyArrayObject *py_grid_address;
  PyArrayObject *py_bz_map;
  PyArrayObject *py_mesh;
  PyArrayObject *py_fc3;
  PyArrayObject *py_shortest_vectors;
  PyArrayObject *py_multiplicities;
  PyArrayObject *py_masses;
  PyArrayObject *py_p2s_map;
  PyArrayObject *py_s2p_map;
  PyArrayObject *py_band_indices;
  PyArrayObject *py_temperatures;
  double cutoff_frequency;
  int is_NU;
  int symmetrize_fc3_q;

  double *gamma;
  int (*relative_grid_address)[4][3];
  double *frequencies;
  lapack_complex_double *eigenvectors;
  size_t (*triplets)[3];
  npy_intp num_triplets;
  int *triplet_weights;
  int *grid_address;
  size_t *bz_map;
  int *mesh;
  double *fc3;
  double *svecs;
  int *multi;
  double *masses;
  int *p2s;
  int *s2p;
  Iarray *band_indices;
  Darray *temperatures;
  int svecs_dims[3];
  int i;
  int is_compact_fc3;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOiid",
                        &py_gamma,
                        &py_relative_grid_address,
                        &py_frequencies,
                        &py_eigenvectors,
                        &py_triplets,
                        &py_triplet_weights,
                        &py_grid_address,
                        &py_bz_map,
                        &py_mesh,
                        &py_fc3,
                        &py_shortest_vectors,
                        &py_multiplicities,
                        &py_masses,
                        &py_p2s_map,
                        &py_s2p_map,
                        &py_band_indices,
                        &py_temperatures,
                        &is_NU,
                        &symmetrize_fc3_q,
                        &cutoff_frequency)) {
    return NULL;
  }

  gamma = (double*)PyArray_DATA(py_gamma);
  relative_grid_address = (int(*)[4][3])PyArray_DATA(py_relative_grid_address);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  eigenvectors = (lapack_complex_double*)PyArray_DATA(py_eigenvectors);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  num_triplets = PyArray_DIMS(py_triplets)[0];
  triplet_weights = (int*)PyArray_DATA(py_triplet_weights);
  grid_address = (int*)PyArray_DATA(py_grid_address);
  bz_map = (size_t*)PyArray_DATA(py_bz_map);
  mesh = (int*)PyArray_DATA(py_mesh);
  fc3 = (double*)PyArray_DATA(py_fc3);
  if (PyArray_DIMS(py_fc3)[0] == PyArray_DIMS(py_fc3)[1]) {
    is_compact_fc3 = 0;
  } else {
    is_compact_fc3 = 1;
  }
  svecs = (double*)PyArray_DATA(py_shortest_vectors);
  for (i = 0; i < 3; i++) {
    svecs_dims[i] = PyArray_DIMS(py_shortest_vectors)[i];
  }
  multi = (int*)PyArray_DATA(py_multiplicities);
  masses = (double*)PyArray_DATA(py_masses);
  p2s = (int*)PyArray_DATA(py_p2s_map);
  s2p = (int*)PyArray_DATA(py_s2p_map);
  band_indices = convert_to_iarray(py_band_indices);
  temperatures = convert_to_darray(py_temperatures);

  ph3py_get_pp_collision(gamma,
                         relative_grid_address,
                         frequencies,
                         eigenvectors,
                         triplets,
                         num_triplets,
                         triplet_weights,
                         grid_address,
                         bz_map,
                         mesh,
                         fc3,
                         is_compact_fc3,
                         svecs,
                         svecs_dims,
                         multi,
                         masses,
                         p2s,
                         s2p,
                         band_indices,
                         temperatures,
                         is_NU,
                         symmetrize_fc3_q,
                         cutoff_frequency);

  free(band_indices);
  band_indices = NULL;
  free(temperatures);
  temperatures = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_get_pp_collision_with_sigma(PyObject *self, PyObject *args)
{
  PyArrayObject *py_gamma;
  PyArrayObject *py_frequencies;
  PyArrayObject *py_eigenvectors;
  PyArrayObject *py_triplets;
  PyArrayObject *py_triplet_weights;
  PyArrayObject *py_grid_address;
  PyArrayObject *py_mesh;
  PyArrayObject *py_fc3;
  PyArrayObject *py_shortest_vectors;
  PyArrayObject *py_multiplicities;
  PyArrayObject *py_masses;
  PyArrayObject *py_p2s_map;
  PyArrayObject *py_s2p_map;
  PyArrayObject *py_band_indices;
  PyArrayObject *py_temperatures;
  int is_NU;
  int symmetrize_fc3_q;
  double sigma;
  double sigma_cutoff;
  double cutoff_frequency;

  double *gamma;
  double *frequencies;
  lapack_complex_double *eigenvectors;
  size_t (*triplets)[3];
  npy_intp num_triplets;
  int *triplet_weights;
  int *grid_address;
  int *mesh;
  double *fc3;
  double *svecs;
  int *multi;
  double *masses;
  int *p2s;
  int *s2p;
  Iarray *band_indices;
  Darray *temperatures;
  int svecs_dims[3];
  int i;
  int is_compact_fc3;

  if (!PyArg_ParseTuple(args, "OddOOOOOOOOOOOOOOiid",
                        &py_gamma,
                        &sigma,
                        &sigma_cutoff,
                        &py_frequencies,
                        &py_eigenvectors,
                        &py_triplets,
                        &py_triplet_weights,
                        &py_grid_address,
                        &py_mesh,
                        &py_fc3,
                        &py_shortest_vectors,
                        &py_multiplicities,
                        &py_masses,
                        &py_p2s_map,
                        &py_s2p_map,
                        &py_band_indices,
                        &py_temperatures,
                        &is_NU,
                        &symmetrize_fc3_q,
                        &cutoff_frequency)) {
    return NULL;
  }

  gamma = (double*)PyArray_DATA(py_gamma);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  eigenvectors = (lapack_complex_double*)PyArray_DATA(py_eigenvectors);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  num_triplets = PyArray_DIMS(py_triplets)[0];
  triplet_weights = (int*)PyArray_DATA(py_triplet_weights);
  grid_address = (int*)PyArray_DATA(py_grid_address);
  mesh = (int*)PyArray_DATA(py_mesh);
  fc3 = (double*)PyArray_DATA(py_fc3);
  if (PyArray_DIMS(py_fc3)[0] == PyArray_DIMS(py_fc3)[1]) {
    is_compact_fc3 = 0;
  } else {
    is_compact_fc3 = 1;
  }
  svecs = (double*)PyArray_DATA(py_shortest_vectors);
  for (i = 0; i < 3; i++) {
    svecs_dims[i] = PyArray_DIMS(py_shortest_vectors)[i];
  }
  multi = (int*)PyArray_DATA(py_multiplicities);
  masses = (double*)PyArray_DATA(py_masses);
  p2s = (int*)PyArray_DATA(py_p2s_map);
  s2p = (int*)PyArray_DATA(py_s2p_map);
  band_indices = convert_to_iarray(py_band_indices);
  temperatures = convert_to_darray(py_temperatures);

  ph3py_get_pp_collision_with_sigma(gamma,
                                    sigma,
                                    sigma_cutoff,
                                    frequencies,
                                    eigenvectors,
                                    triplets,
                                    num_triplets,
                                    triplet_weights,
                                    grid_address,
                                    mesh,
                                    fc3,
                                    is_compact_fc3,
                                    svecs,
                                    svecs_dims,
                                    multi,
                                    masses,
                                    p2s,
                                    s2p,
                                    band_indices,
                                    temperatures,
                                    is_NU,
                                    symmetrize_fc3_q,
                                    cutoff_frequency);

  free(band_indices);
  band_indices = NULL;
  free(temperatures);
  temperatures = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_get_imag_self_energy_with_g(PyObject *self, PyObject *args)
{
  PyArrayObject *py_gamma;
  PyArrayObject *py_fc3_normal_squared;
  PyArrayObject *py_frequencies;
  PyArrayObject *py_triplets;
  PyArrayObject *py_triplet_weights;
  PyArrayObject *py_g;
  PyArrayObject *py_g_zero;
  double cutoff_frequency, temperature;
  int frequency_point_index;

  Darray *fc3_normal_squared;
  double *gamma;
  double *g;
  char* g_zero;
  double *frequencies;
  size_t (*triplets)[3];
  int *triplet_weights;
  int num_frequency_points;

  if (!PyArg_ParseTuple(args, "OOOOOdOOdi",
                        &py_gamma,
                        &py_fc3_normal_squared,
                        &py_triplets,
                        &py_triplet_weights,
                        &py_frequencies,
                        &temperature,
                        &py_g,
                        &py_g_zero,
                        &cutoff_frequency,
                        &frequency_point_index)) {
    return NULL;
  }

  fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
  gamma = (double*)PyArray_DATA(py_gamma);
  g = (double*)PyArray_DATA(py_g);
  g_zero = (char*)PyArray_DATA(py_g_zero);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  triplet_weights = (int*)PyArray_DATA(py_triplet_weights);
  num_frequency_points = PyArray_DIMS(py_g)[2];

  ph3py_get_imag_self_energy_at_bands_with_g(gamma,
                                             fc3_normal_squared,
                                             frequencies,
                                             triplets,
                                             triplet_weights,
                                             g,
                                             g_zero,
                                             temperature,
                                             cutoff_frequency,
                                             num_frequency_points,
                                             frequency_point_index);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;

  Py_RETURN_NONE;
}

static PyObject *
py_get_detailed_imag_self_energy_with_g(PyObject *self, PyObject *args)
{
  PyArrayObject *py_gamma_detail;
  PyArrayObject *py_gamma_N;
  PyArrayObject *py_gamma_U;
  PyArrayObject *py_fc3_normal_squared;
  PyArrayObject *py_frequencies;
  PyArrayObject *py_triplets;
  PyArrayObject *py_triplet_weights;
  PyArrayObject *py_grid_address;
  PyArrayObject *py_g;
  PyArrayObject *py_g_zero;
  double cutoff_frequency, temperature;

  Darray *fc3_normal_squared;
  double *gamma_detail;
  double *gamma_N;
  double *gamma_U;
  double *g;
  char* g_zero;
  double *frequencies;
  size_t (*triplets)[3];
  int *triplet_weights;
  int *grid_address;

  if (!PyArg_ParseTuple(args, "OOOOOOOOdOOd",
                        &py_gamma_detail,
                        &py_gamma_N,
                        &py_gamma_U,
                        &py_fc3_normal_squared,
                        &py_triplets,
                        &py_triplet_weights,
                        &py_grid_address,
                        &py_frequencies,
                        &temperature,
                        &py_g,
                        &py_g_zero,
                        &cutoff_frequency)) {
    return NULL;
  }

  fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
  gamma_detail = (double*)PyArray_DATA(py_gamma_detail);
  gamma_N = (double*)PyArray_DATA(py_gamma_N);
  gamma_U = (double*)PyArray_DATA(py_gamma_U);
  g = (double*)PyArray_DATA(py_g);
  g_zero = (char*)PyArray_DATA(py_g_zero);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  triplet_weights = (int*)PyArray_DATA(py_triplet_weights);
  grid_address = (int*)PyArray_DATA(py_grid_address);

  ph3py_get_detailed_imag_self_energy_at_bands_with_g(gamma_detail,
                                                      gamma_N,
                                                      gamma_U,
                                                      fc3_normal_squared,
                                                      frequencies,
                                                      triplets,
                                                      triplet_weights,
                                                      grid_address,
                                                      g,
                                                      g_zero,
                                                      temperature,
                                                      cutoff_frequency);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_get_real_self_energy_at_bands(PyObject *self,
                                                   PyObject *args)
{
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
  int *band_indices;
  size_t (*triplets)[3];
  int *triplet_weights;

  if (!PyArg_ParseTuple(args, "OOOOOOdddd",
                        &py_shift,
                        &py_fc3_normal_squared,
                        &py_triplets,
                        &py_triplet_weights,
                        &py_frequencies,
                        &py_band_indices,
                        &temperature,
                        &epsilon,
                        &unit_conversion_factor,
                        &cutoff_frequency)) {
    return NULL;
  }


  fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
  shift = (double*)PyArray_DATA(py_shift);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  band_indices = (int*)PyArray_DATA(py_band_indices);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  triplet_weights = (int*)PyArray_DATA(py_triplet_weights);

  ph3py_get_real_self_energy_at_bands(shift,
                                      fc3_normal_squared,
                                      band_indices,
                                      frequencies,
                                      triplets,
                                      triplet_weights,
                                      epsilon,
                                      temperature,
                                      unit_conversion_factor,
                                      cutoff_frequency);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_get_real_self_energy_at_frequency_point(PyObject *self,
                                                             PyObject *args)
{
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
  int *band_indices;
  size_t (*triplets)[3];
  int *triplet_weights;

  if (!PyArg_ParseTuple(args, "OdOOOOOdddd",
                        &py_shift,
                        &frequency_point,
                        &py_fc3_normal_squared,
                        &py_triplets,
                        &py_triplet_weights,
                        &py_frequencies,
                        &py_band_indices,
                        &temperature,
                        &epsilon,
                        &unit_conversion_factor,
                        &cutoff_frequency)) {
    return NULL;
  }


  fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
  shift = (double*)PyArray_DATA(py_shift);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  band_indices = (int*)PyArray_DATA(py_band_indices);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  triplet_weights = (int*)PyArray_DATA(py_triplet_weights);

  ph3py_get_real_self_energy_at_frequency_point(shift,
                                                frequency_point,
                                                fc3_normal_squared,
                                                band_indices,
                                                frequencies,
                                                triplets,
                                                triplet_weights,
                                                epsilon,
                                                temperature,
                                                unit_conversion_factor,
                                                cutoff_frequency);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_get_collision_matrix(PyObject *self, PyObject *args)
{
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
  size_t (*triplets)[3];
  size_t *triplets_map;
  size_t *map_q;
  size_t *rotated_grid_points;
  npy_intp num_gp, num_ir_gp, num_rot;
  double *rotations_cartesian;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOddd",
                        &py_collision_matrix,
                        &py_fc3_normal_squared,
                        &py_frequencies,
                        &py_g,
                        &py_triplets,
                        &py_triplets_map,
                        &py_map_q,
                        &py_rotated_grid_points,
                        &py_rotations_cartesian,
                        &temperature,
                        &unit_conversion_factor,
                        &cutoff_frequency)) {
    return NULL;
  }

  fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
  collision_matrix = (double*)PyArray_DATA(py_collision_matrix);
  g = (double*)PyArray_DATA(py_g);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  triplets_map = (size_t*)PyArray_DATA(py_triplets_map);
  num_gp = PyArray_DIMS(py_triplets_map)[0];
  map_q = (size_t*)PyArray_DATA(py_map_q);
  rotated_grid_points = (size_t*)PyArray_DATA(py_rotated_grid_points);
  num_ir_gp = PyArray_DIMS(py_rotated_grid_points)[0];
  num_rot = PyArray_DIMS(py_rotated_grid_points)[1];
  rotations_cartesian = (double*)PyArray_DATA(py_rotations_cartesian);

  assert(num_rot == PyArray_DIMS(py_rotations_cartesian)[0]);
  assert(num_gp == PyArray_DIMS(py_frequencies)[0]);

  ph3py_get_collision_matrix(collision_matrix,
                             fc3_normal_squared,
                             frequencies,
                             triplets,
                             triplets_map,
                             map_q,
                             rotated_grid_points,
                             rotations_cartesian,
                             g,
                             num_ir_gp,
                             num_gp,
                             num_rot,
                             temperature,
                             unit_conversion_factor,
                             cutoff_frequency);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_get_reducible_collision_matrix(PyObject *self, PyObject *args)
{
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
  size_t (*triplets)[3];
  size_t *triplets_map;
  npy_intp num_gp;
  size_t *map_q;

  if (!PyArg_ParseTuple(args, "OOOOOOOddd",
                        &py_collision_matrix,
                        &py_fc3_normal_squared,
                        &py_frequencies,
                        &py_g,
                        &py_triplets,
                        &py_triplets_map,
                        &py_map_q,
                        &temperature,
                        &unit_conversion_factor,
                        &cutoff_frequency)) {
    return NULL;
  }

  fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
  collision_matrix = (double*)PyArray_DATA(py_collision_matrix);
  g = (double*)PyArray_DATA(py_g);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  triplets_map = (size_t*)PyArray_DATA(py_triplets_map);
  num_gp = PyArray_DIMS(py_triplets_map)[0];
  map_q = (size_t*)PyArray_DATA(py_map_q);

  ph3py_get_reducible_collision_matrix(collision_matrix,
                                       fc3_normal_squared,
                                       frequencies,
                                       triplets,
                                       triplets_map,
                                       map_q,
                                       g,
                                       num_gp,
                                       temperature,
                                       unit_conversion_factor,
                                       cutoff_frequency);

  free(fc3_normal_squared);
  fc3_normal_squared = NULL;

  Py_RETURN_NONE;
}

static PyObject * py_symmetrize_collision_matrix(PyObject *self, PyObject *args)
{
  PyArrayObject *py_collision_matrix;

  double *collision_matrix;
  long num_band, num_grid_points, num_temp, num_sigma;
  long num_column;

  if (!PyArg_ParseTuple(args, "O",
                        &py_collision_matrix)) {
    return NULL;
  }

  collision_matrix = (double*)PyArray_DATA(py_collision_matrix);
  num_sigma = PyArray_DIMS(py_collision_matrix)[0];
  num_temp = PyArray_DIMS(py_collision_matrix)[1];
  num_grid_points = PyArray_DIMS(py_collision_matrix)[2];
  num_band = PyArray_DIMS(py_collision_matrix)[3];

  if (PyArray_NDIM(py_collision_matrix) == 8) {
    num_column = num_grid_points * num_band * 3;
  } else {
    num_column = num_grid_points * num_band;
  }

  ph3py_symmetrize_collision_matrix(collision_matrix,
                                    num_column,
                                    num_temp,
                                    num_sigma);

  Py_RETURN_NONE;
}

static PyObject * py_expand_collision_matrix(PyObject *self, PyObject *args)
{
  PyArrayObject *py_collision_matrix;
  PyArrayObject *py_ir_grid_points;
  PyArrayObject *py_rot_grid_points;

  double *collision_matrix;
  size_t *rot_grid_points;
  size_t *ir_grid_points;
  long num_band, num_grid_points, num_temp, num_sigma, num_rot, num_ir_gp;

  if (!PyArg_ParseTuple(args, "OOO",
                        &py_collision_matrix,
                        &py_ir_grid_points,
                        &py_rot_grid_points)) {
    return NULL;
  }

  collision_matrix = (double*)PyArray_DATA(py_collision_matrix);
  rot_grid_points = (size_t*)PyArray_DATA(py_rot_grid_points);
  ir_grid_points = (size_t*)PyArray_DATA(py_ir_grid_points);
  num_sigma = (long)PyArray_DIMS(py_collision_matrix)[0];
  num_temp = (long)PyArray_DIMS(py_collision_matrix)[1];
  num_grid_points = (long)PyArray_DIMS(py_collision_matrix)[2];
  num_band = (long)PyArray_DIMS(py_collision_matrix)[3];
  num_rot = (long)PyArray_DIMS(py_rot_grid_points)[0];
  num_ir_gp = (long)PyArray_DIMS(py_ir_grid_points)[0];

  ph3py_expand_collision_matrix(collision_matrix,
                                rot_grid_points,
                                ir_grid_points,
                                num_ir_gp,
                                num_grid_points,
                                num_rot,
                                num_sigma,
                                num_temp,
                                num_band);

  Py_RETURN_NONE;
}

static PyObject * py_get_isotope_strength(PyObject *self, PyObject *args)
{
  PyArrayObject *py_gamma;
  PyArrayObject *py_frequencies;
  PyArrayObject *py_eigenvectors;
  PyArrayObject *py_band_indices;
  PyArrayObject *py_mass_variances;
  long grid_point;
  long num_grid_points;
  double cutoff_frequency;
  double sigma;

  double *gamma;
  double *frequencies;
  lapack_complex_double *eigenvectors;
  int *band_indices;
  double *mass_variances;
  npy_intp num_band, num_band0;

  if (!PyArg_ParseTuple(args, "OlOOOOldd",
                        &py_gamma,
                        &grid_point,
                        &py_mass_variances,
                        &py_frequencies,
                        &py_eigenvectors,
                        &py_band_indices,
                        &num_grid_points,
                        &sigma,
                        &cutoff_frequency)) {
    return NULL;
  }


  gamma = (double*)PyArray_DATA(py_gamma);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  eigenvectors = (lapack_complex_double*)PyArray_DATA(py_eigenvectors);
  band_indices = (int*)PyArray_DATA(py_band_indices);
  mass_variances = (double*)PyArray_DATA(py_mass_variances);
  num_band = PyArray_DIMS(py_frequencies)[1];
  num_band0 = PyArray_DIMS(py_band_indices)[0];

  ph3py_get_isotope_scattering_strength(gamma,
                                        grid_point,
                                        mass_variances,
                                        frequencies,
                                        eigenvectors,
                                        num_grid_points,
                                        band_indices,
                                        num_band,
                                        num_band0,
                                        sigma,
                                        cutoff_frequency);

  Py_RETURN_NONE;
}

static PyObject * py_get_thm_isotope_strength(PyObject *self, PyObject *args)
{
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
  size_t *ir_grid_points;
  int *weights;
  lapack_complex_double *eigenvectors;
  int *band_indices;
  double *mass_variances;
  npy_intp num_band, num_band0, num_ir_grid_points;
  double *integration_weights;

  if (!PyArg_ParseTuple(args, "OlOOOOOOOd",
                        &py_gamma,
                        &grid_point,
                        &py_ir_grid_points,
                        &py_weights,
                        &py_mass_variances,
                        &py_frequencies,
                        &py_eigenvectors,
                        &py_band_indices,
                        &py_integration_weights,
                        &cutoff_frequency)) {
    return NULL;
  }


  gamma = (double*)PyArray_DATA(py_gamma);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  ir_grid_points = (size_t*)PyArray_DATA(py_ir_grid_points);
  weights = (int*)PyArray_DATA(py_weights);
  eigenvectors = (lapack_complex_double*)PyArray_DATA(py_eigenvectors);
  band_indices = (int*)PyArray_DATA(py_band_indices);
  mass_variances = (double*)PyArray_DATA(py_mass_variances);
  num_band = PyArray_DIMS(py_frequencies)[1];
  num_band0 = PyArray_DIMS(py_band_indices)[0];
  integration_weights = (double*)PyArray_DATA(py_integration_weights);
  num_ir_grid_points = PyArray_DIMS(py_ir_grid_points)[0];

  ph3py_get_thm_isotope_scattering_strength(gamma,
                                            grid_point,
                                            ir_grid_points,
                                            weights,
                                            mass_variances,
                                            frequencies,
                                            eigenvectors,
                                            num_ir_grid_points,
                                            band_indices,
                                            num_band,
                                            num_band0,
                                            integration_weights,
                                            cutoff_frequency);

  Py_RETURN_NONE;
}

static PyObject * py_distribute_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject *force_constants_third;
  int target;
  int source;
  PyArrayObject *rotation_cart_inv;
  PyArrayObject *atom_mapping_py;

  double *fc3;
  double *rot_cart_inv;
  int *atom_mapping;
  npy_intp num_atom;

  if (!PyArg_ParseTuple(args, "OiiOO",
                        &force_constants_third,
                        &target,
                        &source,
                        &atom_mapping_py,
                        &rotation_cart_inv)) {
    return NULL;
  }

  fc3 = (double*)PyArray_DATA(force_constants_third);
  rot_cart_inv = (double*)PyArray_DATA(rotation_cart_inv);
  atom_mapping = (int*)PyArray_DATA(atom_mapping_py);
  num_atom = PyArray_DIMS(atom_mapping_py)[0];

  ph3py_distribute_fc3(fc3,
                       target,
                       source,
                       atom_mapping,
                       num_atom,
                       rot_cart_inv);

  Py_RETURN_NONE;
}

static PyObject * py_rotate_delta_fc2s(PyObject *self, PyObject *args)
{
  PyArrayObject *py_fc3;
  PyArrayObject *py_delta_fc2s;
  PyArrayObject *py_inv_U;
  PyArrayObject *py_site_sym_cart;
  PyArrayObject *py_rot_map_syms;

  double (*fc3)[3][3][3];
  double (*delta_fc2s)[3][3];
  double *inv_U;
  double (*site_sym_cart)[3][3];
  int *rot_map_syms;
  npy_intp num_atom, num_disp, num_site_sym;

  if (!PyArg_ParseTuple(args, "OOOOO",
                        &py_fc3,
                        &py_delta_fc2s,
                        &py_inv_U,
                        &py_site_sym_cart,
                        &py_rot_map_syms)) {
    return NULL;
  }

  /* (num_atom, num_atom, 3, 3, 3) */
  fc3 = (double(*)[3][3][3])PyArray_DATA(py_fc3);
  /* (n_u1, num_atom, num_atom, 3, 3) */
  delta_fc2s = (double(*)[3][3])PyArray_DATA(py_delta_fc2s);
  /* (3, n_u1 * n_sym) */
  inv_U = (double*)PyArray_DATA(py_inv_U);
  /* (n_sym, 3, 3) */
  site_sym_cart = (double(*)[3][3])PyArray_DATA(py_site_sym_cart);
  /* (n_sym, natom) */
  rot_map_syms = (int*)PyArray_DATA(py_rot_map_syms);

  num_atom = PyArray_DIMS(py_fc3)[0];
  num_disp = PyArray_DIMS(py_delta_fc2s)[0];
  num_site_sym = PyArray_DIMS(py_site_sym_cart)[0];

  ph3py_rotate_delta_fc2(fc3,
                         delta_fc2s,
                         inv_U,
                         site_sym_cart,
                         rot_map_syms,
                         num_atom,
                         num_site_sym,
                         num_disp);

  Py_RETURN_NONE;
}

static PyObject *
py_set_permutation_symmetry_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject *py_fc3;

  double *fc3;
  npy_intp num_atom;

  if (!PyArg_ParseTuple(args, "O", &py_fc3)) {
    return NULL;
  }

  fc3 = (double*)PyArray_DATA(py_fc3);
  num_atom = PyArray_DIMS(py_fc3)[0];

  ph3py_set_permutation_symmetry_fc3(fc3, num_atom);

  Py_RETURN_NONE;
}

static PyObject *
py_set_permutation_symmetry_compact_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject* py_fc3;
  PyArrayObject* py_permutations;
  PyArrayObject* py_s2pp_map;
  PyArrayObject* py_p2s_map;
  PyArrayObject* py_nsym_list;

  double *fc3;
  int *s2pp;
  int *p2s;
  int *nsym_list;
  int *perms;
  npy_intp n_patom, n_satom;

  if (!PyArg_ParseTuple(args, "OOOOO",
                        &py_fc3,
                        &py_permutations,
                        &py_s2pp_map,
                        &py_p2s_map,
                        &py_nsym_list)) {
    return NULL;
  }

  fc3 = (double*)PyArray_DATA(py_fc3);
  perms = (int*)PyArray_DATA(py_permutations);
  s2pp = (int*)PyArray_DATA(py_s2pp_map);
  p2s = (int*)PyArray_DATA(py_p2s_map);
  nsym_list = (int*)PyArray_DATA(py_nsym_list);
  n_patom = PyArray_DIMS(py_fc3)[0];
  n_satom = PyArray_DIMS(py_fc3)[1];

  ph3py_set_permutation_symmetry_compact_fc3(fc3,
                                             p2s,
                                             s2pp,
                                             nsym_list,
                                             perms,
                                             n_satom,
                                             n_patom);

  Py_RETURN_NONE;
}

static PyObject * py_transpose_compact_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject* py_fc3;
  PyArrayObject* py_permutations;
  PyArrayObject* py_s2pp_map;
  PyArrayObject* py_p2s_map;
  PyArrayObject* py_nsym_list;
  int t_type;

  double *fc3;
  int *s2pp;
  int *p2s;
  int *nsym_list;
  int *perms;
  npy_intp n_patom, n_satom;

  if (!PyArg_ParseTuple(args, "OOOOOi",
                        &py_fc3,
                        &py_permutations,
                        &py_s2pp_map,
                        &py_p2s_map,
                        &py_nsym_list,
                        &t_type)) {
    return NULL;
  }

  fc3 = (double*)PyArray_DATA(py_fc3);
  perms = (int*)PyArray_DATA(py_permutations);
  s2pp = (int*)PyArray_DATA(py_s2pp_map);
  p2s = (int*)PyArray_DATA(py_p2s_map);
  nsym_list = (int*)PyArray_DATA(py_nsym_list);
  n_patom = PyArray_DIMS(py_fc3)[0];
  n_satom = PyArray_DIMS(py_fc3)[1];

  ph3py_transpose_compact_fc3(fc3,
                              p2s,
                              s2pp,
                              nsym_list,
                              perms,
                              n_satom,
                              n_patom,
                              t_type);

  Py_RETURN_NONE;
}

static PyObject * py_get_neighboring_gird_points(PyObject *self, PyObject *args)
{
  PyArrayObject *py_relative_grid_points;
  PyArrayObject *py_grid_points;
  PyArrayObject *py_relative_grid_address;
  PyArrayObject *py_mesh;
  PyArrayObject *py_bz_grid_address;
  PyArrayObject *py_bz_map;

  size_t *relative_grid_points;
  size_t *grid_points;
  npy_intp num_grid_points, num_relative_grid_address;
  int (*relative_grid_address)[3];
  int *mesh;
  int (*bz_grid_address)[3];
  size_t *bz_map;

  if (!PyArg_ParseTuple(args, "OOOOOO",
                        &py_relative_grid_points,
                        &py_grid_points,
                        &py_relative_grid_address,
                        &py_mesh,
                        &py_bz_grid_address,
                        &py_bz_map)) {
    return NULL;
  }

  relative_grid_points = (size_t*)PyArray_DATA(py_relative_grid_points);
  grid_points = (size_t*)PyArray_DATA(py_grid_points);
  num_grid_points = PyArray_DIMS(py_grid_points)[0];
  relative_grid_address = (int(*)[3])PyArray_DATA(py_relative_grid_address);
  num_relative_grid_address = PyArray_DIMS(py_relative_grid_address)[0];
  mesh = (int*)PyArray_DATA(py_mesh);
  bz_grid_address = (int(*)[3])PyArray_DATA(py_bz_grid_address);
  bz_map = (size_t*)PyArray_DATA(py_bz_map);

  ph3py_get_neighboring_gird_points(relative_grid_points,
                                    grid_points,
                                    relative_grid_address,
                                    mesh,
                                    bz_grid_address,
                                    bz_map,
                                    num_grid_points,
                                    num_relative_grid_address);

  Py_RETURN_NONE;
}

static PyObject * py_set_integration_weights(PyObject *self, PyObject *args)
{
  PyArrayObject *py_iw;
  PyArrayObject *py_frequency_points;
  PyArrayObject *py_relative_grid_address;
  PyArrayObject *py_mesh;
  PyArrayObject *py_grid_points;
  PyArrayObject *py_frequencies;
  PyArrayObject *py_bz_grid_address;
  PyArrayObject *py_bz_map;

  double *iw;
  double *frequency_points;
  npy_intp num_band0, num_band, num_gp;
  int (*relative_grid_address)[4][3];
  int *mesh;
  size_t *grid_points;
  int (*bz_grid_address)[3];
  size_t *bz_map;
  double *frequencies;

  if (!PyArg_ParseTuple(args, "OOOOOOOO",
                        &py_iw,
                        &py_frequency_points,
                        &py_relative_grid_address,
                        &py_mesh,
                        &py_grid_points,
                        &py_frequencies,
                        &py_bz_grid_address,
                        &py_bz_map)) {
    return NULL;
  }

  iw = (double*)PyArray_DATA(py_iw);
  frequency_points = (double*)PyArray_DATA(py_frequency_points);
  num_band0 = PyArray_DIMS(py_frequency_points)[0];
  relative_grid_address = (int(*)[4][3])PyArray_DATA(py_relative_grid_address);
  mesh = (int*)PyArray_DATA(py_mesh);
  grid_points = (size_t*)PyArray_DATA(py_grid_points);
  num_gp = PyArray_DIMS(py_grid_points)[0];
  bz_grid_address = (int(*)[3])PyArray_DATA(py_bz_grid_address);
  bz_map = (size_t*)PyArray_DATA(py_bz_map);
  frequencies = (double*)PyArray_DATA(py_frequencies);
  num_band = PyArray_DIMS(py_frequencies)[1];

  ph3py_set_integration_weights(iw,
                                frequency_points,
                                num_band0,
                                num_band,
                                num_gp,
                                relative_grid_address,
                                mesh,
                                grid_points,
                                bz_grid_address,
                                bz_map,
                                frequencies);

  Py_RETURN_NONE;
}

static PyObject *
py_tpl_get_triplets_reciprocal_mesh_at_q(PyObject *self, PyObject *args)
{
  PyArrayObject *py_map_triplets;
  PyArrayObject *py_grid_address;
  PyArrayObject *py_map_q;
  PyArrayObject *py_mesh;
  PyArrayObject *py_rotations;
  long fixed_grid_number;
  int is_time_reversal;
  int swappable;

  int (*grid_address)[3];
  size_t *map_triplets;
  size_t *map_q;
  int *mesh;
  int (*rot)[3][3];
  npy_intp num_rot;
  size_t num_ir;

  if (!PyArg_ParseTuple(args, "OOOlOiOi",
                        &py_map_triplets,
                        &py_map_q,
                        &py_grid_address,
                        &fixed_grid_number,
                        &py_mesh,
                        &is_time_reversal,
                        &py_rotations,
                        &swappable)) {
    return NULL;
  }

  grid_address = (int(*)[3])PyArray_DATA(py_grid_address);
  map_triplets = (size_t*)PyArray_DATA(py_map_triplets);
  map_q = (size_t*)PyArray_DATA(py_map_q);
  mesh = (int*)PyArray_DATA(py_mesh);
  rot = (int(*)[3][3])PyArray_DATA(py_rotations);
  num_rot = PyArray_DIMS(py_rotations)[0];
  num_ir = ph3py_get_triplets_reciprocal_mesh_at_q(map_triplets,
                                                   map_q,
                                                   grid_address,
                                                   fixed_grid_number,
                                                   mesh,
                                                   is_time_reversal,
                                                   num_rot,
                                                   rot,
                                                   swappable);

  return PyLong_FromSize_t(num_ir);
}

static PyObject * py_tpl_get_BZ_triplets_at_q(PyObject *self, PyObject *args)
{
  PyArrayObject *py_triplets;
  PyArrayObject *py_bz_grid_address;
  PyArrayObject *py_bz_map;
  PyArrayObject *py_map_triplets;
  PyArrayObject *py_mesh;
  long grid_point;

  size_t (*triplets)[3];
  int (*bz_grid_address)[3];
  size_t *bz_map;
  size_t *map_triplets;
  npy_intp num_map_triplets;
  int *mesh;
  size_t num_ir;

  if (!PyArg_ParseTuple(args, "OlOOOO",
                        &py_triplets,
                        &grid_point,
                        &py_bz_grid_address,
                        &py_bz_map,
                        &py_map_triplets,
                        &py_mesh)) {
    return NULL;
  }

  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  bz_grid_address = (int(*)[3])PyArray_DATA(py_bz_grid_address);
  bz_map = (size_t*)PyArray_DATA(py_bz_map);
  map_triplets = (size_t*)PyArray_DATA(py_map_triplets);
  num_map_triplets = PyArray_DIMS(py_map_triplets)[0];
  mesh = (int*)PyArray_DATA(py_mesh);

  num_ir = ph3py_get_BZ_triplets_at_q(triplets,
                                      grid_point,
                                      bz_grid_address,
                                      bz_map,
                                      map_triplets,
                                      num_map_triplets,
                                      mesh);

  return PyLong_FromSize_t(num_ir);
}

static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args)
{
  PyArrayObject *py_iw;
  PyArrayObject *py_iw_zero;
  PyArrayObject *py_frequency_points;
  PyArrayObject *py_relative_grid_address;
  PyArrayObject *py_mesh;
  PyArrayObject *py_triplets;
  PyArrayObject *py_frequencies1;
  PyArrayObject *py_frequencies2;
  PyArrayObject *py_bz_grid_address;
  PyArrayObject *py_bz_map;
  int tp_type;

  double *iw;
  char *iw_zero;
  double *frequency_points;
  int (*relative_grid_address)[4][3];
  int *mesh;
  size_t (*triplets)[3];
  int (*bz_grid_address)[3];
  size_t *bz_map;
  double *frequencies1, *frequencies2;
  npy_intp num_band0, num_band1, num_band2, num_triplets;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOi",
                        &py_iw,
                        &py_iw_zero,
                        &py_frequency_points,
                        &py_relative_grid_address,
                        &py_mesh,
                        &py_triplets,
                        &py_frequencies1,
                        &py_frequencies2,
                        &py_bz_grid_address,
                        &py_bz_map,
                        &tp_type)) {
    return NULL;
  }

  iw = (double*)PyArray_DATA(py_iw);
  iw_zero = (char*)PyArray_DATA(py_iw_zero);
  frequency_points = (double*)PyArray_DATA(py_frequency_points);
  num_band0 = PyArray_DIMS(py_frequency_points)[0];
  relative_grid_address = (int(*)[4][3])PyArray_DATA(py_relative_grid_address);
  mesh = (int*)PyArray_DATA(py_mesh);
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  num_triplets = PyArray_DIMS(py_triplets)[0];
  bz_grid_address = (int(*)[3])PyArray_DATA(py_bz_grid_address);
  bz_map = (size_t*)PyArray_DATA(py_bz_map);
  frequencies1 = (double*)PyArray_DATA(py_frequencies1);
  frequencies2 = (double*)PyArray_DATA(py_frequencies2);
  num_band1 = PyArray_DIMS(py_frequencies1)[1];
  num_band2 = PyArray_DIMS(py_frequencies2)[1];

  ph3py_get_integration_weight(iw,
                               iw_zero,
                               frequency_points,
                               num_band0,
                               relative_grid_address,
                               mesh,
                               triplets,
                               num_triplets,
                               bz_grid_address,
                               bz_map,
                               frequencies1,
                               num_band1,
                               frequencies2,
                               num_band2,
                               tp_type,
                               1,
                               0);

  Py_RETURN_NONE;
}

static PyObject *
py_set_triplets_integration_weights_with_sigma(PyObject *self, PyObject *args)
{
  PyArrayObject *py_iw;
  PyArrayObject *py_iw_zero;
  PyArrayObject *py_frequency_points;
  PyArrayObject *py_triplets;
  PyArrayObject *py_frequencies;
  double sigma, sigma_cutoff;

  double *iw;
  char *iw_zero;
  double *frequency_points;
  size_t (*triplets)[3];
  double *frequencies;
  npy_intp num_band0, num_band, num_iw, num_triplets;

  if (!PyArg_ParseTuple(args, "OOOOOdd",
                        &py_iw,
                        &py_iw_zero,
                        &py_frequency_points,
                        &py_triplets,
                        &py_frequencies,
                        &sigma,
                        &sigma_cutoff)) {
    return NULL;
  }

  iw = (double*)PyArray_DATA(py_iw);
  iw_zero = (char*)PyArray_DATA(py_iw_zero);
  frequency_points = (double*)PyArray_DATA(py_frequency_points);
  num_band0 = PyArray_DIMS(py_frequency_points)[0];
  triplets = (size_t(*)[3])PyArray_DATA(py_triplets);
  num_triplets = PyArray_DIMS(py_triplets)[0];
  frequencies = (double*)PyArray_DATA(py_frequencies);
  num_band = PyArray_DIMS(py_frequencies)[1];
  num_iw = PyArray_DIMS(py_iw)[0];

  ph3py_get_integration_weight_with_sigma(iw,
                                          iw_zero,
                                          sigma,
                                          sigma_cutoff,
                                          frequency_points,
                                          num_band0,
                                          triplets,
                                          num_triplets,
                                          frequencies,
                                          num_band,
                                          num_iw);

  Py_RETURN_NONE;
}

static PyObject *
py_diagonalize_collision_matrix(PyObject *self, PyObject *args)
{
  PyArrayObject *py_collision_matrix;
  PyArrayObject *py_eigenvalues;
  double cutoff;
  int i_sigma, i_temp, is_pinv, solver;

  double *collision_matrix;
  double *eigvals;
  npy_intp num_temp, num_grid_point, num_band;
  size_t num_column, adrs_shift;
  int info;

  if (!PyArg_ParseTuple(args, "OOiidii",
                        &py_collision_matrix,
                        &py_eigenvalues,
                        &i_sigma,
                        &i_temp,
                        &cutoff,
                        &solver,
                        &is_pinv)) {
    return NULL;
  }

  collision_matrix = (double*)PyArray_DATA(py_collision_matrix);
  eigvals = (double*)PyArray_DATA(py_eigenvalues);

  if (PyArray_NDIM(py_collision_matrix) == 2) {
    num_temp = 1;
    num_column = PyArray_DIM(py_collision_matrix, 1);
  } else {
    num_temp = PyArray_DIM(py_collision_matrix, 1);
    num_grid_point = PyArray_DIM(py_collision_matrix, 2);
    num_band = PyArray_DIM(py_collision_matrix, 3);
    if (PyArray_NDIM(py_collision_matrix) == 8) {
      num_column = num_grid_point * num_band * 3;
    } else {
      num_column = num_grid_point * num_band;
    }
  }
  adrs_shift = (i_sigma * num_column * num_column * num_temp +
                i_temp * num_column * num_column);

  /* show_colmat_info(py_collision_matrix, i_sigma, i_temp, adrs_shift); */

  info = phonopy_dsyev(collision_matrix + adrs_shift,
                       eigvals, num_column, solver);
  if (is_pinv) {
    pinv_from_eigensolution(collision_matrix + adrs_shift,
                            eigvals, num_column, cutoff, 0);
  }

  return PyLong_FromLong((long) info);
}

static PyObject * py_pinv_from_eigensolution(PyObject *self, PyObject *args)
{
  PyArrayObject *py_collision_matrix;
  PyArrayObject *py_eigenvalues;
  double cutoff;
  int i_sigma, i_temp, pinv_method;

  double *collision_matrix;
  double *eigvals;
  npy_intp num_temp, num_grid_point, num_band;
  size_t num_column, adrs_shift;

  if (!PyArg_ParseTuple(args, "OOiidi",
                        &py_collision_matrix,
                        &py_eigenvalues,
                        &i_sigma,
                        &i_temp,
                        &cutoff,
                        &pinv_method)) {
    return NULL;
  }

  collision_matrix = (double*)PyArray_DATA(py_collision_matrix);
  eigvals = (double*)PyArray_DATA(py_eigenvalues);
  num_temp = PyArray_DIMS(py_collision_matrix)[1];
  num_grid_point = PyArray_DIMS(py_collision_matrix)[2];
  num_band = PyArray_DIMS(py_collision_matrix)[3];

  if (PyArray_NDIM(py_collision_matrix) == 8) {
    num_column = num_grid_point * num_band * 3;
  } else {
    num_column = num_grid_point * num_band;
  }
  adrs_shift = (i_sigma * num_column * num_column * num_temp +
                i_temp * num_column * num_column);

  /* show_colmat_info(py_collision_matrix, i_sigma, i_temp, adrs_shift); */

  pinv_from_eigensolution(collision_matrix + adrs_shift,
                          eigvals, num_column, cutoff, pinv_method);

  Py_RETURN_NONE;
}

static PyObject * py_get_default_colmat_solver(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

#ifdef MKL_LAPACKE
  return PyLong_FromLong((long) 1);
#else
  return PyLong_FromLong((long) 4);
#endif

}

static void pinv_from_eigensolution(double *data,
                                    const double *eigvals,
                                    const size_t size,
                                    const double cutoff,
                                    const int pinv_method)
{
  size_t i, ib, j, k, max_l, i_s, j_s;
  double *tmp_data;
  double e, sum;
  size_t *l;

  l = NULL;
  tmp_data = NULL;

  tmp_data = (double*)malloc(sizeof(double) * size * size);

#pragma omp parallel for
  for (i = 0; i < size * size; i++) {
    tmp_data[i] = data[i];
  }

  l = (size_t*)malloc(sizeof(size_t) * size);
  max_l = 0;
  for (i = 0; i < size; i++) {
    if (pinv_method == 0) {
      e = fabs(eigvals[i]);
    } else {
      e = eigvals[i];
    }
    if (e > cutoff) {
      l[max_l] = i;
      max_l++;
    }
  }

#pragma omp parallel for private(ib, j, k, i_s, j_s, sum)
  for (i = 0; i < size / 2; i++) {
    /* from front */
    i_s = i * size;
    for (j = i; j < size; j++) {
      j_s = j * size;
      sum = 0;
      for (k = 0; k < max_l; k++) {
        sum += tmp_data[i_s + l[k]] * tmp_data[j_s + l[k]] / eigvals[l[k]];
      }
      data[i_s + j] = sum;
      data[j_s + i] = sum;
    }
    /* from back */
    ib = size - i - 1;
    i_s = ib * size;
    for (j = ib; j < size; j++) {
      j_s = j * size;
      sum = 0;
      for (k = 0; k < max_l; k++) {
        sum += tmp_data[i_s + l[k]] * tmp_data[j_s + l[k]] / eigvals[l[k]];
      }
      data[i_s + j] = sum;
      data[j_s + ib] = sum;
    }
  }

  /* when size is odd */
  if ((size % 2) == 1) {
    i = (size - 1) / 2;
    i_s = i * size;
    for (j = i; j < size; j++) {
      j_s = j * size;
      sum = 0;
      for (k = 0; k < max_l; k++) {
        sum += tmp_data[i_s + l[k]] * tmp_data[j_s + l[k]] / eigvals[l[k]];
      }
      data[i_s + j] = sum;
      data[j_s + i] = sum;
    }
  }

  free(l);
  l = NULL;

  free(tmp_data);
  tmp_data = NULL;
}

static void show_colmat_info(const PyArrayObject *py_collision_matrix,
                             const size_t i_sigma,
                             const size_t i_temp,
                             const size_t adrs_shift)
{
  int i;

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


static Iarray* convert_to_iarray(const PyArrayObject* npyary)
{
  int i;
  Iarray *ary;

  ary = (Iarray*) malloc(sizeof(Iarray));
  for (i = 0; i < PyArray_NDIM(npyary); i++) {
    ary->dims[i] = PyArray_DIMS(npyary)[i];
  }
  ary->data = (int*)PyArray_DATA(npyary);
  return ary;
}


static Darray* convert_to_darray(const PyArrayObject* npyary)
{
  int i;
  Darray *ary;

  ary = (Darray*) malloc(sizeof(Darray));
  for (i = 0; i < PyArray_NDIM(npyary); i++) {
    ary->dims[i] = PyArray_DIMS(npyary)[i];
  }
  ary->data = (double*)PyArray_DATA(npyary);
  return ary;
}
