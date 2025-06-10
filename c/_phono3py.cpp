#include <assert.h>
#include <math.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdint.h>

#include "phono3py.h"
#include "phonoc_array.h"

namespace nb = nanobind;

static Larray *convert_to_larray(nb::ndarray<> npyary) {
    int64_t i;
    Larray *ary;

    ary = (Larray *)malloc(sizeof(Larray));
    for (i = 0; i < npyary.ndim(); i++) {
        ary->dims[i] = npyary.shape(i);
    }
    ary->data = (int64_t *)npyary.data();
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

// static void show_colmat_info(const PyArrayObject *py_collision_matrix,
//                              const int64_t i_sigma, const int64_t i_temp,
//                              const int64_t adrs_shift) {
//     int64_t i;

//     printf(" Array_shape:(");
//     for (i = 0; i < PyArray_NDIM(py_collision_matrix); i++) {
//         printf("%d", (int)PyArray_DIM(py_collision_matrix, i));
//         if (i < PyArray_NDIM(py_collision_matrix) - 1) {
//             printf(",");
//         } else {
//             printf("), ");
//         }
//     }
//     printf("Data shift:%lu [%lu, %lu]\n", adrs_shift, i_sigma, i_temp);
// }

void py_get_interaction(
    nb::ndarray<> py_fc3_normal_squared, nb::ndarray<> py_g_zero,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_triplets, nb::ndarray<> py_bz_grid_addresses,
    nb::ndarray<> py_D_diag, nb::ndarray<> py_Q, nb::ndarray<> py_fc3,
    nb::ndarray<> py_fc3_nonzero_indices, nb::ndarray<> py_svecs,
    nb::ndarray<> py_multi, nb::ndarray<> py_masses, nb::ndarray<> py_p2s_map,
    nb::ndarray<> py_s2p_map, nb::ndarray<> py_band_indices,
    int64_t symmetrize_fc3_q, int64_t make_r0_average,
    nb::ndarray<> py_all_shortest, double cutoff_frequency,
    int64_t openmp_per_triplets) {
    Darray *fc3_normal_squared;
    Darray *freqs;
    _lapack_complex_double *eigvecs;
    int64_t (*triplets)[3];
    int64_t num_triplets;
    char *g_zero;
    int64_t (*bz_grid_addresses)[3];
    int64_t *D_diag;
    int64_t (*Q)[3];
    double *fc3;
    char *fc3_nonzero_indices;
    double (*svecs)[3];
    int64_t (*multi)[2];
    double *masses;
    char *all_shortest;
    int64_t *p2s;
    int64_t *s2p;
    int64_t *band_indices;
    int64_t multi_dims[2];
    int64_t i;
    int64_t is_compact_fc3;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    freqs = convert_to_darray(py_frequencies);
    /* npy_cdouble and lapack_complex_double may not be compatible. */
    /* So eigenvectors should not be used in Python side */
    eigvecs = (_lapack_complex_double *)py_eigenvectors.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    num_triplets = (int64_t)py_triplets.shape(0);
    g_zero = (char *)py_g_zero.data();
    bz_grid_addresses = (int64_t (*)[3])py_bz_grid_addresses.data();
    D_diag = (int64_t *)py_D_diag.data();
    Q = (int64_t (*)[3])py_Q.data();
    fc3 = (double *)py_fc3.data();
    if (py_fc3.shape(0) == py_fc3.shape(1)) {
        is_compact_fc3 = 0;
    } else {
        is_compact_fc3 = 1;
    }
    fc3_nonzero_indices = (char *)py_fc3_nonzero_indices.data();
    svecs = (double (*)[3])py_svecs.data();
    for (i = 0; i < 2; i++) {
        multi_dims[i] = py_multi.shape(i);
    }
    multi = (int64_t (*)[2])py_multi.data();
    masses = (double *)py_masses.data();
    p2s = (int64_t *)py_p2s_map.data();
    s2p = (int64_t *)py_s2p_map.data();
    band_indices = (int64_t *)py_band_indices.data();
    all_shortest = (char *)py_all_shortest.data();

    ph3py_get_interaction(fc3_normal_squared, g_zero, freqs, eigvecs, triplets,
                          num_triplets, bz_grid_addresses, D_diag, Q, fc3,
                          fc3_nonzero_indices, is_compact_fc3, svecs,
                          multi_dims, multi, masses, p2s, s2p, band_indices,
                          symmetrize_fc3_q, make_r0_average, all_shortest,
                          cutoff_frequency, openmp_per_triplets);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
    free(freqs);
    freqs = NULL;
}

void py_get_pp_collision(
    nb::ndarray<> py_gamma, nb::ndarray<> py_relative_grid_address,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_bz_map,
    int64_t bz_grid_type, nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
    nb::ndarray<> py_fc3, nb::ndarray<> py_fc3_nonzero_indices,
    nb::ndarray<> py_svecs, nb::ndarray<> py_multi, nb::ndarray<> py_masses,
    nb::ndarray<> py_p2s_map, nb::ndarray<> py_s2p_map,
    nb::ndarray<> py_band_indices, nb::ndarray<> py_temperatures_THz,
    int64_t is_NU, int64_t symmetrize_fc3_q, int64_t make_r0_average,
    nb::ndarray<> py_all_shortest, double cutoff_frequency,
    int64_t openmp_per_triplets) {
    double *gamma;
    int64_t (*relative_grid_address)[4][3];
    double *frequencies;
    _lapack_complex_double *eigenvectors;
    int64_t (*triplets)[3];
    int64_t num_triplets;
    int64_t *triplet_weights;
    int64_t (*bz_grid_addresses)[3];
    int64_t *bz_map;
    int64_t *D_diag;
    int64_t (*Q)[3];
    double *fc3;
    char *fc3_nonzero_indices;
    double (*svecs)[3];
    int64_t (*multi)[2];
    double *masses;
    int64_t *p2s;
    int64_t *s2p;
    Larray *band_indices;
    Darray *temperatures_THz;
    char *all_shortest;
    int64_t multi_dims[2];
    int64_t i;
    int64_t is_compact_fc3;

    gamma = (double *)py_gamma.data();
    relative_grid_address = (int64_t (*)[4][3])py_relative_grid_address.data();
    frequencies = (double *)py_frequencies.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    num_triplets = (int64_t)py_triplets.shape(0);
    triplet_weights = (int64_t *)py_triplet_weights.data();
    bz_grid_addresses = (int64_t (*)[3])py_bz_grid_addresses.data();
    bz_map = (int64_t *)py_bz_map.data();
    D_diag = (int64_t *)py_D_diag.data();
    Q = (int64_t (*)[3])py_Q.data();
    fc3 = (double *)py_fc3.data();
    if (py_fc3.shape(0) == py_fc3.shape(1)) {
        is_compact_fc3 = 0;
    } else {
        is_compact_fc3 = 1;
    }
    fc3_nonzero_indices = (char *)py_fc3_nonzero_indices.data();
    svecs = (double (*)[3])py_svecs.data();
    for (i = 0; i < 2; i++) {
        multi_dims[i] = py_multi.shape(i);
    }
    multi = (int64_t (*)[2])py_multi.data();
    masses = (double *)py_masses.data();
    p2s = (int64_t *)py_p2s_map.data();
    s2p = (int64_t *)py_s2p_map.data();
    band_indices = convert_to_larray(py_band_indices);
    temperatures_THz = convert_to_darray(py_temperatures_THz);
    all_shortest = (char *)py_all_shortest.data();

    ph3py_get_pp_collision(
        gamma, relative_grid_address, frequencies, eigenvectors, triplets,
        num_triplets, triplet_weights, bz_grid_addresses, bz_map, bz_grid_type,
        D_diag, Q, fc3, fc3_nonzero_indices, is_compact_fc3, svecs, multi_dims,
        multi, masses, p2s, s2p, band_indices, temperatures_THz, is_NU,
        symmetrize_fc3_q, make_r0_average, all_shortest, cutoff_frequency,
        openmp_per_triplets);

    free(band_indices);
    band_indices = NULL;
    free(temperatures_THz);
    temperatures_THz = NULL;
}

void py_get_pp_collision_with_sigma(
    nb::ndarray<> py_gamma, double sigma, double sigma_cutoff,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_D_diag,
    nb::ndarray<> py_Q, nb::ndarray<> py_fc3,
    nb::ndarray<> py_fc3_nonzero_indices, nb::ndarray<> py_svecs,
    nb::ndarray<> py_multi, nb::ndarray<> py_masses, nb::ndarray<> py_p2s_map,
    nb::ndarray<> py_s2p_map, nb::ndarray<> py_band_indices,
    nb::ndarray<> py_temperatures_THz, int64_t is_NU, int64_t symmetrize_fc3_q,
    int64_t make_r0_average, nb::ndarray<> py_all_shortest,
    double cutoff_frequency, int64_t openmp_per_triplets) {
    double *gamma;
    double *frequencies;
    _lapack_complex_double *eigenvectors;
    int64_t (*triplets)[3];
    int64_t num_triplets;
    int64_t *triplet_weights;
    int64_t (*bz_grid_addresses)[3];
    int64_t *D_diag;
    int64_t (*Q)[3];
    double *fc3;
    char *fc3_nonzero_indices;
    double (*svecs)[3];
    int64_t (*multi)[2];
    double *masses;
    int64_t *p2s;
    int64_t *s2p;
    Larray *band_indices;
    Darray *temperatures_THz;
    char *all_shortest;
    int64_t multi_dims[2];
    int64_t i;
    int64_t is_compact_fc3;

    gamma = (double *)py_gamma.data();
    frequencies = (double *)py_frequencies.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    num_triplets = (int64_t)py_triplets.shape(0);
    triplet_weights = (int64_t *)py_triplet_weights.data();
    bz_grid_addresses = (int64_t (*)[3])py_bz_grid_addresses.data();
    D_diag = (int64_t *)py_D_diag.data();
    Q = (int64_t (*)[3])py_Q.data();
    fc3 = (double *)py_fc3.data();
    if (py_fc3.shape(0) == py_fc3.shape(1)) {
        is_compact_fc3 = 0;
    } else {
        is_compact_fc3 = 1;
    }
    fc3_nonzero_indices = (char *)py_fc3_nonzero_indices.data();
    svecs = (double (*)[3])py_svecs.data();
    for (i = 0; i < 2; i++) {
        multi_dims[i] = py_multi.shape(i);
    }
    multi = (int64_t (*)[2])py_multi.data();
    masses = (double *)py_masses.data();
    p2s = (int64_t *)py_p2s_map.data();
    s2p = (int64_t *)py_s2p_map.data();
    band_indices = convert_to_larray(py_band_indices);
    temperatures_THz = convert_to_darray(py_temperatures_THz);
    all_shortest = (char *)py_all_shortest.data();

    ph3py_get_pp_collision_with_sigma(
        gamma, sigma, sigma_cutoff, frequencies, eigenvectors, triplets,
        num_triplets, triplet_weights, bz_grid_addresses, D_diag, Q, fc3,
        fc3_nonzero_indices, is_compact_fc3, svecs, multi_dims, multi, masses,
        p2s, s2p, band_indices, temperatures_THz, is_NU, symmetrize_fc3_q,
        make_r0_average, all_shortest, cutoff_frequency, openmp_per_triplets);

    free(band_indices);
    band_indices = NULL;
    free(temperatures_THz);
    temperatures_THz = NULL;
}

void py_get_imag_self_energy_with_g(
    nb::ndarray<> py_gamma, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_frequencies, double temperature_THz, nb::ndarray<> py_g,
    nb::ndarray<> py_g_zero, double cutoff_frequency,
    int64_t frequency_point_index) {
    Darray *fc3_normal_squared;
    double *gamma;
    double *g;
    char *g_zero;
    double *frequencies;
    int64_t (*triplets)[3];
    int64_t *triplet_weights;
    int64_t num_frequency_points;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    gamma = (double *)py_gamma.data();
    g = (double *)py_g.data();
    g_zero = (char *)py_g_zero.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    triplet_weights = (int64_t *)py_triplet_weights.data();
    num_frequency_points = (int64_t)py_g.shape(2);

    ph3py_get_imag_self_energy_at_bands_with_g(
        gamma, fc3_normal_squared, frequencies, triplets, triplet_weights, g,
        g_zero, temperature_THz, cutoff_frequency, num_frequency_points,
        frequency_point_index);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_detailed_imag_self_energy_with_g(
    nb::ndarray<> py_gamma_detail, nb::ndarray<> py_gamma_N,
    nb::ndarray<> py_gamma_U, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_frequencies,
    double temperature_THz, nb::ndarray<> py_g, nb::ndarray<> py_g_zero,
    double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *gamma_detail;
    double *gamma_N;
    double *gamma_U;
    double *g;
    char *g_zero;
    double *frequencies;
    int64_t (*triplets)[3];
    int64_t *triplet_weights;
    int64_t (*bz_grid_addresses)[3];

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    gamma_detail = (double *)py_gamma_detail.data();
    gamma_N = (double *)py_gamma_N.data();
    gamma_U = (double *)py_gamma_U.data();
    g = (double *)py_g.data();
    g_zero = (char *)py_g_zero.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    triplet_weights = (int64_t *)py_triplet_weights.data();
    bz_grid_addresses = (int64_t (*)[3])py_bz_grid_addresses.data();

    ph3py_get_detailed_imag_self_energy_at_bands_with_g(
        gamma_detail, gamma_N, gamma_U, fc3_normal_squared, frequencies,
        triplets, triplet_weights, bz_grid_addresses, g, g_zero,
        temperature_THz, cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_real_self_energy_at_bands(
    nb::ndarray<> py_shift, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_band_indices,
    double temperature_THz, double epsilon, double unit_conversion_factor,
    double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *shift;
    double *frequencies;
    int64_t *band_indices;
    int64_t (*triplets)[3];
    int64_t *triplet_weights;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    shift = (double *)py_shift.data();
    frequencies = (double *)py_frequencies.data();
    band_indices = (int64_t *)py_band_indices.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    triplet_weights = (int64_t *)py_triplet_weights.data();

    ph3py_get_real_self_energy_at_bands(
        shift, fc3_normal_squared, band_indices, frequencies, triplets,
        triplet_weights, epsilon, temperature_THz, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_real_self_energy_at_frequency_point(
    nb::ndarray<> py_shift, double frequency_point,
    nb::ndarray<> py_fc3_normal_squared, nb::ndarray<> py_triplets,
    nb::ndarray<> py_triplet_weights, nb::ndarray<> py_frequencies,
    nb::ndarray<> py_band_indices, double temperature_THz, double epsilon,
    double unit_conversion_factor, double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *shift;
    double *frequencies;
    int64_t *band_indices;
    int64_t (*triplets)[3];
    int64_t *triplet_weights;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    shift = (double *)py_shift.data();
    frequencies = (double *)py_frequencies.data();
    band_indices = (int64_t *)py_band_indices.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    triplet_weights = (int64_t *)py_triplet_weights.data();

    ph3py_get_real_self_energy_at_frequency_point(
        shift, frequency_point, fc3_normal_squared, band_indices, frequencies,
        triplets, triplet_weights, epsilon, temperature_THz,
        unit_conversion_factor, cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_collision_matrix(
    nb::ndarray<> py_collision_matrix, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_g, nb::ndarray<> py_triplets,
    nb::ndarray<> py_triplets_map, nb::ndarray<> py_map_q,
    nb::ndarray<> py_rotated_grid_points, nb::ndarray<> py_rotations_cartesian,
    double temperature_THz, double unit_conversion_factor,
    double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *collision_matrix;
    double *g;
    double *frequencies;
    int64_t (*triplets)[3];
    int64_t *triplets_map;
    int64_t *map_q;
    int64_t *rotated_grid_points;
    int64_t num_gp, num_ir_gp, num_rot;
    double *rotations_cartesian;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    collision_matrix = (double *)py_collision_matrix.data();
    g = (double *)py_g.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    triplets_map = (int64_t *)py_triplets_map.data();
    num_gp = (int64_t)py_triplets_map.shape(0);
    map_q = (int64_t *)py_map_q.data();
    rotated_grid_points = (int64_t *)py_rotated_grid_points.data();
    num_ir_gp = (int64_t)py_rotated_grid_points.shape(0);
    num_rot = (int64_t)py_rotated_grid_points.shape(1);
    rotations_cartesian = (double *)py_rotations_cartesian.data();

    assert(num_rot == py_rotations_cartesian.shape(0));
    assert(num_gp == py_frequencies.shape(0));

    ph3py_get_collision_matrix(collision_matrix, fc3_normal_squared,
                               frequencies, triplets, triplets_map, map_q,
                               rotated_grid_points, rotations_cartesian, g,
                               num_ir_gp, num_gp, num_rot, temperature_THz,
                               unit_conversion_factor, cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_reducible_collision_matrix(
    nb::ndarray<> py_collision_matrix, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_g, nb::ndarray<> py_triplets,
    nb::ndarray<> py_triplets_map, nb::ndarray<> py_map_q,
    double temperature_THz, double unit_conversion_factor,
    double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *collision_matrix;
    double *g;
    double *frequencies;
    int64_t (*triplets)[3];
    int64_t *triplets_map;
    int64_t num_gp;
    int64_t *map_q;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    collision_matrix = (double *)py_collision_matrix.data();
    g = (double *)py_g.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    triplets_map = (int64_t *)py_triplets_map.data();
    num_gp = (int64_t)py_triplets_map.shape(0);
    map_q = (int64_t *)py_map_q.data();

    ph3py_get_reducible_collision_matrix(
        collision_matrix, fc3_normal_squared, frequencies, triplets,
        triplets_map, map_q, g, num_gp, temperature_THz, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_symmetrize_collision_matrix(nb::ndarray<> py_collision_matrix) {
    double *collision_matrix;
    int64_t num_band, num_grid_points, num_temp, num_sigma;
    int64_t num_column;

    collision_matrix = (double *)py_collision_matrix.data();
    num_sigma = (int64_t)py_collision_matrix.shape(0);
    num_temp = (int64_t)py_collision_matrix.shape(1);
    num_grid_points = (int64_t)py_collision_matrix.shape(2);
    num_band = (int64_t)py_collision_matrix.shape(3);

    if (py_collision_matrix.ndim() == 8) {
        num_column = num_grid_points * num_band * 3;
    } else {
        num_column = num_grid_points * num_band;
    }

    ph3py_symmetrize_collision_matrix(collision_matrix, num_column, num_temp,
                                      num_sigma);
}

void py_expand_collision_matrix(nb::ndarray<> py_collision_matrix,
                                nb::ndarray<> py_ir_grid_points,
                                nb::ndarray<> py_rot_grid_points) {
    double *collision_matrix;
    int64_t *rot_grid_points;
    int64_t *ir_grid_points;
    int64_t num_band, num_grid_points, num_temp, num_sigma, num_rot, num_ir_gp;

    collision_matrix = (double *)py_collision_matrix.data();
    rot_grid_points = (int64_t *)py_rot_grid_points.data();
    ir_grid_points = (int64_t *)py_ir_grid_points.data();
    num_sigma = (int64_t)py_collision_matrix.shape(0);
    num_temp = (int64_t)py_collision_matrix.shape(1);
    num_grid_points = (int64_t)py_collision_matrix.shape(2);
    num_band = (int64_t)py_collision_matrix.shape(3);
    num_rot = (int64_t)py_rot_grid_points.shape(0);
    num_ir_gp = (int64_t)py_ir_grid_points.shape(0);

    ph3py_expand_collision_matrix(collision_matrix, rot_grid_points,
                                  ir_grid_points, num_ir_gp, num_grid_points,
                                  num_rot, num_sigma, num_temp, num_band);
}

void py_distribute_fc3(nb::ndarray<> force_constants_third, int64_t target,
                       int64_t source, nb::ndarray<> atom_mapping_py,
                       nb::ndarray<> rotation_cart_inv) {
    double *fc3;
    double *rot_cart_inv;
    int64_t *atom_mapping;
    int64_t num_atom;

    fc3 = (double *)force_constants_third.data();
    rot_cart_inv = (double *)rotation_cart_inv.data();
    atom_mapping = (int64_t *)atom_mapping_py.data();
    num_atom = (int64_t)atom_mapping_py.shape(0);

    ph3py_distribute_fc3(fc3, target, source, atom_mapping, num_atom,
                         rot_cart_inv);
}

void py_rotate_delta_fc2s(nb::ndarray<> py_fc3, nb::ndarray<> py_delta_fc2s,
                          nb::ndarray<> py_inv_U,
                          nb::ndarray<> py_site_sym_cart,
                          nb::ndarray<> py_rot_map_syms) {
    double (*fc3)[3][3][3];
    double (*delta_fc2s)[3][3];
    double *inv_U;
    double (*site_sym_cart)[3][3];
    int64_t *rot_map_syms;
    int64_t num_atom, num_disp, num_site_sym;

    /* (num_atom, num_atom, 3, 3, 3) */
    fc3 = (double (*)[3][3][3])py_fc3.data();
    /* (n_u1, num_atom, num_atom, 3, 3) */
    delta_fc2s = (double (*)[3][3])py_delta_fc2s.data();
    /* (3, n_u1 * n_sym) */
    inv_U = (double *)py_inv_U.data();
    /* (n_sym, 3, 3) */
    site_sym_cart = (double (*)[3][3])py_site_sym_cart.data();
    /* (n_sym, natom) */
    rot_map_syms = (int64_t *)py_rot_map_syms.data();

    num_atom = (int64_t)py_fc3.shape(0);
    num_disp = (int64_t)py_delta_fc2s.shape(0);
    num_site_sym = (int64_t)py_site_sym_cart.shape(0);

    ph3py_rotate_delta_fc2(fc3, delta_fc2s, inv_U, site_sym_cart, rot_map_syms,
                           num_atom, num_site_sym, num_disp);
}

void py_get_isotope_strength(
    nb::ndarray<> py_gamma, int64_t grid_point, nb::ndarray<> py_ir_grid_points,
    nb::ndarray<> py_weights, nb::ndarray<> py_mass_variances,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_band_indices, double sigma, double cutoff_frequency) {
    double *gamma;
    double *frequencies;
    int64_t *ir_grid_points;
    double *weights;
    _lapack_complex_double *eigenvectors;
    int64_t *band_indices;
    double *mass_variances;
    int64_t num_band, num_band0, num_ir_grid_points;

    gamma = (double *)py_gamma.data();
    frequencies = (double *)py_frequencies.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    ir_grid_points = (int64_t *)py_ir_grid_points.data();
    weights = (double *)py_weights.data();
    band_indices = (int64_t *)py_band_indices.data();
    mass_variances = (double *)py_mass_variances.data();
    num_band = (int64_t)py_frequencies.shape(1);
    num_band0 = (int64_t)py_band_indices.shape(0);
    num_ir_grid_points = (int64_t)py_ir_grid_points.shape(0);

    ph3py_get_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        eigenvectors, num_ir_grid_points, band_indices, num_band, num_band0,
        sigma, cutoff_frequency);
}

void py_get_thm_isotope_strength(
    nb::ndarray<> py_gamma, int64_t grid_point, nb::ndarray<> py_ir_grid_points,
    nb::ndarray<> py_weights, nb::ndarray<> py_mass_variances,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_band_indices, nb::ndarray<> py_integration_weights,
    double cutoff_frequency) {
    double *gamma;
    double *frequencies;
    int64_t *ir_grid_points;
    double *weights;
    _lapack_complex_double *eigenvectors;
    int64_t *band_indices;
    double *mass_variances;
    int64_t num_band, num_band0, num_ir_grid_points;
    double *integration_weights;

    gamma = (double *)py_gamma.data();
    frequencies = (double *)py_frequencies.data();
    ir_grid_points = (int64_t *)py_ir_grid_points.data();
    weights = (double *)py_weights.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    band_indices = (int64_t *)py_band_indices.data();
    mass_variances = (double *)py_mass_variances.data();
    num_band = (int64_t)py_frequencies.shape(1);
    num_band0 = (int64_t)py_band_indices.shape(0);
    integration_weights = (double *)py_integration_weights.data();
    num_ir_grid_points = (int64_t)py_ir_grid_points.shape(0);

    ph3py_get_thm_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        eigenvectors, num_ir_grid_points, band_indices, num_band, num_band0,
        integration_weights, cutoff_frequency);
}

void py_get_permutation_symmetry_fc3(nb::ndarray<> py_fc3) {
    double *fc3;
    int64_t num_atom;

    fc3 = (double *)py_fc3.data();
    num_atom = (int64_t)py_fc3.shape(0);

    ph3py_get_permutation_symmetry_fc3(fc3, num_atom);
}

void py_get_permutation_symmetry_compact_fc3(nb::ndarray<> py_fc3,
                                             nb::ndarray<> py_permutations,
                                             nb::ndarray<> py_s2pp_map,
                                             nb::ndarray<> py_p2s_map,
                                             nb::ndarray<> py_nsym_list) {
    double *fc3;
    int64_t *s2pp;
    int64_t *p2s;
    int64_t *nsym_list;
    int64_t *perms;
    int64_t n_patom, n_satom;

    fc3 = (double *)py_fc3.data();
    perms = (int64_t *)py_permutations.data();
    s2pp = (int64_t *)py_s2pp_map.data();
    p2s = (int64_t *)py_p2s_map.data();
    nsym_list = (int64_t *)py_nsym_list.data();
    n_patom = (int64_t)py_fc3.shape(0);
    n_satom = (int64_t)py_fc3.shape(1);

    ph3py_get_permutation_symmetry_compact_fc3(fc3, p2s, s2pp, nsym_list, perms,
                                               n_satom, n_patom);
}

void py_transpose_compact_fc3(nb::ndarray<> py_fc3,
                              nb::ndarray<> py_permutations,
                              nb::ndarray<> py_s2pp_map,
                              nb::ndarray<> py_p2s_map,
                              nb::ndarray<> py_nsym_list, int64_t t_type) {
    double *fc3;
    int64_t *s2pp;
    int64_t *p2s;
    int64_t *nsym_list;
    int64_t *perms;
    int64_t n_patom, n_satom;

    fc3 = (double *)py_fc3.data();
    perms = (int64_t *)py_permutations.data();
    s2pp = (int64_t *)py_s2pp_map.data();
    p2s = (int64_t *)py_p2s_map.data();
    nsym_list = (int64_t *)py_nsym_list.data();
    n_patom = (int64_t)py_fc3.shape(0);
    n_satom = (int64_t)py_fc3.shape(1);

    ph3py_transpose_compact_fc3(fc3, p2s, s2pp, nsym_list, perms, n_satom,
                                n_patom, t_type);
}

void py_get_thm_relative_grid_address(nb::ndarray<> py_relative_grid_address,
                                      nb::ndarray<> py_reciprocal_lattice_py) {
    int64_t (*relative_grid_address)[4][3];
    double (*reciprocal_lattice)[3];

    relative_grid_address = (int64_t (*)[4][3])py_relative_grid_address.data();
    reciprocal_lattice = (double (*)[3])py_reciprocal_lattice_py.data();

    ph3py_get_relative_grid_address(relative_grid_address, reciprocal_lattice);
}

void py_get_neighboring_grid_points(nb::ndarray<> py_relative_grid_points,
                                    nb::ndarray<> py_grid_points,
                                    nb::ndarray<> py_relative_grid_address,
                                    nb::ndarray<> py_D_diag,
                                    nb::ndarray<> py_bz_grid_address,
                                    nb::ndarray<> py_bz_map,
                                    int64_t bz_grid_type) {
    int64_t *relative_grid_points;
    int64_t *grid_points;
    int64_t num_grid_points, num_relative_grid_address;
    int64_t (*relative_grid_address)[3];
    int64_t *D_diag;
    int64_t (*bz_grid_address)[3];
    int64_t *bz_map;

    relative_grid_points = (int64_t *)py_relative_grid_points.data();
    grid_points = (int64_t *)py_grid_points.data();
    num_grid_points = (int64_t)py_grid_points.shape(0);
    relative_grid_address = (int64_t (*)[3])py_relative_grid_address.data();
    num_relative_grid_address = (int64_t)py_relative_grid_address.shape(0);
    D_diag = (int64_t *)py_D_diag.data();
    bz_grid_address = (int64_t (*)[3])py_bz_grid_address.data();
    bz_map = (int64_t *)py_bz_map.data();

    ph3py_get_neighboring_gird_points(
        relative_grid_points, grid_points, relative_grid_address, D_diag,
        bz_grid_address, bz_map, bz_grid_type, num_grid_points,
        num_relative_grid_address);
}

void py_get_thm_integration_weights_at_grid_points(
    nb::ndarray<> py_iw, nb::ndarray<> py_frequency_points,
    nb::ndarray<> py_relative_grid_address, nb::ndarray<> py_D_diag,
    nb::ndarray<> py_grid_points, nb::ndarray<> py_frequencies,
    nb::ndarray<> py_bz_grid_address, nb::ndarray<> py_bz_map,
    nb::ndarray<> py_gp2irgp_map, int64_t bz_grid_type, const char *function) {
    double *iw;
    double *frequency_points;
    int64_t num_frequency_points, num_band, num_gp;
    int64_t (*relative_grid_address)[4][3];
    int64_t *D_diag;
    int64_t *grid_points;
    int64_t (*bz_grid_address)[3];
    int64_t *bz_map;
    int64_t *gp2irgp_map;
    double *frequencies;

    iw = (double *)py_iw.data();
    frequency_points = (double *)py_frequency_points.data();
    num_frequency_points = (int64_t)py_frequency_points.shape(0);
    relative_grid_address = (int64_t (*)[4][3])py_relative_grid_address.data();
    D_diag = (int64_t *)py_D_diag.data();
    grid_points = (int64_t *)py_grid_points.data();
    num_gp = (int64_t)py_grid_points.shape(0);
    bz_grid_address = (int64_t (*)[3])py_bz_grid_address.data();
    bz_map = (int64_t *)py_bz_map.data();
    gp2irgp_map = (int64_t *)py_gp2irgp_map.data();
    frequencies = (double *)py_frequencies.data();
    num_band = (int64_t)py_frequencies.shape(1);

    ph3py_get_thm_integration_weights_at_grid_points(
        iw, frequency_points, num_frequency_points, num_band, num_gp,
        relative_grid_address, D_diag, grid_points, bz_grid_address, bz_map,
        bz_grid_type, frequencies, gp2irgp_map, function[0]);
}

int64_t py_tpl_get_triplets_reciprocal_mesh_at_q(
    nb::ndarray<> py_map_triplets, nb::ndarray<> py_map_q,
    int64_t fixed_grid_number, nb::ndarray<> py_D_diag,
    int64_t is_time_reversal, nb::ndarray<> py_rotations, int64_t swappable) {
    int64_t *map_triplets;
    int64_t *map_q;
    int64_t *D_diag;
    int64_t (*rot)[3][3];
    int64_t num_rot;
    int64_t num_ir;

    map_triplets = (int64_t *)py_map_triplets.data();
    map_q = (int64_t *)py_map_q.data();
    D_diag = (int64_t *)py_D_diag.data();
    rot = (int64_t (*)[3][3])py_rotations.data();
    num_rot = (int64_t)py_rotations.shape(0);

    num_ir = ph3py_get_triplets_reciprocal_mesh_at_q(
        map_triplets, map_q, fixed_grid_number, D_diag, is_time_reversal,
        num_rot, rot, swappable);

    return num_ir;
}

int64_t py_tpl_get_BZ_triplets_at_q(
    nb::ndarray<> py_triplets, int64_t grid_point,
    nb::ndarray<> py_bz_grid_address, nb::ndarray<> py_bz_map,
    nb::ndarray<> py_map_triplets, nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
    nb::ndarray<> py_reciprocal_lattice, int64_t bz_grid_type) {
    int64_t (*triplets)[3];
    int64_t (*bz_grid_address)[3];
    int64_t *bz_map;
    int64_t *map_triplets;
    int64_t num_map_triplets;
    int64_t *D_diag;
    int64_t (*Q)[3];
    double (*reciprocal_lattice)[3];
    int64_t num_ir;

    triplets = (int64_t (*)[3])py_triplets.data();
    bz_grid_address = (int64_t (*)[3])py_bz_grid_address.data();
    bz_map = (int64_t *)py_bz_map.data();
    map_triplets = (int64_t *)py_map_triplets.data();
    num_map_triplets = (int64_t)py_map_triplets.shape(0);
    D_diag = (int64_t *)py_D_diag.data();
    Q = (int64_t (*)[3])py_Q.data();
    reciprocal_lattice = (double (*)[3])py_reciprocal_lattice.data();

    num_ir = ph3py_get_BZ_triplets_at_q(
        triplets, grid_point, bz_grid_address, bz_map, map_triplets,
        num_map_triplets, D_diag, Q, reciprocal_lattice, bz_grid_type);

    return num_ir;
}

void py_get_triplets_integration_weights(
    nb::ndarray<> py_iw, nb::ndarray<> py_iw_zero,
    nb::ndarray<> py_frequency_points, nb::ndarray<> py_relative_grid_address,
    nb::ndarray<> py_D_diag, nb::ndarray<> py_triplets,
    nb::ndarray<> py_frequencies1, nb::ndarray<> py_frequencies2,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_bz_map,
    int64_t bz_grid_type, int64_t tp_type) {
    double *iw;
    char *iw_zero;
    double *frequency_points;
    int64_t (*relative_grid_address)[4][3];
    int64_t *D_diag;
    int64_t (*triplets)[3];
    int64_t (*bz_grid_addresses)[3];
    int64_t *bz_map;
    double *frequencies1, *frequencies2;
    int64_t num_band0, num_band1, num_band2, num_triplets;

    iw = (double *)py_iw.data();
    iw_zero = (char *)py_iw_zero.data();
    frequency_points = (double *)py_frequency_points.data();
    num_band0 = (int64_t)py_frequency_points.shape(0);
    relative_grid_address = (int64_t (*)[4][3])py_relative_grid_address.data();
    D_diag = (int64_t *)py_D_diag.data();
    triplets = (int64_t (*)[3])py_triplets.data();
    num_triplets = (int64_t)py_triplets.shape(0);
    bz_grid_addresses = (int64_t (*)[3])py_bz_grid_addresses.data();
    bz_map = (int64_t *)py_bz_map.data();
    frequencies1 = (double *)py_frequencies1.data();
    frequencies2 = (double *)py_frequencies2.data();
    num_band1 = (int64_t)py_frequencies1.shape(1);
    num_band2 = (int64_t)py_frequencies2.shape(1);

    ph3py_get_integration_weight(
        iw, iw_zero, frequency_points, num_band0, relative_grid_address, D_diag,
        triplets, num_triplets, bz_grid_addresses, bz_map, bz_grid_type,
        frequencies1, num_band1, frequencies2, num_band2, tp_type, 1);
}

void py_get_triplets_integration_weights_with_sigma(
    nb::ndarray<> py_iw, nb::ndarray<> py_iw_zero,
    nb::ndarray<> py_frequency_points, nb::ndarray<> py_triplets,
    nb::ndarray<> py_frequencies, double sigma, double sigma_cutoff) {
    double *iw;
    char *iw_zero;
    double *frequency_points;
    int64_t (*triplets)[3];
    double *frequencies;
    int64_t num_band0, num_band, num_iw, num_triplets;

    iw = (double *)py_iw.data();
    iw_zero = (char *)py_iw_zero.data();
    frequency_points = (double *)py_frequency_points.data();
    num_band0 = (int64_t)py_frequency_points.shape(0);
    triplets = (int64_t (*)[3])py_triplets.data();
    num_triplets = (int64_t)py_triplets.shape(0);
    frequencies = (double *)py_frequencies.data();
    num_band = (int64_t)py_frequencies.shape(1);
    num_iw = (int64_t)py_iw.shape(0);

    ph3py_get_integration_weight_with_sigma(
        iw, iw_zero, sigma, sigma_cutoff, frequency_points, num_band0, triplets,
        num_triplets, frequencies, num_band, num_iw);
}

int64_t py_get_default_colmat_solver() {
#if defined(MKL_BLAS) || defined(SCIPY_MKL_H) || !defined(NO_INCLUDE_LAPACKE)
    return (int64_t)1;
#else
    return (int64_t)4;
#endif
}

int64_t py_get_omp_max_threads() { return ph3py_get_max_threads(); }

#ifdef NO_INCLUDE_LAPACKE
int64_t py_include_lapacke() { return 0; }
#else
int64_t py_include_lapacke() { return 1; }
int64_t py_diagonalize_collision_matrix(nb::ndarray<> py_collision_matrix,
                                        nb::ndarray<> py_eigenvalues,
                                        int64_t i_sigma, int64_t i_temp,
                                        double cutoff, int64_t solver,
                                        int64_t is_pinv) {
    double *collision_matrix;
    double *eigvals;
    int64_t num_temp, num_grid_point, num_band;
    int64_t num_column, adrs_shift;
    int64_t info;

    collision_matrix = (double *)py_collision_matrix.data();
    eigvals = (double *)py_eigenvalues.data();

    if (py_collision_matrix.ndim() == 2) {
        num_temp = 1;
        num_column = py_collision_matrix.shape(1);
    } else {
        num_temp = py_collision_matrix.shape(1);
        num_grid_point = py_collision_matrix.shape(2);
        num_band = py_collision_matrix.shape(3);
        if (py_collision_matrix.ndim() == 8) {
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

    return info;
}

void py_pinv_from_eigensolution(nb::ndarray<> py_collision_matrix,
                                nb::ndarray<> py_eigenvalues, int64_t i_sigma,
                                int64_t i_temp, double cutoff,
                                int64_t pinv_method) {
    double *collision_matrix;
    double *eigvals;
    int64_t num_temp, num_grid_point, num_band;
    int64_t num_column, adrs_shift;

    collision_matrix = (double *)py_collision_matrix.data();
    eigvals = (double *)py_eigenvalues.data();
    num_temp = py_collision_matrix.shape(1);
    num_grid_point = py_collision_matrix.shape(2);
    num_band = py_collision_matrix.shape(3);

    if (py_collision_matrix.ndim() == 8) {
        num_column = num_grid_point * num_band * 3;
    } else {
        num_column = num_grid_point * num_band;
    }
    adrs_shift = (i_sigma * num_column * num_column * num_temp +
                  i_temp * num_column * num_column);

    /* show_colmat_info(py_collision_matrix, i_sigma, i_temp, adrs_shift); */

    ph3py_pinv_from_eigensolution(collision_matrix + adrs_shift, eigvals,
                                  num_column, cutoff, pinv_method);
}

int64_t py_lapacke_pinv(nb::ndarray<> data_out_py, nb::ndarray<> data_in_py,
                        double cutoff) {
    int64_t m;
    int64_t n;
    double *data_in;
    double *data_out;
    int64_t info;

    m = data_in_py.shape(0);
    n = data_in_py.shape(1);
    data_in = (double *)data_in_py.data();
    data_out = (double *)data_out_py.data();

    info = ph3py_phonopy_pinv(data_out, data_in, m, n, cutoff);

    return info;
}
#endif

NB_MODULE(_phono3py, m) {
    m.def("interaction", &py_get_interaction);
    m.def("pp_collision", &py_get_pp_collision);
    m.def("pp_collision_with_sigma", &py_get_pp_collision_with_sigma);
    m.def("imag_self_energy_with_g", &py_get_imag_self_energy_with_g);
    m.def("detailed_imag_self_energy_with_g",
          &py_get_detailed_imag_self_energy_with_g);
    m.def("real_self_energy_at_bands", &py_get_real_self_energy_at_bands);
    m.def("real_self_energy_at_frequency_point",
          &py_get_real_self_energy_at_frequency_point);
    m.def("collision_matrix", &py_get_collision_matrix);
    m.def("reducible_collision_matrix", &py_get_reducible_collision_matrix);
    m.def("symmetrize_collision_matrix", &py_symmetrize_collision_matrix);
    m.def("expand_collision_matrix", &py_expand_collision_matrix);
    m.def("distribute_fc3", &py_distribute_fc3);
    m.def("rotate_delta_fc2s", &py_rotate_delta_fc2s);
    m.def("isotope_strength", &py_get_isotope_strength);
    m.def("thm_isotope_strength", &py_get_thm_isotope_strength);
    m.def("permutation_symmetry_fc3", &py_get_permutation_symmetry_fc3);
    m.def("permutation_symmetry_compact_fc3",
          &py_get_permutation_symmetry_compact_fc3);
    m.def("transpose_compact_fc3", &py_transpose_compact_fc3);
    m.def("tetrahedra_relative_grid_address",
          &py_get_thm_relative_grid_address);
    m.def("neighboring_grid_points", &py_get_neighboring_grid_points);
    m.def("integration_weights_at_grid_points",
          &py_get_thm_integration_weights_at_grid_points);
    m.def("triplets_reciprocal_mesh_at_q",
          &py_tpl_get_triplets_reciprocal_mesh_at_q);
    m.def("BZ_triplets_at_q", &py_tpl_get_BZ_triplets_at_q);
    m.def("triplets_integration_weights", &py_get_triplets_integration_weights);
    m.def("triplets_integration_weights_with_sigma",
          &py_get_triplets_integration_weights_with_sigma);
    m.def("default_colmat_solver", &py_get_default_colmat_solver);
    m.def("omp_max_threads", &py_get_omp_max_threads);
    m.def("include_lapacke", &py_include_lapacke);
#ifndef NO_INCLUDE_LAPACKE
    m.def("diagonalize_collision_matrix", &py_diagonalize_collision_matrix);
    m.def("pinv_from_eigensolution", &py_pinv_from_eigensolution);
    m.def("lapacke_pinv", &py_lapacke_pinv);
#endif
}
