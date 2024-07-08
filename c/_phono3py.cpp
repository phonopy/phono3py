#include <assert.h>
#include <math.h>
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

// static void show_colmat_info(const PyArrayObject *py_collision_matrix,
//                              const long i_sigma, const long i_temp,
//                              const long adrs_shift) {
//     long i;

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

void py_get_pp_collision(
    nb::ndarray<> py_gamma, nb::ndarray<> py_relative_grid_address,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_bz_map,
    long bz_grid_type, nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
    nb::ndarray<> py_fc3, nb::ndarray<> py_svecs, nb::ndarray<> py_multi,
    nb::ndarray<> py_masses, nb::ndarray<> py_p2s_map, nb::ndarray<> py_s2p_map,
    nb::ndarray<> py_band_indices, nb::ndarray<> py_temperatures, long is_NU,
    long symmetrize_fc3_q, long make_r0_average, nb::ndarray<> py_all_shortest,
    double cutoff_frequency, long openmp_per_triplets) {
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

    gamma = (double *)py_gamma.data();
    relative_grid_address = (long(*)[4][3])py_relative_grid_address.data();
    frequencies = (double *)py_frequencies.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    triplets = (long(*)[3])py_triplets.data();
    num_triplets = (long)py_triplets.shape(0);
    triplet_weights = (long *)py_triplet_weights.data();
    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();
    bz_map = (long *)py_bz_map.data();
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
    band_indices = convert_to_larray(py_band_indices);
    temperatures = convert_to_darray(py_temperatures);
    all_shortest = (char *)py_all_shortest.data();

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
}

void py_get_pp_collision_with_sigma(
    nb::ndarray<> py_gamma, double sigma, double sigma_cutoff,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_D_diag,
    nb::ndarray<> py_Q, nb::ndarray<> py_fc3, nb::ndarray<> py_svecs,
    nb::ndarray<> py_multi, nb::ndarray<> py_masses, nb::ndarray<> py_p2s_map,
    nb::ndarray<> py_s2p_map, nb::ndarray<> py_band_indices,
    nb::ndarray<> py_temperatures, long is_NU, long symmetrize_fc3_q,
    long make_r0_average, nb::ndarray<> py_all_shortest,
    double cutoff_frequency, long openmp_per_triplets) {
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

    gamma = (double *)py_gamma.data();
    frequencies = (double *)py_frequencies.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    triplets = (long(*)[3])py_triplets.data();
    num_triplets = (long)py_triplets.shape(0);
    triplet_weights = (long *)py_triplet_weights.data();
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
    band_indices = convert_to_larray(py_band_indices);
    temperatures = convert_to_darray(py_temperatures);
    all_shortest = (char *)py_all_shortest.data();

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
}

void py_get_imag_self_energy_with_g(
    nb::ndarray<> py_gamma, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_frequencies, double temperature, nb::ndarray<> py_g,
    nb::ndarray<> py_g_zero, double cutoff_frequency,
    long frequency_point_index) {
    Darray *fc3_normal_squared;
    double *gamma;
    double *g;
    char *g_zero;
    double *frequencies;
    long(*triplets)[3];
    long *triplet_weights;
    long num_frequency_points;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    gamma = (double *)py_gamma.data();
    g = (double *)py_g.data();
    g_zero = (char *)py_g_zero.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (long(*)[3])py_triplets.data();
    triplet_weights = (long *)py_triplet_weights.data();
    num_frequency_points = (long)py_g.shape(2);

    ph3py_get_imag_self_energy_at_bands_with_g(
        gamma, fc3_normal_squared, frequencies, triplets, triplet_weights, g,
        g_zero, temperature, cutoff_frequency, num_frequency_points,
        frequency_point_index);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_detailed_imag_self_energy_with_g(
    nb::ndarray<> py_gamma_detail, nb::ndarray<> py_gamma_N,
    nb::ndarray<> py_gamma_U, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_frequencies,
    double temperature, nb::ndarray<> py_g, nb::ndarray<> py_g_zero,
    double cutoff_frequency) {
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

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    gamma_detail = (double *)py_gamma_detail.data();
    gamma_N = (double *)py_gamma_N.data();
    gamma_U = (double *)py_gamma_U.data();
    g = (double *)py_g.data();
    g_zero = (char *)py_g_zero.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (long(*)[3])py_triplets.data();
    triplet_weights = (long *)py_triplet_weights.data();
    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();

    ph3py_get_detailed_imag_self_energy_at_bands_with_g(
        gamma_detail, gamma_N, gamma_U, fc3_normal_squared, frequencies,
        triplets, triplet_weights, bz_grid_addresses, g, g_zero, temperature,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_real_self_energy_at_bands(
    nb::ndarray<> py_shift, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_triplets, nb::ndarray<> py_triplet_weights,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_band_indices,
    double temperature, double epsilon, double unit_conversion_factor,
    double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *shift;
    double *frequencies;
    long *band_indices;
    long(*triplets)[3];
    long *triplet_weights;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    shift = (double *)py_shift.data();
    frequencies = (double *)py_frequencies.data();
    band_indices = (long *)py_band_indices.data();
    triplets = (long(*)[3])py_triplets.data();
    triplet_weights = (long *)py_triplet_weights.data();

    ph3py_get_real_self_energy_at_bands(
        shift, fc3_normal_squared, band_indices, frequencies, triplets,
        triplet_weights, epsilon, temperature, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_real_self_energy_at_frequency_point(
    nb::ndarray<> py_shift, double frequency_point,
    nb::ndarray<> py_fc3_normal_squared, nb::ndarray<> py_triplets,
    nb::ndarray<> py_triplet_weights, nb::ndarray<> py_frequencies,
    nb::ndarray<> py_band_indices, double temperature, double epsilon,
    double unit_conversion_factor, double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *shift;
    double *frequencies;
    long *band_indices;
    long(*triplets)[3];
    long *triplet_weights;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    shift = (double *)py_shift.data();
    frequencies = (double *)py_frequencies.data();
    band_indices = (long *)py_band_indices.data();
    triplets = (long(*)[3])py_triplets.data();
    triplet_weights = (long *)py_triplet_weights.data();

    ph3py_get_real_self_energy_at_frequency_point(
        shift, frequency_point, fc3_normal_squared, band_indices, frequencies,
        triplets, triplet_weights, epsilon, temperature, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_collision_matrix(
    nb::ndarray<> py_collision_matrix, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_g, nb::ndarray<> py_triplets,
    nb::ndarray<> py_triplets_map, nb::ndarray<> py_map_q,
    nb::ndarray<> py_rotated_grid_points, nb::ndarray<> py_rotations_cartesian,
    double temperature, double unit_conversion_factor,
    double cutoff_frequency) {
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

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    collision_matrix = (double *)py_collision_matrix.data();
    g = (double *)py_g.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (long(*)[3])py_triplets.data();
    triplets_map = (long *)py_triplets_map.data();
    num_gp = (long)py_triplets_map.shape(0);
    map_q = (long *)py_map_q.data();
    rotated_grid_points = (long *)py_rotated_grid_points.data();
    num_ir_gp = (long)py_rotated_grid_points.shape(0);
    num_rot = (long)py_rotated_grid_points.shape(1);
    rotations_cartesian = (double *)py_rotations_cartesian.data();

    assert(num_rot == py_rotations_cartesian.shape(0));
    assert(num_gp == py_frequencies.shape(0));

    ph3py_get_collision_matrix(collision_matrix, fc3_normal_squared,
                               frequencies, triplets, triplets_map, map_q,
                               rotated_grid_points, rotations_cartesian, g,
                               num_ir_gp, num_gp, num_rot, temperature,
                               unit_conversion_factor, cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_get_reducible_collision_matrix(
    nb::ndarray<> py_collision_matrix, nb::ndarray<> py_fc3_normal_squared,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_g, nb::ndarray<> py_triplets,
    nb::ndarray<> py_triplets_map, nb::ndarray<> py_map_q, double temperature,
    double unit_conversion_factor, double cutoff_frequency) {
    Darray *fc3_normal_squared;
    double *collision_matrix;
    double *g;
    double *frequencies;
    long(*triplets)[3];
    long *triplets_map;
    long num_gp;
    long *map_q;

    fc3_normal_squared = convert_to_darray(py_fc3_normal_squared);
    collision_matrix = (double *)py_collision_matrix.data();
    g = (double *)py_g.data();
    frequencies = (double *)py_frequencies.data();
    triplets = (long(*)[3])py_triplets.data();
    triplets_map = (long *)py_triplets_map.data();
    num_gp = (long)py_triplets_map.shape(0);
    map_q = (long *)py_map_q.data();

    ph3py_get_reducible_collision_matrix(
        collision_matrix, fc3_normal_squared, frequencies, triplets,
        triplets_map, map_q, g, num_gp, temperature, unit_conversion_factor,
        cutoff_frequency);

    free(fc3_normal_squared);
    fc3_normal_squared = NULL;
}

void py_symmetrize_collision_matrix(nb::ndarray<> py_collision_matrix) {
    double *collision_matrix;
    long num_band, num_grid_points, num_temp, num_sigma;
    long num_column;

    collision_matrix = (double *)py_collision_matrix.data();
    num_sigma = (long)py_collision_matrix.shape(0);
    num_temp = (long)py_collision_matrix.shape(1);
    num_grid_points = (long)py_collision_matrix.shape(2);
    num_band = (long)py_collision_matrix.shape(3);

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
    long *rot_grid_points;
    long *ir_grid_points;
    long num_band, num_grid_points, num_temp, num_sigma, num_rot, num_ir_gp;

    collision_matrix = (double *)py_collision_matrix.data();
    rot_grid_points = (long *)py_rot_grid_points.data();
    ir_grid_points = (long *)py_ir_grid_points.data();
    num_sigma = (long)py_collision_matrix.shape(0);
    num_temp = (long)py_collision_matrix.shape(1);
    num_grid_points = (long)py_collision_matrix.shape(2);
    num_band = (long)py_collision_matrix.shape(3);
    num_rot = (long)py_rot_grid_points.shape(0);
    num_ir_gp = (long)py_ir_grid_points.shape(0);

    ph3py_expand_collision_matrix(collision_matrix, rot_grid_points,
                                  ir_grid_points, num_ir_gp, num_grid_points,
                                  num_rot, num_sigma, num_temp, num_band);
}

void py_distribute_fc3(nb::ndarray<> force_constants_third, long target,
                       long source, nb::ndarray<> atom_mapping_py,
                       nb::ndarray<> rotation_cart_inv) {
    double *fc3;
    double *rot_cart_inv;
    long *atom_mapping;
    long num_atom;

    fc3 = (double *)force_constants_third.data();
    rot_cart_inv = (double *)rotation_cart_inv.data();
    atom_mapping = (long *)atom_mapping_py.data();
    num_atom = (long)atom_mapping_py.shape(0);

    ph3py_distribute_fc3(fc3, target, source, atom_mapping, num_atom,
                         rot_cart_inv);
}

void py_rotate_delta_fc2s(nb::ndarray<> py_fc3, nb::ndarray<> py_delta_fc2s,
                          nb::ndarray<> py_inv_U,
                          nb::ndarray<> py_site_sym_cart,
                          nb::ndarray<> py_rot_map_syms) {
    double(*fc3)[3][3][3];
    double(*delta_fc2s)[3][3];
    double *inv_U;
    double(*site_sym_cart)[3][3];
    long *rot_map_syms;
    long num_atom, num_disp, num_site_sym;

    /* (num_atom, num_atom, 3, 3, 3) */
    fc3 = (double(*)[3][3][3])py_fc3.data();
    /* (n_u1, num_atom, num_atom, 3, 3) */
    delta_fc2s = (double(*)[3][3])py_delta_fc2s.data();
    /* (3, n_u1 * n_sym) */
    inv_U = (double *)py_inv_U.data();
    /* (n_sym, 3, 3) */
    site_sym_cart = (double(*)[3][3])py_site_sym_cart.data();
    /* (n_sym, natom) */
    rot_map_syms = (long *)py_rot_map_syms.data();

    num_atom = (long)py_fc3.shape(0);
    num_disp = (long)py_delta_fc2s.shape(0);
    num_site_sym = (long)py_site_sym_cart.shape(0);

    ph3py_rotate_delta_fc2(fc3, delta_fc2s, inv_U, site_sym_cart, rot_map_syms,
                           num_atom, num_site_sym, num_disp);
}

void py_get_isotope_strength(
    nb::ndarray<> py_gamma, long grid_point, nb::ndarray<> py_ir_grid_points,
    nb::ndarray<> py_weights, nb::ndarray<> py_mass_variances,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_band_indices, double sigma, double cutoff_frequency) {
    double *gamma;
    double *frequencies;
    long *ir_grid_points;
    double *weights;
    _lapack_complex_double *eigenvectors;
    long *band_indices;
    double *mass_variances;
    long num_band, num_band0, num_ir_grid_points;

    gamma = (double *)py_gamma.data();
    frequencies = (double *)py_frequencies.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    ir_grid_points = (long *)py_ir_grid_points.data();
    weights = (double *)py_weights.data();
    band_indices = (long *)py_band_indices.data();
    mass_variances = (double *)py_mass_variances.data();
    num_band = (long)py_frequencies.shape(1);
    num_band0 = (long)py_band_indices.shape(0);
    num_ir_grid_points = (long)py_ir_grid_points.shape(0);

    ph3py_get_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        eigenvectors, num_ir_grid_points, band_indices, num_band, num_band0,
        sigma, cutoff_frequency);
}

void py_get_thm_isotope_strength(
    nb::ndarray<> py_gamma, long grid_point, nb::ndarray<> py_ir_grid_points,
    nb::ndarray<> py_weights, nb::ndarray<> py_mass_variances,
    nb::ndarray<> py_frequencies, nb::ndarray<> py_eigenvectors,
    nb::ndarray<> py_band_indices, nb::ndarray<> py_integration_weights,
    double cutoff_frequency) {
    double *gamma;
    double *frequencies;
    long *ir_grid_points;
    double *weights;
    _lapack_complex_double *eigenvectors;
    long *band_indices;
    double *mass_variances;
    long num_band, num_band0, num_ir_grid_points;
    double *integration_weights;

    gamma = (double *)py_gamma.data();
    frequencies = (double *)py_frequencies.data();
    ir_grid_points = (long *)py_ir_grid_points.data();
    weights = (double *)py_weights.data();
    eigenvectors = (_lapack_complex_double *)py_eigenvectors.data();
    band_indices = (long *)py_band_indices.data();
    mass_variances = (double *)py_mass_variances.data();
    num_band = (long)py_frequencies.shape(1);
    num_band0 = (long)py_band_indices.shape(0);
    integration_weights = (double *)py_integration_weights.data();
    num_ir_grid_points = (long)py_ir_grid_points.shape(0);

    ph3py_get_thm_isotope_scattering_strength(
        gamma, grid_point, ir_grid_points, weights, mass_variances, frequencies,
        eigenvectors, num_ir_grid_points, band_indices, num_band, num_band0,
        integration_weights, cutoff_frequency);
}

void py_get_permutation_symmetry_fc3(nb::ndarray<> py_fc3) {
    double *fc3;
    long num_atom;

    fc3 = (double *)py_fc3.data();
    num_atom = (long)py_fc3.shape(0);

    ph3py_get_permutation_symmetry_fc3(fc3, num_atom);
}

void py_get_permutation_symmetry_compact_fc3(nb::ndarray<> py_fc3,
                                             nb::ndarray<> py_permutations,
                                             nb::ndarray<> py_s2pp_map,
                                             nb::ndarray<> py_p2s_map,
                                             nb::ndarray<> py_nsym_list) {
    double *fc3;
    long *s2pp;
    long *p2s;
    long *nsym_list;
    long *perms;
    long n_patom, n_satom;

    fc3 = (double *)py_fc3.data();
    perms = (long *)py_permutations.data();
    s2pp = (long *)py_s2pp_map.data();
    p2s = (long *)py_p2s_map.data();
    nsym_list = (long *)py_nsym_list.data();
    n_patom = (long)py_fc3.shape(0);
    n_satom = (long)py_fc3.shape(1);

    ph3py_get_permutation_symmetry_compact_fc3(fc3, p2s, s2pp, nsym_list, perms,
                                               n_satom, n_patom);
}

void py_transpose_compact_fc3(nb::ndarray<> py_fc3,
                              nb::ndarray<> py_permutations,
                              nb::ndarray<> py_s2pp_map,
                              nb::ndarray<> py_p2s_map,
                              nb::ndarray<> py_nsym_list, long t_type) {
    double *fc3;
    long *s2pp;
    long *p2s;
    long *nsym_list;
    long *perms;
    long n_patom, n_satom;

    fc3 = (double *)py_fc3.data();
    perms = (long *)py_permutations.data();
    s2pp = (long *)py_s2pp_map.data();
    p2s = (long *)py_p2s_map.data();
    nsym_list = (long *)py_nsym_list.data();
    n_patom = (long)py_fc3.shape(0);
    n_satom = (long)py_fc3.shape(1);

    ph3py_transpose_compact_fc3(fc3, p2s, s2pp, nsym_list, perms, n_satom,
                                n_patom, t_type);
}

void py_get_thm_relative_grid_address(nb::ndarray<> py_relative_grid_address,
                                      nb::ndarray<> py_reciprocal_lattice_py) {
    long(*relative_grid_address)[4][3];
    double(*reciprocal_lattice)[3];

    relative_grid_address = (long(*)[4][3])py_relative_grid_address.data();
    reciprocal_lattice = (double(*)[3])py_reciprocal_lattice_py.data();

    ph3py_get_relative_grid_address(relative_grid_address, reciprocal_lattice);
}

void py_get_neighboring_grid_points(nb::ndarray<> py_relative_grid_points,
                                    nb::ndarray<> py_grid_points,
                                    nb::ndarray<> py_relative_grid_address,
                                    nb::ndarray<> py_D_diag,
                                    nb::ndarray<> py_bz_grid_address,
                                    nb::ndarray<> py_bz_map,
                                    long bz_grid_type) {
    long *relative_grid_points;
    long *grid_points;
    long num_grid_points, num_relative_grid_address;
    long(*relative_grid_address)[3];
    long *D_diag;
    long(*bz_grid_address)[3];
    long *bz_map;

    relative_grid_points = (long *)py_relative_grid_points.data();
    grid_points = (long *)py_grid_points.data();
    num_grid_points = (long)py_grid_points.shape(0);
    relative_grid_address = (long(*)[3])py_relative_grid_address.data();
    num_relative_grid_address = (long)py_relative_grid_address.shape(0);
    D_diag = (long *)py_D_diag.data();
    bz_grid_address = (long(*)[3])py_bz_grid_address.data();
    bz_map = (long *)py_bz_map.data();

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
    nb::ndarray<> py_gp2irgp_map, long bz_grid_type, const char *function) {
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

    iw = (double *)py_iw.data();
    frequency_points = (double *)py_frequency_points.data();
    num_frequency_points = (long)py_frequency_points.shape(0);
    relative_grid_address = (long(*)[4][3])py_relative_grid_address.data();
    D_diag = (long *)py_D_diag.data();
    grid_points = (long *)py_grid_points.data();
    num_gp = (long)py_grid_points.shape(0);
    bz_grid_address = (long(*)[3])py_bz_grid_address.data();
    bz_map = (long *)py_bz_map.data();
    gp2irgp_map = (long *)py_gp2irgp_map.data();
    frequencies = (double *)py_frequencies.data();
    num_band = (long)py_frequencies.shape(1);

    ph3py_get_thm_integration_weights_at_grid_points(
        iw, frequency_points, num_frequency_points, num_band, num_gp,
        relative_grid_address, D_diag, grid_points, bz_grid_address, bz_map,
        bz_grid_type, frequencies, gp2irgp_map, function[0]);
}

long py_tpl_get_triplets_reciprocal_mesh_at_q(
    nb::ndarray<> py_map_triplets, nb::ndarray<> py_map_q,
    long fixed_grid_number, nb::ndarray<> py_D_diag, long is_time_reversal,
    nb::ndarray<> py_rotations, long swappable) {
    long *map_triplets;
    long *map_q;
    long *D_diag;
    long(*rot)[3][3];
    long num_rot;
    long num_ir;

    map_triplets = (long *)py_map_triplets.data();
    map_q = (long *)py_map_q.data();
    D_diag = (long *)py_D_diag.data();
    rot = (long(*)[3][3])py_rotations.data();
    num_rot = (long)py_rotations.shape(0);

    num_ir = ph3py_get_triplets_reciprocal_mesh_at_q(
        map_triplets, map_q, fixed_grid_number, D_diag, is_time_reversal,
        num_rot, rot, swappable);

    return num_ir;
}

long py_tpl_get_BZ_triplets_at_q(nb::ndarray<> py_triplets, long grid_point,
                                 nb::ndarray<> py_bz_grid_address,
                                 nb::ndarray<> py_bz_map,
                                 nb::ndarray<> py_map_triplets,
                                 nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
                                 long bz_grid_type) {
    long(*triplets)[3];
    long(*bz_grid_address)[3];
    long *bz_map;
    long *map_triplets;
    long num_map_triplets;
    long *D_diag;
    long(*Q)[3];
    long num_ir;

    triplets = (long(*)[3])py_triplets.data();
    bz_grid_address = (long(*)[3])py_bz_grid_address.data();
    bz_map = (long *)py_bz_map.data();
    map_triplets = (long *)py_map_triplets.data();
    num_map_triplets = (long)py_map_triplets.shape(0);
    D_diag = (long *)py_D_diag.data();
    Q = (long(*)[3])py_Q.data();

    num_ir = ph3py_get_BZ_triplets_at_q(triplets, grid_point, bz_grid_address,
                                        bz_map, map_triplets, num_map_triplets,
                                        D_diag, Q, bz_grid_type);

    return num_ir;
}

void py_get_triplets_integration_weights(
    nb::ndarray<> py_iw, nb::ndarray<> py_iw_zero,
    nb::ndarray<> py_frequency_points, nb::ndarray<> py_relative_grid_address,
    nb::ndarray<> py_D_diag, nb::ndarray<> py_triplets,
    nb::ndarray<> py_frequencies1, nb::ndarray<> py_frequencies2,
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_bz_map,
    long bz_grid_type, long tp_type) {
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

    iw = (double *)py_iw.data();
    iw_zero = (char *)py_iw_zero.data();
    frequency_points = (double *)py_frequency_points.data();
    num_band0 = (long)py_frequency_points.shape(0);
    relative_grid_address = (long(*)[4][3])py_relative_grid_address.data();
    D_diag = (long *)py_D_diag.data();
    triplets = (long(*)[3])py_triplets.data();
    num_triplets = (long)py_triplets.shape(0);
    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();
    bz_map = (long *)py_bz_map.data();
    frequencies1 = (double *)py_frequencies1.data();
    frequencies2 = (double *)py_frequencies2.data();
    num_band1 = (long)py_frequencies1.shape(1);
    num_band2 = (long)py_frequencies2.shape(1);

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
    long(*triplets)[3];
    double *frequencies;
    long num_band0, num_band, num_iw, num_triplets;

    iw = (double *)py_iw.data();
    iw_zero = (char *)py_iw_zero.data();
    frequency_points = (double *)py_frequency_points.data();
    num_band0 = (long)py_frequency_points.shape(0);
    triplets = (long(*)[3])py_triplets.data();
    num_triplets = (long)py_triplets.shape(0);
    frequencies = (double *)py_frequencies.data();
    num_band = (long)py_frequencies.shape(1);
    num_iw = (long)py_iw.shape(0);

    ph3py_get_integration_weight_with_sigma(
        iw, iw_zero, sigma, sigma_cutoff, frequency_points, num_band0, triplets,
        num_triplets, frequencies, num_band, num_iw);
}

long py_get_grid_index_from_address(nb::ndarray<> py_address,
                                    nb::ndarray<> py_D_diag) {
    long *address;
    long *D_diag;
    long gp;

    address = (long *)py_address.data();
    D_diag = (long *)py_D_diag.data();

    gp = ph3py_get_grid_index_from_address(address, D_diag);

    return gp;
}

long py_get_ir_grid_map(nb::ndarray<> py_grid_mapping_table,
                        nb::ndarray<> py_D_diag, nb::ndarray<> py_is_shift,
                        nb::ndarray<> py_rotations) {
    long *D_diag;
    long *is_shift;
    long(*rot)[3][3];
    long num_rot;

    long *grid_mapping_table;
    long num_ir;

    D_diag = (long *)py_D_diag.data();
    is_shift = (long *)py_is_shift.data();
    rot = (long(*)[3][3])py_rotations.data();
    num_rot = (long)py_rotations.shape(0);
    grid_mapping_table = (long *)py_grid_mapping_table.data();

    num_ir = ph3py_get_ir_grid_map(grid_mapping_table, D_diag, is_shift, rot,
                                   num_rot);
    return num_ir;
}

void py_get_gr_grid_addresses(nb::ndarray<> py_gr_grid_addresses,
                              nb::ndarray<> py_D_diag) {
    long(*gr_grid_addresses)[3];
    long *D_diag;

    gr_grid_addresses = (long(*)[3])py_gr_grid_addresses.data();
    D_diag = (long *)py_D_diag.data();

    ph3py_get_gr_grid_addresses(gr_grid_addresses, D_diag);
}

long py_get_reciprocal_rotations(nb::ndarray<> py_rec_rotations,
                                 nb::ndarray<> py_rotations,
                                 long is_time_reversal) {
    long(*rec_rotations)[3][3];
    long(*rotations)[3][3];
    long num_rot, num_rec_rot;

    rec_rotations = (long(*)[3][3])py_rec_rotations.data();
    rotations = (long(*)[3][3])py_rotations.data();
    num_rot = (long)py_rotations.shape(0);

    num_rec_rot = ph3py_get_reciprocal_rotations(rec_rotations, rotations,
                                                 num_rot, is_time_reversal);

    return num_rec_rot;
}

bool py_transform_rotations(nb::ndarray<> py_transformed_rotations,
                            nb::ndarray<> py_rotations, nb::ndarray<> py_D_diag,
                            nb::ndarray<> py_Q) {
    long(*transformed_rotations)[3][3];
    long(*rotations)[3][3];
    long *D_diag;
    long(*Q)[3];
    long num_rot, succeeded;

    transformed_rotations = (long(*)[3][3])py_transformed_rotations.data();
    rotations = (long(*)[3][3])py_rotations.data();
    D_diag = (long *)py_D_diag.data();
    Q = (long(*)[3])py_Q.data();
    num_rot = (long)py_transformed_rotations.shape(0);

    succeeded = ph3py_transform_rotations(transformed_rotations, rotations,
                                          num_rot, D_diag, Q);
    if (succeeded) {
        return true;
    } else {
        return false;
    }
}

bool py_get_snf3x3(nb::ndarray<> py_D_diag, nb::ndarray<> py_P,
                   nb::ndarray<> py_Q, nb::ndarray<> py_A) {
    long *D_diag;
    long(*P)[3];
    long(*Q)[3];
    long(*A)[3];
    long succeeded;

    D_diag = (long *)py_D_diag.data();
    P = (long(*)[3])py_P.data();
    Q = (long(*)[3])py_Q.data();
    A = (long(*)[3])py_A.data();

    succeeded = ph3py_get_snf3x3(D_diag, P, Q, A);
    if (succeeded) {
        return true;
    } else {
        return false;
    }
}

long py_get_bz_grid_addresses(nb::ndarray<> py_bz_grid_addresses,
                              nb::ndarray<> py_bz_map, nb::ndarray<> py_bzg2grg,
                              nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
                              nb::ndarray<> py_PS,
                              nb::ndarray<> py_reciprocal_lattice, long type) {
    long(*bz_grid_addresses)[3];
    long *bz_map;
    long *bzg2grg;
    long *D_diag;
    long(*Q)[3];
    long *PS;
    double(*reciprocal_lattice)[3];
    long num_total_gp;

    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();
    bz_map = (long *)py_bz_map.data();
    bzg2grg = (long *)py_bzg2grg.data();
    D_diag = (long *)py_D_diag.data();
    Q = (long(*)[3])py_Q.data();
    PS = (long *)py_PS.data();
    reciprocal_lattice = (double(*)[3])py_reciprocal_lattice.data();

    num_total_gp =
        ph3py_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg, D_diag,
                                    Q, PS, reciprocal_lattice, type);

    return num_total_gp;
}

long py_rotate_bz_grid_addresses(long bz_grid_index, nb::ndarray<> py_rotation,
                                 nb::ndarray<> py_bz_grid_addresses,
                                 nb::ndarray<> py_bz_map,
                                 nb::ndarray<> py_D_diag, nb::ndarray<> py_PS,
                                 long type) {
    long(*bz_grid_addresses)[3];
    long(*rotation)[3];
    long *bz_map;
    long *D_diag;
    long *PS;
    long ret_bz_gp;

    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();
    rotation = (long(*)[3])py_rotation.data();
    bz_map = (long *)py_bz_map.data();
    D_diag = (long *)py_D_diag.data();
    PS = (long *)py_PS.data();

    ret_bz_gp = ph3py_rotate_bz_grid_index(
        bz_grid_index, rotation, bz_grid_addresses, bz_map, D_diag, PS, type);

    return ret_bz_gp;
}

long py_diagonalize_collision_matrix(nb::ndarray<> py_collision_matrix,
                                     nb::ndarray<> py_eigenvalues, long i_sigma,
                                     long i_temp, double cutoff, long solver,
                                     long is_pinv) {
    double *collision_matrix;
    double *eigvals;
    long num_temp, num_grid_point, num_band;
    long num_column, adrs_shift;
    long info;

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
                                nb::ndarray<> py_eigenvalues, long i_sigma,
                                long i_temp, double cutoff, long pinv_method) {
    double *collision_matrix;
    double *eigvals;
    long num_temp, num_grid_point, num_band;
    long num_column, adrs_shift;

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

long py_get_default_colmat_solver() {
#if defined(MKL_LAPACKE) || defined(SCIPY_MKL_H)
    return (long)1;
#else
    return (long)4;
#endif
}

long py_lapacke_pinv(nb::ndarray<> data_out_py, nb::ndarray<> data_in_py,
                     double cutoff) {
    long m;
    long n;
    double *data_in;
    double *data_out;
    long info;

    m = data_in_py.shape(0);
    n = data_in_py.shape(1);
    data_in = (double *)data_in_py.data();
    data_out = (double *)data_out_py.data();

    info = ph3py_phonopy_pinv(data_out, data_in, m, n, cutoff);

    return info;
}

long py_get_omp_max_threads() { return ph3py_get_max_threads(); }

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
    m.def("grid_index_from_address", &py_get_grid_index_from_address);
    m.def("ir_grid_map", &py_get_ir_grid_map);
    m.def("gr_grid_addresses", &py_get_gr_grid_addresses);
    m.def("reciprocal_rotations", &py_get_reciprocal_rotations);
    m.def("transform_rotations", &py_transform_rotations);
    m.def("snf3x3", &py_get_snf3x3);
    m.def("bz_grid_addresses", &py_get_bz_grid_addresses);
    m.def("rotate_bz_grid_index", &py_rotate_bz_grid_addresses);
    m.def("diagonalize_collision_matrix", &py_diagonalize_collision_matrix);
    m.def("pinv_from_eigensolution", &py_pinv_from_eigensolution);
    m.def("default_colmat_solver", &py_get_default_colmat_solver);
    m.def("lapacke_pinv", &py_lapacke_pinv);
    m.def("omp_max_threads", &py_get_omp_max_threads);
}
