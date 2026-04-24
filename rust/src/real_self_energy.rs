//! Real-part of the bubble-diagram self-energy.
//!
//! Port of `rse_get_real_self_energy_at_bands` and
//! `rse_get_real_self_energy_at_frequency_point` in
//! `c/real_self_energy.c`.  Frequencies and Bose-Einstein occupations
//! are in THz units throughout.
//!
//! `fc3_normal_squared` is the `|V^(3)|^2` array laid out as
//! ``(num_triplets, num_band0, num_band, num_band)`` in row-major
//! order.  Both entry points return `real_self_energy[0..num_band0]`
//! summed over triplets and scaled by `unit_conversion_factor`.

use rayon::prelude::*;

use crate::funcs::bose_einstein;

/// Finite-temperature inner sum at one triplet and one band0 index.
///
/// Mirrors `sum_real_self_energy_at_band` in `c/real_self_energy.c`.
/// The Cauchy principal value `1 / x` is regularised as
/// `x / (x^2 + epsilon^2)`.
fn sum_at_band(
    num_band: usize,
    fc3: &[f64],
    fpoint: f64,
    freqs1: &[f64],
    freqs2: &[f64],
    epsilon: f64,
    temperature_thz: f64,
    cutoff_frequency: f64,
) -> f64 {
    let eps2 = epsilon * epsilon;
    let mut shift = 0.0;
    for i in 0..num_band {
        let f1i = freqs1[i];
        if f1i <= cutoff_frequency {
            continue;
        }
        let n1 = bose_einstein(f1i, temperature_thz);
        let row = &fc3[i * num_band..(i + 1) * num_band];
        for j in 0..num_band {
            let f2j = freqs2[j];
            if f2j <= cutoff_frequency {
                continue;
            }
            let n2 = bose_einstein(f2j, temperature_thz);
            let f1 = fpoint + f1i + f2j;
            let f2 = fpoint - f1i - f2j;
            let f3 = fpoint - f1i + f2j;
            let f4 = fpoint + f1i - f2j;
            let d = -(n1 + n2 + 1.0) * f1 / (f1 * f1 + eps2)
                + (n1 + n2 + 1.0) * f2 / (f2 * f2 + eps2)
                - (n1 - n2) * f3 / (f3 * f3 + eps2)
                + (n1 - n2) * f4 / (f4 * f4 + eps2);
            shift += d * row[j];
        }
    }
    shift
}

/// Zero-temperature inner sum.  Mirrors
/// `sum_real_self_energy_at_band_0K` in `c/real_self_energy.c`.
fn sum_at_band_0k(
    num_band: usize,
    fc3: &[f64],
    fpoint: f64,
    freqs1: &[f64],
    freqs2: &[f64],
    epsilon: f64,
    cutoff_frequency: f64,
) -> f64 {
    let eps2 = epsilon * epsilon;
    let mut shift = 0.0;
    for i in 0..num_band {
        let f1i = freqs1[i];
        if f1i <= cutoff_frequency {
            continue;
        }
        let row = &fc3[i * num_band..(i + 1) * num_band];
        for j in 0..num_band {
            let f2j = freqs2[j];
            if f2j <= cutoff_frequency {
                continue;
            }
            let f1 = fpoint + f1i + f2j;
            let f2 = fpoint - f1i - f2j;
            let d = -f1 / (f1 * f1 + eps2) + f2 / (f2 * f2 + eps2);
            shift += d * row[j];
        }
    }
    shift
}

/// Per-band driver corresponding to `get_real_self_energy_at_band`.
/// Reduces across triplets in parallel.
#[allow(clippy::too_many_arguments)]
fn real_self_energy_at_band(
    band_index: usize,
    fc3_normal_squared: &[f64],
    num_triplets: usize,
    num_band0: usize,
    num_band: usize,
    fpoint: f64,
    frequencies: &[f64],
    triplets: &[[i64; 3]],
    triplet_weights: &[i64],
    epsilon: f64,
    temperature_thz: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) -> f64 {
    let stride_triplet = num_band0 * num_band * num_band;
    let stride_band0 = num_band * num_band;
    let band_base = band_index * stride_band0;

    (0..num_triplets)
        .into_par_iter()
        .map(|i| {
            let gp1 = triplets[i][1] as usize;
            let gp2 = triplets[i][2] as usize;
            let fc3 = &fc3_normal_squared
                [i * stride_triplet + band_base..i * stride_triplet + band_base + stride_band0];
            let freqs1 = &frequencies[gp1 * num_band..(gp1 + 1) * num_band];
            let freqs2 = &frequencies[gp2 * num_band..(gp2 + 1) * num_band];
            let inner = if temperature_thz > 0.0 {
                sum_at_band(
                    num_band,
                    fc3,
                    fpoint,
                    freqs1,
                    freqs2,
                    epsilon,
                    temperature_thz,
                    cutoff_frequency,
                )
            } else {
                sum_at_band_0k(
                    num_band,
                    fc3,
                    fpoint,
                    freqs1,
                    freqs2,
                    epsilon,
                    cutoff_frequency,
                )
            };
            inner * triplet_weights[i] as f64 * unit_conversion_factor
        })
        .sum()
}

/// Port of `rse_get_real_self_energy_at_bands`.  Writes
/// `real_self_energy[0..num_band0]` where band0 indices are taken from
/// `band_indices` and evaluated at the on-shell phonon frequency of the
/// first grid point of the triplet.
#[allow(clippy::too_many_arguments)]
pub(crate) fn real_self_energy_at_bands(
    real_self_energy: &mut [f64],
    fc3_normal_squared: &[f64],
    num_triplets: usize,
    num_band0: usize,
    num_band: usize,
    band_indices: &[i64],
    frequencies: &[f64],
    triplets: &[[i64; 3]],
    triplet_weights: &[i64],
    epsilon: f64,
    temperature_thz: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) {
    let gp0 = triplets[0][0] as usize;
    for i in 0..num_band0 {
        let bi = band_indices[i] as usize;
        let fpoint = frequencies[gp0 * num_band + bi];
        if fpoint < cutoff_frequency {
            real_self_energy[i] = 0.0;
        } else {
            real_self_energy[i] = real_self_energy_at_band(
                i,
                fc3_normal_squared,
                num_triplets,
                num_band0,
                num_band,
                fpoint,
                frequencies,
                triplets,
                triplet_weights,
                epsilon,
                temperature_thz,
                unit_conversion_factor,
                cutoff_frequency,
            );
        }
    }
}

/// Port of `rse_get_real_self_energy_at_frequency_point`.  Same as
/// `real_self_energy_at_bands` but all band0 entries share a single
/// external `frequency_point`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn real_self_energy_at_frequency_point(
    real_self_energy: &mut [f64],
    frequency_point: f64,
    fc3_normal_squared: &[f64],
    num_triplets: usize,
    num_band0: usize,
    num_band: usize,
    frequencies: &[f64],
    triplets: &[[i64; 3]],
    triplet_weights: &[i64],
    epsilon: f64,
    temperature_thz: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) {
    for i in 0..num_band0 {
        if frequency_point < cutoff_frequency {
            real_self_energy[i] = 0.0;
        } else {
            real_self_energy[i] = real_self_energy_at_band(
                i,
                fc3_normal_squared,
                num_triplets,
                num_band0,
                num_band,
                frequency_point,
                frequencies,
                triplets,
                triplet_weights,
                epsilon,
                temperature_thz,
                unit_conversion_factor,
                cutoff_frequency,
            );
        }
    }
}
