//! Third-order force constant (fc3) helpers.
//!
//! Mirrors `c/fc3.c`.  `fc3` has layout
//! `[num_atom, num_atom, num_atom, 3, 3, 3]` packed in C order, so each
//! atom triple occupies a contiguous block of 27 doubles.

use rayon::prelude::*;

/// `*mut T` wrapper opting into Send + Sync for rayon.  Only used in
/// parallel kernels where the offsets touched by each task are manually
/// verified to be disjoint.
#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    fn ptr(self) -> *mut T {
        self.0
    }
}

/// Rotate a single 3x3x3 cartesian tensor by `r` (3x3 row-major).
///
/// `out[l, m, n] = sum_{i,j,k} r[l,i] r[m,j] r[n,k] * tensor[i,j,k]`.
/// Matches `tensor3_rotation` in `c/fc3.c`.
#[inline]
fn tensor3_rotation(out: &mut [f64], tensor: &[f64], r: &[f64]) {
    for pos in 0..27usize {
        out[pos] = tensor3_rotation_elem(tensor, r, pos);
    }
}

#[inline]
fn tensor3_rotation_elem(tensor: &[f64], r: &[f64], pos: usize) -> f64 {
    let l = pos / 9;
    let m = (pos % 9) / 3;
    let n = pos % 3;
    let mut sum = 0.0;
    for i in 0..3usize {
        for j in 0..3usize {
            for k in 0..3usize {
                sum += r[l * 3 + i] * r[m * 3 + j] * r[n * 3 + k] * tensor[i * 9 + j * 3 + k];
            }
        }
    }
    sum
}

/// Distribute fc3 from `source` triplet to `target` triplet via `rot_cart`.
///
/// For each atom pair `(i, j)` the 27-element block at
/// `(target, i, j)` is written from the rotated block at
/// `(source, atom_mapping[i], atom_mapping[j])`.  Matches
/// `fc3_distribute_fc3` in `c/fc3.c`.
pub(crate) fn distribute_fc3(
    fc3: &mut [f64],
    target: usize,
    source: usize,
    atom_mapping: &[i64],
    num_atom: usize,
    rot_cart: &[f64],
) {
    let stride_i = num_atom * 27;
    let stride_src = num_atom * num_atom * 27;
    let src_base = source * stride_src;
    let tgt_base = target * stride_src;
    for i in 0..num_atom {
        for j in 0..num_atom {
            let adrs_out = tgt_base + i * stride_i + j * 27;
            let adrs_in =
                src_base + (atom_mapping[i] as usize) * stride_i + (atom_mapping[j] as usize) * 27;
            // Source and target addresses never overlap (target != source
            // is guaranteed by the caller); copy via a local buffer to
            // satisfy the borrow checker without unsafe.
            let mut buf = [0.0f64; 27];
            let src = &fc3[adrs_in..adrs_in + 27];
            tensor3_rotation(&mut buf, src, rot_cart);
            fc3[adrs_out..adrs_out + 27].copy_from_slice(&buf);
        }
    }
}

/// Enforce permutation symmetry on full fc3 in place.
///
/// For every unordered atom triple `{a, b, c}` (iterated as sorted
/// `a <= b <= c`), compute the 6-way mean of the six index permutations
/// and write it back to all six positions.  Matches
/// `fc3_set_permutation_symmetry_fc3` in `c/fc3.c`.
///
/// Parallelized over the outermost `a` index.  The six (index, offset)
/// positions touched by one `(a, b, c)` task lie inside atom triples
/// whose sorted form is `(a, b, c)`; since this sorted form is unique
/// to the task, writes across tasks are disjoint.
pub(crate) fn set_permutation_symmetry_fc3(fc3: &mut [f64], num_atom: usize) {
    let stride_a = num_atom * num_atom * 27;
    let stride_b = num_atom * 27;
    let fc3_ptr = SyncMutPtr(fc3.as_mut_ptr());

    (0..num_atom).into_par_iter().for_each(|a| {
        for b in a..num_atom {
            for c in b..num_atom {
                let base_abc = a * stride_a + b * stride_b + c * 27;
                let base_acb = a * stride_a + c * stride_b + b * 27;
                let base_bac = b * stride_a + a * stride_b + c * 27;
                let base_bca = b * stride_a + c * stride_b + a * 27;
                let base_cab = c * stride_a + a * stride_b + b * 27;
                let base_cba = c * stride_a + b * stride_b + a * 27;

                let mut elem = [0.0f64; 27];
                // elem[i, j, k] = mean over the 6 index permutations.
                unsafe {
                    let p = fc3_ptr.ptr();
                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                let sum = *p.add(base_abc + i * 9 + j * 3 + k)
                                    + *p.add(base_acb + i * 9 + k * 3 + j)
                                    + *p.add(base_bac + j * 9 + i * 3 + k)
                                    + *p.add(base_bca + j * 9 + k * 3 + i)
                                    + *p.add(base_cab + k * 9 + i * 3 + j)
                                    + *p.add(base_cba + k * 9 + j * 3 + i);
                                elem[i * 9 + j * 3 + k] = sum / 6.0;
                            }
                        }
                    }
                    // Write the mean back to all six positions.  When a, b,
                    // c are not all distinct, some bases coincide but the
                    // written value is identical, so repeated writes are
                    // safe.
                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                let v = elem[i * 9 + j * 3 + k];
                                *p.add(base_abc + i * 9 + j * 3 + k) = v;
                                *p.add(base_acb + i * 9 + k * 3 + j) = v;
                                *p.add(base_bac + j * 9 + i * 3 + k) = v;
                                *p.add(base_bca + j * 9 + k * 3 + i) = v;
                                *p.add(base_cab + k * 9 + i * 3 + j) = v;
                                *p.add(base_cba + k * 9 + j * 3 + i) = v;
                            }
                        }
                    }
                }
            }
        }
    });
}

/// Transpose two spatial indices of a compact fc3 in place.
///
/// `fc3` has layout `[n_patom, n_satom, n_satom, 3, 3, 3]`.  The
/// variants are:
/// * `t_type = 0`: swap dim 0 and dim 1
/// * `t_type = 1`: swap dim 0 and dim 2
/// * `t_type = 2`: swap dim 1 and dim 2
/// Other values are no-ops, matching `fc3_transpose_compact_fc3` in
/// `c/fc3.c`.  Types 0 and 1 require symmetry information (`p2s`,
/// `s2pp`, `nsym_list`, `perms`) because one of the swapped indices
/// runs over supercell atoms, while the compact layout only stores
/// primitive-cell atoms in dim 0; the swap crosses slabs and is
/// serialised via a `done` table.
pub(crate) fn transpose_compact_fc3(
    fc3: &mut [f64],
    p2s: &[i64],
    s2pp: &[i64],
    nsym_list: &[i64],
    perms: &[i64],
    n_satom: usize,
    n_patom: usize,
    t_type: i64,
) {
    match t_type {
        0 | 1 => {
            transpose_compact_fc3_type01(fc3, p2s, s2pp, nsym_list, perms, n_satom, n_patom, t_type)
        }
        2 => transpose_compact_fc3_type2(fc3, n_satom, n_patom),
        _ => {}
    }
}

fn transpose_compact_fc3_type01(
    fc3: &mut [f64],
    p2s: &[i64],
    s2pp: &[i64],
    nsym_list: &[i64],
    perms: &[i64],
    n_satom: usize,
    n_patom: usize,
    t_type: i64,
) {
    let mut done = vec![false; n_satom * n_patom];
    for i_p in 0..n_patom {
        let i = p2s[i_p] as usize;
        for j in 0..n_satom {
            if done[i_p * n_satom + j] {
                continue;
            }
            let j_p = s2pp[j] as usize;
            let nsym = nsym_list[j] as usize;
            let i_trans = perms[nsym * n_satom + i] as usize;
            done[i_p * n_satom + j] = true;
            done[j_p * n_satom + i_trans] = true;
            for k in 0..n_satom {
                let k_trans = perms[nsym * n_satom + k] as usize;
                if t_type == 0 {
                    let adrs = (i_p * n_satom * n_satom + j * n_satom + k) * 27;
                    let adrs_t = (j_p * n_satom * n_satom + i_trans * n_satom + k_trans) * 27;
                    swap_blocks_lm(fc3, adrs, adrs_t);
                } else {
                    let adrs = (i_p * n_satom * n_satom + k * n_satom + j) * 27;
                    let adrs_t = (j_p * n_satom * n_satom + k_trans * n_satom + i_trans) * 27;
                    swap_blocks_ln(fc3, adrs, adrs_t);
                }
            }
        }
    }
}

// Swap two 27-blocks with index permutation (l, m, n) <-> (m, l, n).
// Mirrors the t_type = 0 branch of the C reference.
fn swap_blocks_lm(fc3: &mut [f64], adrs: usize, adrs_t: usize) {
    let mut elem_src = [0.0f64; 27];
    let mut elem_dst = [0.0f64; 27];
    elem_src.copy_from_slice(&fc3[adrs..adrs + 27]);
    elem_dst.copy_from_slice(&fc3[adrs_t..adrs_t + 27]);
    if adrs != adrs_t {
        for l in 0..3 {
            for m in 0..3 {
                for n in 0..3 {
                    // fc3[adrs][l, m, n] = fc3[adrs_t][m, l, n]
                    fc3[adrs + l * 9 + m * 3 + n] = elem_dst[m * 9 + l * 3 + n];
                }
            }
        }
    }
    for l in 0..3 {
        for m in 0..3 {
            for n in 0..3 {
                // fc3[adrs_t][m, l, n] = elem_src[l, m, n]
                fc3[adrs_t + m * 9 + l * 3 + n] = elem_src[l * 9 + m * 3 + n];
            }
        }
    }
}

// Swap two 27-blocks with index permutation (l, m, n) <-> (n, m, l).
// Mirrors the t_type = 1 branch of the C reference.
fn swap_blocks_ln(fc3: &mut [f64], adrs: usize, adrs_t: usize) {
    let mut elem_src = [0.0f64; 27];
    let mut elem_dst = [0.0f64; 27];
    elem_src.copy_from_slice(&fc3[adrs..adrs + 27]);
    elem_dst.copy_from_slice(&fc3[adrs_t..adrs_t + 27]);
    if adrs != adrs_t {
        for l in 0..3 {
            for m in 0..3 {
                for n in 0..3 {
                    // fc3[adrs][l, m, n] = fc3[adrs_t][n, m, l]
                    fc3[adrs + l * 9 + m * 3 + n] = elem_dst[n * 9 + m * 3 + l];
                }
            }
        }
    }
    for l in 0..3 {
        for m in 0..3 {
            for n in 0..3 {
                // fc3[adrs_t][n, m, l] = elem_src[l, m, n]
                fc3[adrs_t + n * 9 + m * 3 + l] = elem_src[l * 9 + m * 3 + n];
            }
        }
    }
}

fn transpose_compact_fc3_type2(fc3: &mut [f64], n_satom: usize, n_patom: usize) {
    let stride_ip = n_satom * n_satom * 27;
    let fc3_ptr = SyncMutPtr(fc3.as_mut_ptr());
    // Different i_p slabs are disjoint, so parallelize over i_p.
    (0..n_patom).into_par_iter().for_each(|i_p| {
        let base_ip = i_p * stride_ip;
        for j in 0..n_satom {
            for k in j..n_satom {
                let adrs = base_ip + (j * n_satom + k) * 27;
                let adrs_t = base_ip + (k * n_satom + j) * 27;
                let mut elem_src = [0.0f64; 27];
                let mut elem_dst = [0.0f64; 27];
                unsafe {
                    let p = fc3_ptr.ptr();
                    for off in 0..27 {
                        elem_src[off] = *p.add(adrs + off);
                        elem_dst[off] = *p.add(adrs_t + off);
                    }
                    if k != j {
                        for l in 0..3 {
                            for m in 0..3 {
                                for n in 0..3 {
                                    // fc3[adrs][l, m, n] = fc3[adrs_t][l, n, m]
                                    *p.add(adrs + l * 9 + m * 3 + n) = elem_dst[l * 9 + n * 3 + m];
                                }
                            }
                        }
                    }
                    for l in 0..3 {
                        for m in 0..3 {
                            for n in 0..3 {
                                // fc3[adrs_t][l, n, m] = elem_src[l, m, n]
                                *p.add(adrs_t + l * 9 + n * 3 + m) = elem_src[l * 9 + m * 3 + n];
                            }
                        }
                    }
                }
            }
        }
    });
}

/// Enforce permutation symmetry on a compact fc3 in place.
///
/// `fc3` has layout `[n_patom, n_satom, n_satom, 3, 3, 3]`.  For every
/// unordered orbit of six index permutations `{ijk, ikj, jik, jki, kij,
/// kji}`, compute the mean and write it back to the six positions
/// (within the compact layout, supercell indices are mapped to primitive
/// slabs via `s2pp`, `p2s`, `nsym_list`, `perms`).  Matches
/// `fc3_set_permutation_symmetry_compact_fc3` in `c/fc3.c`.  Serial
/// because `done` is shared state across orbits.
pub(crate) fn set_permutation_symmetry_compact_fc3(
    fc3: &mut [f64],
    p2s: &[i64],
    s2pp: &[i64],
    nsym_list: &[i64],
    perms: &[i64],
    n_satom: usize,
    n_patom: usize,
) {
    let n2 = n_satom * n_satom;
    let mut done = vec![false; n_patom * n2];

    for i_p in 0..n_patom {
        let i = p2s[i_p] as usize;
        for j in 0..n_satom {
            let j_p = s2pp[j] as usize;
            let nsym_j = nsym_list[j] as usize;
            let i_trans_j = perms[nsym_j * n_satom + i] as usize;
            for k in 0..n_satom {
                let k_p = s2pp[k] as usize;
                let k_trans_j = perms[nsym_j * n_satom + k] as usize;
                let nsym_k = nsym_list[k] as usize;
                let i_trans_k = perms[nsym_k * n_satom + i] as usize;
                let j_trans_k = perms[nsym_k * n_satom + j] as usize;

                // ijk, ikj, jik, jki, kij, kji
                let mut adrs = [
                    i_p * n2 + j * n_satom + k,
                    i_p * n2 + k * n_satom + j,
                    j_p * n2 + i_trans_j * n_satom + k_trans_j,
                    j_p * n2 + k_trans_j * n_satom + i_trans_j,
                    k_p * n2 + i_trans_k * n_satom + j_trans_k,
                    k_p * n2 + j_trans_k * n_satom + i_trans_k,
                ];

                if adrs.iter().any(|&a| done[a]) {
                    continue;
                }
                for &a in &adrs {
                    done[a] = true;
                }
                for a in &mut adrs {
                    *a *= 27;
                }

                let mut elem = [0.0f64; 27];
                for l in 0..3 {
                    for m in 0..3 {
                        for n in 0..3 {
                            let sum = fc3[adrs[0] + l * 9 + m * 3 + n]
                                + fc3[adrs[1] + l * 9 + n * 3 + m]
                                + fc3[adrs[2] + m * 9 + l * 3 + n]
                                + fc3[adrs[3] + m * 9 + n * 3 + l]
                                + fc3[adrs[4] + n * 9 + l * 3 + m]
                                + fc3[adrs[5] + n * 9 + m * 3 + l];
                            elem[l * 9 + m * 3 + n] = sum / 6.0;
                        }
                    }
                }
                for l in 0..3 {
                    for m in 0..3 {
                        for n in 0..3 {
                            let v = elem[l * 9 + m * 3 + n];
                            fc3[adrs[0] + l * 9 + m * 3 + n] = v;
                            fc3[adrs[1] + l * 9 + n * 3 + m] = v;
                            fc3[adrs[2] + m * 9 + l * 3 + n] = v;
                            fc3[adrs[3] + m * 9 + n * 3 + l] = v;
                            fc3[adrs[4] + n * 9 + l * 3 + m] = v;
                            fc3[adrs[5] + n * 9 + m * 3 + l] = v;
                        }
                    }
                }
            }
        }
    }
}

/// Rotate displacement-difference fc2 tensors and project them onto fc3.
///
/// Shapes:
/// * `fc3`: `(num_atom, num_atom, 3, 3, 3)` (flat, `num_atom * num_atom`
///   blocks of 27)
/// * `delta_fc2s`: `(num_disp, num_atom, num_atom, 3, 3)`
/// * `inv_u`: `(3, num_disp * num_site_sym)`
/// * `site_sym_cart`: `(num_site_sym, 3, 3)` (row-major 3x3 rotations)
/// * `rot_map_syms`: `(num_site_sym, num_atom)`
///
/// Mirrors `fc3_rotate_delta_fc2` in `c/fc3.c`.  Parallelised over the
/// `(i_atom, j_atom)` pair since each pair writes to its own 27-block
/// of `fc3` and only reads `delta_fc2s`, `inv_u`, `site_sym_cart`, and
/// `rot_map_syms` (all disjoint from the writes).
pub(crate) fn rotate_delta_fc2s(
    fc3: &mut [f64],
    delta_fc2s: &[f64],
    inv_u: &[f64],
    site_sym_cart: &[f64],
    rot_map_syms: &[i64],
    num_atom: usize,
    num_site_sym: usize,
    num_disp: usize,
) {
    let total = num_disp * num_site_sym;
    fc3.par_chunks_mut(27)
        .enumerate()
        .for_each(|(i_atoms, fc3_block)| {
            let i_a = i_atoms / num_atom;
            let j_a = i_atoms % num_atom;
            let mut rot_delta = vec![0.0f64; total * 9];
            for i_d in 0..num_disp {
                for j_s in 0..num_site_sym {
                    let src_i = rot_map_syms[j_s * num_atom + i_a] as usize;
                    let src_j = rot_map_syms[j_s * num_atom + j_a] as usize;
                    let src_base = ((i_d * num_atom + src_i) * num_atom + src_j) * 9;
                    let dst_base = (i_d * num_site_sym + j_s) * 9;
                    let r = &site_sym_cart[j_s * 9..j_s * 9 + 9];
                    // tensor2_rotation:
                    // out[i, j] = sum_{k, l} r[i, k] r[j, l] tensor[k, l]
                    for i in 0..3 {
                        for j in 0..3 {
                            rot_delta[dst_base + i * 3 + j] = 0.0;
                            for k in 0..3 {
                                for l in 0..3 {
                                    rot_delta[dst_base + i * 3 + j] += r[i * 3 + k]
                                        * r[j * 3 + l]
                                        * delta_fc2s[src_base + k * 3 + l];
                                }
                            }
                        }
                    }
                }
            }
            // fc3[i_atoms][k, l, m] = sum_n inv_u[k, n] * rot_delta[n, l, m]
            for k in 0..3 {
                for l in 0..3 {
                    for m in 0..3 {
                        fc3_block[k * 9 + l * 3 + m] = 0.0;
                        for n in 0..total {
                            fc3_block[k * 9 + l * 3 + m] +=
                                inv_u[k * total + n] * rot_delta[n * 9 + l * 3 + m];
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_rot() -> [f64; 9] {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn tensor3_rotation_identity_is_noop() {
        let tensor: Vec<f64> = (0..27).map(|v| v as f64).collect();
        let mut out = vec![0.0f64; 27];
        tensor3_rotation(&mut out, &tensor, &identity_rot());
        for k in 0..27 {
            assert!((out[k] - tensor[k]).abs() < 1e-15);
        }
    }

    #[test]
    fn tensor3_rotation_swap_xy_permutes_indices() {
        // r swaps x<->y: r[0,1]=r[1,0]=1, r[2,2]=1 (others 0).
        let r = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        // tensor[i,j,k] = 100*i + 10*j + k.
        let mut tensor = [0.0f64; 27];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    tensor[i * 9 + j * 3 + k] = (100 * i + 10 * j + k) as f64;
                }
            }
        }
        let mut out = [0.0f64; 27];
        tensor3_rotation(&mut out, &tensor, &r);
        // For this r, out[l,m,n] = tensor[sigma(l), sigma(m), sigma(n)]
        // where sigma swaps 0 and 1, leaves 2.
        let sigma = |x: usize| match x {
            0 => 1,
            1 => 0,
            _ => 2,
        };
        for l in 0..3 {
            for m in 0..3 {
                for n in 0..3 {
                    let expected = tensor[sigma(l) * 9 + sigma(m) * 3 + sigma(n)];
                    assert!((out[l * 9 + m * 3 + n] - expected).abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn set_permutation_symmetry_fc3_idempotent() {
        // After one call, a second call must not change anything.
        let num_atom = 3;
        let mut fc3 = vec![0.0f64; num_atom * num_atom * num_atom * 27];
        for (i, v) in fc3.iter_mut().enumerate() {
            *v = ((i * 17) % 101) as f64 - 50.0;
        }
        set_permutation_symmetry_fc3(&mut fc3, num_atom);
        let snapshot = fc3.clone();
        set_permutation_symmetry_fc3(&mut fc3, num_atom);
        for (a, b) in snapshot.iter().zip(fc3.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn set_permutation_symmetry_fc3_enforces_all_permutations() {
        let num_atom = 3;
        let mut fc3 = vec![0.0f64; num_atom * num_atom * num_atom * 27];
        for (i, v) in fc3.iter_mut().enumerate() {
            *v = ((i * 13) % 97) as f64 - 48.0;
        }
        set_permutation_symmetry_fc3(&mut fc3, num_atom);
        // For every triple and every (i, j, k), all six permuted entries
        // must be equal.
        let stride_a = num_atom * num_atom * 27;
        let stride_b = num_atom * 27;
        for a in 0..num_atom {
            for b in 0..num_atom {
                for c in 0..num_atom {
                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                let v_abc =
                                    fc3[a * stride_a + b * stride_b + c * 27 + i * 9 + j * 3 + k];
                                let v_acb =
                                    fc3[a * stride_a + c * stride_b + b * 27 + i * 9 + k * 3 + j];
                                let v_bac =
                                    fc3[b * stride_a + a * stride_b + c * 27 + j * 9 + i * 3 + k];
                                let v_bca =
                                    fc3[b * stride_a + c * stride_b + a * 27 + j * 9 + k * 3 + i];
                                let v_cab =
                                    fc3[c * stride_a + a * stride_b + b * 27 + k * 9 + i * 3 + j];
                                let v_cba =
                                    fc3[c * stride_a + b * stride_b + a * 27 + k * 9 + j * 3 + i];
                                for v in [v_acb, v_bac, v_bca, v_cab, v_cba] {
                                    assert!((v - v_abc).abs() < 1e-14);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn transpose_compact_fc3_type2_is_involution() {
        // type=2 swaps dim 1 <-> dim 2 along with the (m <-> n) inner
        // swap.  Applied twice, fc3 must return to its original state.
        let n_patom = 2;
        let n_satom = 3;
        let mut fc3 = vec![0.0f64; n_patom * n_satom * n_satom * 27];
        for (i, v) in fc3.iter_mut().enumerate() {
            *v = ((i * 11) % 73) as f64 - 36.0;
        }
        let original = fc3.clone();
        // Args for t_type=2 are ignored by that branch.
        let p2s = [0i64, 1];
        let s2pp = [0i64, 1, 0];
        let nsym_list = [0i64; 3];
        let perms = [0i64, 1, 2];
        transpose_compact_fc3(
            &mut fc3, &p2s, &s2pp, &nsym_list, &perms, n_satom, n_patom, 2,
        );
        transpose_compact_fc3(
            &mut fc3, &p2s, &s2pp, &nsym_list, &perms, n_satom, n_patom, 2,
        );
        for (a, b) in original.iter().zip(fc3.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn transpose_compact_fc3_type01_is_involution() {
        // With trivial (identity) symmetry and n_patom = n_satom, the
        // type=0 / type=1 swap is its own inverse as well.
        let n = 3;
        let size = n * n * n * 27;
        let p2s: Vec<i64> = (0..n as i64).collect();
        let s2pp = p2s.clone();
        let nsym_list = vec![0i64; n];
        let perms: Vec<i64> = (0..n as i64).collect();
        for t_type in [0i64, 1] {
            let mut fc3 = vec![0.0f64; size];
            for (i, v) in fc3.iter_mut().enumerate() {
                *v = ((i * 7 + t_type as usize * 5) % 59) as f64 - 29.0;
            }
            let original = fc3.clone();
            transpose_compact_fc3(&mut fc3, &p2s, &s2pp, &nsym_list, &perms, n, n, t_type);
            transpose_compact_fc3(&mut fc3, &p2s, &s2pp, &nsym_list, &perms, n, n, t_type);
            for (a, b) in original.iter().zip(fc3.iter()) {
                assert!((a - b).abs() < 1e-15, "t_type {} not involutive", t_type);
            }
        }
    }

    #[test]
    fn set_permutation_symmetry_compact_fc3_idempotent() {
        // Trivial symmetry (n_patom == n_satom, identity perms).  Under
        // this setup, compact fc3 coincides with full fc3 and a second
        // call must not change the result.
        let n = 3;
        let size = n * n * n * 27;
        let p2s: Vec<i64> = (0..n as i64).collect();
        let s2pp = p2s.clone();
        let nsym_list = vec![0i64; n];
        let perms: Vec<i64> = (0..n as i64).collect();

        let mut fc3 = vec![0.0f64; size];
        for (i, v) in fc3.iter_mut().enumerate() {
            *v = ((i * 19) % 89) as f64 - 44.0;
        }
        set_permutation_symmetry_compact_fc3(&mut fc3, &p2s, &s2pp, &nsym_list, &perms, n, n);
        let snapshot = fc3.clone();
        set_permutation_symmetry_compact_fc3(&mut fc3, &p2s, &s2pp, &nsym_list, &perms, n, n);
        for (a, b) in snapshot.iter().zip(fc3.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn set_permutation_symmetry_compact_fc3_matches_full_under_trivial_symmetry() {
        // Under trivial symmetry, the compact routine must produce the
        // same fc3 as the full routine on the same data.
        let n = 3;
        let size = n * n * n * 27;
        let p2s: Vec<i64> = (0..n as i64).collect();
        let s2pp = p2s.clone();
        let nsym_list = vec![0i64; n];
        let perms: Vec<i64> = (0..n as i64).collect();

        let mut base = vec![0.0f64; size];
        for (i, v) in base.iter_mut().enumerate() {
            *v = ((i * 23) % 97) as f64 - 48.0;
        }
        let mut fc3_full = base.clone();
        let mut fc3_compact = base.clone();
        set_permutation_symmetry_fc3(&mut fc3_full, n);
        set_permutation_symmetry_compact_fc3(
            &mut fc3_compact,
            &p2s,
            &s2pp,
            &nsym_list,
            &perms,
            n,
            n,
        );
        for (a, b) in fc3_full.iter().zip(fc3_compact.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn rotate_delta_fc2s_identity_symmetry_is_projection() {
        // num_site_sym = 1 with identity rotation, num_disp = 1.
        // Then rot_delta[0, l, m] = delta_fc2s[0, i_a, j_a, l, m]
        // and fc3[i_atoms, k, l, m] = inv_u[k, 0] * delta_fc2s[0, i_a, j_a, l, m].
        let num_atom = 2;
        let num_disp = 1;
        let num_site_sym = 1;
        let total = num_disp * num_site_sym;

        // Seed delta_fc2s with unique values.
        let mut delta_fc2s = vec![0.0f64; num_disp * num_atom * num_atom * 9];
        for (i, v) in delta_fc2s.iter_mut().enumerate() {
            *v = (i + 1) as f64;
        }
        // inv_u of shape (3, 1); pick distinct values.
        let inv_u = vec![0.25f64, -0.5, 2.0];
        // Identity 3x3.
        let site_sym_cart = vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // rot_map_syms is identity: rot_map[0, a] = a.
        let rot_map_syms: Vec<i64> = (0..num_atom as i64).collect();

        let mut fc3 = vec![0.0f64; num_atom * num_atom * 27];
        rotate_delta_fc2s(
            &mut fc3,
            &delta_fc2s,
            &inv_u,
            &site_sym_cart,
            &rot_map_syms,
            num_atom,
            num_site_sym,
            num_disp,
        );

        for i_a in 0..num_atom {
            for j_a in 0..num_atom {
                let out_base = (i_a * num_atom + j_a) * 27;
                let src_base = ((0 * num_atom + i_a) * num_atom + j_a) * 9;
                for k in 0..3 {
                    for l in 0..3 {
                        for m in 0..3 {
                            let expected = inv_u[k * total] * delta_fc2s[src_base + l * 3 + m];
                            let got = fc3[out_base + k * 9 + l * 3 + m];
                            assert!((got - expected).abs() < 1e-14);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn distribute_fc3_identity_copies_block() {
        // num_atom = 2, so fc3 has 2*2*2 = 8 blocks of 27.
        let num_atom = 2;
        let mut fc3 = vec![0.0f64; num_atom * num_atom * num_atom * 27];
        // Fill source (target index = 1) block of source = 0 with unique values.
        for i in 0..num_atom {
            for j in 0..num_atom {
                let base = (0 * num_atom * num_atom + i * num_atom + j) * 27;
                for k in 0..27 {
                    fc3[base + k] = (1000 * i + 100 * j + k) as f64;
                }
            }
        }
        let atom_mapping: Vec<i64> = (0..num_atom as i64).collect();
        distribute_fc3(&mut fc3, 1, 0, &atom_mapping, num_atom, &identity_rot());
        // Target slab must equal source slab.
        for i in 0..num_atom {
            for j in 0..num_atom {
                let src = (0 * num_atom * num_atom + i * num_atom + j) * 27;
                let tgt = (1 * num_atom * num_atom + i * num_atom + j) * 27;
                for k in 0..27 {
                    assert!((fc3[tgt + k] - fc3[src + k]).abs() < 1e-15);
                }
            }
        }
    }
}
