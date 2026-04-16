//! Smith Normal Form of a 3x3 integer matrix.
//!
//! Port of `c/snf3x3.c`.  Given an integer matrix `A`, compute
//! integer unimodular matrices `P` and `Q` and a diagonal matrix
//! `D` such that `P * A * Q = D`.

type Mat = [[i64; 3]; 3];

const IDENTITY: Mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snf3x3 {
    pub d_diag: [i64; 3],
    pub p: Mat,
    pub q: Mat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Snf3x3Error {
    Singular,
    NotConverged,
}

pub fn snf3x3(a: Mat) -> Result<Snf3x3, Snf3x3Error> {
    if det(&a) == 0 {
        return Err(Snf3x3Error::Singular);
    }

    let mut a = a;
    let mut p = IDENTITY;
    let mut q = IDENTITY;

    for _ in 0..100 {
        if first(&mut a, &mut p, &mut q) && second(&mut a, &mut p, &mut q) {
            finalize(&mut a, &mut p, &mut q);
            transpose(&mut q);
            let d_diag = [a[0][0], a[1][1], a[2][2]];
            return Ok(Snf3x3 { d_diag, p, q });
        }
    }
    Err(Snf3x3Error::NotConverged)
}

// ---------- first row/column reduction ----------

fn first(a: &mut Mat, p: &mut Mat, q: &mut Mat) -> bool {
    first_one_loop(a, p, q);

    if a[1][0] == 0 && a[2][0] == 0 {
        return true;
    }

    if a[1][0] % a[0][0] == 0 && a[2][0] % a[0][0] == 0 {
        let l = first_finalize(a);
        *a = matmul(&l, a);
        *p = matmul(&l, p);
        return true;
    }
    false
}

fn first_one_loop(a: &mut Mat, p: &mut Mat, q: &mut Mat) {
    first_column(a, p);
    transpose(a);
    first_column(a, q);
    transpose(a);
}

fn first_column(a: &mut Mat, p: &mut Mat) {
    let Some(i) = search_first_pivot(a) else {
        return;
    };
    if i > 0 {
        let l = elementary_swap_rows(0, i);
        *a = matmul(&l, a);
        *p = matmul(&l, p);
    }
    if a[1][0] != 0 {
        let l = zero_first_column(1, a);
        *a = matmul(&l, a);
        *p = matmul(&l, p);
    }
    if a[2][0] != 0 {
        let l = zero_first_column(2, a);
        *a = matmul(&l, a);
        *p = matmul(&l, p);
    }
}

fn zero_first_column(j: usize, a: &Mat) -> Mat {
    let [r, s, t] = extended_gcd(a[0][0], a[j][0]);
    elementary_set_zero(0, j, a[0][0], a[j][0], r, s, t)
}

fn search_first_pivot(a: &Mat) -> Option<usize> {
    (0..3).find(|&i| a[i][0] != 0)
}

fn first_finalize(a: &Mat) -> Mat {
    let mut l = IDENTITY;
    l[1][0] = -a[1][0] / a[0][0];
    l[2][0] = -a[2][0] / a[0][0];
    l
}

// ---------- second row/column reduction ----------

fn second(a: &mut Mat, p: &mut Mat, q: &mut Mat) -> bool {
    second_one_loop(a, p, q);

    if a[2][1] == 0 {
        return true;
    }
    if a[2][1] % a[1][1] == 0 {
        let l = second_finalize(a);
        *a = matmul(&l, a);
        *p = matmul(&l, p);
        return true;
    }
    false
}

fn second_one_loop(a: &mut Mat, p: &mut Mat, q: &mut Mat) {
    second_column(a, p);
    transpose(a);
    second_column(a, q);
    transpose(a);
}

fn second_column(a: &mut Mat, p: &mut Mat) {
    if a[1][1] == 0 && a[2][1] != 0 {
        let l = elementary_swap_rows(1, 2);
        *a = matmul(&l, a);
        *p = matmul(&l, p);
    }
    if a[2][1] != 0 {
        let l = zero_second_column(a);
        *a = matmul(&l, a);
        *p = matmul(&l, p);
    }
}

fn zero_second_column(a: &Mat) -> Mat {
    let [r, s, t] = extended_gcd(a[1][1], a[2][1]);
    elementary_set_zero(1, 2, a[1][1], a[2][1], r, s, t)
}

fn second_finalize(a: &Mat) -> Mat {
    let mut l = IDENTITY;
    l[2][1] = -a[2][1] / a[1][1];
    l
}

// ---------- finalize ----------

fn finalize(a: &mut Mat, p: &mut Mat, q: &mut Mat) {
    make_diag_positive(a, p);
    finalize_sort(a, p, q);
    finalize_disturb(a, q, 0, 1);
    first(a, p, q);
    finalize_sort(a, p, q);
    finalize_disturb(a, q, 1, 2);
    second(a, p, q);
    flip_pq(p, q);
}

fn finalize_sort(a: &mut Mat, p: &mut Mat, q: &mut Mat) {
    if a[0][0] > a[1][1] {
        swap_diag_elems(a, p, q, 0, 1);
    }
    if a[1][1] > a[2][2] {
        swap_diag_elems(a, p, q, 1, 2);
    }
    if a[0][0] > a[1][1] {
        swap_diag_elems(a, p, q, 0, 1);
    }
}

fn finalize_disturb(a: &mut Mat, q: &mut Mat, i: usize, j: usize) {
    if a[j][j] % a[i][i] != 0 {
        transpose(a);
        let l = elementary_disturb_rows(i, j);
        *a = matmul(&l, a);
        *q = matmul(&l, q);
        transpose(a);
    }
}

fn swap_diag_elems(a: &mut Mat, p: &mut Mat, q: &mut Mat, i: usize, j: usize) {
    let l = elementary_swap_rows(i, j);
    *a = matmul(&l, a);
    *p = matmul(&l, p);
    transpose(a);
    *a = matmul(&l, a);
    *q = matmul(&l, q);
    transpose(a);
}

fn make_diag_positive(a: &mut Mat, p: &mut Mat) {
    for i in 0..3 {
        if a[i][i] < 0 {
            let l = elementary_flip_sign_row(i);
            *a = matmul(&l, a);
            *p = matmul(&l, p);
        }
    }
}

fn flip_pq(p: &mut Mat, q: &mut Mat) {
    if det(p) < 0 {
        for i in 0..3 {
            for j in 0..3 {
                p[i][j] = -p[i][j];
                q[i][j] = -q[i][j];
            }
        }
    }
}

// ---------- elementary matrices ----------

fn elementary_swap_rows(r1: usize, r2: usize) -> Mat {
    let mut l = IDENTITY;
    l[r1][r1] = 0;
    l[r2][r2] = 0;
    l[r1][r2] = 1;
    l[r2][r1] = 1;
    l
}

fn elementary_set_zero(i: usize, j: usize, a: i64, b: i64, r: i64, s: i64, t: i64) -> Mat {
    let mut l = IDENTITY;
    l[i][i] = s;
    l[i][j] = t;
    l[j][i] = -b / r;
    l[j][j] = a / r;
    l
}

fn elementary_flip_sign_row(i: usize) -> Mat {
    let mut l = IDENTITY;
    l[i][i] = -1;
    l
}

fn elementary_disturb_rows(i: usize, j: usize) -> Mat {
    let mut l = IDENTITY;
    l[i][j] = 1;
    l
}

// ---------- extended gcd ----------

// Returns [gcd, s, t] such that gcd == s*a + t*b.  Uses Euclidean
// (non-negative remainder) division, matching the C version's
// manual normalization of a truncated remainder.
fn extended_gcd(a: i64, b: i64) -> [i64; 3] {
    let mut r = [a, b];
    let mut s = [1i64, 0];
    let mut t = [0i64, 1];

    for _ in 0..1000 {
        if r[1] == 0 {
            break;
        }
        let q = r[0].div_euclid(r[1]);
        let r2 = r[0].rem_euclid(r[1]);
        let s2 = s[0] - q * s[1];
        let t2 = t[0] - q * t[1];
        r[0] = r[1];
        r[1] = r2;
        s[0] = s[1];
        s[1] = s2;
        t[0] = t[1];
        t[1] = t2;
    }

    debug_assert_eq!(r[0], a * s[0] + b * t[0]);
    [r[0], s[0], t[0]]
}

// ---------- matrix utilities ----------

fn transpose(m: &mut Mat) {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let tmp = m[i][j];
            m[i][j] = m[j][i];
            m[j][i] = tmp;
        }
    }
}

// Returns a * b.  Safe against aliasing because the result is
// returned by value, not written through a shared pointer.
fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut c = [[0i64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn det(m: &Mat) -> i64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
