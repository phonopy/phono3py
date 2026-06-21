(inter_band_transport)=

# Inter-band transport

The standard lattice thermal conductivity calculation of phono3py (selected by
`--br` or `--lbte`) treats phonons as particles and accumulates only the
diagonal (intra-band) contributions of the heat flux. In systems with many phonon
bands that are close in frequency (e.g. crystals of large unit cells or
glass-like), the off-diagonal (inter-band) terms
of the heat-flux operator between such distinct bands provide an additional
contribution to the thermal conductivity.

The `--tt` (`--transport-type`) option selects which transport formulation is
used to evaluate these contributions:

```
phono3py ... --br --tt smm19
phono3py ... --lbte --tt njc23
```

- `--tt` is combined with `--br` (RTA) or `--lbte` (direct solution). Without
  `--tt`, the standard particle-like formulation is used.
- The built-in variants `njc23`, `ibdb19`, and `smm19` are **experimental**.
  Their interface and behavior may change without notice.
- `wte` invokes the external
  [phono3py-wte](https://github.com/MSimoncelli/phono3py-wte) plugin, which must
  be installed separately (see {ref}`wigner_solution`).

In the following, the phonon frequency, eigenvector, Bose-Einstein
distribution, and scattering half-linewidth (HWHM) of mode $(\mathbf{q}, j)$ are
denoted by $\omega_{\mathbf{q}j}$, $e_{\mathbf{q}j}$, $n_{\mathbf{q}j}$, and
$\Gamma_{\mathbf{q}j}$, respectively. $T$ is the temperature and $D(\mathbf{q})$
is the dynamical matrix.

(velocity_matrix)=

## Velocity matrix

The off-diagonal contributions are built from the velocity matrix, which
generalizes the group velocity to pairs of bands $(j, j')$ at the same q-point.
For the Cartesian direction $\alpha$,

$$
v^{\alpha}_{\mathbf{q}jj'} =
\frac{1}{2\sqrt{\omega_{\mathbf{q}j}\omega_{\mathbf{q}j'}}}
\langle e_{\mathbf{q}j} |
\frac{\partial D(\mathbf{q})}{\partial q_{\alpha}}
| e_{\mathbf{q}j'} \rangle .
$$

The diagonal element ($j = j'$) reduces to the ordinary group velocity
$v_{\mathbf{q}j}$. Degenerate bands are handled by rotating the corresponding
eigenvectors so that the velocity matrix is well defined. In the conductivity
formulas below, the outer product
$v^{\alpha}_{\mathbf{q}jj'} v^{\beta *}_{\mathbf{q}jj'}$ is averaged over the star
of $\mathbf{q}$. This star average is real (its imaginary part cancels by
symmetry) and recovers the crystal symmetry of the conductivity tensor.

(mode_conductivity)=

## Mode thermal conductivity

All inter-band variants build the thermal conductivity from the velocity matrix
and a heat capacity matrix $C_{\mathbf{q}jj'}$. They differ only in the
definition of this matrix, given in the sections below. The mode contribution
to the thermal conductivity tensor for the band pair $(j, j')$ at q-point
$\mathbf{q}$ is

$$
\kappa^{\alpha\beta}_{\mathbf{q}jj'} =
C_{\mathbf{q}jj'}\,
v^{\alpha}_{\mathbf{q}jj'} v^{\beta *}_{\mathbf{q}jj'}
\frac{\Gamma_{\mathbf{q}j} + \Gamma_{\mathbf{q}j'}}
{(\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'})^{2}
+ (\Gamma_{\mathbf{q}j} + \Gamma_{\mathbf{q}j'})^{2}} .
$$

Each heat capacity matrix reduces on the diagonal ($j = j'$) to the ordinary
mode heat capacity $C_{\mathbf{q}j}$, so the diagonal terms reproduce the
standard particle-like (populations) conductivity, while the off-diagonal terms
($j \ne j'$) give the additional inter-band contribution. The thermal
conductivity tensor is obtained by summing the mode contributions over the
Brillouin zone and all band pairs (with the usual normalization by the cell
volume and the number of grid points).

Of the built-in variants, `njc23` and `ibdb19` arise from the Green-Kubo
formulation and `smm19` from the Wigner / unified-theory picture. All reduce to
the formula above with a variant-specific heat capacity matrix.

The `njc23` and `ibdb19` heat capacity matrices are closely related. `ibdb19`
evaluates the main-text Eq. (9) of Isaeva, Barbalinardo, Donadio, and Baroni
(2019), a further approximation of a more detailed formula derived in the
Supplementary Information of the same paper. The expression of Ndour, Jund, and
Chaput (`njc23`) was obtained independently and turns out to coincide with the
first term of Eq. (19) of that Supplementary Information, so `njc23` is the more
accurate of the two. The sections below are ordered from the more accurate
(`njc23`) to the more approximate (`ibdb19`), followed by `smm19`.

## `--tt njc23`

This variant evaluates the expression of Ndour, Jund, and Chaput. Its heat
capacity matrix, generalizing the scalar mode heat capacity to a band pair, is

$$
C_{\mathbf{q}jj'} =
-\frac{\hbar\,(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})^{2}}
{4 T (\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'})}
(n_{\mathbf{q}j} - n_{\mathbf{q}j'}) .
$$

- M. Ndour, P. Jund, and L. Chaput, "Practical approach to thermal conductivity
  calculations of small SiO2 samples", J. Non-Cryst. Solids **621**, 122618
  (2023). DOI:
  [10.1016/j.jnoncrysol.2023.122618](https://doi.org/10.1016/j.jnoncrysol.2023.122618)

## `--tt ibdb19`

This variant evaluates the main-text Eq. (9) of Isaeva, Barbalinardo, Donadio,
and Baroni. Its heat capacity matrix is

$$
C_{\mathbf{q}jj'} =
-\frac{\hbar\,\omega_{\mathbf{q}j} \omega_{\mathbf{q}j'}}
{T (\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'})}
(n_{\mathbf{q}j} - n_{\mathbf{q}j'}) .
$$

The two heat capacity matrices differ only in the frequency prefactor: `njc23`
uses the arithmetic-mean-square
$(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})^{2} / 4$, while `ibdb19` uses the
geometric-mean-square $\omega_{\mathbf{q}j} \omega_{\mathbf{q}j'}$. Since
$\omega_{\mathbf{q}j} \omega_{\mathbf{q}j'} \le (\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})^{2} / 4$
with equality only at $\omega_{\mathbf{q}j} = \omega_{\mathbf{q}j'}$, the two
coincide on the diagonal (and thus give the same particle-like conductivity),
while the off-diagonal contribution of `ibdb19` is slightly smaller than that of
`njc23`.

- L. Isaeva, G. Barbalinardo, D. Donadio, and S. Baroni, "Modeling heat
  transport in crystals and glasses from a unified lattice-dynamical approach",
  Nat. Commun. **10**, 3853 (2019). DOI:
  [10.1038/s41467-019-11572-4](https://doi.org/10.1038/s41467-019-11572-4)

## `--tt smm19`

This variant evaluates a Wigner / unified-theory term in the spirit of the
formulation by Simoncelli, Marzari, and Mauri. The original formulation and its
reference implementation are due to those authors and are provided in the
phono3py-wte plugin, invoked with `--tt wte` (see {ref}`wigner_solution`). That
plugin should be used for results consistent with the published Wigner
formulation. The built-in `smm19` here is an experimental, simplified in-tree
variant that adopts the velocity matrix defined in the {ref}`Velocity matrix
<velocity_matrix>` section above (the same definition as the other variants)
instead of the original velocity-operator definition. Its off-diagonal results
are therefore not identical to those of the original Wigner formulation.

This in-tree `smm19` shares the same velocity matrix and Lorentzian linewidth
factor as the other variants. The difference is its heat capacity matrix, an
effective matrix built from the scalar mode heat capacities $C_{\mathbf{q}j}$ in
a symmetrized form,

$$
C_{\mathbf{q}jj'} =
\frac{1}{4}
(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})
\left(
\frac{C_{\mathbf{q}j}}{\omega_{\mathbf{q}j}}
+ \frac{C_{\mathbf{q}j'}}{\omega_{\mathbf{q}j'}}
\right) ,
$$

which reduces on the diagonal ($j = j'$) to the ordinary mode heat capacity
$C_{\mathbf{q}j}$.

- M. Simoncelli, N. Marzari, and F. Mauri, "Unified theory of thermal transport
  in crystals and glasses", Nat. Phys. **15**, 809 (2019). DOI:
  [10.1038/s41567-019-0520-x](https://doi.org/10.1038/s41567-019-0520-x)

## `--tt wte`

The solution of the Wigner transport equation is provided as a separate plugin,
phono3py-wte, and is invoked with `--tt wte`. The plugin must be installed
separately (it is not bundled with phono3py). See {ref}`wigner_solution` for
installation and usage.

```{toctree}
:hidden:
wigner-solution
```
