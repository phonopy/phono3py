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
- The built-in variants `smm19` and `njc23` are **experimental**; their
  interface and behavior may change without notice.
- `wte` invokes the external
  [phono3py-wte](https://github.com/MSimoncelli/phono3py-wte) plugin (see
  {ref}`wigner_solution`).

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
of $\mathbf{q}$; this star average is real (its imaginary part cancels by
symmetry) and recovers the crystal symmetry of the conductivity tensor.

## `--tt njc23`

This variant evaluates the Green-Kubo term. In addition to the velocity matrix
defined above, it uses the off-diagonal heat capacity matrix, obtained by
generalizing the scalar mode heat capacity to a band pair,

$$
C_{\mathbf{q}jj'} =
-\frac{\hbar\,(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})^{2}}
{4 T (\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'})}
(n_{\mathbf{q}j} - n_{\mathbf{q}j'}) ,
$$

whose diagonal element ($j = j'$) reduces to the ordinary mode heat capacity
$C_{\mathbf{q}j}$. The mode contribution to the thermal conductivity tensor for
the band pair $(j, j')$ at q-point $\mathbf{q}$ is

$$
\kappa^{\alpha\beta}_{\mathbf{q}jj'} =
C_{\mathbf{q}jj'}\,
v^{\alpha}_{\mathbf{q}jj'} v^{\beta *}_{\mathbf{q}jj'}
\frac{\Gamma_{\mathbf{q}j} + \Gamma_{\mathbf{q}j'}}
{(\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'})^{2}
+ (\Gamma_{\mathbf{q}j} + \Gamma_{\mathbf{q}j'})^{2}} .
$$

The diagonal terms ($j = j'$) reproduce the standard particle-like (populations)
conductivity, while the off-diagonal terms ($j \ne j'$) give the additional
inter-band contribution. The thermal conductivity tensor is obtained by summing
the mode contributions over the Brillouin zone and all band pairs (with the
usual normalization by the cell volume and the number of grid points).

phono3py implements the expression of Ndour, Jund, and Chaput, which can be seen
as an approximation of the quasi-harmonic Green-Kubo formula derived in the
appendix of Isaeva, Barbalinardo, Donadio, and Baroni:

- M. Ndour, P. Jund, and L. Chaput, "Practical approach to thermal conductivity
  calculations of small SiO2 samples", J. Non-Cryst. Solids **621**, 122618
  (2023). DOI:
  [10.1016/j.jnoncrysol.2023.122618](https://doi.org/10.1016/j.jnoncrysol.2023.122618)

- L. Isaeva, G. Barbalinardo, D. Donadio, and S. Baroni, "Modeling heat
  transport in crystals and glasses from a unified lattice-dynamical approach",
  Nat. Commun. **10**, 3853 (2019). DOI:
  [10.1038/s41467-019-11572-4](https://doi.org/10.1038/s41467-019-11572-4)

## `--tt smm19`

This variant evaluates a Wigner / unified-theory coherence term in the spirit of
the formulation by Simoncelli, Marzari, and Mauri. The original formulation and
its reference implementation are due to those authors and are provided in the
phono3py-wte plugin, invoked with `--tt wte` (see {ref}`wigner_solution`); that
plugin should be used for results consistent with the published Wigner
formulation. The built-in `smm19` here is an experimental, simplified in-tree
variant that adopts the velocity matrix defined in the {ref}`Velocity matrix
<velocity_matrix>` section above (the same definition as `njc23`) instead of the
original velocity-operator definition. Its off-diagonal results are therefore not
identical to those of the original Wigner formulation.

This in-tree `smm19` shares the same velocity matrix and Lorentzian linewidth
factor as `njc23`; the difference is the heat-capacity prefactor, which is built
from the scalar mode heat capacities $C_{\mathbf{q}j}$ in a symmetrized form. The
mode contribution to the thermal conductivity tensor for the band pair $(j, j')$
at q-point $\mathbf{q}$ is

$$
\kappa^{\alpha\beta}_{\mathbf{q}jj'} =
\frac{1}{4}
(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})
\left(
\frac{C_{\mathbf{q}j}}{\omega_{\mathbf{q}j}}
+ \frac{C_{\mathbf{q}j'}}{\omega_{\mathbf{q}j'}}
\right)
v^{\alpha}_{\mathbf{q}jj'} v^{\beta *}_{\mathbf{q}jj'}
\frac{\Gamma_{\mathbf{q}j} + \Gamma_{\mathbf{q}j'}}
{(\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'})^{2}
+ (\Gamma_{\mathbf{q}j} + \Gamma_{\mathbf{q}j'})^{2}} .
$$

As in `njc23`, the diagonal terms ($j = j'$) reproduce the standard
particle-like (populations) conductivity, while the off-diagonal terms
($j \ne j'$) give the coherence contribution.

If the symmetrized prefactor is regarded as an effective heat capacity matrix,

$$
C^{\mathrm{SMM}}_{\mathbf{q}jj'} =
\frac{1}{4}
(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})
\left(
\frac{C_{\mathbf{q}j}}{\omega_{\mathbf{q}j}}
+ \frac{C_{\mathbf{q}j'}}{\omega_{\mathbf{q}j'}}
\right) ,
$$

the formula takes the same form as `njc23`, with $C^{\mathrm{SMM}}_{\mathbf{q}jj'}$
in place of $C_{\mathbf{q}jj'}$ and the same velocity matrix. In this sense the
two variants differ only in the definition of the heat capacity matrix.

For the formulation and background, see
{ref}`Simoncelli, Marzari, and Mauri (2019) <citation_unified_theory>`.

## `--tt wte`

The solution of the Wigner transport equation is provided as a separate plugin,
phono3py-wte, and is invoked with `--tt wte`. See {ref}`wigner_solution` for
installation and usage.

```{toctree}
:hidden:
wigner-solution
```
