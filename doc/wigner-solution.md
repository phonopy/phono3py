(wigner_solution)=

# Solution of the Wigner transport equation

This page explains how to compute the thermal conductivity from the solution of the Wigner transport equation (WTE)
[M. Simoncelli, N. Marzari, F. Mauri; Nat. Phys. 15, 809 (2019)](https://doi.org/10.1038/s41567-019-0520-x)
({ref}`citation <citation_unified_theory>`) and
[M. Simoncelli, N. Marzari, F. Mauri; arXiv:2112.06897 (2021)](https://arxiv.org/pdf/2112.06897)
({ref}`citation <citation_wigner_formulation>`).


The Wigner formulation of thermal transport in solids encompasses the emergence and coexistence of the particle-like propagation of phonon wavepackets discusses by Peierls for crystals [Peierls, Quantum theory of solids (Oxford Classics Series, 2001)], and the wave-like interband conduction mechanisms discussed by Allen and Feldman for harmonic glasses [
Allen and Feldman, Phys. Rev. Lett. 62, 645 (1989)]. As discussed in the references above, the Wigner formulation allows to describe the thermal conductivity of ordered crystals (where it yields practically the same result of the LBTE), of disordered glasses (where it generalizes Allen-Feldman theory accounting for anharmonicity), as well as of materials with intermediate characteristics (in this intermediate regime, both particle-like and wave-like conduction mechanisms are relevant, thus the Wigner formulation has to be used to obtain accurately predict the thermal conductivity).

In practice, the solution of the Wigner transport equation yields the following expression for the thermal conductivity tensor (we use $\alpha \beta$ to denote Cartesian directions) $\kappa^{\alpha \beta}_{\rm TOT}=\kappa_P^{\alpha \beta}+\kappa_{\rm C}^{\alpha \beta}$, where $\kappa_{\rm P}^{\alpha \beta}$ accounts for the particle-like propagation of phonon wavepackets and is exactly equivalent to the conductivity obtained solving the LBTE, while the other term $\kappa_{\rm C}^{\alpha \beta}$ is the "coherences" conductivity and accounts for the wave-like tunneling of phonons between bands with an energy differences smaller than their linewidths.
Specifically, the expression for $\kappa_{\rm C}^{\alpha \beta}$ reads:

$$
\kappa^{\alpha \beta}_{\rm C}=\frac{\hbar^2}{k_{B} {T}^2}\frac{1}{\mathcal{V}N_{\rm c}}\sum_{\mathbf{q}}\sum_{s\neq s'}\frac{\omega(\mathbf{q})_{s}+\omega(\mathbf{q})_{s'}}{2}{{V}^\alpha}(\mathbf{q})_{s,s'}{{V}}^\beta(\mathbf{q})_{s',s}
\frac{\omega(\mathbf{q})_{s}\bar{{N}}^{T}({\mathbf{q}})_{s}[\bar{{N}}^{T}({\mathbf{q}})_{s}+1]+\omega(\mathbf{q})_{s'}\bar{{N}}^{T}({\mathbf{q}})_{s'}[\bar{{N}}^{T}({\mathbf{q}})_{s'}+1]}{4[\omega(\mathbf{q})_{s'}-\omega(\mathbf{q})_{s}]^2+[\Gamma(\mathbf{q})_{s}+\Gamma(\mathbf{q})_{s'}]^2}[\Gamma(\mathbf{q})_{s}+\Gamma(\mathbf{q})_{s'}],
$$

where $k_{B}$ is the Boltzmann constant, $\mathcal{V}$ is the volume of the primitive cell, $N_{\rm c}$ is the number of phonon wavevectors $\mathbf{q}$ used to sample the Brillouin zone, $\hbar\omega(\mathbf{q})_s$ is the energy of the phonon with wavevector $\mathbf{q}$ and mode $s$, ${{V}^\alpha}(\mathbf{q})_{s,s'}$ is the velocity operator in direction $\alpha$, $\bar{{N}}^{T}({\mathbf{q}})_{s}$ is the Bose-Einstein distribution at temperature $T$, and $\Gamma(\mathbf{q})_{s}$ is the phonon linewidth (full width at half maximum, i.e. the inverse phonon lifetime $\Gamma(\mathbf{q})_{s}=[\tau(\mathbf{q})_{s}]^{-1}$).


As discussed in the references above, the term $\kappa_{\rm P}^{\alpha \beta}$ can be evaluated exactly or in the RTA approximation (the former corresponds to account for all the repumping/depumping scattering events, while the latter only for depumping scattering events). In contrast, the term $\kappa_{\rm C}^{\alpha \beta}$ depends only on the depumping scattering events, thus it remains unchanged if scattering is considered exactly or in the RTA approximation.



```{contents}
:depth: 2
:local:
```

## How to use

### Solution of the WTE, scattering in the RTA approximation
To compute the Wigner conductivity with scattering in the RTA approximation, specify `--br --wigner`. For `example/Si-PBEsol`, the command is:
```bash
% phono3py-load --mesh 11 11 11 --ts 1600 --br --wigner
```
and the output is
```bash
...
=================== End of collection of collisions ===================
----------- Thermal conductivity (W/m-k) with tetrahedron method -----------
#           T(K)        xx         yy         zz         yz         xz         xy
K_P    1600.0      20.059     20.059     20.059     -0.000     -0.000      0.000

K_C    1600.0       0.277      0.277      0.277     -0.000     -0.000      0.000

K_T    1600.0      20.335     20.335     20.335     -0.000     -0.000      0.000
...
```

### Solution of the WTE, exact treatment of scattering
To compute the Wigner conductivity treating scattering exactly, specify `--lbte --wigner`. For `example/Si-PBEsol`, the command is:

```bash
% phono3py-load --mesh 11 11 11 --ts 1600 --lbte --wigner
```
and the output is
```bash
...
=================== End of collection of collisions ===================
- Averaging collision matrix elements by phonon degeneracy [0.035s]
- Making collision matrix symmetric (built-in) [0.000s]
----------- Thermal conductivity (W/m-k) with tetrahedron method -----------
Diagonalizing by lapacke dsyev... [0.148s]
Calculating pseudo-inv with cutoff=1.0e-08 (np.dot) [0.002s]
#                T(K)        xx         yy         zz         yz         xz         xy
K_P_exact       1600.0      21.009     21.009     21.009     -0.000     -0.000      0.000
(K_P_RTA)       1600.0      20.059     20.059     20.059     -0.000     -0.000      0.000
K_C             1600.0       0.277      0.277      0.277     -0.000     -0.000      0.000

K_TOT=K_P_exact+K_C   1600.0      21.286     21.286     21.286     -0.000     -0.000      0.000
----------------------------------------------------------------------------
...
```

We also note that the examples above are performed at very high temperature for illustrative purposes.
The coherences conductivity is often a non-negligible fraction of the total conductivity in materials with glass-like or ultralow thermal conductivity ($\frac{1}{3}\sum_{\alpha=1}^3\kappa^{\alpha\alpha}_{\rm TOT}\lesssim 1 \frac{W}{m\cdot K}$).


## Computational cost

Using the code with the `--wigner` option has a negligible effect on the duration of the calculation.
