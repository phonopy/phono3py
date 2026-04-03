"""Factory functions for Wigner transport equation calculators.

These functions are registered as built-in entries in factory._REGISTRY and
serve as reference implementations for the plugin API.  External code can
override them by calling register_calculator() with the same method name.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.calculator_factory import (
    build_lbte_base_components,
    build_rta_base_components,
)
from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.rta_calculator import RTACalculator
from phono3py.conductivity.wigner.kappa_accumulators import (
    WignerLBTEKappaAccumulator,
    WignerRTAKappaAccumulator,
)
from phono3py.conductivity.wigner.kappa_formulas import get_conversion_factor_WTE
from phono3py.conductivity.wigner.velocity_providers import VelocityOperatorProvider
from phono3py.phonon3.interaction import Interaction


def make_wigner_rta_calculator(
    interaction: Interaction,
    *,
    grid_points: NDArray[np.int64] | None = None,
    temperatures: NDArray[np.double] | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    is_isotope: bool = False,
    mass_variances: NDArray[np.double] | None = None,
    boundary_mfp: float | None = None,
    use_ave_pp: bool = False,
    is_kappa_star: bool = True,
    gv_delta_q: float | None = None,
    is_full_pp: bool = False,
    read_pp: bool = False,
    store_pp: bool = False,
    pp_filename: str | os.PathLike | None = None,
    is_N_U: bool = False,
    is_gamma_detail: bool = False,
    log_level: int = 0,
    **_ignored: Any,
) -> RTACalculator:
    """Build a RTACalculator for the Wigner-RTA method.

    Implements the Wigner transport equation in the RTA, adding the
    off-diagonal coherence contribution (C-term) to the standard
    population contribution (P-term).

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    grid_points : array-like or None, optional
        BZ grid point indices. None uses all irreducible grid points.
    temperatures : array-like or None, optional
        Temperatures in Kelvin.
    sigmas : sequence or None, optional
        Smearing widths. None selects the tetrahedron method.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma.
    is_isotope : bool, optional
        Include isotope scattering.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.
    use_ave_pp : bool, optional
        Use pre-averaged ph-ph interaction.
    is_kappa_star : bool, optional
        Use k-star symmetry.
    gv_delta_q : float or None, optional
        Finite-difference step for group velocity (NAC).
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix.
    read_pp : bool, optional
        Read ph-ph interaction from file.
    store_pp : bool, optional
        Store ph-ph interaction to file.
    pp_filename : str or path or None, optional
        Filename for ph-ph interaction I/O.
    is_N_U : bool, optional
        Decompose gamma into Normal and Umklapp.
    is_gamma_detail : bool, optional
        Store per-triplet gamma.
    log_level : int, optional
        Verbosity.
    **_ignored
        Absorbs LBTE-only keyword arguments passed by the dispatcher.

    Returns
    -------
    RTACalculator

    """
    base = build_rta_base_components(
        interaction,
        grid_points=grid_points,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        temperatures=temperatures,
        boundary_mfp=boundary_mfp,
        is_kappa_star=is_kappa_star,
        is_full_pp=is_full_pp,
        use_ave_pp=use_ave_pp,
        read_pp=read_pp,
        store_pp=store_pp,
        pp_filename=pp_filename,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )

    velocity_provider = VelocityOperatorProvider(
        interaction,
        point_operations=base.point_ops,
        rotations_cartesian=base.rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    cv_provider = ModeHeatCapacityProvider(interaction)

    accumulator = WignerRTAKappaAccumulator(
        context=base.context,
        conversion_factor_WTE=get_conversion_factor_WTE(interaction.primitive.volume),
        log_level=log_level,
    )

    return RTACalculator(
        interaction,
        velocity_provider=velocity_provider,  # type: ignore[arg-type]
        cv_provider=cv_provider,
        scattering_provider=base.scattering_provider,
        accumulator=accumulator,  # type: ignore[arg-type]
        context=base.context,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )


def make_wigner_lbte_calculator(
    interaction: Interaction,
    *,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    temperatures: NDArray[np.double] | None = None,
    is_isotope: bool = False,
    mass_variances: NDArray[np.double] | None = None,
    boundary_mfp: float | None = None,
    is_kappa_star: bool = True,
    is_reducible_collision_matrix: bool = False,
    solve_collective_phonon: bool = False,
    is_full_pp: bool = False,
    read_pp: bool = False,
    pp_filename: str | os.PathLike | None = None,
    pinv_cutoff: float = 1.0e-8,
    pinv_solver: int = 0,
    pinv_method: int = 0,
    lang: Literal["C", "Python"] = "C",
    gv_delta_q: float | None = None,
    log_level: int = 0,
    **_ignored: Any,
) -> LBTECalculator:
    """Build an LBTECalculator with a WignerLBTEKappaAccumulator.

    Implements the Wigner transport equation via direct LBTE solution,
    adding the coherence (C) term on top of the LBTE population (P) terms.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    sigmas : sequence or None, optional
        Smearing widths. None selects the tetrahedron method.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma.
    temperatures : array-like or None, optional
        Temperatures in Kelvin. Default [300.0].
    is_isotope : bool, optional
        Include isotope scattering.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.
    is_kappa_star : bool, optional
        Use k-star symmetry.
    is_reducible_collision_matrix : bool, optional
        Use the full reducible collision matrix.
    solve_collective_phonon : bool, optional
        Use Chaput collective-phonon method.
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix.
    read_pp : bool, optional
        Read ph-ph interaction from file.
    pp_filename : str or path or None, optional
        Filename for ph-ph interaction I/O.
    pinv_cutoff : float, optional
        Eigenvalue cutoff for pseudo-inversion.
    pinv_solver : int, optional
        Solver selection index.
    pinv_method : int, optional
        Pseudo-inverse criterion.
    lang : {"C", "Python"}, optional
        Backend for C-extension operations.
    gv_delta_q : float or None, optional
        Finite-difference step for group velocity (NAC).
    log_level : int, optional
        Verbosity.
    **_ignored
        Absorbs RTA-only keyword arguments passed by the dispatcher.

    Returns
    -------
    LBTECalculator
        Configured with a WignerLBTEKappaAccumulator that computes both the P-term
        (via standard LBTE) and the C-term (Wigner coherence).

    """
    base = build_lbte_base_components(
        interaction,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        temperatures=temperatures,
        is_kappa_star=is_kappa_star,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        solve_collective_phonon=solve_collective_phonon,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        pp_filename=pp_filename,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        lang=lang,
        boundary_mfp=boundary_mfp,
        log_level=log_level,
    )

    velocity_provider = VelocityOperatorProvider(
        interaction,
        point_operations=base.point_ops,
        rotations_cartesian=base.rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    cv_provider = ModeHeatCapacityProvider(interaction)

    wigner_accumulator = WignerLBTEKappaAccumulator(
        solver=base.solver,
        context=base.context,
        conversion_factor_WTE=get_conversion_factor_WTE(interaction.primitive.volume),
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        log_level=log_level,
    )

    return LBTECalculator(
        interaction,
        velocity_provider=velocity_provider,  # type: ignore[arg-type]
        cv_provider=cv_provider,
        collision_provider=base.collision_provider,
        accumulator=wigner_accumulator,  # type: ignore[arg-type]
        context=base.context,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        is_full_pp=is_full_pp,
        log_level=log_level,
    )
