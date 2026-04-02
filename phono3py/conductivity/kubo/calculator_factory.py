"""Factory functions for the Green-Kubo calculators.

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
from phono3py.conductivity.kubo.heat_capacity_providers import (
    HeatCapacityMatrixProvider,
)
from phono3py.conductivity.kubo.kappa_accumulators import (
    KuboLBTEKappaAccumulator,
    KuboRTAKappaAccumulator,
)
from phono3py.conductivity.kubo.velocity_providers import VelocityMatrixProvider
from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.rta_calculator import RTACalculator
from phono3py.conductivity.utils import get_unit_to_WmK
from phono3py.phonon3.interaction import Interaction


def make_kubo_rta_calculator(
    interaction: Interaction,
    *,
    grid_points: Sequence[int] | NDArray[np.int64] | None = None,
    temperatures: Sequence[float] | NDArray[np.double] | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    is_isotope: bool = False,
    mass_variances: Sequence[float] | NDArray[np.double] | None = None,
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
    """Build a RTACalculator for the Green-Kubo RTA method.

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
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        temperatures=temperatures,
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

    cv_provider = HeatCapacityMatrixProvider(interaction)

    velocity_provider = VelocityMatrixProvider(
        interaction,
        point_operations=base.point_ops,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume
    accumulator = KuboRTAKappaAccumulator(
        cutoff_frequency=interaction.cutoff_frequency,
        conversion_factor=conversion_factor,
        temperatures=temperatures,
        sigmas=base.sigmas,
        log_level=log_level,
    )

    return RTACalculator(
        interaction,
        velocity_provider=velocity_provider,  # type: ignore[arg-type]
        cv_provider=cv_provider,  # type: ignore[arg-type]
        scattering_provider=base.scattering_provider,
        accumulator=accumulator,  # type: ignore[arg-type]
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=base.sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        is_kappa_star=is_kappa_star,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )


def make_kubo_lbte_calculator(
    interaction: Interaction,
    *,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    temperatures: Sequence[float] | NDArray[np.double] | None = None,
    is_isotope: bool = False,
    mass_variances: Sequence[float] | NDArray[np.double] | None = None,
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
    """Build an LBTECalculator with a KuboLBTEKappaAccumulator.

    Implements the Green-Kubo formula via direct LBTE solution. The intra-band
    (diagonal) kappa comes from the standard LBTE solve; the inter-band
    (off-diagonal) kappa uses the Kubo formula with LBTE linewidths.

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
        Configured with a KuboLBTEKappaAccumulator that computes both the
        intra-band kappa (via standard LBTE) and the inter-band kappa
        (via Kubo formula with LBTE linewidths).

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

    velocity_provider = VelocityMatrixProvider(
        interaction,
        point_operations=base.point_ops,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    cv_provider = HeatCapacityMatrixProvider(interaction)

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume
    kubo_accumulator = KuboLBTEKappaAccumulator(
        inner=base.accumulator,
        ir_grid_points=base.ir_grid_points,
        frequencies=base.frequencies,
        cutoff_frequency=interaction.cutoff_frequency,
        conversion_factor=conversion_factor,
        sigmas=base.sigmas,
        log_level=log_level,
    )

    return LBTECalculator(
        interaction,
        velocity_provider=velocity_provider,  # type: ignore[arg-type]
        cv_provider=cv_provider,  # type: ignore[arg-type]
        collision_provider=base.collision_provider,
        accumulator=kubo_accumulator,  # type: ignore[arg-type]
        temperatures=base.temperatures,
        sigmas=base.sigmas,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        is_full_pp=is_full_pp,
        log_level=log_level,
    )
