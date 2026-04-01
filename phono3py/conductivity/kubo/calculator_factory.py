"""Factory function for the Green-Kubo RTA calculator.

This function is registered as a built-in entry in factory._REGISTRY and
serves as a reference implementation for the plugin API.  External code can
override it by calling register_calculator() with the same method name.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.kubo.heat_capacity_providers import (
    HeatCapacityMatrixProvider,
)
from phono3py.conductivity.kubo.kappa_accumulators import KuboKappaAccumulator
from phono3py.conductivity.kubo.kappa_formulas import KuboKappaFormula
from phono3py.conductivity.kubo.velocity_providers import VelocityMatrixProvider
from phono3py.conductivity.rta_calculator import ConductivityCalculator
from phono3py.conductivity.scattering_providers import RTAScatteringProvider
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
) -> ConductivityCalculator:
    """Build a ConductivityCalculator for the Green-Kubo RTA method.

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
    ConductivityCalculator

    """
    _sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
    _temperatures: NDArray[np.double] | None = (
        np.asarray(temperatures, dtype="double") if temperatures is not None else None
    )

    if is_kappa_star:
        point_ops = interaction.bz_grid.reciprocal_operations
    else:
        point_ops = np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)

    cv_provider = HeatCapacityMatrixProvider(interaction)

    velocity_provider = VelocityMatrixProvider(
        interaction,
        point_operations=point_ops,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    scattering_provider = RTAScatteringProvider(
        interaction,
        sigmas=_sigmas,
        temperatures=(
            _temperatures
            if _temperatures is not None
            else np.arange(0, 1001, 10, dtype="double")
        ),
        sigma_cutoff=sigma_cutoff,
        is_full_pp=is_full_pp,
        use_ave_pp=use_ave_pp,
        read_pp=read_pp,
        store_pp=store_pp,
        pp_filename=pp_filename,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume
    kappa_formula = KuboKappaFormula(
        cutoff_frequency=interaction.cutoff_frequency,
        conversion_factor=conversion_factor,
    )
    accumulator = KuboKappaAccumulator(kappa_formula)

    return ConductivityCalculator(
        interaction,
        velocity_provider=velocity_provider,
        cv_provider=cv_provider,
        scattering_provider=scattering_provider,
        accumulator=accumulator,
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=_sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        is_kappa_star=is_kappa_star,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )
