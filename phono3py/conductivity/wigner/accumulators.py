"""Kappa accumulator for the Wigner transport equation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointResult
from phono3py.conductivity.wigner.formulas import WignerKappaFormula


class WignerKappaAccumulator:
    """Kappa accumulator for the Wigner transport equation (WTE).

    Decomposes kappa into a population term (kappa_P_RTA) and a coherence
    term (kappa_C).  The total kappa is kappa_TOT_RTA = kappa_P_RTA + kappa_C.

    The P arrays are pre-allocated in ``prepare()``.  The C arrays are
    allocated lazily on the first grid point because their shape depends on
    ``num_band`` (nat3), which is not known until the first formula call.

    Parameters
    ----------
    formula : WignerKappaFormula
        Formula instance used to compute mode kappa at each grid point.

    """

    def __init__(self, formula: WignerKappaFormula) -> None:
        """Init method."""
        self._formula = formula
        self._kappa_P: NDArray[np.double]
        self._mode_kappa_P: NDArray[np.double]
        self._kappa_C: NDArray[np.double] | None = None
        self._mode_kappa_C: NDArray[np.double] | None = None

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate population-term arrays; coherence arrays are lazy."""
        self._kappa_P = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa_P = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6), dtype="double", order="C"
        )

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and accumulate population and coherence kappa at ``i_gp``."""
        mode_kappa_P = self._formula.compute(
            result
        )  # sets result.extra["wigner_mode_kappa_C"]
        self._mode_kappa_P[:, :, i_gp, :, :] = mode_kappa_P
        self._kappa_P += np.sum(mode_kappa_P, axis=2)

        mode_kappa_C = result.extra.get("wigner_mode_kappa_C")
        if mode_kappa_C is not None:
            if self._mode_kappa_C is None:
                # Lazy allocation: shape depends on num_band (nat3).
                ns, nt, nb0, nb, _ = mode_kappa_C.shape
                num_gp = self._mode_kappa_P.shape[2]
                self._mode_kappa_C = np.zeros(
                    (ns, nt, num_gp, nb0, nb, 6), dtype="double", order="C"
                )
                self._kappa_C = np.zeros((ns, nt, 6), dtype="double", order="C")
            self._mode_kappa_C[:, :, i_gp, :, :, :] = mode_kappa_C
            assert self._kappa_C is not None
            self._kappa_C += np.sum(mode_kappa_C, axis=(2, 3))

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise both population and coherence terms."""
        if num_sampling_grid_points > 0:
            self._kappa_P /= num_sampling_grid_points
            if self._kappa_C is not None:
                self._kappa_C /= num_sampling_grid_points

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total Wigner kappa (kappa_P + kappa_C)."""
        if self._kappa_C is not None:
            return self._kappa_P + self._kappa_C
        return self._kappa_P

    @property
    def kappa_TOT_RTA(self) -> NDArray[np.double]:
        """Return total Wigner kappa (same as ``kappa``)."""
        return self.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double]:
        """Return population kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_P

    @property
    def kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence kappa, shape (num_sigma, num_temp, 6).

        None if not computed.

        """
        return self._kappa_C

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return population mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._mode_kappa_P

    @property
    def mode_kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence mode kappa; None if not computed.

        Shape: (num_sigma, num_temp, num_gp, num_band0, num_band, 6).

        """
        return self._mode_kappa_C

    def show_rta_progress(self, br: object, log_level: int) -> None:
        """Print K_P, K_C, K_T rows for the Wigner-RTA conductivity.

        Called via duck typing from ShowCalcProgress.kappa_RTA so that
        all Wigner-specific display logic stays in this subpackage.

        Parameters
        ----------
        br :
            ConductivityCalculator instance (typed as object to avoid
            circular imports).
        log_level :
            Verbosity level.

        """

        def _req(v: object, name: str) -> "NDArray[np.double]":
            assert v is not None, f"{name} must not be None"
            return v  # type: ignore[return-value]

        temperatures = _req(getattr(br, "temperatures", None), "temperatures")
        sigmas = br.sigmas
        kappa_tot = _req(getattr(br, "kappa", None), "kappa")
        num_ignored = _req(
            getattr(br, "number_of_ignored_phonon_modes", None),
            "number_of_ignored_phonon_modes",
        )
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.number_of_sampling_grid_points * num_band

        kappa_P_RTA = _req(self._kappa_P, "kappa_P")
        kappa_C = _req(self._kappa_C, "kappa_C")

        for i, sigma in enumerate(sigmas):
            kappa_p_i = kappa_P_RTA[i]
            kappa_c_i = kappa_C[i]
            kappa_tot_i = kappa_tot[i]
            text = "----------- Thermal conductivity (W/m-k) "
            if sigma:
                text += "for sigma=%s -----------" % sigma
            else:
                text += "with tetrahedron method -----------"
            print(text)
            if log_level > 1:
                print(
                    ("#%6s       " + " %-10s" * 6 + "#ipm")
                    % ("      \t   T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                for j, (t, k) in enumerate(zip(temperatures, kappa_p_i, strict=True)):
                    print(
                        "K_P\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % ((t,) + tuple(k) + (num_ignored[i, j], num_phonon_modes))
                    )
                print(" ")
                for j, (t, k) in enumerate(zip(temperatures, kappa_c_i, strict=True)):
                    print(
                        "K_C\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % ((t,) + tuple(k) + (num_ignored[i, j], num_phonon_modes))
                    )
                print(" ")
                for j, (t, k) in enumerate(zip(temperatures, kappa_tot_i, strict=True)):
                    print(
                        "K_T\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % ((t,) + tuple(k) + (num_ignored[i, j], num_phonon_modes))
                    )
            else:
                print(
                    ("#%6s       " + " %-10s" * 6)
                    % ("      \t   T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                for t, k in zip(temperatures, kappa_p_i, strict=True):
                    print("K_P\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_c_i, strict=True):
                    print("K_C\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_tot_i, strict=True):
                    print("K_T\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("", flush=True)

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return Wigner-specific kappa arrays keyed by HDF5 dataset name.

        Each value has sigma as its first axis so that callers can slice by
        sigma index with ``value[i_sigma]``.

        Keys
        ----
        kappa_TOT_RTA : (num_sigma, num_temp, 6)
        kappa_P_RTA   : (num_sigma, num_temp, 6)
        kappa_C       : (num_sigma, num_temp, 6) or None
        mode_kappa_P_RTA : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_C  : (num_sigma, num_temp, num_gp, num_band0, num_band, 6) or None

        """
        return {
            "kappa_TOT_RTA": self.kappa_TOT_RTA,
            "kappa_P_RTA": self._kappa_P,
            "kappa_C": self._kappa_C,
            "mode_kappa_P_RTA": self._mode_kappa_P,
            "mode_kappa_C": self._mode_kappa_C,
        }
