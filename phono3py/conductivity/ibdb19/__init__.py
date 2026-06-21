"""IBDB19 (quasi-harmonic Green-Kubo) plugin for phono3py conductivity.

Importing this package registers the ``"IBDB19-rta"`` and ``"IBDB19-lbte"``
methods with the conductivity factory so that
``conductivity_calculator("IBDB19-rta", ...)`` and
``conductivity_calculator("IBDB19-lbte", ...)`` work out of the box.

This variant implements Eq. (9) of L. Isaeva, G. Barbalinardo, D. Donadio,
and S. Baroni, Nat. Commun. 10, 3853 (2019). It shares the inter-band
mode-kappa kernel and velocity matrix with NJC23 and differs only in the
heat capacity matrix (``mode_cv_matrix_ibdb19`` instead of
``mode_cv_matrix_njc23``).

"""

from phono3py.conductivity.interband_variant import register_interband_variant
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix_ibdb19

register_interband_variant("IBDB19", mode_cv_matrix_ibdb19)
