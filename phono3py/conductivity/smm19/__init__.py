"""SMM19 plugin for phono3py conductivity.

Importing this package registers the ``"SMM19-rta"`` and ``"SMM19-lbte"``
methods with the conductivity factory so that
``conductivity_calculator("SMM19-rta", ...)`` and
``conductivity_calculator("SMM19-lbte", ...)`` work out of the box.

This variant implements the Wigner / unified-theory coherence term of
G. Simoncelli, N. Marzari, and F. Mauri, Nat. Phys. 15, 809 (2019). It
shares the inter-band mode-kappa kernel and velocity matrix with NJC23 and
differs only in the heat capacity matrix (the effective
``mode_cv_matrix_smm19`` built from scalar mode heat capacities).

"""

from phono3py.conductivity.interband_variant import register_interband_variant
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix_smm19

register_interband_variant("SMM19", mode_cv_matrix_smm19)
