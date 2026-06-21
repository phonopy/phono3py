"""NJC23 (Green-Kubo) plugin for phono3py conductivity.

Importing this package registers the ``"NJC23-rta"`` and ``"NJC23-lbte"``
methods with the conductivity factory so that
``conductivity_calculator("NJC23-rta", ...)`` and
``conductivity_calculator("NJC23-lbte", ...)`` work out of the box.

"""

from phono3py.conductivity.interband_variant import register_interband_variant
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix_njc23

register_interband_variant("NJC23", mode_cv_matrix_njc23)
