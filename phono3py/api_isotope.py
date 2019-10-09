# Copyright (C) 2019 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from phonopy.units import VaspToTHz
from phono3py.other.isotope import Isotope


class Phono3pyIsotope(object):
    def __init__(self,
                 mesh,
                 primitive,
                 mass_variances=None,  # length of list is num_atom.
                 band_indices=None,
                 sigmas=None,
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        if sigmas is None:
            self._sigmas = [None]
        else:
            self._sigmas = sigmas
        self._mesh_numbers = mesh
        self._iso = Isotope(mesh,
                            primitive,
                            mass_variances=mass_variances,
                            band_indices=band_indices,
                            frequency_factor_to_THz=frequency_factor_to_THz,
                            symprec=symprec,
                            cutoff_frequency=cutoff_frequency,
                            lapack_zheev_uplo=lapack_zheev_uplo)

    def run(self, grid_points):
        for gp in grid_points:
            self._iso.set_grid_point(gp)

            print("--------------- Isotope scattering ---------------")
            print("Grid point: %d" % gp)
            adrs = self._iso.get_grid_address()[gp]
            q = adrs.astype('double') / self._mesh_numbers
            print("q-point: %s" % q)

            if self._sigmas:
                for sigma in self._sigmas:
                    if sigma is None:
                        print("Tetrahedron method")
                    else:
                        print("Sigma: %s" % sigma)
                    self._iso.set_sigma(sigma)
                    self._iso.run()

                    frequencies = self._iso.get_phonons()[0]
                    print('')
                    print("Phonon-isotope scattering rate in THz (1/4pi-tau)")
                    print(" Frequency     Rate")
                    for g, f in zip(self._iso.get_gamma(), frequencies[gp]):
                        print("%8.3f     %5.3e" % (f, g))
            else:
                print("sigma or tetrahedron method has to be set.")

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._primitive = primitive
        self._iso.set_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals)

    def set_sigma(self, sigma):
        self._iso.set_sigma(sigma)
