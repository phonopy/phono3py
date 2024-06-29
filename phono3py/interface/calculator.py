"""Utilities of calculator interfaces."""

# Copyright (C) 2020 Atsushi Togo
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

calculator_info = {
    "abinit": {"option": {"name": "--abinit", "help": "Invoke Abinit mode"}},
    # 'aims': {'option': {'name': "--aims",
    #                     'help': "Invoke FHI-aims mode"}},
    # 'cp2k': {'option': {'name': "--cp2k",
    #                     'help': "Invoke CP2K mode"}},
    "crystal": {"option": {"name": "--crystal", "help": "Invoke CRYSTAL mode"}},
    # 'dftbp': {'option': {'name': "--dftb+",
    #                      'help': "Invoke dftb+ mode"}},
    # 'elk': {'option': {'name': "--elk",
    #                    'help': "Invoke elk mode"}},
    "qe": {"option": {"name": "--qe", "help": "Invoke Quantum espresso (QE) mode"}},
    # 'siesta': {'option': {'name': "--siesta",
    #                       'help': "Invoke Siesta mode"}},
    "turbomole": {"option": {"name": "--turbomole", "help": "Invoke TURBOMOLE mode"}},
    "vasp": {"option": {"name": "--vasp", "help": "Invoke Vasp mode"}},
    # 'wien2k': {'option': {'name': "--wien2k",
    #                       'help': "Invoke Wien2k mode"}},
}


def get_default_displacement_distance(interface_mode):
    """Return default displacement distances for calculators."""
    if interface_mode in ("qe", "abinit", "turbomole"):
        displacement_distance = 0.06
    elif interface_mode == "crystal":
        displacement_distance = 0.03
    else:
        displacement_distance = 0.03
    return displacement_distance


def get_additional_info_to_write_supercells(interface_mode, supercell_matrix):
    """Return additional information to write supercells for calculators."""
    additional_info = {}
    if interface_mode == "crystal":
        additional_info["template_file"] = "TEMPLATE3"
        additional_info["supercell_matrix"] = supercell_matrix
    return additional_info


def get_additional_info_to_write_fc2_supercells(
    interface_mode, phonon_supercell_matrix, suffix: str = "fc2"
):
    """Return additional information to write fc2-supercells for calculators."""
    additional_info = {}
    if interface_mode == "qe":
        additional_info["pre_filename"] = "supercell_%s" % suffix
    elif interface_mode == "crystal":
        additional_info["template_file"] = "TEMPLATE"
        additional_info["pre_filename"] = "supercell_%s" % suffix
        additional_info["supercell_matrix"] = phonon_supercell_matrix
    elif interface_mode == "abinit":
        additional_info["pre_filename"] = "supercell_%s" % suffix
    elif interface_mode == "turbomole":
        additional_info["pre_filename"] = "supercell_%s" % suffix
    else:
        additional_info["pre_filename"] = "POSCAR_%s" % suffix.upper()
    return additional_info
