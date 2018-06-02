#!/usr/bin/env python

import numpy as np
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_BORN
from phonopy.units import Bohr, Hartree
from phonopy.harmonic.force_constants import show_drift_force_constants
from phono3py.phonon3.fc3 import show_drift_fc3
from phono3py.phonon3 import Phono3py
from phono3py.file_IO import (parse_disp_fc3_yaml,
                              parse_disp_fc2_yaml,
                              parse_FORCES_FC2,
                              parse_FORCES_FC3,
                              read_fc3_from_hdf5,
                              read_fc2_from_hdf5)

def run_thermal_conductivity(phono3py):
    # Create fc3 and fc2 from disp_fc3.yaml and FORCES_FC3
    disp_dataset = parse_disp_fc3_yaml(filename="disp_fc3.yaml")
    forces_fc3 = parse_FORCES_FC3(disp_dataset, filename="FORCES_FC3")
    phono3py.produce_fc3(forces_fc3,
                         displacement_dataset=disp_dataset,
                         symmetrize_fc3r=True)
    fc3 = phono3py.get_fc3()
    fc2 = phono3py.get_fc2()

    # # Create fc2 from disp_fc2.yaml and FORCES_FC2
    # disp_dataset2 = parse_disp_fc2_yaml(filename="disp_fc2.yaml")
    # forces_fc2 = parse_FORCES_FC2(disp_dataset2, filename="FORCES_FC2")
    # phono3py.produce_fc2(
    #     forces_fc2,
    #     displacement_dataset=disp_dataset2,
    #     is_translational_symmetry=True,
    #     is_permutation_symmetry=True)

    # # Read fc3 and fc2 from c3.hdf5 and fc2.hdf5
    # fc3 = read_fc3_from_hdf5(filename="fc3.hdf5")
    # fc2 = read_fc2_from_hdf5(filename="fc2.hdf5")
    # phono3py.set_fc3(fc3)
    # phono3py.set_fc2(fc2)

    show_drift_fc3(fc3)
    show_drift_force_constants(fc2, name='fc2')

    # # For special cases like NAC
    # primitive = phono3py.get_phonon_primitive()
    # nac_params = parse_BORN(primitive, filename="BORN")
    # nac_params['factor'] = Hartree * Bohr
    # phono3py.set_phph_interaction(nac_params=nac_params)

    phono3py.run_thermal_conductivity(
        temperatures=range(0, 1001, 10),
        boundary_mfp=1e6, # This is to avoid divergence of phonon life time.
        write_kappa=True)

    # Conductivity_RTA object (https://git.io/vVRUW)
    cond_rta = phono3py.get_thermal_conductivity()

def create_supercells_with_displacements(phono3py):
    phono3py.generate_displacements(distance=0.03)
    scells_with_disps = phono3py.get_supercells_with_displacements()

    # from phonopy.interface.vasp import write_vasp
    # for i, scell in enumerate(scells_with_disps):
    #     write_vasp("POSCAR-%05d" % (i + 1), scell)
    #     # print(scell)

    # A dataset of displacements. The dictionary format is shown at
    # phono3py.phonon3.displacement_fc3.get_third_order_displacements.
    print("Displacement sets")
    disp_dataset = phono3py.get_displacement_dataset()
    count = 0
    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        print("%4d: %4d                %s" % (
            count + 1,
            disp1['number'] + 1,
            np.around(disp1['displacement'], decimals=3)))
        count += 1

    distances = []
    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        for j, disp2 in enumerate(disp1['second_atoms']):
            print("%4d: %4d-%4d (%6.3f)  %s %s" % (
                count + 1,
                disp1['number'] + 1,
                disp2['number'] + 1,
                disp2['pair_distance'],
                np.around(disp1['displacement'], decimals=3),
                np.around(disp2['displacement'], decimals=3)))
            distances.append(disp2['pair_distance'])
            count += 1

    # Find unique pair distances
    distances = np.array(distances)
    distances_int = (distances * 1e5).astype(int)
    unique_distances = np.unique(distances_int) * 1e-5 # up to 5 decimals
    print("Unique pair distances")
    print(unique_distances)

    # FORCES_FC3 is created as follows:
    from phono3py.file_IO import write_FORCES_FC3
    # force_sets is a simple force array:
    #   [len(scells_with_disps), num_supercell_atoms, 3].
    force_sets = parse_FORCES_FC3(disp_dataset, filename="FORCES_FC3")
    write_FORCES_FC3(disp_dataset, force_sets, filename="FORCES_FC3_new")

if __name__ == '__main__':
    cell = read_vasp("POSCAR-unitcell")
    mesh = [11, 11, 11]
    phono3py = Phono3py(cell,
                        np.diag([2, 2, 2]),
                        primitive_matrix=[[0, 0.5, 0.5],
                                          [0.5, 0, 0.5],
                                          [0.5, 0.5, 0]],
                        mesh=mesh,
                        log_level=1) # log_level=0 make phono3py quiet

    create_supercells_with_displacements(phono3py)
    run_thermal_conductivity(phono3py)
