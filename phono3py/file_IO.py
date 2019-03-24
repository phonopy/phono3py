import os
import numpy as np
import h5py

from phonopy.file_IO import (write_force_constants_to_hdf5,
                             read_force_constants_hdf5,
                             check_force_constants_indices,
                             get_cell_from_disp_yaml)


def write_cell_yaml(w, supercell):
    w.write("lattice:\n")
    for axis in supercell.get_cell():
        w.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_scaled_positions()
    w.write("atoms:\n")
    for i, (s, v) in enumerate(zip(symbols, positions)):
        w.write("- symbol: %-2s # %d\n" % (s, i+1))
        w.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % tuple(v))


def write_disp_fc3_yaml(dataset, supercell, filename='disp_fc3.yaml'):
    w = open(filename, 'w')
    w.write("natom: %d\n" % dataset['natom'])

    num_first = len(dataset['first_atoms'])
    w.write("num_first_displacements: %d\n" % num_first)
    if 'cutoff_distance' in dataset:
        w.write("cutoff_distance: %f\n" % dataset['cutoff_distance'])

    num_second = 0
    num_disp_files = 0
    for d1 in dataset['first_atoms']:
        num_disp_files += 1
        num_second += len(d1['second_atoms'])
        for d2 in d1['second_atoms']:
            if 'included' in d2:
                if d2['included']:
                    num_disp_files += 1
            else:
                num_disp_files += 1

    w.write("num_second_displacements: %d\n" % num_second)
    w.write("num_displacements_created: %d\n" % num_disp_files)

    w.write("first_atoms:\n")
    count1 = 1
    count2 = num_first + 1
    for disp1 in dataset['first_atoms']:
        disp_cart1 = disp1['displacement']
        w.write("- number: %5d\n" % (disp1['number'] + 1))
        w.write("  displacement:\n")
        w.write("    [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                (disp_cart1[0], disp_cart1[1], disp_cart1[2], count1))
        w.write("  second_atoms:\n")
        count1 += 1

        included = None
        atom2 = -1
        for disp2 in disp1['second_atoms']:
            if atom2 != disp2['number']:
                atom2 = disp2['number']
                if 'included' in disp2:
                    included = disp2['included']
                pair_distance = disp2['pair_distance']
                w.write("  - number: %5d\n" % (atom2 + 1))
                w.write("    distance: %f\n" % pair_distance)
                if included is not None:
                    if included:
                        w.write("    included: %s\n" % "true")
                    else:
                        w.write("    included: %s\n" % "false")
                w.write("    displacements:\n")

            disp_cart2 = disp2['displacement']
            w.write("    - [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                    (disp_cart2[0], disp_cart2[1], disp_cart2[2], count2))
            count2 += 1

    write_cell_yaml(w, supercell)

    w.close()

    return num_first + num_second, num_disp_files


def write_disp_fc2_yaml(dataset, supercell, filename='disp_fc2.yaml'):
    w = open(filename, 'w')
    w.write("natom: %d\n" % dataset['natom'])

    num_first = len(dataset['first_atoms'])
    w.write("num_first_displacements: %d\n" % num_first)
    w.write("first_atoms:\n")
    for i, disp1 in enumerate(dataset['first_atoms']):
        disp_cart1 = disp1['displacement']
        w.write("- number: %5d\n" % (disp1['number'] + 1))
        w.write("  displacement:\n")
        w.write("    [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                (disp_cart1[0], disp_cart1[1], disp_cart1[2], i + 1))

    if supercell is not None:
        write_cell_yaml(w, supercell)

    w.close()

    return num_first


def write_FORCES_FC2(disp_dataset,
                     forces_fc2=None,
                     fp=None,
                     filename="FORCES_FC2"):
    if fp is None:
        w = open(filename, 'w')
    else:
        w = fp

    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        w.write("# File: %-5d\n" % (i + 1))
        w.write("# %-5d " % (disp1['number'] + 1))
        w.write("%20.16f %20.16f %20.16f\n" % tuple(disp1['displacement']))
        if forces_fc2 is None:
            force_set = disp1['forces']
        else:
            force_set = forces_fc2[i]
        for forces in force_set:
            w.write("%15.10f %15.10f %15.10f\n" % tuple(forces))


def write_FORCES_FC3(disp_dataset, forces_fc3, fp=None, filename="FORCES_FC3"):
    if fp is None:
        w = open(filename, 'w')
    else:
        w = fp

    natom = disp_dataset['natom']
    num_disp1 = len(disp_dataset['first_atoms'])
    count = num_disp1
    file_count = num_disp1

    write_FORCES_FC2(disp_dataset, forces_fc2=forces_fc3, fp=w)

    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        atom1 = disp1['number']
        for disp2 in disp1['second_atoms']:
            atom2 = disp2['number']
            w.write("# File: %-5d\n" % (count + 1))
            w.write("# %-5d " % (atom1 + 1))
            w.write("%20.16f %20.16f %20.16f\n" % tuple(disp1['displacement']))
            w.write("# %-5d " % (atom2 + 1))
            w.write("%20.16f %20.16f %20.16f\n" % tuple(disp2['displacement']))

            # For supercell calculation reduction
            included = True
            if 'included' in disp2:
                included = disp2['included']
            if included:
                for forces in forces_fc3[file_count]:
                    w.write("%15.10f %15.10f %15.10f\n" % tuple(forces))
                file_count += 1
            else:
                # for forces in forces_fc3[i]:
                #     w.write("%15.10f %15.10f %15.10f\n" % (tuple(forces)))
                for j in range(natom):
                    w.write("%15.10f %15.10f %15.10f\n" % (0, 0, 0))
            count += 1


def write_fc3_dat(force_constants_third, filename='fc3.dat'):
    w = open(filename, 'w')
    for i in range(force_constants_third.shape[0]):
        for j in range(force_constants_third.shape[1]):
            for k in range(force_constants_third.shape[2]):
                tensor3 = force_constants_third[i, j, k]
                w.write(" %d - %d - %d  (%f)\n" % (i + 1, j + 1, k + 1,
                                                   np.abs(tensor3).sum()))
                for tensor2 in tensor3:
                    for vec in tensor2:
                        w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
                    w.write("\n")


def write_fc3_to_hdf5(fc3,
                      filename='fc3.hdf5',
                      p2s_map=None,
                      compression=None):
    """Write third-order force constants in hdf5 format.

    Parameters
    ----------
    force_constants : ndarray
        Force constants
        shape=(n_satom, n_satom, n_satom, 3, 3, 3) or
        (n_patom, n_satom, n_satom,3,3,3), dtype=double
    filename : str
        Filename to be used.
    p2s_map : ndarray, optional
        Primitive atom indices in supercell index system
        shape=(n_patom,), dtype=intc
    compression : str or int, optional
        h5py's lossless compression filters (e.g., "gzip", "lzf").
        See the detail at docstring of h5py.Group.create_dataset. Default is
        None.

    """

    with h5py.File(filename, 'w') as w:
        w.create_dataset('fc3', data=fc3, compression=compression)
        if p2s_map is not None:
            w.create_dataset('p2s_map', data=p2s_map)


def read_fc3_from_hdf5(filename='fc3.hdf5', p2s_map=None):
    with h5py.File(filename, 'r') as f:
        fc3 = f['fc3'][:]
        if 'p2s_map' in f:
            p2s_map_in_file = f['p2s_map'][:]
            check_force_constants_indices(fc3.shape[:2],
                                          p2s_map_in_file,
                                          p2s_map,
                                          filename)
        if fc3.dtype == np.double and fc3.flags.c_contiguous:
            return fc3
        else:
            msg = ("%s has to be read by h5py as numpy ndarray of "
                   "dtype='double' and c_contiguous." % filename)
            raise TypeError(msg)
    return None


def write_fc2_dat(force_constants, filename='fc2.dat'):
    w = open(filename, 'w')
    for i, fcs in enumerate(force_constants):
        for j, fcb in enumerate(fcs):
            w.write(" %d - %d\n" % (i+1, j+1))
            for vec in fcb:
                w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
            w.write("\n")


def write_fc2_to_hdf5(force_constants,
                      filename='fc2.hdf5',
                      p2s_map=None,
                      compression=None):
    try:
        write_force_constants_to_hdf5(force_constants,
                                      filename=filename,
                                      p2s_map=p2s_map,
                                      compression=compression)
    except TypeError:
        # This fills the gap between versions with/without compression
        # in phonopy.
        write_force_constants_to_hdf5(force_constants,
                                      filename=filename,
                                      p2s_map=p2s_map)


def read_fc2_from_hdf5(filename='fc2.hdf5',
                       p2s_map=None):
    return read_force_constants_hdf5(filename=filename,
                                     p2s_map=p2s_map)


def write_triplets(triplets,
                   weights,
                   mesh,
                   grid_address,
                   grid_point=None,
                   filename=None):
    triplets_filename = "triplets"
    suffix = "-m%d%d%d" % tuple(mesh)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if filename is not None:
        suffix += "." + filename
    suffix += ".dat"
    triplets_filename += suffix
    w = open(triplets_filename, 'w')
    for weight, g3 in zip(weights, triplets):
        w.write("%4d    " % weight)
        for q3 in grid_address[g3]:
            w.write("%4d %4d %4d    " % tuple(q3))
        w.write("\n")
    w.close()


def write_grid_address(grid_address, mesh, filename=None):
    grid_address_filename = "grid_address"
    suffix = "-m%d%d%d" % tuple(mesh)
    if filename is not None:
        suffix += "." + filename
    suffix += ".dat"
    grid_address_filename += suffix

    w = open(grid_address_filename, 'w')
    w.write("# Grid addresses for %dx%dx%d mesh\n" % tuple(mesh))
    w.write("#%9s    %8s %8s %8s     %8s %8s %8s\n" %
            ("index", "a", "b", "c",
             ("a%%%d" % mesh[0]), ("b%%%d" % mesh[1]), ("c%%%d" % mesh[2])))
    for i, bz_q in enumerate(grid_address):
        if i == np.prod(mesh):
            w.write("#" + "-" * 78 + "\n")
        q = bz_q % mesh
        w.write("%10d    %8d %8d %8d     " % (i, bz_q[0], bz_q[1], bz_q[2]))
        w.write("%8d %8d %8d\n" % tuple(q))

    return grid_address_filename


def write_grid_address_to_hdf5(grid_address,
                               mesh,
                               grid_mapping_table,
                               compression=None,
                               filename=None):
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "grid_address" + suffix + ".hdf5"
    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('mesh', data=mesh)
        w.create_dataset('grid_address', data=grid_address,
                         compression=compression)
        w.create_dataset('grid_mapping_table', data=grid_mapping_table,
                         compression=compression)
        return full_filename
    return None


def write_freq_shifts_to_hdf5(freq_shifts, filename='freq_shifts.hdf5'):
    with h5py.File(filename, 'w') as w:
        w.create_dataset('shift', data=freq_shifts)


def write_imag_self_energy_at_grid_point(gp,
                                         band_indices,
                                         mesh,
                                         frequencies,
                                         gammas,
                                         sigma=None,
                                         temperature=None,
                                         scattering_event_class=None,
                                         filename=None,
                                         is_mesh_symmetry=True):

    gammas_filename = "gammas"
    gammas_filename += "-m%d%d%d-g%d-" % (mesh[0],
                                          mesh[1],
                                          mesh[2],
                                          gp)
    if sigma is not None:
        gammas_filename += ("s%f" % sigma).rstrip('0').rstrip('\.') + "-"

    if temperature is not None:
        gammas_filename += ("t%f" % temperature).rstrip('0').rstrip('\.') + "-"

    for i in band_indices:
        gammas_filename += "b%d" % (i + 1)

    if scattering_event_class is not None:
        gammas_filename += "-c%d" % scattering_event_class

    if filename is not None:
        gammas_filename += ".%s" % filename
    elif not is_mesh_symmetry:
        gammas_filename += ".nosym"
    gammas_filename += ".dat"

    w = open(gammas_filename, 'w')
    for freq, g in zip(frequencies, gammas):
        w.write("%15.7f %20.15e\n" % (freq, g))
    w.close()


def write_joint_dos(gp,
                    mesh,
                    frequencies,
                    jdos,
                    sigma=None,
                    temperatures=None,
                    filename=None,
                    is_mesh_symmetry=True):
    if temperatures is None:
        return _write_joint_dos_at_t(gp,
                                     mesh,
                                     frequencies,
                                     jdos,
                                     sigma=sigma,
                                     temperature=None,
                                     filename=filename,
                                     is_mesh_symmetry=is_mesh_symmetry)
    else:
        for jdos_at_t, t in zip(jdos, temperatures):
            return _write_joint_dos_at_t(gp,
                                         mesh,
                                         frequencies,
                                         jdos_at_t,
                                         sigma=sigma,
                                         temperature=t,
                                         filename=filename,
                                         is_mesh_symmetry=is_mesh_symmetry)


def _write_joint_dos_at_t(grid_point,
                          mesh,
                          frequencies,
                          jdos,
                          sigma=None,
                          temperature=None,
                          filename=None,
                          is_mesh_symmetry=True):
    suffix = _get_filename_suffix(mesh,
                                  grid_point=grid_point,
                                  sigma=sigma,
                                  filename=filename)
    jdos_filename = "jdos%s" % suffix
    if temperature is not None:
        jdos_filename += ("-t%f" % temperature).rstrip('0').rstrip('\.')
    if not is_mesh_symmetry:
        jdos_filename += ".nosym"
    if filename is not None:
        jdos_filename += ".%s" % filename
    jdos_filename += ".dat"

    with open(jdos_filename, 'w') as w:
        for omega, vals in zip(frequencies, jdos):
            w.write("%15.7f" % omega)
            w.write((" %20.15e" * len(vals)) % tuple(vals))
            w.write("\n")
        return jdos_filename

def write_linewidth_at_grid_point(gp,
                                  band_indices,
                                  temperatures,
                                  gamma,
                                  mesh,
                                  sigma=None,
                                  filename=None,
                                  is_mesh_symmetry=True):

    lw_filename = "linewidth"
    lw_filename += "-m%d%d%d-g%d-" % (mesh[0], mesh[1], mesh[2], gp)
    if sigma is not None:
        lw_filename += ("s%f" % sigma).rstrip('0') + "-"

    for i in band_indices:
        lw_filename += "b%d" % (i + 1)

    if filename is not None:
        lw_filename += ".%s" % filename
    elif not is_mesh_symmetry:
        lw_filename += ".nosym"
    lw_filename += ".dat"

    w = open(lw_filename, 'w')
    for v, t in zip(gamma.sum(axis=1) * 2 / gamma.shape[1], temperatures):
        w.write("%15.7f %20.15e\n" % (t, v))
    w.close()


def write_frequency_shift(gp,
                          band_indices,
                          temperatures,
                          delta,
                          mesh,
                          epsilon=None,
                          filename=None,
                          is_mesh_symmetry=True):

    fst_filename = "frequency_shift"
    fst_filename += "-m%d%d%d-g%d-" % (mesh[0], mesh[1], mesh[2], gp)
    if epsilon is not None:
        if epsilon > 1e-5:
            fst_filename += ("s%f" % epsilon).rstrip('0') + "-"
        else:
            fst_filename += ("s%.3e" % epsilon) + "-"
    for i in band_indices:
        fst_filename += "b%d" % (i + 1)
    if filename is not None:
        fst_filename += ".%s" % filename
    elif not is_mesh_symmetry:
        fst_filename += ".nosym"
    fst_filename += ".dat"

    w = open(fst_filename, 'w')
    for v, t in zip(delta.sum(axis=1) / delta.shape[1], temperatures):
        w.write("%15.7f %20.15e\n" % (t, v))
    w.close()


def write_collision_to_hdf5(temperature,
                            mesh,
                            gamma=None,
                            gamma_isotope=None,
                            collision_matrix=None,
                            grid_point=None,
                            band_index=None,
                            sigma=None,
                            sigma_cutoff=None,
                            filename=None):
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(mesh,
                                  grid_point=grid_point,
                                  band_indices=band_indices,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    full_filename = "collision" + suffix + ".hdf5"
    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('temperature', data=temperature)
        if gamma is not None:
            w.create_dataset('gamma', data=gamma)
        if gamma_isotope is not None:
            w.create_dataset('gamma_isotope', data=gamma_isotope)
        if collision_matrix is not None:
            w.create_dataset('collision_matrix', data=collision_matrix)
        if grid_point is not None:
            w.create_dataset('grid_point', data=grid_point)
        if band_index is not None:
            w.create_dataset('band_index', data=(band_index + 1))
        if sigma is not None:
            w.create_dataset('sigma', data=sigma)
        if sigma_cutoff is not None:
            w.create_dataset('sigma_cutoff_width', data=sigma_cutoff)

        text = "Collisions "
        if grid_point is not None:
            text += "at grid adress %d " % grid_point
        if sigma is not None:
            if grid_point is not None:
                text += "and "
            else:
                text += "at "
            text += "sigma %s " % _del_zeros(sigma)
        text += "were written into "
        if sigma is not None:
            text += "\n"
        text += "\"%s\"." % ("collision" + suffix + ".hdf5")
        print(text)

    return full_filename


def write_full_collision_matrix(collision_matrix, filename='fcm.hdf5'):
    with h5py.File(filename, 'w') as w:
        w.create_dataset('collision_matrix', data=collision_matrix)


def write_unitary_matrix_to_hdf5(temperature,
                                 mesh,
                                 unitary_matrix=None,
                                 sigma=None,
                                 sigma_cutoff=None,
                                 solver=None,
                                 filename=None,
                                 verbose=False):
    """Write eigenvectors of collision matrices at temperatures.

    Depending on the choice of the solver, eigenvectors are sotred in
    either column-wise or row-wise.

    """

    suffix = _get_filename_suffix(mesh,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    hdf5_filename = "unitary" + suffix + ".hdf5"
    with h5py.File(hdf5_filename, 'w') as w:
        w.create_dataset('temperature', data=temperature)
        if unitary_matrix is not None:
            w.create_dataset('unitary_matrix', data=unitary_matrix)
        if solver is not None:
            w.create_dataset('solver', data=solver)

        if verbose:
            if len(temperature) > 1:
                text = "Unitary matrices "
            else:
                text = "Unitary matrix "
            if sigma is not None:
                text += "at sigma %s " % _del_zeros(sigma)
                if sigma_cutoff is not None:
                    text += "(%4.2f SD) " % sigma_cutoff
            if len(temperature) > 1:
                text += "were written into "
            else:
                text += "was written into "
            if sigma is not None:
                text += "\n"
            text += "\"%s\"." % hdf5_filename
            print(text)


def write_collision_eigenvalues_to_hdf5(temperatures,
                                        mesh,
                                        collision_eigenvalues,
                                        sigma=None,
                                        sigma_cutoff=None,
                                        filename=None,
                                        verbose=True):
    suffix = _get_filename_suffix(mesh,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    with h5py.File("coleigs" + suffix + ".hdf5", 'w') as w:
        w.create_dataset('temperature', data=temperatures)
        w.create_dataset('collision_eigenvalues', data=collision_eigenvalues)
        w.close()

        if verbose:
            text = "Eigenvalues of collision matrix "
            if sigma is not None:
                text += "with sigma %s\n" % sigma
            text += "were written into "
            text += "\"%s\"" % ("coleigs" + suffix + ".hdf5")
            print(text)


def write_kappa_to_hdf5(temperature,
                        mesh,
                        frequency=None,
                        group_velocity=None,
                        gv_by_gv=None,
                        mean_free_path=None,
                        heat_capacity=None,
                        kappa=None,
                        mode_kappa=None,
                        kappa_RTA=None,  # RTA calculated in LBTE
                        mode_kappa_RTA=None,  # RTA calculated in LBTE
                        f_vector=None,
                        gamma=None,
                        gamma_isotope=None,
                        gamma_N=None,
                        gamma_U=None,
                        averaged_pp_interaction=None,
                        qpoint=None,
                        weight=None,
                        mesh_divisors=None,
                        grid_point=None,
                        band_index=None,
                        sigma=None,
                        sigma_cutoff=None,
                        kappa_unit_conversion=None,
                        compression=None,
                        filename=None,
                        verbose=True):
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(mesh,
                                  mesh_divisors=mesh_divisors,
                                  grid_point=grid_point,
                                  band_indices=band_indices,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    full_filename = "kappa" + suffix + ".hdf5"
    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('temperature', data=temperature)
        w.create_dataset('mesh', data=mesh)
        if frequency is not None:
            w.create_dataset('frequency', data=frequency,
                             compression=compression)
        if group_velocity is not None:
            w.create_dataset('group_velocity', data=group_velocity,
                             compression=compression)
        if gv_by_gv is not None:
            w.create_dataset('gv_by_gv', data=gv_by_gv)
        if mean_free_path is not None:
            w.create_dataset('mean_free_path', data=mean_free_path,
                             compression=compression)
        if heat_capacity is not None:
            w.create_dataset('heat_capacity', data=heat_capacity,
                             compression=compression)
        if kappa is not None:
            w.create_dataset('kappa', data=kappa)
        if mode_kappa is not None:
            w.create_dataset('mode_kappa', data=mode_kappa,
                             compression=compression)
        if kappa_RTA is not None:
            w.create_dataset('kappa_RTA', data=kappa_RTA)
        if mode_kappa_RTA is not None:
            w.create_dataset('mode_kappa_RTA', data=mode_kappa_RTA,
                             compression=compression)
        if f_vector is not None:
            w.create_dataset('f_vector', data=f_vector,
                             compression=compression)
        if gamma is not None:
            w.create_dataset('gamma', data=gamma,
                             compression=compression)
        if gamma_isotope is not None:
            w.create_dataset('gamma_isotope', data=gamma_isotope,
                             compression=compression)
        if gamma_N is not None:
            w.create_dataset('gamma_N', data=gamma_N,
                             compression=compression)
        if gamma_U is not None:
            w.create_dataset('gamma_U', data=gamma_U,
                             compression=compression)
        if averaged_pp_interaction is not None:
            w.create_dataset('ave_pp', data=averaged_pp_interaction,
                             compression=compression)
        if qpoint is not None:
            w.create_dataset('qpoint', data=qpoint,
                             compression=compression)
        if weight is not None:
            w.create_dataset('weight', data=weight,
                             compression=compression)
        if grid_point is not None:
            w.create_dataset('grid_point', data=grid_point)
        if band_index is not None:
            w.create_dataset('band_index', data=(band_index + 1))
        if sigma is not None:
            w.create_dataset('sigma', data=sigma)
        if sigma_cutoff is not None:
            w.create_dataset('sigma_cutoff_width', data=sigma_cutoff)
        if kappa_unit_conversion is not None:
            w.create_dataset('kappa_unit_conversion',
                             data=kappa_unit_conversion)

        if verbose:
            text = ""
            if kappa is not None:
                text += "Thermal conductivity and related properties "
            else:
                text += "Thermal conductivity related properties "
            if grid_point is not None:
                text += "at gp-%d " % grid_point
                if band_index is not None:
                    text += "and band_index-%d\n" % (band_index + 1)
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % sigma
                if sigma_cutoff is None:
                    text += "\n"
                else:
                    text += "(%4.2f SD)\n" % sigma_cutoff
                text += "were written into "
            else:
                text += "were written into "
                if band_index is None:
                    text += "\n"
            text += "\"%s\"." % full_filename
            print(text)

        return full_filename


def read_gamma_from_hdf5(mesh,
                         mesh_divisors=None,
                         grid_point=None,
                         band_index=None,
                         sigma=None,
                         sigma_cutoff=None,
                         filename=None,
                         verbose=True):
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(mesh,
                                  mesh_divisors=mesh_divisors,
                                  grid_point=grid_point,
                                  band_indices=band_indices,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    full_filename = "kappa" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        if verbose:
            print("%s not found." % full_filename)
        return None

    read_data = {}

    with h5py.File(full_filename, 'r') as f:
        read_data['gamma'] = f['gamma'][:]
        for key in ('gamma_isotope',
                    'ave_pp',
                    'gamma_N',
                    'gamma_U'):
            if key in f.keys():
                if len(f[key].shape) > 0:
                    read_data[key] = f[key][:]
                else:
                    read_data[key] = f[key][()]
        if verbose:
            print("Read data from %s." % full_filename)

    return read_data


def read_collision_from_hdf5(mesh,
                             indices=None,
                             grid_point=None,
                             band_index=None,
                             sigma=None,
                             sigma_cutoff=None,
                             filename=None,
                             verbose=True):
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(mesh,
                                  grid_point=grid_point,
                                  band_indices=band_indices,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    full_filename = "collision" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        if verbose:
            print("%s not found." % full_filename)
        return None

    with h5py.File(full_filename, 'r') as f:
        if indices == 'all':
            colmat_shape = (1,) + f['collision_matrix'].shape
            collision_matrix = np.zeros(colmat_shape,
                                        dtype='double', order='C')
            gamma = np.array(f['gamma'][:], dtype='double', order='C')
            collision_matrix[0] = f['collision_matrix'][:]
            temperatures = np.array(f['temperature'][:], dtype='double')
        else:
            colmat_shape = (1, len(indices)) + f['collision_matrix'].shape[1:]
            collision_matrix = np.zeros(colmat_shape, dtype='double')
            gamma = np.array(f['gamma'][indices], dtype='double', order='C')
            collision_matrix[0] = f['collision_matrix'][indices]
            temperatures = np.array(f['temperature'][indices], dtype='double')

        if verbose:
            text = "Collisions "
            if band_index is None:
                if grid_point is not None:
                    text += "at grid point %d " % grid_point
            else:
                if grid_point is not None:
                    text += ("at (grid point %d, band index %d) " %
                             (grid_point, band_index))
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % _del_zeros(sigma)
                if sigma_cutoff is not None:
                    text += "(%4.2f SD)" % sigma_cutoff
            if band_index is None and grid_point is not None:
                text += " were read from "
                text += "\n"
            else:
                text += "\n"
                text += "were read from "
            text += "\"%s\"." % full_filename
            print(text)

        return collision_matrix, gamma, temperatures

    return None


def write_pp_to_hdf5(mesh,
                     pp=None,
                     g_zero=None,
                     grid_point=None,
                     triplet=None,
                     weight=None,
                     triplet_map=None,
                     triplet_all=None,
                     sigma=None,
                     sigma_cutoff=None,
                     filename=None,
                     verbose=True,
                     check_consistency=False,
                     compression=None):
    suffix = _get_filename_suffix(mesh,
                                  grid_point=grid_point,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    full_filename = "pp" + suffix + ".hdf5"

    with h5py.File(full_filename, 'w') as w:
        if pp is not None:
            if g_zero is None:
                w.create_dataset('pp', data=pp,
                                 compression=compression)
                if triplet is not None:
                    w.create_dataset('triplet', data=triplet,
                                     compression=compression)
                if weight is not None:
                    w.create_dataset('weight', data=weight,
                                     compression=compression)
                if triplet_map is not None:
                    w.create_dataset('triplet_map', data=triplet_map,
                                     compression=compression)
                if triplet_all is not None:
                    w.create_dataset('triplet_all', data=triplet_all,
                                     compression=compression)
            else:
                x = g_zero.ravel()
                nonzero_pp = np.array(pp.ravel()[x == 0], dtype='double')
                bytelen = len(x) // 8
                remlen = len(x) % 8
                y = x[:bytelen * 8].reshape(-1, 8)
                z = np.packbits(y)
                if remlen != 0:
                    z_rem = np.packbits(x[bytelen * 8:])

                w.create_dataset('nonzero_pp', data=nonzero_pp,
                                 compression=compression)
                w.create_dataset('pp_shape', data=pp.shape,
                                 compression=compression)
                w.create_dataset('g_zero_bits', data=z,
                                 compression=compression)
                if remlen != 0:
                    w.create_dataset('g_zero_bits_reminder', data=z_rem)

                # This is only for the test and coupled with read_pp_from_hdf5.
                if check_consistency:
                    w.create_dataset('pp', data=pp,
                                     compression=compression)
                    w.create_dataset('g_zero', data=g_zero,
                                     compression=compression)

        if verbose:
            text = ""
            text += "Ph-ph interaction strength "
            if grid_point is not None:
                text += "at gp-%d " % grid_point
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % sigma
                if sigma_cutoff is None:
                    text += "\n"
                else:
                    text += "(%4.2f SD)\n" % sigma_cutoff
                text += "were written into "
            else:
                text += "were written into "
                text += "\n"
            text += "\"%s\"." % full_filename
            print(text)

        return full_filename


def read_pp_from_hdf5(mesh,
                      grid_point=None,
                      sigma=None,
                      sigma_cutoff=None,
                      filename=None,
                      verbose=True,
                      check_consistency=False):
    suffix = _get_filename_suffix(mesh,
                                  grid_point=grid_point,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    full_filename = "pp" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        if verbose:
            print("%s not found." % full_filename)
        return None

    with h5py.File(full_filename, 'r') as f:
        if 'nonzero_pp' in f:
            nonzero_pp = f['nonzero_pp'][:]
            pp_shape = f['pp_shape'][:]
            z = f['g_zero_bits'][:]
            bytelen = np.prod(pp_shape) // 8
            remlen = 0
            if 'g_zero_bits_reminder' in f:
                z_rem = f['g_zero_bits_reminder'][:]
                remlen = np.prod(pp_shape) - bytelen * 8

            bits = np.unpackbits(z)
            if not bits.flags['C_CONTIGUOUS']:
                bits = np.array(bits, dtype='uint8')

            g_zero = np.zeros(pp_shape, dtype='byte', order='C')
            b = g_zero.ravel()
            b[:(bytelen * 8)] = bits
            if remlen != 0:
                b[-remlen:] = np.unpackbits(z_rem)[:remlen]

            pp = np.zeros(pp_shape, dtype='double', order='C')
            pp_ravel = pp.ravel()
            pp_ravel[g_zero.ravel() == 0] = nonzero_pp

            # check_consistency==True in write_pp_to_hdf5 required.
            if check_consistency and g_zero is not None:
                if verbose:
                    print("Checking consistency of ph-ph interanction "
                          "strength.")
                assert (g_zero == f['g_zero'][:]).all()
                assert np.allclose(pp, f['pp'][:])
        else:
            pp = np.zeros(f['pp'].shape, dtype='double', order='C')
            pp[:] = f['pp'][:]
            g_zero = None

        if verbose:
            print("Ph-ph interaction strength was read from \"%s\"." %
                  full_filename)

        return pp, g_zero

    return None


def write_gamma_detail_to_hdf5(temperature,
                               mesh,
                               gamma_detail=None,
                               grid_point=None,
                               triplet=None,
                               weight=None,
                               triplet_map=None,
                               triplet_all=None,
                               frequency_points=None,
                               band_index=None,
                               sigma=None,
                               sigma_cutoff=None,
                               compression=None,
                               filename=None,
                               verbose=True):
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(mesh,
                                  grid_point=grid_point,
                                  band_indices=band_indices,
                                  sigma=sigma,
                                  sigma_cutoff=sigma_cutoff,
                                  filename=filename)
    full_filename = "gamma_detail" + suffix + ".hdf5"

    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('temperature', data=temperature)
        w.create_dataset('mesh', data=mesh)
        if gamma_detail is not None:
            w.create_dataset('gamma_detail', data=gamma_detail,
                             compression=compression)
        if triplet is not None:
            w.create_dataset('triplet', data=triplet,
                             compression=compression)
        if weight is not None:
            w.create_dataset('weight', data=weight,
                             compression=compression)
        if triplet_map is not None:
            w.create_dataset('triplet_map', data=triplet_map,
                             compression=compression)
        if triplet_all is not None:
            w.create_dataset('triplet_all', data=triplet_all,
                             compression=compression)
        if grid_point is not None:
            w.create_dataset('grid_point', data=grid_point)
        if band_index is not None:
            w.create_dataset('band_index', data=(band_index + 1))
        if sigma is not None:
            w.create_dataset('sigma', data=sigma)
        if sigma_cutoff is not None:
            w.create_dataset('sigma_cutoff_width', data=sigma_cutoff)
        if frequency_points is not None:
            w.create_dataset('frequency_point', data=frequency_points)

        if verbose:
            text = ""
            text += "Phonon triplets contributions to Gamma "
            if grid_point is not None:
                text += "at gp-%d " % grid_point
                if band_index is not None:
                    text += "and band_index-%d\n" % (band_index + 1)
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s" % sigma
                if sigma_cutoff is None:
                    text += "\n"
                else:
                    text += "(%4.2f SD)\n" % sigma_cutoff
                text += "were written into "
            else:
                text += "were written into "
                if band_index is None:
                    text += "\n"
            text += "\"%s\"." % full_filename
            print(text)

        return full_filename

    return None


def write_phonon_to_hdf5(frequency,
                         eigenvector,
                         grid_address,
                         mesh,
                         compression=None,
                         filename=None):
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "phonon" + suffix + ".hdf5"

    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('mesh', data=mesh)
        w.create_dataset('grid_address', data=grid_address,
                         compression=compression)
        w.create_dataset('frequency', data=frequency,
                         compression=compression)
        w.create_dataset('eigenvector', data=eigenvector,
                         compression=compression)
        return full_filename

    return None


def read_phonon_from_hdf5(mesh,
                          filename=None,
                          verbose=True):
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "phonon" + suffix + ".hdf5"
    if not os.path.exists(full_filename):
        if verbose:
            print("%s not found." % full_filename)
        return None

    with h5py.File(full_filename, 'r') as f:
        frequencies = np.array(f['frequency'][:], dtype='double', order='C')
        itemsize = frequencies.itemsize
        eigenvectors = np.array(f['eigenvector'][:],
                                dtype=("c%d" % (itemsize * 2)), order='C')
        mesh_in_file = np.array(f['mesh'][:], dtype='intc')
        grid_address = np.array(f['grid_address'][:], dtype='intc', order='C')

        assert (mesh_in_file == mesh).all(), "Mesh numbers are inconsistent."

        if verbose:
            print("Phonons are read from \"%s\"." % full_filename)

        return frequencies, eigenvectors, grid_address

    return None


def write_ir_grid_points(mesh,
                         mesh_divs,
                         grid_points,
                         coarse_grid_weights,
                         grid_address,
                         primitive_lattice):
    w = open("ir_grid_points.yaml", 'w')
    w.write("mesh: [ %d, %d, %d ]\n" % tuple(mesh))
    if mesh_divs is not None:
        w.write("mesh_divisors: [ %d, %d, %d ]\n" % tuple(mesh_divs))
    w.write("reciprocal_lattice:\n")
    for vec, axis in zip(primitive_lattice.T, ('a*', 'b*', 'c*')):
        w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n"
                % (tuple(vec) + (axis,)))
    w.write("num_reduced_ir_grid_points: %d\n" % len(grid_points))
    w.write("ir_grid_points:  # [address, weight]\n")

    for g, weight in zip(grid_points, coarse_grid_weights):
        w.write("- grid_point: %d\n" % g)
        w.write("  weight: %d\n" % weight)
        w.write("  grid_address: [ %12d, %12d, %12d ]\n" %
                tuple(grid_address[g]))
        w.write("  q-point:      [ %12.7f, %12.7f, %12.7f ]\n" %
                tuple(grid_address[g].astype('double') / mesh))


def parse_disp_fc2_yaml(filename="disp_fc2.yaml", return_cell=False):
    dataset = _parse_yaml(filename)
    natom = dataset['natom']
    new_dataset = {}
    new_dataset['natom'] = natom
    new_first_atoms = []
    for first_atoms in dataset['first_atoms']:
        first_atoms['number'] -= 1
        atom1 = first_atoms['number']
        disp1 = first_atoms['displacement']
        new_first_atoms.append({'number': atom1, 'displacement': disp1})
    new_dataset['first_atoms'] = new_first_atoms

    if return_cell:
        cell = get_cell_from_disp_yaml(dataset)
        return new_dataset, cell
    else:
        return new_dataset


def parse_disp_fc3_yaml(filename="disp_fc3.yaml", return_cell=False):
    dataset = _parse_yaml(filename)
    natom = dataset['natom']
    new_dataset = {}
    new_dataset['natom'] = natom
    if 'cutoff_distance' in dataset:
        new_dataset['cutoff_distance'] = dataset['cutoff_distance']
    new_first_atoms = []
    for first_atoms in dataset['first_atoms']:
        atom1 = first_atoms['number'] - 1
        disp1 = first_atoms['displacement']
        new_second_atoms = []
        for second_atom in first_atoms['second_atoms']:
            disp2_dataset = {'number': second_atom['number'] - 1}
            if 'included' in second_atom:
                included = second_atom['included']
            else:
                included = True
            disp2_dataset.update({'included': included})
            if 'distance' in second_atom:
                disp2_dataset.update(
                    {'pair_distance': second_atom['distance']})
            for disp2 in second_atom['displacements']:
                disp2_dataset.update({'displacement': disp2})
                new_second_atoms.append(disp2_dataset.copy())
        new_first_atoms.append({'number': atom1,
                                'displacement': disp1,
                                'second_atoms': new_second_atoms})
    new_dataset['first_atoms'] = new_first_atoms

    if return_cell:
        cell = get_cell_from_disp_yaml(dataset)
        return new_dataset, cell
    else:
        return new_dataset


def parse_FORCES_FC2(disp_dataset, filename="FORCES_FC2"):
    num_atom = disp_dataset['natom']
    num_disp = len(disp_dataset['first_atoms'])
    forces_fc2 = []
    with open(filename, 'r') as f2:
        for i in range(num_disp):
            forces = _parse_force_lines(f2, num_atom)
            if forces is None:
                return []
            else:
                forces_fc2.append(forces)
    return forces_fc2


def parse_FORCES_FC3(disp_dataset, filename="FORCES_FC3", use_loadtxt=False):
    num_atom = disp_dataset['natom']
    num_disp = len(disp_dataset['first_atoms'])
    for disp1 in disp_dataset['first_atoms']:
        num_disp += len(disp1['second_atoms'])

    if use_loadtxt:
        forces_fc3 = np.loadtxt(filename)
        return forces_fc3.reshape((num_disp, -1, 3))
    else:
        forces_fc3 = np.zeros((num_disp, num_atom, 3),
                              dtype='double', order='C')
        with open(filename, 'r') as f3:
            for i in range(num_disp):
                forces = _parse_force_lines(f3, num_atom)
                if forces is None:
                    raise RuntimeError("Failed to parse %s." % filename)
                else:
                    forces_fc3[i] = forces
        return forces_fc3


def parse_QPOINTS3(filename='QPOINTS3'):
    f = open(filename)
    num = int(f.readline().strip())
    count = 0
    qpoints3 = []
    for line in f:
        line_array = [float(x) for x in line.strip().split()]

        if len(line_array) < 9:
            raise RuntimeError("Failed to parse %s." % filename)
        else:
            qpoints3.append(line_array[0:9])

        count += 1
        if count == num:
            break

    return np.array(qpoints3)


def parse_fc3(num_atom, filename='fc3.dat'):
    f = open(filename)
    fc3 = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3), dtype=float)
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                f.readline()
                for l in range(3):
                    fc3[i, j, k, l] = [
                        [float(x) for x in f.readline().split()],
                        [float(x) for x in f.readline().split()],
                        [float(x) for x in f.readline().split()]]
                    f.readline()
    return fc3


def parse_fc2(num_atom, filename='fc2.dat'):
    f = open(filename)
    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype=float)
    for i in range(num_atom):
        for j in range(num_atom):
            f.readline()
            fc2[i, j] = [[float(x) for x in f.readline().split()],
                         [float(x) for x in f.readline().split()],
                         [float(x) for x in f.readline().split()]]
            f.readline()

    return fc2


def parse_triplets(filename):
    f = open(filename)
    triplets = []
    weights = []
    for line in f:
        if line.strip()[0] == "#":
            continue

        line_array = [int(x) for x in line.split()]
        triplets.append(line_array[:3])
        weights.append(line_array[3])

    return np.array(triplets), np.array(weights)


def parse_grid_address(filename):
    f = open(filename, 'r')
    grid_address = []
    for line in f:
        if line.strip()[0] == "#":
            continue

        line_array = [int(x) for x in line.split()]
        grid_address.append(line_array[1:4])

    return np.array(grid_address)


def _get_filename_suffix(mesh,
                         mesh_divisors=None,
                         grid_point=None,
                         band_indices=None,
                         sigma=None,
                         sigma_cutoff=None,
                         filename=None):
    suffix = "-m%d%d%d" % tuple(mesh)
    if mesh_divisors is not None:
        if (np.array(mesh_divisors, dtype=int) != 1).any():
            suffix += "-d%d%d%d" % tuple(mesh_divisors)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if band_indices is not None:
        suffix += "-"
        for bi in band_indices:
            suffix += "b%d" % (bi + 1)
    if sigma is not None:
        suffix += "-s" + _del_zeros(sigma)
        if sigma_cutoff is not None:
            sigma_cutoff_str = _del_zeros(sigma_cutoff)
            suffix += "-sd" + sigma_cutoff_str
    if filename is not None:
        suffix += "." + filename

    return suffix


def _del_zeros(val):
    return ("%f" % val).rstrip('0').rstrip('\.')


def _parse_yaml(file_yaml):
    import yaml
    try:
        from yaml import CLoader as Loader
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    with open(file_yaml) as f:
        string = f.read()
    data = yaml.load(string, Loader=Loader)
    return data


def _parse_force_lines(forcefile, num_atom):
    forces = []
    for line in forcefile:
        if line.strip() == '':
            continue
        if line.strip()[0] == '#':
            continue
        forces.append([float(x) for x in line.strip().split()])
        if len(forces) == num_atom:
            break

    if not len(forces) == num_atom:
        return None
    else:
        return np.array(forces)


def _parse_force_constants_lines(fcthird_file, num_atom):
    fc2 = []
    for line in fcthird_file:
        if line.strip() == '':
            continue
        if line.strip()[0] == '#':
            continue
        fc2.append([float(x) for x in line.strip().split()])
        if len(fc2) == num_atom ** 2 * 3:
            break

    if not len(fc2) == num_atom ** 2 * 3:
        return None
    else:
        return np.array(fc2).reshape(num_atom, num_atom, 3, 3)
