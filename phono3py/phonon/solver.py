import numpy as np


def set_phonon_c(dm,
                 frequencies,
                 eigenvectors,
                 phonon_done,
                 grid_points,
                 grid_address,
                 mesh,
                 frequency_factor_to_THz,
                 nac_q_direction,  # in reduced coordinates
                 lapack_zheev_uplo,
                 verbose=False):
    import phono3py._phono3py as phono3c

    (svecs,
     multiplicity,
     masses,
     rec_lattice,  # column vectors
     positions,
     born,
     nac_factor,
     dielectric) = _extract_params(dm)

    if dm.is_nac() and dm.get_nac_method() == 'gonze':
        gonze_nac_dataset = dm.get_Gonze_nac_dataset()
        if gonze_nac_dataset[0] is None:
            dm.make_Gonze_nac_dataset(verbose=verbose)
            gonze_nac_dataset = dm.get_Gonze_nac_dataset()
        (gonze_fc,  # fc where the dipole-diple contribution is removed.
         dd_q0,     # second term of dipole-dipole expression.
         G_cutoff,  # Cutoff radius in reciprocal space. This will not be used.
         G_list,    # List of G points where d-d interactions are integrated.
         Lambda) = gonze_nac_dataset  # Convergence parameter
        fc = gonze_fc
    else:
        positions = None
        dd_q0 = None
        G_list = None
        Lambda = 0
        fc = dm.get_force_constants()

    assert grid_points.dtype == 'uintp'
    assert grid_points.flags.c_contiguous

    fc_p2s, fc_s2p = _get_fc_elements_mapping(dm, fc)
    phono3c.phonons_at_gridpoints(
        frequencies,
        eigenvectors,
        phonon_done,
        grid_points,
        grid_address,
        np.array(mesh, dtype='intc'),
        fc,
        svecs,
        multiplicity,
        positions,
        masses,
        fc_p2s,
        fc_s2p,
        frequency_factor_to_THz,
        born,
        dielectric,
        rec_lattice,
        nac_q_direction,
        nac_factor,
        dd_q0,
        G_list,
        Lambda,
        lapack_zheev_uplo)


def set_phonon_py(grid_point,
                  phonon_done,
                  frequencies,
                  eigenvectors,
                  grid_address,
                  mesh,
                  dynamical_matrix,
                  frequency_factor_to_THz,
                  lapack_zheev_uplo):
    gp = grid_point
    if phonon_done[gp] == 0:
        phonon_done[gp] = 1
        q = grid_address[gp].astype('double') / mesh
        dynamical_matrix.set_dynamical_matrix(q)
        dm = dynamical_matrix.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm, UPLO=lapack_zheev_uplo)
        eigvals = eigvals.real
        frequencies[gp] = (np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
                           * frequency_factor_to_THz)
        eigenvectors[gp] = eigvecs


def _extract_params(dm):
    svecs, multiplicity = dm.get_shortest_vectors()
    masses = np.array(dm.get_primitive().get_masses(), dtype='double')
    rec_lattice = np.array(np.linalg.inv(dm.get_primitive().get_cell()),
                           dtype='double', order='C')
    positions = np.array(dm.get_primitive().get_positions(),
                         dtype='double', order='C')
    if dm.is_nac():
        born = dm.get_born_effective_charges()
        nac_factor = dm.get_nac_factor()
        dielectric = dm.get_dielectric_constant()
    else:
        born = None
        nac_factor = 0
        dielectric = None

    return (svecs,
            multiplicity,
            masses,
            rec_lattice,
            positions,
            born,
            nac_factor,
            dielectric)


def _get_fc_elements_mapping(dm, fc):
    p2s_map = dm.get_primitive().get_primitive_to_supercell_map()
    s2p_map = dm.get_primitive().get_supercell_to_primitive_map()
    if fc.shape[0] == fc.shape[1]:  # full fc
        fc_p2s = p2s_map
        fc_s2p = s2p_map
    else:  # compact fc
        primitive = dm.get_primitive()
        p2p_map = primitive.get_primitive_to_primitive_map()
        s2pp_map = np.array([p2p_map[s2p_map[i]] for i in range(len(s2p_map))],
                            dtype='intc')
        fc_p2s = np.arange(len(p2s_map), dtype='intc')
        fc_s2p = s2pp_map

    return fc_p2s, fc_s2p
