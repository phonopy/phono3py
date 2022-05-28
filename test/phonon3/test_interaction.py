"""Test Interaction class."""
import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.phonon3.interaction import Interaction

itr_RTA_Si = [
    4.522052e-08,
    4.896362e-08,
    4.614211e-08,
    4.744361e-08,
    4.832248e-08,
    4.698535e-08,
    4.597876e-08,
    4.645423e-08,
    4.659572e-08,
    4.730222e-08,
]

itr_RTA_AlN = [
    7.456796e-08,
    7.242121e-08,
    7.068141e-08,
    7.059521e-08,
    7.289497e-08,
    7.127172e-08,
    7.082734e-08,
    7.394367e-08,
    7.084351e-08,
    7.083299e-08,
    7.085792e-08,
    7.124150e-08,
    7.048386e-08,
    7.062840e-08,
    7.036795e-08,
    7.043995e-08,
    7.366440e-08,
    7.136803e-08,
    6.988469e-08,
    6.989518e-08,
    7.179516e-08,
    7.038043e-08,
    7.011416e-08,
    7.278196e-08,
    6.999028e-08,
    7.009615e-08,
    7.018236e-08,
    7.025054e-08,
    6.977425e-08,
    6.993095e-08,
    6.962119e-08,
    6.964423e-08,
    7.121739e-08,
    6.939940e-08,
    6.834705e-08,
    6.847351e-08,
    6.977063e-08,
    6.872065e-08,
    6.863218e-08,
    7.055696e-08,
    6.836064e-08,
    6.854052e-08,
    6.864199e-08,
    6.849059e-08,
    6.826958e-08,
    6.837379e-08,
    6.808307e-08,
    6.804480e-08,
    6.961289e-08,
    6.816170e-08,
    6.730028e-08,
    6.746055e-08,
    6.851460e-08,
    6.764892e-08,
    6.754060e-08,
    6.913662e-08,
    6.729303e-08,
    6.736722e-08,
    6.734663e-08,
    6.743441e-08,
    6.713107e-08,
    6.710084e-08,
    6.698233e-08,
    6.694871e-08,
]


@pytest.mark.parametrize("lang", ["C", "Py"])
def test_interaction_RTA_si(si_pbesol, lang):
    """Test interaction_strength of Si."""
    itr = _get_irt(si_pbesol, [4, 4, 4])
    itr.set_grid_point(1)
    itr.run(lang=lang)
    # _show(itr)
    # (10, 6, 6, 6)
    np.testing.assert_allclose(
        itr.interaction_strength.sum(axis=(1, 2, 3)), itr_RTA_Si, rtol=0, atol=1e-6
    )


def test_interaction_RTA_AlN(aln_lda):
    """Test interaction_strength of AlN."""
    itr = _get_irt(aln_lda, [7, 7, 7])
    itr.set_grid_point(1)
    itr.run()
    # _show(itr)
    np.testing.assert_allclose(
        itr.interaction_strength.sum(axis=(1, 2, 3)), itr_RTA_AlN, rtol=0, atol=1e-6
    )


def test_interaction_nac_direction_phonon_NaCl(nacl_pbe: Phono3py):
    """Test interaction_strength of NaCl with nac_q_direction."""
    itr = _get_irt(nacl_pbe, [7, 7, 7], nac_params=nacl_pbe.nac_params)
    itr.nac_q_direction = [1, 0, 0]
    itr.set_grid_point(0)
    frequencies, _, _ = itr.get_phonons()
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 7.41183870], rtol=0, atol=1e-6
    )


def test_interaction_nac_direction_phonon_NaCl_second_error(nacl_pbe: Phono3py):
    """Test interaction_strength of NaCl with nac_q_direction.

    Second setting non-gamma grid point must raise exception.

    """
    itr = _get_irt(nacl_pbe, [7, 7, 7], nac_params=nacl_pbe.nac_params)
    itr.nac_q_direction = [1, 0, 0]
    itr.set_grid_point(0)
    with pytest.raises(RuntimeError):
        itr.set_grid_point(1)


def test_interaction_nac_direction_phonon_NaCl_second_no_error(nacl_pbe: Phono3py):
    """Test interaction_strength of NaCl with nac_q_direction.

    Second setting non-gamma grid point should not raise exception because
    nac_q_direction = None is set, but the phonons at Gamma is updated to those without
    NAC.

    """
    itr = _get_irt(nacl_pbe, [7, 7, 7], nac_params=nacl_pbe.nac_params)
    itr.nac_q_direction = [1, 0, 0]
    itr.set_grid_point(0)
    itr.nac_q_direction = None
    itr.set_grid_point(1)
    frequencies, _, _ = itr.get_phonons()
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 4.59488262], rtol=0, atol=1e-6
    )


def test_interaction_run_phonon_solver_at_gamma_NaCl(nacl_pbe: Phono3py):
    """Test run_phonon_solver_at_gamma with nac_q_direction on NaCl.

    Phonon calculation at Gamma without NAC is peformed at itr.init_dynamical_matrix().
    The phonons at Gamma without NAC are saved in dedicated variables.

    Phonon calculation at Gamma with NAC is peformed at itr.set_grid_point(0) and
    stored in phonon variables on grid.

    itr.run_phonon_solver_at_gamma() stored phonons at Gamma without NAC are copied
    to phonon variables on grid.

    itr.run_phonon_solver_at_gamma(is_nac=True) runs phonon calculation at Gamma with
    NAC and stores them in phonon variables on grid.

    """
    itr = _get_irt(nacl_pbe, [7, 7, 7], nac_params=nacl_pbe.nac_params)
    itr.nac_q_direction = [1, 0, 0]
    frequencies, _, _ = itr.get_phonons()
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 4.59488262], rtol=0, atol=1e-6
    )
    itr.set_grid_point(0)
    frequencies, _, _ = itr.get_phonons()
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 7.41183870], rtol=0, atol=1e-6
    )
    itr.run_phonon_solver_at_gamma()
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 4.59488262], rtol=0, atol=1e-6
    )
    itr.run_phonon_solver_at_gamma(is_nac=True)
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 7.41183870], rtol=0, atol=1e-6
    )


def test_phonon_solver_expand_RTA_si(si_pbesol):
    """Test phonon solver with eigenvector rotation of Si.

    Eigenvectors can be different but frequencies must be almost the same.

    """
    itr = _get_irt(si_pbesol, [4, 4, 4])
    freqs, _, phonon_done = itr.get_phonons()
    assert (phonon_done == 1).all()
    itr = _get_irt(si_pbesol, [4, 4, 4], solve_dynamical_matrices=False)
    itr.run_phonon_solver_with_eigvec_rotation()
    freqs_expanded, _, _ = itr.get_phonons()
    np.testing.assert_allclose(freqs, freqs_expanded, rtol=0, atol=1e-6)


def _get_irt(ph3: Phono3py, mesh, nac_params=None, solve_dynamical_matrices=True):
    ph3.mesh_numbers = mesh
    itr = Interaction(
        ph3.primitive, ph3.grid, ph3.primitive_symmetry, ph3.fc3, cutoff_frequency=1e-4
    )
    if nac_params is None:
        itr.init_dynamical_matrix(
            ph3.fc2,
            ph3.phonon_supercell,
            ph3.phonon_primitive,
        )
    else:
        itr.init_dynamical_matrix(
            ph3.fc2,
            ph3.phonon_supercell,
            ph3.phonon_primitive,
            nac_params=nac_params,
        )
    if solve_dynamical_matrices:
        itr.run_phonon_solver()
    return itr


def _show(itr):
    itr_vals = itr.interaction_strength.sum(axis=(1, 2, 3))
    for i, v in enumerate(itr_vals):
        print("%e, " % v, end="")
        if (i + 1) % 5 == 0:
            print("")
