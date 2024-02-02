"""Test Interaction class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
import pytest
from phonopy.structure.cells import get_smallest_vectors

from phono3py import Phono3py
from phono3py.phonon3.interaction import Interaction


@pytest.mark.parametrize("lang", ["C", "Python"])
def test_interaction_RTA_si(si_pbesol: Phono3py, lang: Literal["C", "Python"]):
    """Test interaction_strength of Si."""
    if si_pbesol._make_r0_average:
        ref_itr_RTA_Si = [
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
    else:
        ref_itr_RTA_Si = [
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

    itr = _get_irt(si_pbesol, [4, 4, 4])
    itr.set_grid_point(1)
    itr.run(lang=lang)
    # _show(itr)
    # (10, 6, 6, 6)
    np.testing.assert_allclose(
        itr.interaction_strength.sum(axis=(1, 2, 3)), ref_itr_RTA_Si, rtol=0, atol=1e-10
    )


def test_interaction_RTA_AlN(aln_lda: Phono3py):
    """Test interaction_strength of AlN."""
    if aln_lda._make_r0_average:
        ref_itr_RTA_AlN = [
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
    else:
        ref_itr_RTA_AlN = [
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

    itr = _get_irt(aln_lda, [7, 7, 7])
    itr.set_grid_point(1)
    itr.run()
    _show(itr)
    np.testing.assert_allclose(
        itr.interaction_strength.sum(axis=(1, 2, 3)),
        ref_itr_RTA_AlN,
        rtol=0,
        atol=1e-10,
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


def test_phonon_solver_expand_RTA_si(si_pbesol: Phono3py):
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


def test_get_all_shortest(aln_lda: Phono3py):
    """Test Interaction._get_all_shortest."""
    ph3 = aln_lda
    ph3.mesh_numbers = 30
    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        cutoff_frequency=1e-5,
    )
    s_svecs, s_multi = get_smallest_vectors(
        ph3.supercell.cell,
        ph3.supercell.scaled_positions,
        ph3.supercell.scaled_positions,
        store_dense_svecs=True,
    )
    s_lattice = ph3.supercell.cell
    p_lattice = itr.primitive.cell
    shortests = itr._all_shortest
    svecs, multi, _, _, _ = itr.get_primitive_and_supercell_correspondence()
    n_satom, n_patom, _ = multi.shape
    for i, j, k in np.ndindex((n_patom, n_satom, n_satom)):
        is_found = 0
        if multi[j, i, 0] == 1 and multi[k, i, 0] == 1 and s_multi[j, k, 0] == 1:
            d_jk_shortest = np.linalg.norm(s_svecs[s_multi[j, k, 1]] @ s_lattice)
            vec_ij = svecs[multi[j, i, 1]]
            vec_ik = svecs[multi[k, i, 1]]
            vec_jk = vec_ik - vec_ij
            d_jk = np.linalg.norm(vec_jk @ p_lattice)
            if abs(d_jk - d_jk_shortest) < ph3.symmetry.tolerance:
                is_found = 1
        assert shortests[i, j, k] == is_found


def _get_irt(
    ph3: Phono3py,
    mesh: Union[int, float, Sequence, np.ndarray],
    nac_params: Optional[dict] = None,
    solve_dynamical_matrices: bool = True,
    make_r0_average: bool = False,
):
    ph3.mesh_numbers = mesh
    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        fc3=ph3.fc3,
        make_r0_average=make_r0_average,
        cutoff_frequency=1e-4,
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


def _show(itr: Interaction):
    itr_vals = itr.interaction_strength.sum(axis=(1, 2, 3))
    for i, v in enumerate(itr_vals):
        print("%e, " % v, end="")
        if (i + 1) % 5 == 0:
            print("")
