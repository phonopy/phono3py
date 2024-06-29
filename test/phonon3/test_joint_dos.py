"""Tests for joint-density-of-states."""

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.api_jointdos import Phono3pyJointDos
from phono3py.phonon.grid import BZGrid
from phono3py.phonon3.joint_dos import JointDos

si_freq_points = [
    0.0000000,
    3.4102469,
    6.8204938,
    10.2307406,
    13.6409875,
    17.0512344,
    20.4614813,
    23.8717281,
    27.2819750,
    30.6922219,
]
si_jdos_12 = [
    10.8993284,
    0.0000000,
    1.9825862,
    0.0000000,
    1.6458638,
    0.4147573,
    3.7550744,
    0.8847213,
    0.0176267,
    1.0774414,
    0.0000000,
    2.1981098,
    0.0000000,
    1.4959386,
    0.0000000,
    2.0987108,
    0.0000000,
    1.1648722,
    0.0000000,
    0.0000000,
]
si_jdos_nomeshsym_12 = [
    10.9478722,
    0.0000000,
    1.9825862,
    0.0000000,
    1.6458638,
    0.4147573,
    3.7550744,
    0.8847213,
    0.0176267,
    1.0774414,
    0.0000000,
    2.1981098,
    0.0000000,
    1.4959386,
    0.0000000,
    2.0987108,
    0.0000000,
    1.1648722,
    0.0000000,
    0.0000000,
]

nacl_freq_points = [
    0.0000000,
    1.6322306,
    3.2644613,
    4.8966919,
    6.5289225,
    8.1611531,
    9.7933838,
    11.4256144,
    13.0578450,
    14.6900756,
]
nacl_jdos_12 = [
    20.5529946,
    0.0000000,
    11.3095088,
    0.0000000,
    2.3068141,
    0.1854566,
    0.2624358,
    1.1781852,
    0.0000000,
    4.9673048,
    0.0000000,
    8.0794774,
    0.0000000,
    5.3993210,
    0.0000000,
    1.3717314,
    0.0000000,
    0.1144440,
    0.0000000,
    0.0000000,
]


nacl_freq_points_gamma = [
    0.0000000,
    1.6322306,
    3.2644613,
    4.8966919,
    6.5289225,
    8.1611531,
    9.7933838,
    11.4256144,
    13.0578450,
    14.6900756,
]
nacl_jdos_12_gamma = [
    1638874677039989.2500000,
    0.0000000,
    9.1146639,
    0.0403982,
    1.5633599,
    0.2967087,
    0.3866759,
    0.9911241,
    0.0446820,
    5.2759325,
    0.0000000,
    9.3698670,
    0.0000000,
    5.1064113,
    0.0000000,
    0.8748614,
    0.0000000,
    0.1474919,
    0.0000000,
    0.0112830,
]

nacl_freq_points_at_300K = [
    0.0000000,
    1.6322306,
    3.2644613,
    4.8966919,
    6.5289225,
    8.1611531,
    9.7933838,
    11.4256144,
    13.0578450,
    14.6900756,
]
nacl_jdos_12_at_300K = [
    0.0000000,
    0.0000000,
    8.4768159,
    0.0000000,
    4.1241485,
    1.4712023,
    0.8016066,
    6.7628440,
    0.0000000,
    21.2134161,
    0.0000000,
    26.9803216,
    0.0000000,
    14.9103483,
    0.0000000,
    3.2833064,
    0.0000000,
    0.2398336,
    0.0000000,
    0.0000000,
]
nacl_freq_points_gamma_at_300K = [
    0.0000000,
    1.6322306,
    3.2644613,
    4.8966919,
    6.5289225,
    8.1611531,
    9.7933838,
    11.4256144,
    13.0578450,
    14.6900756,
]
nacl_jdos_gamma_at_300K = [
    0.0000000,
    0.0000000,
    6.3607672,
    0.4210009,
    2.9647113,
    2.3994749,
    0.9360874,
    5.2286115,
    0.1977176,
    22.0282005,
    0.0000000,
    32.0059314,
    0.0000000,
    13.9738865,
    0.0000000,
    2.1095895,
    0.0000000,
    0.3079461,
    0.0000000,
    0.0213677,
]


def test_jdos_si(si_pbesol: Phono3py):
    """Test joint-DOS by Si."""
    si_pbesol.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        si_pbesol.fc2,
        mesh=si_pbesol.mesh_numbers,
        num_frequency_points=10,
        log_level=1,
    )
    jdos.run([105])

    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(si_freq_points, jdos.frequency_points, atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        si_jdos_12[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdso_si_nomeshsym(si_pbesol: Phono3py):
    """Test joint-DOS without considering mesh symmetry by Si."""
    si_pbesol.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        si_pbesol.fc2,
        mesh=si_pbesol.mesh_numbers,
        num_frequency_points=10,
        is_mesh_symmetry=False,
        log_level=1,
    )
    jdos.run([105])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(si_freq_points, jdos.frequency_points, atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        si_jdos_nomeshsym_12[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdos_nacl(nacl_pbe: Phono3py):
    """Test joint-DOS by NaCl."""
    nacl_pbe.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        nacl_pbe.phonon_supercell,
        nacl_pbe.phonon_primitive,
        nacl_pbe.fc2,
        mesh=nacl_pbe.mesh_numbers,
        nac_params=nacl_pbe.nac_params,
        num_frequency_points=10,
        log_level=1,
    )
    jdos.run([105])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(nacl_freq_points, jdos.frequency_points, atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        nacl_jdos_12[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdos_nacl_gamma(nacl_pbe: Phono3py):
    """Test joint-DOS at Gamma-point by NaCl."""
    nacl_pbe.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        nacl_pbe.phonon_supercell,
        nacl_pbe.phonon_primitive,
        nacl_pbe.fc2,
        mesh=nacl_pbe.mesh_numbers,
        nac_params=nacl_pbe.nac_params,
        nac_q_direction=[1, 0, 0],
        num_frequency_points=10,
        log_level=1,
    )
    jdos.run([0])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(nacl_freq_points_gamma, jdos.frequency_points, atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        nacl_jdos_12_gamma[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdos_nacl_at_300K(nacl_pbe: Phono3py):
    """Test joint-DOS at 300K by NaCl."""
    nacl_pbe.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        nacl_pbe.phonon_supercell,
        nacl_pbe.phonon_primitive,
        nacl_pbe.fc2,
        mesh=nacl_pbe.mesh_numbers,
        nac_params=nacl_pbe.nac_params,
        num_frequency_points=10,
        temperatures=[
            300,
        ],
        log_level=1,
    )
    jdos.run([105])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(
        nacl_freq_points_at_300K, jdos.frequency_points, atol=1e-5
    )
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        nacl_jdos_12_at_300K[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdos_nacl_nac_gamma_at_300K_npoints(nacl_pbe: Phono3py):
    """Real part of self energy spectrum of NaCl.

    * at 10 frequency points sampled uniformly.
    * at q->0

    """
    nacl_pbe.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        nacl_pbe.phonon_supercell,
        nacl_pbe.phonon_primitive,
        nacl_pbe.fc2,
        mesh=nacl_pbe.mesh_numbers,
        nac_params=nacl_pbe.nac_params,
        nac_q_direction=[1, 0, 0],
        num_frequency_points=10,
        temperatures=[
            300,
        ],
        log_level=1,
    )
    jdos.run([nacl_pbe.grid.gp_Gamma])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(
        nacl_freq_points_gamma_at_300K, jdos.frequency_points, atol=1e-5
    )
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        nacl_jdos_gamma_at_300K[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdos_nac_direction_phonon_NaCl(nacl_pbe: Phono3py):
    """Test JDOS of NaCl with nac_q_direction."""
    jdos = _get_jdos(
        nacl_pbe,
        [7, 7, 7],
        nac_params=nacl_pbe.nac_params,
    )
    jdos.nac_q_direction = [1, 0, 0]
    jdos.set_grid_point(0)
    frequencies, _, _ = jdos.get_phonons()
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 7.41183870], rtol=0, atol=1e-6
    )


def test_jdos_nac_direction_phonon_NaCl_second_error(nacl_pbe: Phono3py):
    """Test JDOS of NaCl with nac_q_direction.

    Second setting non-gamma grid point must raise exception.

    """
    jdos = _get_jdos(
        nacl_pbe,
        [7, 7, 7],
        nac_params=nacl_pbe.nac_params,
    )
    jdos.nac_q_direction = [1, 0, 0]
    jdos.set_grid_point(0)
    with pytest.raises(RuntimeError):
        jdos.set_grid_point(1)


def test_jdos_nac_direction_phonon_NaCl_second_no_error(nacl_pbe: Phono3py):
    """Test JDOS of NaCl with nac_q_direction.

    Second setting non-gamma grid point should not raise exception because
    nac_q_direction = None is set, but the phonons at Gamma is updated to those without
    NAC.

    """
    jdos = _get_jdos(
        nacl_pbe,
        [7, 7, 7],
        nac_params=nacl_pbe.nac_params,
    )
    jdos.nac_q_direction = [1, 0, 0]
    jdos.set_grid_point(0)
    jdos.nac_q_direction = None
    jdos.set_grid_point(1)
    frequencies, _, _ = jdos.get_phonons()
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.59488262, 4.59488262, 4.59488262], rtol=0, atol=1e-6
    )


def test_jdos_nac_NaCl_300K_C(nacl_pbe: Phono3py):
    """Test running JDOS of NaCl in C mode."""
    jdos = _get_jdos(
        nacl_pbe,
        [9, 9, 9],
        nac_params=nacl_pbe.nac_params,
    )
    jdos.set_grid_point(105)
    jdos.frequency_points = nacl_freq_points_at_300K
    jdos.temperature = 300
    jdos.run_phonon_solver()
    jdos.run_integration_weights()
    jdos.run_jdos()
    np.testing.assert_allclose(
        nacl_jdos_12_at_300K[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdos_nac_NaCl_300K_Py(nacl_pbe: Phono3py):
    """Test running JDOS of NaCl in Py (JDOS) mode."""
    jdos = _get_jdos(
        nacl_pbe,
        [9, 9, 9],
        nac_params=nacl_pbe.nac_params,
    )
    jdos.set_grid_point(105)
    jdos.frequency_points = nacl_freq_points_at_300K
    jdos.temperature = 300
    jdos.run_phonon_solver()
    jdos.run_integration_weights()
    jdos.run_jdos(lang="Py")
    np.testing.assert_allclose(
        nacl_jdos_12_at_300K[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def test_jdos_nac_NaCl_300K_PyPy(nacl_pbe: Phono3py):
    """Test running JDOS of NaCl in Py (JDOS) and Py (tetrahedron) mode."""
    jdos = _get_jdos(
        nacl_pbe,
        [9, 9, 9],
        nac_params=nacl_pbe.nac_params,
    )
    jdos.set_grid_point(105)
    jdos.frequency_points = nacl_freq_points_at_300K
    jdos.temperature = 300
    jdos.run_phonon_solver()
    jdos.run_integration_weights(lang="Py")
    jdos.run_jdos(lang="Py")
    np.testing.assert_allclose(
        nacl_jdos_12_at_300K[2:], jdos.joint_dos.ravel()[2:], rtol=1e-2, atol=1e-5
    )


def _get_jdos(ph3: Phono3py, mesh, nac_params=None):
    bz_grid = BZGrid(
        mesh,
        lattice=ph3.primitive.cell,
        symmetry_dataset=ph3.primitive_symmetry.dataset,
        store_dense_gp_map=True,
    )
    jdos = JointDos(
        ph3.primitive,
        ph3.supercell,
        bz_grid,
        ph3.fc2,
        nac_params=nac_params,
        cutoff_frequency=1e-4,
    )
    return jdos
