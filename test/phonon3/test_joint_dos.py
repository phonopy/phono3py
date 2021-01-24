import numpy as np
from phono3py.api_jointdos import Phono3pyJointDos

si_freq_points = [0.0000000, 3.4102469, 6.8204938, 10.2307406, 13.6409875,
                  17.0512344, 20.4614813, 23.8717281, 27.2819750, 30.6922219]
si_jdos_12 = [10.8993284, 0.0000000,
              1.9825862, 0.0000000,
              1.6458638, 0.4147573,
              3.7550744, 0.8847213,
              0.0176267, 1.0774414,
              0.0000000, 2.1981098,
              0.0000000, 1.4959386,
              0.0000000, 2.0987108,
              0.0000000, 1.1648722,
              0.0000000, 0.0000000]
nacl_freq_points = [0.0000000, 1.6322306, 3.2644613, 4.8966919, 6.5289225,
                    8.1611531, 9.7933838, 11.4256144, 13.0578450, 14.6900756]
nacl_jdos_12 = [20.5529946, 0.0000000,
                11.3095088, 0.0000000,
                2.3068141, 0.1854566,
                0.2624358, 1.1781852,
                0.0000000, 4.9673048,
                0.0000000, 8.0794774,
                0.0000000, 5.3993210,
                0.0000000, 1.3717314,
                0.0000000, 0.1144440,
                0.0000000, 0.0000000]
nacl_freq_points_gamma = [
    0.0000000, 1.6553106, 3.3106213, 4.9659319, 6.6212426,
    8.2765532, 9.9318639, 11.5871745, 13.2424851, 14.8977958]
nacl_jdos_12_gamma = [1742452844146884.7500000, 0.0000000,
                      8.8165476, 0.0415488,
                      1.4914142, 0.3104766,
                      0.3679421, 1.0509976,
                      0.0358263, 5.8578016,
                      0.0000000, 7.2272898,
                      0.0000000, 5.7740314,
                      0.0000000, 0.6663207,
                      0.0000000, 0.1348658,
                      0.0000000, 0.0000000]

nacl_freq_points_at_300K = [
    0.0000000, 1.6322306, 3.2644613, 4.8966919, 6.5289225,
    8.1611531, 9.7933838, 11.4256144, 13.0578450, 14.6900756]
nacl_jdos_12_at_300K = [0.0000000, 0.0000000,
                        8.4625631, 0.0000000,
                        4.1076174, 1.5151176,
                        0.7992725, 6.7993659,
                        0.0000000, 21.2271309,
                        0.0000000, 26.9803907,
                        0.0000000, 14.9103483,
                        0.0000000, 3.2833064,
                        0.0000000, 0.2398336,
                        0.0000000, 0.0000000]


def test_jdos_si(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        si_pbesol.mesh_numbers,
        si_pbesol.fc2,
        num_frequency_points=10,
        log_level=1)
    jdos.run([103])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(si_freq_points, jdos.frequency_points,
                               atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(si_jdos_12[2:], jdos.joint_dos.ravel()[2:],
                               rtol=1e-2, atol=1e-5)


def test_jdos_nacl(nacl_pbe):
    nacl_pbe.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        nacl_pbe.phonon_supercell,
        nacl_pbe.phonon_primitive,
        nacl_pbe.mesh_numbers,
        nacl_pbe.fc2,
        nac_params=nacl_pbe.nac_params,
        num_frequency_points=10,
        log_level=1)
    jdos.run([103])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(nacl_freq_points, jdos.frequency_points,
                               atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(nacl_jdos_12[2:], jdos.joint_dos.ravel()[2:],
                               rtol=1e-2, atol=1e-5)


def test_jdos_nacl_gamma(nacl_pbe):
    nacl_pbe.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        nacl_pbe.phonon_supercell,
        nacl_pbe.phonon_primitive,
        nacl_pbe.mesh_numbers,
        nacl_pbe.fc2,
        nac_params=nacl_pbe.nac_params,
        nac_q_direction=[1, 0, 0],
        num_frequency_points=10,
        log_level=1)
    jdos.run([0])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(nacl_freq_points_gamma, jdos.frequency_points,
                               atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        nacl_jdos_12_gamma[2:], jdos.joint_dos.ravel()[2:],
        rtol=1e-2, atol=1e-5)


def test_jdos_nacl_at_300K(nacl_pbe):
    nacl_pbe.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        nacl_pbe.phonon_supercell,
        nacl_pbe.phonon_primitive,
        nacl_pbe.mesh_numbers,
        nacl_pbe.fc2,
        nac_params=nacl_pbe.nac_params,
        num_frequency_points=10,
        temperatures=[300, ],
        log_level=1)
    jdos.run([103])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(
        nacl_freq_points_at_300K, jdos.frequency_points,
        atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(
        nacl_jdos_12_at_300K[2:], jdos.joint_dos.ravel()[2:],
        rtol=1e-2, atol=1e-5)
