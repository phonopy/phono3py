import numpy as np
from phono3py.api_jointdos import Phono3pyJointDos

freq_points = [0.0000000, 3.4102469, 6.8204938, 10.2307406, 13.6409875,
               17.0512344, 20.4614813, 23.8717281, 27.2819750, 30.6922219]
jdos_12 = [10.8993284, 0.0000000,
           1.9825862, 0.0000000,
           1.6458638, 0.4147573,
           3.7550744, 0.8847213,
           0.0176267, 1.0774414,
           0.0000000, 2.1981098,
           0.0000000, 1.4959386,
           0.0000000, 2.0987108,
           0.0000000, 1.1648722,
           0.0000000, 0.0000000]


def test_jdos(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    jdos = Phono3pyJointDos(
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        si_pbesol.mesh_numbers,
        si_pbesol.fc2,
        num_frequency_points=10)
    jdos.run([1, 103])
    # print(", ".join(["%.7f" % fp for fp in jdos.frequency_points]))
    np.testing.assert_allclose(freq_points, jdos.frequency_points,
                               atol=1e-5)
    # print(", ".join(["%.7f" % jd for jd in jdos.joint_dos.ravel()]))
    np.testing.assert_allclose(jdos_12, jdos.joint_dos.ravel(), atol=1e-2)
