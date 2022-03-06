"""Tests for velocity operator."""
import numpy as np
import pytest
from phonopy import Phonopy

from phono3py.phonon.velocity_operator import VelocityOperator


@pytest.mark.skipif(True, reason="waiting for being fixed.")
def test_gv_operator_nacl(ph_nacl: Phonopy):
    """Test of GroupVelocity by NaCl."""
    gv_operator_square_modulus_ref = {}
    # direction 0
    gv_operator_square_modulus_ref[0] = np.array(
        [
            [
                2.42970100e02,
                -6.20236119e00,
                -1.82999046e-01,
                -2.09295390e-01,
                4.07782967e01,
                2.10142118e-01,
            ],
            [
                -6.20236119e00,
                2.73261377e02,
                -3.71620328e02,
                8.44686890e-01,
                -1.61985084e00,
                -5.01748557e00,
            ],
            [
                -1.82999046e-01,
                -3.71620328e02,
                1.03489135e03,
                -2.58482318e01,
                1.88975807e-02,
                1.33902037e01,
            ],
            [
                -2.09295390e-01,
                8.44686890e-01,
                -2.58482318e01,
                1.30173952e00,
                -1.56714339e-01,
                3.74138211e00,
            ],
            [
                4.07782967e01,
                -1.61985084e00,
                1.88975807e-02,
                -1.56714339e-01,
                1.07994337e01,
                -2.56681660e00,
            ],
            [
                2.10142118e-01,
                -5.01748557e00,
                1.33902037e01,
                3.74138211e00,
                -2.56681660e00,
                5.34677654e02,
            ],
        ]
    )

    # direction 1
    gv_operator_square_modulus_ref[1] = np.array(
        [
            [190.26961913, 24.42080327, 18.66103712, 3.499692, 17.12151154, 7.82170714],
            [
                24.42080327,
                753.95642101,
                449.00523399,
                56.90864201,
                -3.00513131,
                178.25703584,
            ],
            [
                18.66103712,
                449.00523399,
                282.90229148,
                124.96202594,
                -3.1864197,
                46.43438272,
            ],
            [
                3.499692,
                56.90864201,
                124.96202594,
                628.61784617,
                -13.41223828,
                -240.42545741,
            ],
            [
                17.12151154,
                -3.00513131,
                -3.1864197,
                -13.41223828,
                4.49707039,
                2.33350832,
            ],
            [
                7.82170714,
                178.25703584,
                46.43438272,
                -240.42545741,
                2.33350832,
                394.18935848,
            ],
        ]
    )
    # direction 2
    gv_operator_square_modulus_ref[2] = np.array(
        [
            [
                3.21847047e02,
                -2.87032091e01,
                -2.45808366e01,
                -1.57578343e00,
                -1.32856444e02,
                -1.34389001e01,
            ],
            [
                -2.87032091e01,
                5.54329513e01,
                1.16917174e02,
                -3.18740464e00,
                8.04928014e00,
                6.19917390e01,
            ],
            [
                -2.45808366e01,
                1.16917174e02,
                2.55127746e02,
                5.66979440e-01,
                9.79228317e00,
                7.96414618e01,
            ],
            [
                -1.57578343e00,
                -3.18740464e00,
                5.66979440e-01,
                1.61522304e01,
                1.98704126e01,
                -1.14635470e02,
            ],
            [
                -1.32856444e02,
                8.04928014e00,
                9.79228317e00,
                1.98704126e01,
                8.60482934e02,
                -1.68196151e01,
            ],
            [
                -1.34389001e01,
                6.19917390e01,
                7.96414618e01,
                -1.14635470e02,
                -1.68196151e01,
                8.58940749e02,
            ],
        ]
    )

    gv_operator = VelocityOperator(
        ph_nacl.dynamical_matrix, symmetry=ph_nacl.primitive_symmetry
    )
    # we chose an 'ugly' q-point because we want to avoid degeneracies.
    # degeneracies are tested in phono3py
    gv_operator.run([[0.1, 0.22, 0.33]])
    square_modulus = np.zeros((6, 6, 3))
    for direction in range(2, 3):
        square_modulus[:, :, direction] = np.matmul(
            gv_operator.velocity_operators[0][:, :, direction],
            gv_operator.velocity_operators[0][:, :, direction].conjugate().T,
        ).real
        np.testing.assert_allclose(
            square_modulus[:, :, direction].ravel(),
            gv_operator_square_modulus_ref[direction].ravel(),
            atol=1e-4,
        )


@pytest.mark.skipif(True, reason="waiting for being fixed.")
def test_gv_operator_si(ph_si: Phonopy):
    """Test of GroupVelocity by Si."""
    gv_operator_square_modulus_ref = {}
    # direction 0
    gv_operator_square_modulus_ref[0] = np.array(
        [
            [
                2.73572808e03,
                -4.06210738e00,
                1.33571471e03,
                -1.11264018e00,
                -2.72825378e02,
                3.23442510e-01,
            ],
            [
                -4.06210738e00,
                2.72065300e03,
                -3.20535769e01,
                -4.41527949e02,
                -1.29661972e01,
                5.12752451e02,
            ],
            [
                1.33571471e03,
                -3.20535769e01,
                4.16058788e03,
                -1.19524446e00,
                -1.26829836e01,
                -1.15732958e00,
            ],
            [
                -1.11264018e00,
                -4.41527949e02,
                -1.19524446e00,
                1.45914531e03,
                1.90180987e00,
                -8.45069100e02,
            ],
            [
                -2.72825378e02,
                -1.29661972e01,
                -1.26829836e01,
                1.90180987e00,
                1.78584698e03,
                -2.25355370e-01,
            ],
            [
                3.23442510e-01,
                5.12752451e02,
                -1.15732958e00,
                -8.45069100e02,
                -2.25355370e-01,
                7.25820137e02,
            ],
        ]
    )

    # direction 1
    gv_operator_square_modulus_ref[1] = np.array(
        [
            [
                1.95698890e03,
                -8.55308045e00,
                -6.97133428e02,
                8.21442883e-01,
                -1.08587694e03,
                -9.83370396e-02,
            ],
            [
                -8.55308045e00,
                2.20579768e03,
                4.90281444e00,
                -2.83807879e02,
                3.01470115e00,
                -7.45274905e02,
            ],
            [
                -6.97133428e02,
                4.90281444e00,
                2.96863421e03,
                -2.26085394e-01,
                -7.66976692e00,
                2.51720167e-01,
            ],
            [
                8.21442883e-01,
                -2.83807879e02,
                -2.26085394e-01,
                4.24208287e02,
                6.40518671e-03,
                4.96247028e02,
            ],
            [
                -1.08587694e03,
                3.01470115e00,
                -7.66976692e00,
                6.40518671e-03,
                8.39180146e02,
                -8.39817213e-01,
            ],
            [
                -9.83370396e-02,
                -7.45274905e02,
                2.51720167e-01,
                4.96247028e02,
                -8.39817213e-01,
                7.45082308e02,
            ],
        ]
    )

    # direction 2
    gv_operator_square_modulus_ref[2] = np.array(
        [
            [
                1.53807537e03,
                -3.89302418e00,
                -1.75424484e02,
                -1.65219640e00,
                1.31051315e02,
                2.03331851e-02,
            ],
            [
                -3.89302418e00,
                2.57583012e03,
                1.04592533e01,
                -9.42612420e02,
                -1.42765014e01,
                3.33706787e01,
            ],
            [
                -1.75424484e02,
                1.04592533e01,
                2.92924900e03,
                -8.53691987e-02,
                -2.44376137e02,
                -1.00559293e00,
            ],
            [
                -1.65219640e00,
                -9.42612420e02,
                -8.53691987e-02,
                4.18562364e02,
                -3.54778535e-01,
                -1.90266774e02,
            ],
            [
                1.31051315e02,
                -1.42765014e01,
                -2.44376137e02,
                -3.54778535e-01,
                4.36890621e01,
                6.82908650e-01,
            ],
            [
                2.03331851e-02,
                3.33706787e01,
                -1.00559293e00,
                -1.90266774e02,
                6.82908650e-01,
                1.51072731e03,
            ],
        ]
    )

    gv_operator = VelocityOperator(
        ph_si.dynamical_matrix, symmetry=ph_si.primitive_symmetry
    )
    gv_operator.run([[0.1, 0.22, 0.33]])
    square_modulus = np.zeros((6, 6, 3))
    for direction in range(2, 3):
        square_modulus[:, :, direction] = np.matmul(
            gv_operator.velocity_operators[0][:, :, direction],
            gv_operator.velocity_operators[0][:, :, direction].conj().T,
        )
        np.testing.assert_allclose(
            square_modulus[:, :, direction].ravel(),
            gv_operator_square_modulus_ref[direction].ravel(),
            atol=1e-5,
        )
