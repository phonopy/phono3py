"""Tests for SSCHA routines."""

import numpy as np
import pytest
from phonopy.phonon.qpoints import QpointsPhonon
from phonopy.phonon.random_displacements import RandomDisplacements

from phono3py.sscha.sscha import (
    DispCorrMatrix,
    DispCorrMatrixMesh,
    SupercellPhonon,
    ThirdOrderFC,
    get_sscha_matrices,
)

si_pbesol_upsilon0_0 = [[3.849187e02, 0, 0], [0, 3.849187e02, 0], [0, 0, 3.849187e02]]
si_pbesol_upsilon1_34 = [
    [1.886404, -1.549705, -1.126055],
    [-1.549705, 1.886404, -1.126055],
    [-1.126055, -1.126055, -0.006187],
]
si_pbesol_111_freqs = [
    0.00000,
    0.00000,
    0.00000,
    4.02839,
    4.02839,
    4.02839,
    4.02839,
    4.02839,
    4.02839,
    12.13724,
    12.13724,
    12.13724,
    12.13724,
    12.13724,
    12.13724,
    13.71746,
    13.71746,
    13.71746,
    13.71746,
    13.71746,
    13.71746,
    15.24974,
    15.24974,
    15.24974,
]


def get_supercell_phonon(ph3):
    """Return SupercellPhonon class instance."""
    ph3.mesh_numbers = [1, 1, 1]
    ph3.init_phph_interaction()
    fc2 = ph3.dynamical_matrix.force_constants
    supercell = ph3.phonon_supercell
    factor = ph3.unit_conversion_factor
    return SupercellPhonon(supercell, fc2, frequency_factor_to_THz=factor)


def mass_sand(matrix, mass):
    """Calculate mass sandwich."""
    return ((matrix * mass).T * mass).T


def mass_inv(matrix, mass):
    """Calculate inverse mass sandwich."""
    bare = mass_sand(matrix, mass)
    inv_bare = np.linalg.pinv(bare)
    return mass_sand(inv_bare, mass)


def test_SupercellPhonon(si_pbesol_111):
    """Test of SupercellPhonon class."""
    sph = get_supercell_phonon(si_pbesol_111)
    np.testing.assert_allclose(si_pbesol_111_freqs, sph.frequencies, atol=1e-4)


def test_disp_corr_matrix_mesh(si_pbesol):
    """Test of DispCorrMatrixMesh class."""
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    dynmat = si_pbesol.dynamical_matrix
    uu = DispCorrMatrixMesh(dynmat.primitive, dynmat.supercell)
    qpoints_phonon = QpointsPhonon(
        uu.commensurate_points, dynmat, with_eigenvectors=True
    )
    freqs = qpoints_phonon.frequencies
    eigvecs = qpoints_phonon.eigenvectors
    uu.run(freqs, eigvecs, 300.0)
    np.testing.assert_allclose(
        si_pbesol_upsilon0_0, uu.upsilon_matrix[0:3, 0:3], atol=1e-4
    )
    np.testing.assert_allclose(
        si_pbesol_upsilon1_34,
        uu.upsilon_matrix[1 * 3 : 2 * 3, 34 * 3 : 35 * 3],
        atol=1e-4,
    )

    sqrt_masses = np.repeat(np.sqrt(si_pbesol.supercell.masses), 3)
    uu_inv = mass_inv(uu.psi_matrix, sqrt_masses)
    np.testing.assert_allclose(uu.upsilon_matrix, uu_inv, atol=1e-8, rtol=0)


def test_disp_corr_matrix(si_pbesol):
    """Test of DispCorrMatrix class."""
    supercell_phonon = get_supercell_phonon(si_pbesol)
    uu = DispCorrMatrix(supercell_phonon)
    uu.run(300.0)
    np.testing.assert_allclose(
        si_pbesol_upsilon0_0, uu.upsilon_matrix[0:3, 0:3], atol=1e-4
    )
    np.testing.assert_allclose(
        si_pbesol_upsilon1_34,
        uu.upsilon_matrix[1 * 3 : 2 * 3, 34 * 3 : 35 * 3],
        atol=1e-4,
    )


def test_get_sscha_matrices(si_pbesol_111):
    """Test of get_sscha_matrices.

    Prefactor is compared with sqrt(1/det(Upsilon / M / 2pi)).

    """
    rd_300 = [
        -0.006539522224431159,
        0.1112911766258876,
        -0.04247705573972028,
        0.06605246122112128,
        0.04912371642641159,
        -0.1144800479882498,
        -0.009140217856930866,
        -0.14945291811302244,
        -0.013929568255205775,
        -0.1178444426978898,
        -0.028168641379735657,
        0.10446758683943788,
        0.048510589548951134,
        -0.0820485562356512,
        -0.00031712163840745566,
        -0.056498756168815396,
        0.01717764755513093,
        0.07364167633689502,
        -0.02453770974517977,
        0.1033693900525631,
        0.025644322937051583,
        0.09999759792317454,
        -0.021291814931583906,
        -0.032549792491801156,
        0.04778800530000226,
        0.04694656109627236,
        -0.016231937830557045,
        -0.008373790536921609,
        0.08591652849959887,
        -0.07355907350291352,
        0.019918420826451032,
        -0.002903973626837904,
        0.057544764529824856,
        -0.013670020504842999,
        -0.08801538847559207,
        0.0558986494606823,
        0.014090447119130494,
        -0.1424846038374843,
        0.04676826633743707,
        -0.0214536168504123,
        -0.0256455553870123,
        0.030008606106752298,
        0.02459736136675902,
        0.11381723151838435,
        0.02685301121629841,
        -0.06289680672016591,
        0.012369200212670935,
        -0.12728228631752436,
    ]
    rd_400 = [
        -0.03733576946420855,
        0.04232449735821845,
        0.07294448380484658,
        0.0745821832022657,
        -0.016359814478840513,
        -0.017525333263806897,
        -0.06193675904650906,
        -0.017377926164506056,
        0.01621807801223652,
        0.13476445234723203,
        0.05554389001553833,
        -0.04759177991765211,
        0.04575359046588339,
        0.04562186042469192,
        -0.06396674924341537,
        -0.09392771136887822,
        0.02642624337715112,
        0.014504150286882218,
        0.06107814609935116,
        -0.03949940651501436,
        -0.08898878250385281,
        -0.12297813223513647,
        -0.09667934401723893,
        0.11440593282476184,
        -0.010889706123501479,
        -0.055155392917289325,
        -0.06541419287046933,
        0.0359941397032081,
        0.09612166473863022,
        0.05598952872736431,
        0.08481014701143275,
        0.06420526258056838,
        0.01673404598974324,
        -0.08631437337840182,
        -0.034024949306803956,
        -0.055104092790023464,
        -0.08905675321168185,
        -0.020950621886958537,
        0.0030113320537691144,
        0.05718359487674082,
        0.013079609610128471,
        -0.0038377175980778863,
        0.06297054138192276,
        0.029490841183290026,
        0.024382074574234348,
        -0.05469759025971933,
        -0.09276641400156528,
        0.024239021913459665,
    ]
    prob_300 = [0.0003802683925260675, 0.13450413298841105]
    prob_400 = [0.006465377592145493, 0.14352136412319727]
    n_snapshots = 2

    ph3 = si_pbesol_111
    uu = get_sscha_matrices(ph3.supercell, ph3.fc2)
    # _rd = RandomDisplacements(ph3.supercell, ph3.primitive, ph3.fc2)
    for temp, prob_ref, rd in zip((300, 400), (prob_300, prob_400), (rd_300, rd_400)):
        uu.run(temp)
        # _rd.run(temp, number_of_snapshots=n_snapshots)
        # dmat = _rd.u.reshape(n_snapshots, -1)
        dmat = np.reshape(rd, (n_snapshots, -1))
        vals = -(dmat * np.dot(dmat, uu.upsilon_matrix)).sum(axis=1) / 2
        prob = uu.prefactor * np.exp(vals)
        inv_sqrt_masses = np.repeat(1.0 / np.sqrt(ph3.supercell.masses), 3)
        umat = inv_sqrt_masses * (inv_sqrt_masses * uu.upsilon_matrix).T
        eigs = np.linalg.eigvalsh(umat)
        prefactor = np.sqrt(np.prod(np.extract(eigs > 1e-5, eigs) / np.pi / 2))

        assert prefactor / uu.prefactor == pytest.approx(1, 1e-3)
        np.testing.assert_allclose(prob, prob_ref, atol=0, rtol=1e-07)


def test_disp_corr_matrix_si(si_pbesol):
    """Test of DispCorrMatrix class with Si."""
    _test_disp_corr_matrix(si_pbesol)


def test_disp_corr_matrix_nacl(nacl_pbe):
    """Test of DispCorrMatrix class with NaCl."""
    _test_disp_corr_matrix(nacl_pbe)


def _test_disp_corr_matrix(ph3):
    supercell_phonon = get_supercell_phonon(ph3)
    uu = DispCorrMatrix(supercell_phonon)
    uu.run(300.0)

    sqrt_masses = np.repeat(np.sqrt(ph3.supercell.masses), 3)
    uu_inv = mass_inv(uu.psi_matrix, sqrt_masses)
    np.testing.assert_allclose(uu.upsilon_matrix, uu_inv, atol=1e-8, rtol=0)

    rd = RandomDisplacements(ph3.supercell, ph3.primitive, ph3.fc2)
    rd.run_correlation_matrix(300)
    rd_uu_inv = np.transpose(rd.uu_inv, axes=[0, 2, 1, 3]).reshape(uu_inv.shape)
    np.testing.assert_allclose(uu.upsilon_matrix, rd_uu_inv, atol=1e-8, rtol=0)


def test_fc3(si_pbesol_iterha_111):
    """Test of ThirdOrderFC class."""
    pytest.importorskip("symfc")

    ph = si_pbesol_iterha_111
    ph.produce_force_constants(
        calculate_full_force_constants=True, fc_calculator="symfc"
    )
    supercell_phonon = SupercellPhonon(
        ph.supercell,
        ph.force_constants,
        frequency_factor_to_THz=ph.unit_conversion_factor,
    )
    fc3 = ThirdOrderFC(ph.displacements, ph.forces, supercell_phonon)
    fc3.run(T=300)
