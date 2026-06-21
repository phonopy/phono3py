"""Regression tests for inter-band (coherence) transport by NaMgF3.

NaMgF3 (orthorhombic perovskite, 20 atoms / 60 bands) has many phonon
bands close in frequency, so the inter-band (off-diagonal) contribution to
the lattice thermal conductivity is significant (~19% of the total here).
This makes it a meaningful test of the inter-band variants NJC23, IBDB19,
and SMM19, unlike the small-cell Si/NaCl cases.

The scattering linewidths (gamma) are read from a bundled kappa hdf5
(``--read-gamma``), so fc3 is not needed and the computation is fast. Only
fc2 (for the dynamical matrix, frequencies, and velocity matrix) is loaded.

"""

import os
import pathlib
import shutil

import numpy as np
import pytest

import phono3py

data_dir = pathlib.Path(__file__).parent
mesh = [9, 7, 9]

# Reference kappa at 300 K for mesh 9x7x9 (xx, yy, zz) in W/m-K.
# The diagonal (intra) part is identical for all three variants.
ref_intra = [3.022, 3.230, 3.234]
ref_inter = {
    "NJC23": [0.702, 0.746, 0.740],
    "IBDB19": [0.698, 0.742, 0.736],
    "SMM19": [0.702, 0.746, 0.741],
}
TOLERANCE = 0.05


@pytest.fixture(scope="module")
def namgf3_kappa(tmp_path_factory) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run NJC23/IBDB19/SMM19 RTA for NaMgF3 with read_gamma.

    The bundled gamma file is copied into a temporary working directory
    under the mesh-convention name ``kappa-m979.hdf5`` expected by
    ``--read-gamma``, and the directory is used as the working directory
    while reading.

    Returns
    -------
    dict
        Mapping transport_type -> (kappa_intra, kappa_inter), each raveled
        to a length-6 Voigt vector.

    """
    workdir = tmp_path_factory.mktemp("namgf3")
    shutil.copy(data_dir / "kappa-NaMgF3-m979.hdf5", workdir / "kappa-m979.hdf5")

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        ph3 = phono3py.load(
            str(data_dir / "phono3py_params_NaMgF3.yaml.xz"),
            fc2_filename=str(data_dir / "fc2_NaMgF3.hdf5"),
            log_level=0,
        )
        ph3.mesh_numbers = mesh
        ph3.init_phph_interaction()
        results = {}
        for transport_type in ("NJC23", "IBDB19", "SMM19"):
            ph3.run_thermal_conductivity(
                temperatures=[300],
                read_gamma=True,
                transport_type=transport_type,
            )
            tc = ph3.thermal_conductivity
            results[transport_type] = (
                tc.kappa_intra.ravel().copy(),
                tc.kappa_inter.ravel().copy(),
            )
    finally:
        os.chdir(cwd)
    return results


@pytest.mark.parametrize("transport_type", ["NJC23", "IBDB19", "SMM19"])
def test_namgf3_kappa_intra(namgf3_kappa, transport_type: str):
    """Diagonal (intra-band) kappa is the same for all three variants."""
    kappa_intra = namgf3_kappa[transport_type][0]
    np.testing.assert_allclose(ref_intra, kappa_intra[:3], atol=TOLERANCE)
    np.testing.assert_allclose([0, 0, 0], kappa_intra[3:], atol=TOLERANCE)


@pytest.mark.parametrize("transport_type", ["NJC23", "IBDB19", "SMM19"])
def test_namgf3_kappa_inter(namgf3_kappa, transport_type: str):
    """Off-diagonal (inter-band) kappa matches the reference values."""
    kappa_inter = namgf3_kappa[transport_type][1]
    np.testing.assert_allclose(
        ref_inter[transport_type], kappa_inter[:3], atol=TOLERANCE
    )
    np.testing.assert_allclose([0, 0, 0], kappa_inter[3:], atol=TOLERANCE)


def test_namgf3_kappa_inter_ordering(namgf3_kappa):
    """IBDB19 inter-band kappa is below NJC23 and SMM19, which agree.

    The variants differ only in the heat capacity matrix prefactor:
    IBDB19 uses the geometric-mean-square ``w_j w_j'`` while NJC23/SMM19
    use the arithmetic-mean-square ``(w_j + w_j')^2 / 4``. Since
    ``w_j w_j' <= (w_j + w_j')^2 / 4`` with equality only on the diagonal,
    IBDB19 gives a strictly smaller off-diagonal contribution.

    """
    inter_njc23 = namgf3_kappa["NJC23"][1][:3]
    inter_ibdb19 = namgf3_kappa["IBDB19"][1][:3]
    inter_smm19 = namgf3_kappa["SMM19"][1][:3]

    assert np.all(inter_ibdb19 < inter_njc23)
    assert np.all(inter_ibdb19 < inter_smm19)
    np.testing.assert_allclose(inter_njc23, inter_smm19, atol=0.01)
