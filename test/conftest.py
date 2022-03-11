"""Pytest conftest.py."""
import os

import phonopy
import pytest
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml

import phono3py

current_dir = os.path.dirname(os.path.abspath(__file__))
store_dense_gp_map = True


def pytest_addoption(parser):
    """Activate v1 emulation  with --v1 option."""
    parser.addoption(
        "--v1",
        action="store_false",
        default=True,
        help="Run with phono3py v1.x emulation.",
    )


@pytest.fixture(scope="session")
def agno2_cell():
    """Return AgNO2 cell (Imm2)."""
    cell = read_cell_yaml(os.path.join(current_dir, "AgNO2_cell.yaml"))
    return cell


@pytest.fixture(scope="session")
def si_pbesol(request):
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * full fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_grg(request):
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * full fc
    * GR-grid

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        use_grg=True,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_nosym(request):
    """Return Phono3py instance of Si 2x2x2.

    * without symmetry
    * no fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        is_symmetry=False,
        produce_fc=False,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_nomeshsym(request):
    """Return Phono3py instance of Si 2x2x2.

    * without mesh-symmetry
    * no fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        is_mesh_symmetry=False,
        produce_fc=False,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_compact_fc(request):
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * compact fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        is_compact_fc=True,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111(request):
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_params_Si111.yaml")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_alm(request):
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use alm if available on test side

    """
    pytest.importorskip("alm")

    yaml_filename = os.path.join(current_dir, "phono3py_params_Si111.yaml")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        fc_calculator="alm",
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_iterha_111():
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * no fc

    """
    yaml_filename = os.path.join(current_dir, "phonopy_params-Si111-iterha.yaml.gz")
    return phonopy.load(yaml_filename, log_level=1, produce_fc=False)


@pytest.fixture(scope="session")
def nacl_pbe(request):
    """Return Phono3py instance of NaCl 2x2x2.

    * with symmetry
    * compact fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_params_NaCl222.yaml.xz")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def nacl_pbe_cutoff_fc3(request):
    """Return Phono3py instance of NaCl 2x2x2.

    * cutoff pair with 5

    """
    yaml_filename = os.path.join(current_dir, "phono3py_params_NaCl222.yaml.xz")
    enable_v2 = request.config.getoption("--v1")
    ph3 = phono3py.load(
        yaml_filename,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        produce_fc=False,
        log_level=1,
    )
    forces = ph3.forces
    ph3.generate_displacements(cutoff_pair_distance=5)
    dataset = ph3.dataset
    dataset["first_atoms"][0]["forces"] = forces[0]
    dataset["first_atoms"][1]["forces"] = forces[1]
    count = 2
    for first_atoms in dataset["first_atoms"]:
        for second_atoms in first_atoms["second_atoms"]:
            assert second_atoms["id"] == count + 1
            second_atoms["forces"] = forces[count]
            count += 1
    ph3.dataset = dataset
    ph3.produce_fc3(symmetrize_fc3r=True)
    return ph3


@pytest.fixture(scope="session")
def aln_lda(request):
    """Return Phono3py instance of AlN 3x3x2.

    * with symmetry
    * full fc.

    """
    yaml_filename = os.path.join(current_dir, "phono3py_params_AlN332.yaml.xz")
    enable_v2 = request.config.getoption("--v1")
    return phono3py.load(
        yaml_filename,
        store_dense_gp_map=enable_v2,
        store_dense_svecs=enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def ph_nacl() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2."""
    yaml_filename = os.path.join(current_dir, "phonopy_disp_NaCl.yaml")
    force_sets_filename = os.path.join(current_dir, "FORCE_SETS_NaCl")
    born_filename = os.path.join(current_dir, "BORN_NaCl")
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        born_filename=born_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_si() -> Phonopy:
    """Return Phonopy class instance of Si-prim 2x2x2."""
    yaml_filename = os.path.join(current_dir, "phonopy_params_Si.yaml")
    return phonopy.load(
        yaml_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )
