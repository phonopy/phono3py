"""Pytest conftest.py."""

import tarfile
from pathlib import Path

import numpy as np
import phonopy
import pytest
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms

import phono3py
from phono3py import Phono3py

cwd = Path(__file__).parent


def pytest_addoption(parser):
    """Activate v2 emulation  with --v2 option."""
    parser.addoption(
        "--v2",
        action="store_true",
        default=False,
        help="Run with phono3py v2.x emulation.",
    )


@pytest.fixture(scope="session")
def agno2_cell() -> PhonopyAtoms:
    """Return AgNO2 cell (Imm2)."""
    cell = read_cell_yaml(cwd / "AgNO2_cell.yaml")
    return cell


@pytest.fixture(scope="session")
def aln_cell() -> PhonopyAtoms:
    """Return AlN cell (P6_3mc)."""
    a = 3.111
    c = 4.978
    lattice = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    symbols = ["Al", "Al", "N", "N"]
    positions = [
        [1.0 / 3, 2.0 / 3, 0.0009488200000000],
        [2.0 / 3, 1.0 / 3, 0.5009488200000001],
        [1.0 / 3, 2.0 / 3, 0.6190511800000000],
        [2.0 / 3, 1.0 / 3, 0.1190511800000000],
    ]
    cell = PhonopyAtoms(cell=lattice, symbols=symbols, scaled_positions=positions)
    return cell


@pytest.fixture(scope="session")
def si_pbesol(request) -> Phono3py:
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * full fc

    """
    yaml_filename = cwd / "phono3py_si_pbesol.yaml"
    forces_fc3_filename = cwd / "FORCES_FC3_si_pbesol"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_grg(request) -> Phono3py:
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * full fc
    * GR-grid

    """
    yaml_filename = cwd / "phono3py_si_pbesol.yaml"
    forces_fc3_filename = cwd / "FORCES_FC3_si_pbesol"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        use_grg=True,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_nosym(request) -> Phono3py:
    """Return Phono3py instance of Si 2x2x2.

    * without symmetry
    * no fc

    """
    yaml_filename = cwd / "phono3py_si_pbesol.yaml"
    forces_fc3_filename = cwd / "FORCES_FC3_si_pbesol"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        is_symmetry=False,
        produce_fc=False,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_nomeshsym(request) -> Phono3py:
    """Return Phono3py instance of Si 2x2x2.

    * without mesh-symmetry
    * no fc

    """
    yaml_filename = cwd / "phono3py_si_pbesol.yaml"
    forces_fc3_filename = cwd / "FORCES_FC3_si_pbesol"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        is_mesh_symmetry=False,
        produce_fc=False,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_compact_fc(request) -> Phono3py:
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * compact fc

    """
    yaml_filename = cwd / "phono3py_si_pbesol.yaml"
    forces_fc3_filename = cwd / "FORCES_FC3_si_pbesol"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        forces_fc3_filename=forces_fc3_filename,
        is_compact_fc=True,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc

    """
    yaml_filename = cwd / "phono3py_params_Si111.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_symfc(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use symfc if available on test side

    """
    pytest.importorskip("symfc")

    yaml_filename = cwd / "phono3py_params_Si111.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        fc_calculator="symfc",
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_iterha_111() -> Phonopy:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * no fc

    """
    yaml_filename = cwd / "phonopy_params-Si111-iterha.yaml.gz"
    return phonopy.load(yaml_filename, log_level=1, produce_fc=False)


@pytest.fixture(scope="session")
def si_pbesol_111_222_fd(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use symfc if available on test side

    """
    yaml_filename = cwd / "phono3py_params_Si-111-222.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_222_symfc(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use symfc if available on test side

    """
    pytest.importorskip("symfc")

    yaml_filename = cwd / "phono3py_params_Si-111-222.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        fc_calculator="symfc",
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_222_symfc_fd(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use symfc for fc2 if available on test side

    """
    pytest.importorskip("symfc")

    yaml_filename = cwd / "phono3py_params_Si-111-222.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        fc_calculator="symfc|",
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_222_fd_symfc(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use symfc for fc3 if available on test side

    """
    pytest.importorskip("symfc")

    yaml_filename = cwd / "phono3py_params_Si-111-222.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        fc_calculator="|symfc",
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_222_alm_cutoff(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use alm if available on test side
    * cutoff=3

    """
    pytest.importorskip("alm")

    yaml_filename = cwd / "phono3py_params_Si-111-222.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        fc_calculator="alm",
        fc_calculator_options="cutoff = 3",
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_222_alm_cutoff_fc2(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use alm if available on test side
    * cutoff=3 only for fc2

    """
    pytest.importorskip("alm")

    yaml_filename = cwd / "phono3py_params_Si-111-222.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        fc_calculator="alm",
        fc_calculator_options="cutoff = 3|",
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_pbesol_111_222_alm_cutoff_fc3(request) -> Phono3py:
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc
    * use alm if available on test side
    * cutoff=3 only for fc3

    """
    pytest.importorskip("alm")

    yaml_filename = cwd / "phono3py_params_Si-111-222.yaml"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        fc_calculator="alm",
        fc_calculator_options="|cutoff = 3",
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def nacl_pbe(request) -> Phono3py:
    """Return Phono3py instance of NaCl 2x2x2.

    * with symmetry
    * full fc

    """
    yaml_filename = cwd / "phono3py_params_NaCl222.yaml.xz"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def nacl_pbe_compact_fc(request) -> Phono3py:
    """Return Phono3py instance of NaCl 2x2x2.

    * with symmetry
    * compact fc

    """
    yaml_filename = cwd / "phono3py_params_NaCl222.yaml.xz"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        is_compact_fc=True,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def nacl_pbe_cutoff_fc3(request) -> Phono3py:
    """Return Phono3py instance of NaCl 2x2x2 with cutoff-pair-distance.

    * cutoff pair with 5

    """
    yaml_filename = cwd / "phono3py_params_NaCl222.yaml.xz"
    enable_v2 = request.config.getoption("--v2")
    ph3 = phono3py.load(
        yaml_filename,
        produce_fc=False,
        make_r0_average=not enable_v2,
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
    ph3.produce_fc3()
    return ph3


@pytest.fixture(scope="session")
def nacl_pbe_cutoff_fc3_all_forces(request) -> Phono3py:
    """Return Phono3py instance of NaCl 2x2x2 with cutoff-pair-distance.

    * cutoff pair with 5
    * All forces are set.

    """
    yaml_filename = cwd / "phono3py_params_NaCl222.yaml.xz"
    enable_v2 = request.config.getoption("--v2")
    ph3 = phono3py.load(
        yaml_filename,
        produce_fc=False,
        make_r0_average=not enable_v2,
        log_level=1,
    )
    forces = ph3.forces
    ph3.generate_displacements(cutoff_pair_distance=5)
    ph3.forces = forces
    ph3.produce_fc3()
    return ph3


@pytest.fixture(scope="session")
def nacl_pbe_cutoff_fc3_compact_fc(request) -> Phono3py:
    """Return Phono3py instance of NaCl 2x2x2 with cutoff-pair-distance.

    * cutoff pair with 5
    * All forces are set.
    * Compact FC

    """
    yaml_filename = cwd / "phono3py_params_NaCl222.yaml.xz"
    enable_v2 = request.config.getoption("--v2")
    ph3 = phono3py.load(
        yaml_filename,
        produce_fc=False,
        make_r0_average=not enable_v2,
        log_level=1,
    )
    forces = ph3.forces
    ph3.generate_displacements(cutoff_pair_distance=5)
    ph3.forces = forces
    ph3.produce_fc3(is_compact_fc=True)
    return ph3


@pytest.fixture(scope="session")
def aln_lda(request) -> Phono3py:
    """Return Phono3py instance of AlN 3x3x2.

    * with symmetry
    * full fc.

    """
    yaml_filename = cwd / "phono3py_params_AlN332.yaml.xz"
    enable_v2 = request.config.getoption("--v2")
    return phono3py.load(
        yaml_filename,
        make_r0_average=not enable_v2,
        log_level=1,
    )


@pytest.fixture(scope="session")
def si_111_222_fd() -> Phono3py:
    """Return Phono3py class instance of Si-1x1x1-2x2x2 FD."""
    yaml_filename = cwd / "phono3py_params_Si-111-222-fd.yaml.xz"
    return phono3py.load(yaml_filename, produce_fc=False, log_level=1)


@pytest.fixture(scope="session")
def si_111_222_rd() -> Phono3py:
    """Return Phono3py class instance of Si-1x1x1-2x2x2 RD."""
    yaml_filename = cwd / "phono3py_params_Si-111-222-rd.yaml.xz"
    return phono3py.load(yaml_filename, produce_fc=False, log_level=1)


@pytest.fixture(scope="session")
def mgo_222rd_444rd() -> Phono3py:
    """Return Phono3py class instance of MgO-2x2x2-4x4x4 RD-RD.

    4 and 400 supercells for fc2 and fc3, respectively.

    """
    yaml_filename = cwd / "phono3py_params_MgO-222rd-444rd.yaml.xz"
    return phono3py.load(yaml_filename, produce_fc=False, log_level=1)


@pytest.fixture(scope="session")
def ph_nacl() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    born_filename = cwd / "BORN_NaCl"
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
    yaml_filename = cwd / "phonopy_params_Si.yaml"
    return phonopy.load(
        yaml_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def si_111_222_fd_raw_data() -> tarfile.TarFile:
    """Return Si fc3 111 fc2 222 vasp inputs.

    tar.getnames()
    ['Si-111-222-fd',
     'Si-111-222-fd/vasprun-001.xml',
     'Si-111-222-fd/vasprun-015.xml',
     'Si-111-222-fd/vasprun-014.xml',
     'Si-111-222-fd/vasprun-000.xml',
     'Si-111-222-fd/vasprun-016.xml',
     'Si-111-222-fd/vasprun-002.xml',
     'Si-111-222-fd/vasprun-003.xml',
     'Si-111-222-fd/vasprun-013.xml',
     'Si-111-222-fd/vasprun-007.xml',
     'Si-111-222-fd/vasprun-fc2-000.xml',
     'Si-111-222-fd/vasprun-fc2-001.xml',
     'Si-111-222-fd/vasprun-006.xml',
     'Si-111-222-fd/vasprun-012.xml',
     'Si-111-222-fd/vasprun-004.xml',
     'Si-111-222-fd/vasprun-010.xml',
     'Si-111-222-fd/vasprun-011.xml',
     'Si-111-222-fd/vasprun-005.xml',
     'Si-111-222-fd/phono3py_disp.yaml',
     'Si-111-222-fd/vasprun-008.xml',
     'Si-111-222-fd/vasprun-009.xml']
    member = tar.getmember("Si-111-222-fd/phono3py_disp.yaml")
    tar.extractfile(member)  -> byte file object

    """
    tar = tarfile.open(cwd / "Si-111-222-fd.tar.xz")
    return tar


@pytest.fixture(scope="session")
def si_111_222_rd_raw_data() -> tarfile.TarFile:
    """Return Si fc3 111 fc2 222 vasp inputs.

    tar.getnames()
    ['Si-111-222-rd',
     'Si-111-222-rd/vasprun-001.xml',
     'Si-111-222-rd/vasprun-015.xml',
     'Si-111-222-rd/vasprun-014.xml',
     'Si-111-222-rd/vasprun-000.xml',
     'Si-111-222-rd/vasprun-016.xml',
     'Si-111-222-rd/vasprun-002.xml',
     'Si-111-222-rd/vasprun-003.xml',
     'Si-111-222-rd/vasprun-017.xml',
     'Si-111-222-rd/vasprun-013.xml',
     'Si-111-222-rd/vasprun-007.xml',
     'Si-111-222-rd/vasprun-fc2-000.xml',
     'Si-111-222-rd/vasprun-fc2-001.xml',
     'Si-111-222-rd/vasprun-006.xml',
     'Si-111-222-rd/vasprun-012.xml',
     'Si-111-222-rd/vasprun-004.xml',
     'Si-111-222-rd/vasprun-010.xml',
     'Si-111-222-rd/vasprun-fc2-002.xml',
     'Si-111-222-rd/vasprun-011.xml',
     'Si-111-222-rd/vasprun-005.xml',
     'Si-111-222-rd/phono3py_disp.yaml',
     'Si-111-222-rd/vasprun-020.xml',
     'Si-111-222-rd/vasprun-008.xml',
     'Si-111-222-rd/vasprun-009.xml',
     'Si-111-222-rd/vasprun-019.xml',
     'Si-111-222-rd/vasprun-018.xml']
    member = tar.getmember("Si-111-222-rd/phono3py_disp.yaml")
    tar.extractfile(member)  -> byte file object

    """
    tar = tarfile.open(cwd / "Si-111-222-fd.tar.xz")
    return tar
