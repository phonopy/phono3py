"""Pytest conftest.py."""
import os
import pytest
import phonopy
import phono3py
from phonopy.interface.phonopy_yaml import read_cell_yaml


current_dir = os.path.dirname(os.path.abspath(__file__))
store_dense_gp_map = False


@pytest.fixture(scope='session')
def agno2_cell():
    """Return AgNO2 cell (Imm2)."""
    cell = read_cell_yaml(os.path.join(current_dir, "AgNO2_cell.yaml"))
    return cell


@pytest.fixture(scope='session')
def si_pbesol():
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * full fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         store_dense_gp_map=store_dense_gp_map,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_nosym():
    """Return Phono3py instance of Si 2x2x2.

    * without symmetry
    * no fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         is_symmetry=False,
                         produce_fc=False,
                         store_dense_gp_map=store_dense_gp_map,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_nomeshsym():
    """Return Phono3py instance of Si 2x2x2.

    * without mesh-symmetry
    * no fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         is_mesh_symmetry=False,
                         produce_fc=False,
                         store_dense_gp_map=store_dense_gp_map,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_compact_fc():
    """Return Phono3py instance of Si 2x2x2.

    * with symmetry
    * compact fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         is_compact_fc=True,
                         store_dense_gp_map=store_dense_gp_map,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_111():
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * full fc

    """
    yaml_filename = os.path.join(current_dir, "phono3py_params_Si111.yaml")
    return phono3py.load(yaml_filename,
                         store_dense_gp_map=store_dense_gp_map,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_iterha_111():
    """Return Phono3py instance of Si 1x1x1.

    * with symmetry
    * no fc

    """
    yaml_filename = os.path.join(current_dir,
                                 "phonopy_params-Si111-iterha.yaml.gz")
    return phonopy.load(yaml_filename,
                        log_level=1,
                        produce_fc=False)


@pytest.fixture(scope='session')
def nacl_pbe():
    """Return Phono3py instance of NaCl 2x2x2.

    * with symmetry
    * compact fc

    """
    yaml_filename = os.path.join(current_dir,
                                 "phono3py_params_NaCl222.yaml.xz")
    return phono3py.load(yaml_filename,
                         store_dense_gp_map=store_dense_gp_map,
                         log_level=1)


@pytest.fixture(scope='session')
def nacl_pbe_cutoff_fc3():
    """Return Phono3py instance of NaCl 2x2x2.

    * cutoff pair with 5

    """
    yaml_filename = os.path.join(current_dir,
                                 "phono3py_params_NaCl222.yaml.xz")
    ph3 = phono3py.load(yaml_filename,
                        store_dense_gp_map=store_dense_gp_map,
                        produce_fc=False,
                        log_level=1)
    forces = ph3.forces
    ph3.generate_displacements(cutoff_pair_distance=5)
    dataset = ph3.dataset
    dataset['first_atoms'][0]['forces'] = forces[0]
    dataset['first_atoms'][1]['forces'] = forces[1]
    count = 2
    for first_atoms in dataset['first_atoms']:
        for second_atoms in first_atoms['second_atoms']:
            assert second_atoms['id'] == count + 1
            second_atoms['forces'] = forces[count]
            count += 1
    ph3.dataset = dataset
    ph3.produce_fc3(symmetrize_fc3r=True)
    return ph3


@pytest.fixture(scope='session')
def aln_lda():
    """Return Phono3py instance of AlN 3x3x2.

    * with symmetry
    * full fc.

    """
    yaml_filename = os.path.join(current_dir,
                                 "phono3py_params_AlN332.yaml.xz")
    return phono3py.load(yaml_filename,
                         store_dense_gp_map=store_dense_gp_map,
                         log_level=1)
