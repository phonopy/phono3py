import os
import pytest
import phonopy
import phono3py
from phonopy.interface.phonopy_yaml import read_cell_yaml


current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session')
def agno2_cell():
    cell = read_cell_yaml(os.path.join(current_dir, "AgNO2_cell.yaml"))
    return cell


@pytest.fixture(scope='session')
def si_pbesol():
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_nosym():
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         is_symmetry=False,
                         produce_fc=False,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_nomeshsym():
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         is_mesh_symmetry=False,
                         produce_fc=False,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_compact_fc():
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         is_compact_fc=True,
                         log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_111():
    yaml_filename = os.path.join(current_dir, "phono3py_params_Si111.yaml")
    return phono3py.load(yaml_filename, log_level=1)


@pytest.fixture(scope='session')
def si_pbesol_iterha_111():
    yaml_filename = os.path.join(current_dir,
                                 "phonopy_params-Si111-iterha.yaml.gz")
    return phonopy.load(yaml_filename, log_level=1, produce_fc=False)


@pytest.fixture(scope='session')
def nacl_pbe():
    yaml_filename = os.path.join(current_dir,
                                 "phono3py_params_NaCl222.yaml.xz")
    return phono3py.load(yaml_filename, log_level=1)


@pytest.fixture(scope='session')
def nacl_pbe_cutoff_fc3():
    yaml_filename = os.path.join(current_dir,
                                 "phono3py_params_NaCl222.yaml.xz")
    ph3 = phono3py.load(yaml_filename, log_level=1)
    forces = ph3.forces
    ph3.generate_displacements(cutoff_pair_distance=5)
    dataset = ph3.dataset
    dataset['first_atoms'][0]['forces'] = forces[0]
    dataset['first_atoms'][1]['forces'] = forces[0]
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
    yaml_filename = os.path.join(current_dir,
                                 "phono3py_params_AlN332.yaml.xz")
    return phono3py.load(yaml_filename, log_level=1)
