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
