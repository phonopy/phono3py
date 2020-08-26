import os
import pytest
import phonopy
import phono3py

current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session')
def si_pbesol():
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
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
