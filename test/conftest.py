import os
import pytest
import phono3py

current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session')
def si_pbesol():
    yaml_filename = os.path.join(current_dir, "phono3py_si_pbesol.yaml")
    forces_fc3_filename = os.path.join(current_dir, "FORCES_FC3_si_pbesol")
    return phono3py.load(yaml_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         log_level=1)
