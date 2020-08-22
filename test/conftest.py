import pytest
import phono3py


@pytest.fixture(scope='session')
def si_pbesol():
    return phono3py.load("phono3py_si_pbesol.yaml",
                         forces_fc3_filename="FORCES_FC3_si_pbesol",
                         log_level=1)
