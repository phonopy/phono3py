import sys

import phono3py

ph3 = phono3py.load(sys.argv[1], produce_fc=False, log_level=1)
ph3_new = phono3py.Phono3py(
    ph3.unitcell,
    supercell_matrix=ph3.supercell_matrix,
    primitive_matrix=ph3.primitive_matrix,
)
ph3_new.dataset = ph3.dataset
ph3_new.nac_params = ph3.nac_params
ph3_new.save("phono3py_params.yaml")
