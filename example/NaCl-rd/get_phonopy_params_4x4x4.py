import phonopy

import phono3py

ph3 = phono3py.load("phono3py_params_NaCl.yaml.xz", produce_fc=False, log_level=2)
ph = phonopy.Phonopy(
    unitcell=ph3.unitcell,
    supercell_matrix=ph3.phonon_supercell_matrix,
    primitive_matrix=ph3.primitive_matrix,
)
ph.dataset = ph3.phonon_dataset
ph.nac_params = ph3.nac_params
ph.save("phonopy_params_NaCl.yaml")
