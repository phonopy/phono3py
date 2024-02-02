"""Launch script of ZnTe AiiDA calculation using aiida-phononpy."""

from aiida.engine import submit
from aiida.manage.configuration import load_profile
from aiida.orm import Bool, Float, Str
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_phonopy.common.utils import phonopy_atoms_to_structure
from phonopy.interface.vasp import read_vasp_from_strings

load_profile()

Dict = DataFactory("dict")


def get_settings(cutoff_energy, is_nac=False):
    """Set up parameters."""
    unitcell_str = """  Zn Te
   1.0
     6.0653118499999996    0.0000000000000000    0.0000000000000000
     0.0000000000000000    6.0653118499999996    0.0000000000000000
     0.0000000000000000    0.0000000000000000    6.0653118499999996
 Zn Te
   4   4
Direct
   0.0000000000000000  0.0000000000000000  0.0000000000000000
   0.0000000000000000  0.5000000000000000  0.5000000000000000
   0.5000000000000000  0.0000000000000000  0.5000000000000000
   0.5000000000000000  0.5000000000000000  0.0000000000000000
   0.2500000000000000  0.2500000000000000  0.7500000000000000
   0.2500000000000000  0.7500000000000000  0.2500000000000000
   0.7500000000000000  0.2500000000000000  0.2500000000000000
   0.7500000000000000  0.7500000000000000  0.7500000000000000"""

    cell = read_vasp_from_strings(unitcell_str)
    structure = phonopy_atoms_to_structure(cell)

    base_incar_dict = {
        "PREC": "Accurate",
        "IBRION": -1,
        "EDIFF": 1e-8,
        "NELMIN": 5,
        "NELM": 100,
        "ENCUT": cutoff_energy,
        "IALGO": 38,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "GGA": "PS",
        "LREAL": False,
        "lcharg": False,
        "lwave": False,
    }

    code_string = "vasp544mpi@nancy"
    resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}

    base_config = {
        "code_string": code_string,
        "potential_family": "PBE.54",
        "potential_mapping": {"Zn": "Zn", "Te": "Te"},
        "options": {"resources": resources, "max_wallclock_seconds": 3600 * 10},
    }
    base_parser_settings = {
        "add_energies": True,
        "add_forces": True,
        "add_stress": True,
    }
    forces_config = base_config.copy()
    kpoints_mesh = [2, 2, 2]
    forces_config.update(
        {
            "kpoints_mesh": kpoints_mesh,
            "kpoints_offset": [0.5, 0.5, 0.5],
            "parser_settings": base_parser_settings,
            "parameters": {"incar": base_incar_dict.copy()},
        }
    )
    forces_config["parameters"]["incar"]["NPAR"] = 4
    nac_config = {
        "code_string": code_string,
        "potential_family": "PBE.54",
        "potential_mapping": {"Zn": "Zn", "Te": "Te"},
        "options": {"resources": resources, "max_wallclock_seconds": 3600 * 10},
    }
    nac_parser_settings = {"add_born_charges": True, "add_dielectrics": True}
    nac_parser_settings.update(base_parser_settings)
    nac_incar_dict = {"lepsilon": True}
    nac_incar_dict.update(base_incar_dict.copy())
    nac_config.update(
        {
            "kpoints_mesh": [8, 8, 8],
            "kpoints_offset": [0.5, 0.5, 0.5],
            "parser_settings": nac_parser_settings,
            "parameters": {"incar": nac_incar_dict},
        }
    )
    phonon_settings = {"supercell_matrix": [2, 2, 2], "distance": 0.03}
    if is_nac:
        phonon_settings["is_nac"] = is_nac

    return structure, forces_config, nac_config, phonon_settings


def launch_phono3py(cutoff_energy=350, is_nac=False):
    """Launch calculation."""
    structure, forces_config, nac_config, phonon_settings = get_settings(
        cutoff_energy, is_nac
    )
    Phono3pyWorkChain = WorkflowFactory("phonopy.phono3py")
    builder = Phono3pyWorkChain.get_builder()
    builder.structure = structure
    builder.calculator_settings = Dict(
        dict={"forces": forces_config, "nac": nac_config}
    )
    builder.run_phono3py = Bool(False)
    builder.remote_phono3py = Bool(False)
    builder.code_string = Str("phonopy@nancy")
    builder.phonon_settings = Dict(dict=phonon_settings)
    builder.symmetry_tolerance = Float(1e-5)
    builder.options = Dict(dict=forces_config["options"])
    dim = phonon_settings["supercell_matrix"]
    kpoints_mesh = forces_config["kpoints_mesh"]
    label = "ZnTe phono3py %dx%dx%d kpt %dx%dx%d PBEsol %d eV" % (
        tuple(dim) + tuple(kpoints_mesh) + (cutoff_energy,)
    )
    builder.metadata.label = label
    builder.metadata.description = label

    future = submit(builder)
    print(label)
    print(future)
    print("Running workchain with pk={}".format(future.pk))


if __name__ == "__main__":
    launch_phono3py(cutoff_energy=500, is_nac=True)
