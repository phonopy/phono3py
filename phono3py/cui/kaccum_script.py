"""Phono3py kaccum command line script."""

import argparse
import sys

import h5py
import numpy as np
from phonopy.cui.collect_cell_info import collect_cell_info
from phonopy.interface.calculator import read_crystal_structure
from phonopy.structure.symmetry import Symmetry

from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.other.kaccum import GammaDOSsmearing, get_mfp, run_mfp_dos, run_prop_dos
from phono3py.phonon.grid import BZGrid, get_ir_grid_points

epsilon = 1.0e-8


def _show_tensor(kdos, temperatures, sampling_points, args):
    """Show 2nd rank tensors."""
    for i, kdos_t in enumerate(kdos):
        if not args.gv:
            print("# %d K" % temperatures[i])

        for f, k in zip(sampling_points[i], kdos_t):  # show kappa_xx
            if args.average:
                print(("%13.5f " * 3) % (f, k[0][:3].sum() / 3, k[1][:3].sum() / 3))
            elif args.trace:
                print(("%13.5f " * 3) % (f, k[0][:3].sum(), k[1][:3].sum()))
            else:
                print(("%f " * 13) % ((f,) + tuple(k[0]) + tuple(k[1])))

        print("")
        print("")


def _show_scalar(gdos, temperatures, sampling_points, args):
    """Show scalar values."""
    if args.pqj or args.gruneisen or args.gv_norm:
        for f, g in zip(sampling_points, gdos[0]):
            print("%f %e %e" % (f, g[0], g[1]))
    else:
        for i, gdos_t in enumerate(gdos):
            print("# %d K" % temperatures[i])
            for f, g in zip(sampling_points[i], gdos_t):
                print("%f %f %f" % (f, g[0], g[1]))
            print("")
            print("")


def _set_T_target(temperatures, mode_prop, T_target, mean_freepath=None):
    """Extract property at specified temperature."""
    for i, t in enumerate(temperatures):
        if np.abs(t - T_target) < epsilon:
            temperatures = temperatures[i : i + 1]
            mode_prop = mode_prop[i : i + 1, :, :]
            if mean_freepath is not None:
                mean_freepath = mean_freepath[i : i + 1]
                return temperatures, mode_prop, mean_freepath
            else:
                return temperatures, mode_prop


def _get_ir_grid_info(bz_grid: BZGrid, weights, qpoints=None, ir_grid_points=None):
    """Return ir-grid point information.

    Parameters
    ----------
    bz_grid : BZGrid
        BZ grid information.
    weights : ndarray
        Weights of ir-grid points stored in kappa-xxx.hdf5. This is used to check
        agreement between generated ir-grid points and those underlying in hdf5.
    qpoints : ndarray
        Ir-grid q-point coordinates stored in kappa-xxx.hdf5. This is used to check
        agreement between generated ir-grid points and q-points coordinates in hdf5.

    Returns
    -------
    ir_grid_points : ndarray
        Ir-grid point indices in BZ-grid.
        shape=(ir_grid_points, ), dtype='int_'
    ir_grid_map : ndarray, optional, default=None
        Mapping table to ir-grid point indices in GR-grid.

    """
    (ir_grid_points_ref, weights_ref, ir_grid_map) = get_ir_grid_points(bz_grid)
    ir_grid_points_ref = np.array(bz_grid.grg2bzg[ir_grid_points_ref], dtype="int_")

    _assert_grid_in_hdf5(
        weights, qpoints, ir_grid_points, weights_ref, ir_grid_points_ref, bz_grid
    )

    return ir_grid_points_ref, ir_grid_map


def _assert_grid_in_hdf5(
    weights,
    qpoints,
    ir_grid_points,
    weights_for_check,
    ir_grid_points_ref,
    bz_grid: BZGrid,
):
    try:
        np.testing.assert_equal(weights, weights_for_check)
    except AssertionError:
        print("*******************************")
        print("** Might forget --pa option? **")
        print("*******************************")
        raise

    if ir_grid_points is not None:
        np.testing.assert_equal(ir_grid_points, ir_grid_points_ref)
    if qpoints is not None:
        addresses = bz_grid.addresses[ir_grid_points_ref]
        D_diag = bz_grid.D_diag.astype("double")
        qpoints_for_check = np.dot(addresses / D_diag, bz_grid.Q.T)
        diff_q = qpoints - qpoints_for_check
        np.testing.assert_almost_equal(diff_q, np.rint(diff_q))


def _get_calculator(args):
    """Return calculator name."""
    interface_mode = None
    # if args.qe_mode:
    #     interface_mode = "qe"
    # elif args.crystal_mode:
    #     interface_mode = "crystal"
    # elif args.abinit_mode:
    #     interface_mode = "abinit"
    # elif args.turbomole_mode:
    #     interface_mode = "turbomole"
    return interface_mode


def _read_files(args):
    """Read crystal structure and kappa.hdf5 files."""
    interface_mode = _get_calculator(args)
    if len(args.filenames) > 1:
        cell, _ = read_crystal_structure(
            args.filenames[0], interface_mode=interface_mode
        )
        f = h5py.File(args.filenames[1], "r")
    else:
        cell, _ = read_crystal_structure(
            args.cell_filename, interface_mode=interface_mode
        )
        f = h5py.File(args.filenames[0], "r")

    return cell, f


def _get_mode_property(args, f_kappa):
    """Read property data from hdf5 file object."""
    if args.pqj:
        mode_prop = f_kappa["ave_pp"][:].reshape((1,) + f_kappa["ave_pp"].shape)
    elif args.cv:
        mode_prop = f_kappa["heat_capacity"][:]
    elif args.tau:
        g = f_kappa["gamma"][:]
        g = np.where(g > 0, g, -1)
        mode_prop = np.where(g > 0, 1.0 / (2 * 2 * np.pi * g), 0)
    elif args.gv_norm:
        mode_prop = np.sqrt((f_kappa["group_velocity"][:, :, :] ** 2).sum(axis=2))
        mode_prop = mode_prop.reshape((1,) + mode_prop.shape)
    elif args.gamma:
        mode_prop = f_kappa["gamma"][:]
    elif args.gruneisen:
        mode_prop = f_kappa["gruneisen"][:].reshape((1,) + f_kappa["gruneisen"].shape)
        mode_prop **= 2
    elif args.dos:
        mode_prop = np.ones(
            (1,) + f_kappa["frequency"].shape, dtype="double", order="C"
        )
    else:
        raise RuntimeError("No property target is specified.")
    return mode_prop


def _get_parser():
    """Return args of ArgumentParser."""
    parser = argparse.ArgumentParser(description="Show unit cell volume")
    parser.add_argument(
        "--gv", action="store_true", help="Calculate for gv_x_gv (tensor)"
    )
    parser.add_argument("--pqj", action="store_true", help="Calculate for Pqj (scalar)")
    parser.add_argument("--cv", action="store_true", help="Calculate for Cv (scalar)")
    parser.add_argument(
        "--tau", action="store_true", help="Calculate for lifetimes (scalar)"
    )
    parser.add_argument(
        "--dos", action="store_true", help="Calculate for phonon DOS (scalar)"
    )
    parser.add_argument(
        "--gamma", action="store_true", help="Calculate for Gamma (scalar)"
    )
    parser.add_argument(
        "--gruneisen",
        action="store_true",
        help="Calculate for mode-Gruneisen parameters squared (scalar)",
    )
    parser.add_argument(
        "--gv-norm", action="store_true", help="Calculate for |g_v| (scalar)"
    )
    parser.add_argument(
        "--mfp", action="store_true", help="Mean free path is used instead of frequency"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        dest="temperature",
        help="Temperature to output data at",
    )
    parser.add_argument(
        "--nsp",
        "--num-sampling-points",
        type=int,
        dest="num_sampling_points",
        default=100,
        help="Number of sampling points in frequency or MFP axis",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help=(
            "Output the traces of the tensors divided by 3 "
            "rather than the unique elements"
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=("Output the traces of the tensors " "rather than the unique elements"),
    )
    parser.add_argument(
        "--smearing",
        action="store_true",
        help="Use smearing method (only for scalar density)",
    )
    parser.add_argument(
        "--no-gridsym",
        dest="no_gridsym",
        action="store_true",
        help="Grid symmetry is unused for phonon-xxx.hdf5 inputs.",
    )
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()
    return args


def main():
    """Calculate kappa spectrum.

    Usage
    -----
    If `phono3py_disp.yaml` or `phono3py.yaml` exists in current directory,
    ```
    % phono3py-kaccum kappa-m111111.hdf5
    ```

    Plot by gnuplot
    ---------------
    ```
    % gnuplot
    ...
    gnuplot> p "kaccum.dat" i 30 u 1:2 w l, "kaccum.dat" i 30 u 1:8 w l
    ```

    """
    args = _get_parser()
    primitive = None
    if len(args.filenames) > 1:
        raise RuntimeError(
            'Use of "phono3py-kaccum CRYSTAL_STRUCTURE_FILE" is not supported.'
        )
    else:
        cell_info = collect_cell_info(
            supercell_matrix=np.eye(3, dtype=int),
            phonopy_yaml_cls=Phono3pyYaml,
        )
        cell_filename = cell_info["optional_structure_info"][0]
        print(f'# Crystal structure was read from "{cell_filename}".')
        cell = cell_info["unitcell"]
        phpy_yaml = cell_info.get("phonopy_yaml", None)
        if phpy_yaml is not None:
            primitive = cell_info["phonopy_yaml"].primitive
            if primitive is None:
                primitive = cell
        f_kappa = h5py.File(args.filenames[0], "r")

    if "grid_matrix" in f_kappa:
        mesh = np.array(f_kappa["grid_matrix"][:], dtype="int_")
    else:
        mesh = np.array(f_kappa["mesh"][:], dtype="int_")
    if "temperature" in f_kappa:
        temperatures = f_kappa["temperature"][:]
    else:
        temperatures = np.zeros(1, dtype="double")
    if "weight" in f_kappa:
        frequencies = f_kappa["frequency"][:]
        ir_weights = f_kappa["weight"][:]
        ir_grid_points = None
        qpoints = f_kappa["qpoint"][:]
    elif "ir_grid_weights" in f_kappa and not args.no_gridsym:
        ir_weights = f_kappa["ir_grid_weights"][:]
        ir_grid_points = f_kappa["ir_grid_points"][:]
        qpoints = None
        frequencies = np.array(f_kappa["frequency"][ir_grid_points], dtype="double")
    else:
        frequencies = f_kappa["frequency"][:]
        ir_weights = np.ones(len(frequencies), dtype="int_")
    primitive_symmetry = Symmetry(primitive)
    bz_grid = BZGrid(
        mesh,
        lattice=primitive.cell,
        symmetry_dataset=primitive_symmetry.dataset,
        store_dense_gp_map=True,
    )
    if args.no_gridsym or (ir_weights == 1).all():
        ir_grid_points = bz_grid.grg2bzg
        ir_grid_map = None
    else:
        ir_grid_points, ir_grid_map = _get_ir_grid_info(
            bz_grid,
            ir_weights,
            qpoints=qpoints,
            ir_grid_points=ir_grid_points,
        )
    conditions = frequencies > 0
    if np.logical_not(conditions).sum() > 3:
        sys.stderr.write(
            "# Imaginary frequencies are found. " "They are set to be zero.\n"
        )
        frequencies = np.where(conditions, frequencies, 0)

    # Run for scaler
    if (
        args.gamma
        or args.gruneisen
        or args.pqj
        or args.cv
        or args.tau
        or args.gv_norm
        or args.dos
    ):
        mode_prop = _get_mode_property(args, f_kappa)

        if args.temperature is not None and not (
            args.gv_norm or args.pqj or args.gruneisen or args.dos
        ):
            temperatures, mode_prop = _set_T_target(
                temperatures, mode_prop, args.temperature
            )
        if args.smearing:
            mode_prop_dos = GammaDOSsmearing(
                mode_prop,
                frequencies,
                ir_weights,
                num_sampling_points=args.num_sampling_points,
            )
            sampling_points, gdos = mode_prop_dos.get_gdos()
            sampling_points = np.tile(sampling_points, (len(gdos), 1))
            _show_scalar(gdos[:, :, :], temperatures, sampling_points, args)
        else:
            for i, w in enumerate(ir_weights):
                mode_prop[:, i, :] *= w
            kdos, sampling_points = run_prop_dos(
                frequencies,
                mode_prop[:, :, :, None],
                ir_grid_map,
                ir_grid_points,
                args.num_sampling_points,
                bz_grid,
            )
            _show_scalar(kdos[:, :, :, 0], temperatures, sampling_points, args)

    else:  # Run for tensor
        if args.gv:
            gv_sum2 = f_kappa["gv_by_gv"][:]
            # gv x gv is divied by primitive cell volume.
            unit_conversion = primitive.volume
            mode_prop = gv_sum2.reshape((1,) + gv_sum2.shape) / unit_conversion
        else:
            mode_prop = f_kappa["mode_kappa"][:]

        if args.mfp:
            if "mean_free_path" in f_kappa:
                mfp = f_kappa["mean_free_path"][:]
                mean_freepath = np.sqrt((mfp**2).sum(axis=3))
            else:
                mean_freepath = get_mfp(
                    f_kappa["gamma"][:], f_kappa["group_velocity"][:]
                )
            if args.temperature is not None:
                (temperatures, mode_prop, mean_freepath) = _set_T_target(
                    temperatures,
                    mode_prop,
                    args.temperature,
                    mean_freepath=mean_freepath,
                )

            kdos, sampling_points = run_mfp_dos(
                mean_freepath,
                mode_prop,
                ir_grid_map,
                ir_grid_points,
                args.num_sampling_points,
                bz_grid,
            )
            _show_tensor(kdos, temperatures, sampling_points, args)
        else:
            if args.temperature is not None and not args.gv:
                temperatures, mode_prop = _set_T_target(
                    temperatures, mode_prop, args.temperature
                )
            kdos, sampling_points = run_prop_dos(
                frequencies,
                mode_prop,
                ir_grid_map,
                ir_grid_points,
                args.num_sampling_points,
                bz_grid,
            )
            _show_tensor(kdos, temperatures, sampling_points, args)
