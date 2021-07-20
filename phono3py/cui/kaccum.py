"""Phono3py kaccum command line script."""
import sys
import argparse
import numpy as np
import h5py
from phonopy.cui.settings import fracval
from phonopy.structure.cells import get_primitive
from phonopy.structure.symmetry import Symmetry
from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.dos import NormalDistribution
from phonopy.structure.cells import (
    get_primitive_matrix, guess_primitive_matrix)
from phono3py.phonon.grid import BZGrid, get_ir_grid_points
from phono3py.other.tetrahedron_method import get_integration_weights

epsilon = 1.0e-8


class KappaDOS(object):
    """Class to calculate thermal conductivity spectram."""

    def __init__(self,
                 mode_kappa,
                 frequencies,
                 bz_grid,
                 ir_grid_points,
                 ir_grid_map=None,
                 frequency_points=None,
                 num_sampling_points=100):
        """Init method.

        mode_kappa : ndarray
            Target value.
            shape=(temperatures, ir_grid_points, num_band, num_elem),
            dtype='double'
        frequencies : ndarray
            shape=(ir_grid_points, num_band), dtype='double'
        bz_grid : BZGrid
        ir_grid_points : ndarray
            Ir-grid point indices in GR-grid.
            shape=(ir_grid_points, ), dtype='int_'
        ir_grid_map : ndarray, optional, default=None
            Mapping table to ir-grid point indices in GR-grid.
            None gives `np.arange(len(frequencies), 'int_')`.
        frequency_points : array_like, optional, default=None
            This is used as the frequency points. When None,
            frequency points are created from `num_sampling_points`.
        num_sampling_points : int, optional, default=100
            Number of uniform sampling points.

        """
        min_freq = min(frequencies.ravel())
        max_freq = max(frequencies.ravel()) + epsilon
        if frequency_points is None:
            self._frequency_points = np.linspace(
                min_freq, max_freq, num_sampling_points, dtype='double')
        else:
            self._frequency_points = np.array(
                frequency_points, dtype='double')

        n_temp, _, _, n_elem = mode_kappa.shape
        self._kdos = np.zeros(
            (n_temp, len(self._frequency_points), 2, n_elem), dtype='double')

        if ir_grid_map is None:
            bzgp2irgp_map = None
        else:
            bzgp2irgp_map = self._get_bzgp2irgp_map(bz_grid, ir_grid_map)
        for j, function in enumerate(('J', 'I')):
            iweights = get_integration_weights(self._frequency_points,
                                               frequencies,
                                               bz_grid,
                                               grid_points=ir_grid_points,
                                               bzgp2irgp_map=bzgp2irgp_map,
                                               function=function)
            for i, iw in enumerate(iweights):
                self._kdos[:, :, j] += np.transpose(
                    np.dot(iw, mode_kappa[:, i]), axes=(1, 0, 2))
        self._kdos /= np.prod(bz_grid.D_diag)

    def get_kdos(self):
        """Return thermal conductivity spectram.

        Returns
        -------
        tuple
            frequency_points : ndarray
                shape=(sampling_points, ), dtype='double'
            kdos : ndarray
                shape=(temperatures, sampling_points, 2 (J, I), num_elem),
                dtype='double', order='C'

        """
        return self._frequency_points, self._kdos

    def _get_bzgp2irgp_map(self, bz_grid, ir_grid_map):
        unique_gps = np.unique(ir_grid_map)
        gp_map = {j: i for i, j in enumerate(unique_gps)}
        bzgp2irgp_map = np.array(
            [gp_map[ir_grid_map[grgp]] for grgp in bz_grid.bzg2grg],
            dtype='int_')
        return bzgp2irgp_map


class GammaDOSsmearing(object):
    """Class to calculate Gamma spectram by smearing method."""

    def __init__(self,
                 gamma,
                 frequencies,
                 ir_grid_weights,
                 sigma=None,
                 num_fpoints=200):
        """Init method."""
        self._gamma = gamma
        self._frequencies = frequencies
        self._ir_grid_weights = ir_grid_weights
        self._num_fpoints = num_fpoints
        self._set_frequency_points()
        self._gdos = np.zeros(
            (len(gamma), len(self._frequency_points), 2), dtype='double')
        if sigma is None:
            self._sigma = (max(self._frequency_points) -
                           min(self._frequency_points)) / 100
        else:
            self._sigma = 0.1
        self._smearing_function = NormalDistribution(self._sigma)
        self._run_smearing_method()

    def get_gdos(self):
        """Return Gamma spectram."""
        return self._frequency_points, self._gdos

    def _set_frequency_points(self):
        min_freq = np.min(self._frequencies)
        max_freq = np.max(self._frequencies) + epsilon
        self._frequency_points = np.linspace(min_freq,
                                             max_freq,
                                             self._num_fpoints)

    def _run_smearing_method(self):
        self._dos = []
        num_gp = np.sum(self._ir_grid_weights)
        for i, f in enumerate(self._frequency_points):
            dos = self._smearing_function.calc(self._frequencies - f)
            for j, g_t in enumerate(self._gamma):
                self._gdos[j, i, 1] = np.sum(np.dot(self._ir_grid_weights,
                                                    dos * g_t)) / num_gp


def _show_tensor(kdos, temperatures, sampling_points, args):
    """Show 2nd rank tensors."""
    for i, kdos_t in enumerate(kdos):
        if not args.gv:
            print("# %d K" % temperatures[i])

        for f, k in zip(sampling_points[i], kdos_t):  # show kappa_xx
            if args.average:
                print(("%13.5f " * 3) %
                      (f, k[0][:3].sum() / 3, k[1][:3].sum() / 3))
            elif args.trace:
                print(("%13.5f " * 3) % (f, k[0][:3].sum(), k[1][:3].sum()))
            else:
                print(("%f " * 13) % ((f,) + tuple(k[0]) + tuple(k[1])))

        print('')
        print('')


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
            print('')
            print('')


def _set_T_target(temperatures,
                  mode_prop,
                  T_target,
                  mean_freepath=None):
    """Extract property at specified temperature."""
    for i, t in enumerate(temperatures):
        if np.abs(t - T_target) < epsilon:
            temperatures = temperatures[i:i+1]
            mode_prop = mode_prop[i:i+1, :, :]
            if mean_freepath is not None:
                mean_freepath = mean_freepath[i:i+1]
                return temperatures, mode_prop, mean_freepath
            else:
                return temperatures, mode_prop


def _run_prop_dos(frequencies,
                  mode_prop,
                  ir_grid_map,
                  ir_grid_points,
                  num_sampling_points,
                  bz_grid):
    """Run DOS-like calculation."""
    kappa_dos = KappaDOS(mode_prop,
                         frequencies,
                         bz_grid,
                         ir_grid_points,
                         ir_grid_map=ir_grid_map,
                         num_sampling_points=num_sampling_points)
    freq_points, kdos = kappa_dos.get_kdos()
    sampling_points = np.tile(freq_points, (len(kdos), 1))
    return kdos, sampling_points


def _get_mfp(g, gv):
    """Calculate mean free path from inverse lifetime and group velocity."""
    g = np.where(g > 0, g, -1)
    gv_norm = np.sqrt((gv ** 2).sum(axis=2))
    mean_freepath = np.where(g > 0, gv_norm / (2 * 2 * np.pi * g), 0)
    return mean_freepath


def _run_mfp_dos(mean_freepath,
                 mode_prop,
                 ir_grid_map,
                 ir_grid_points,
                 num_sampling_points,
                 bz_grid):
    """Run DOS-like calculation for mean free path.

    mean_freepath : shape=(temperatures, ir_grid_points, 6)
    mode_prop : shape=(temperatures, ir_grid_points, 6, 6)

    """
    kdos = []
    sampling_points = []
    for i, _ in enumerate(mean_freepath):
        kappa_dos = KappaDOS(mode_prop[i:i+1, :, :],
                             mean_freepath[i],
                             bz_grid,
                             ir_grid_points,
                             ir_grid_map=ir_grid_map,
                             num_sampling_points=num_sampling_points)
        sampling_points_at_T, kdos_at_T = kappa_dos.get_kdos()
        kdos.append(kdos_at_T[0])
        sampling_points.append(sampling_points_at_T)
    kdos = np.array(kdos)
    sampling_points = np.array(sampling_points)

    return kdos, sampling_points


def _get_grid_symmetry(bz_grid, weights, qpoints):
    (ir_grid_points,
     weights_for_check,
     ir_grid_map) = get_ir_grid_points(bz_grid)

    try:
        np.testing.assert_array_equal(weights, weights_for_check)
    except AssertionError:
        print("*******************************")
        print("** Might forget --pa option? **")
        print("*******************************")
        raise

    addresses = bz_grid.addresses[ir_grid_points]
    D_diag = bz_grid.D_diag.astype('double')
    qpoints_for_check = np.dot(addresses / D_diag, bz_grid.Q.T)
    diff_q = qpoints - qpoints_for_check
    np.testing.assert_almost_equal(diff_q, np.rint(diff_q))

    return ir_grid_points, ir_grid_map


def _get_calculator(args):
    """Return calculator name."""
    interface_mode = None
    if args.qe_mode:
        interface_mode = 'qe'
    elif args.crystal_mode:
        interface_mode = 'crystal'
    elif args.abinit_mode:
        interface_mode = 'abinit'
    elif args.turbomole_mode:
        interface_mode = 'turbomole'
    return interface_mode


def _read_files(args):
    """Read crystal structure and kappa.hdf5 files."""
    interface_mode = _get_calculator(args)
    if len(args.filenames) > 1:
        cell, _ = read_crystal_structure(args.filenames[0],
                                         interface_mode=interface_mode)
        f = h5py.File(args.filenames[1], 'r')
    else:
        cell, _ = read_crystal_structure(args.cell_filename,
                                         interface_mode=interface_mode)
        f = h5py.File(args.filenames[0], 'r')

    return cell, f


def _get_mode_property(args, f_kappa):
    """Read property data from hdf5 file object."""
    if args.pqj:
        mode_prop = f_kappa['ave_pp'][:].reshape(
            (1,) + f_kappa['ave_pp'].shape)
    elif args.cv:
        mode_prop = f_kappa['heat_capacity'][:]
    elif args.tau:
        g = f_kappa['gamma'][:]
        g = np.where(g > 0, g, -1)
        mode_prop = np.where(g > 0, 1.0 / (2 * 2 * np.pi * g), 0)
    elif args.gv_norm:
        mode_prop = np.sqrt(
            (f_kappa['group_velocity'][:, :, :] ** 2).sum(axis=2))
        mode_prop = mode_prop.reshape((1,) + mode_prop.shape)
    elif args.gamma:
        mode_prop = f_kappa['gamma'][:]
    elif args.gruneisen:
        mode_prop = f_kappa['gruneisen'][:].reshape(
            (1,) + f_kappa['gruneisen'].shape)
        mode_prop **= 2
    elif args.dos:
        mode_prop = np.ones((1, ) + f_kappa['frequency'].shape,
                            dtype='double', order='C')
    else:
        raise RuntimeError("No property target is specified.")
    return mode_prop


def _get_parser():
    """Return args of ArgumentParser."""
    parser = argparse.ArgumentParser(description="Show unit cell volume")
    parser.add_argument(
        "--pa", "--primitive-axis", "--primitive-axes", nargs='+',
        dest="primitive_matrix", default=None,
        help="Same as PRIMITIVE_AXES tags")
    parser.add_argument(
        "-c", "--cell", dest="cell_filename", metavar="FILE", default=None,
        help="Read unit cell")
    parser.add_argument(
        '--gv', action='store_true',
        help='Calculate for gv_x_gv (tensor)')
    parser.add_argument(
        '--pqj', action='store_true',
        help='Calculate for Pqj (scalar)')
    parser.add_argument(
        '--cv', action='store_true',
        help='Calculate for Cv (scalar)')
    parser.add_argument(
        '--tau', action='store_true',
        help='Calculate for lifetimes (scalar)')
    parser.add_argument(
        '--dos', action='store_true',
        help='Calculate for phonon DOS (scalar)')
    parser.add_argument(
        '--gamma', action='store_true',
        help='Calculate for Gamma (scalar)')
    parser.add_argument(
        '--gruneisen', action='store_true',
        help='Calculate for mode-Gruneisen parameters squared (scalar)')
    parser.add_argument(
        '--gv-norm', action='store_true',
        help='Calculate for |g_v| (scalar)')
    parser.add_argument(
        '--mfp', action='store_true',
        help='Mean free path is used instead of frequency')
    parser.add_argument(
        '--temperature', type=float, dest='temperature',
        help='Temperature to output data at')
    parser.add_argument(
        '--nsp', '--num-sampling-points', type=int, dest='num_sampling_points',
        default=100,
        help="Number of sampling points in frequency or MFP axis")
    parser.add_argument(
        '--average', action='store_true',
        help=("Output the traces of the tensors divided by 3 "
              "rather than the unique elements"))
    parser.add_argument(
        '--trace', action='store_true',
        help=("Output the traces of the tensors "
              "rather than the unique elements"))
    parser.add_argument(
        '--smearing', action='store_true',
        help='Use smearing method (only for scalar density)')
    parser.add_argument(
        '--qe', '--pwscf', dest="qe_mode",
        action="store_true", help="Invoke Pwscf mode")
    parser.add_argument(
        '--crystal', dest="crystal_mode",
        action="store_true", help="Invoke CRYSTAL mode")
    parser.add_argument(
        '--abinit', dest="abinit_mode",
        action="store_true", help="Invoke Abinit mode")
    parser.add_argument(
        '--turbomole', dest="turbomole_mode",
        action="store_true", help="Invoke TURBOMOLE mode")
    parser.add_argument(
        "--noks", "--no-kappa-stars",
        dest="no_kappa_stars", action="store_true",
        help="Deactivate summation of partial kappa at q-stars")
    parser.add_argument('filenames', nargs='*')
    args = parser.parse_args()
    return args


def _analyze_primitive_matrix_option(args, unitcell=None):
    """Analyze --pa option argument."""
    if args.primitive_matrix is not None:
        if type(args.primitive_matrix) is list:
            _primitive_matrix = " ".join(args.primitive_matrix)
        else:
            _primitive_matrix = args.primitive_matrix.strip()

        if _primitive_matrix.lower() == 'auto':
            primitive_matrix = 'auto'
        elif _primitive_matrix.upper() in ('P', 'F', 'I', 'A', 'C', 'R'):
            primitive_matrix = _primitive_matrix.upper()
        elif len(_primitive_matrix.split()) != 9:
            raise SyntaxError(
                "Number of elements in --pa option argument has to be 9.")
        else:
            primitive_matrix = np.reshape(
                [fracval(x) for x in _primitive_matrix.split()], (3, 3))
            if np.linalg.det(primitive_matrix) < 1e-8:
                raise SyntaxError(
                    "Primitive matrix has to have positive determinant.")

    pmat = get_primitive_matrix(primitive_matrix)
    if unitcell is not None and isinstance(pmat, str) and pmat == 'auto':
        return guess_primitive_matrix(unitcell)
    else:
        return pmat


def main():
    """Calculate kappa spectrum."""
    args = _get_parser()
    cell, f_kappa = _read_files(args)
    mesh = np.array(f_kappa['mesh'][:], dtype='int_')
    temperatures = f_kappa['temperature'][:]
    ir_weights = f_kappa['weight'][:]
    primitive_matrix = _analyze_primitive_matrix_option(args, unitcell=cell)
    primitive = get_primitive(cell, primitive_matrix)
    primitive_symmetry = Symmetry(primitive)
    bz_grid = BZGrid(mesh,
                     lattice=primitive.cell,
                     symmetry_dataset=primitive_symmetry.dataset,
                     store_dense_gp_map=False)
    if args.no_kappa_stars or (ir_weights == 1).all():
        ir_grid_points = np.arange(np.prod(mesh), dtype='int_')
        ir_grid_map = np.arange(np.prod(mesh), dtype='int_')
    else:
        ir_grid_points, ir_grid_map = _get_grid_symmetry(
            bz_grid, ir_weights, f_kappa['qpoint'][:])
    frequencies = f_kappa['frequency'][:]
    conditions = frequencies > 0
    if np.logical_not(conditions).sum() > 3:
        sys.stderr.write("# Imaginary frequencies are found. "
                         "They are set to be zero.\n")
        frequencies = np.where(conditions, frequencies, 0)

    # Run for scaler
    if (args.gamma or args.gruneisen or args.pqj or
        args.cv or args.tau or args.gv_norm or args.dos):  # noqa E129

        mode_prop = _get_mode_property(args, f_kappa)

        if (args.temperature is not None and
            not (args.gv_norm or args.pqj or args.gruneisen or args.dos)):  # noqa E129
            temperatures, mode_prop = _set_T_target(temperatures,
                                                    mode_prop,
                                                    args.temperature)
        if args.smearing:
            mode_prop_dos = GammaDOSsmearing(
                mode_prop,
                frequencies,
                ir_weights,
                num_fpoints=args.num_sampling_points)
            sampling_points, gdos = mode_prop_dos.get_gdos()
            sampling_points = np.tile(sampling_points, (len(gdos), 1))
            _show_scalar(gdos[:, :, :], temperatures, sampling_points, args)
        else:
            for i, w in enumerate(ir_weights):
                mode_prop[:, i, :] *= w
            kdos, sampling_points = _run_prop_dos(frequencies,
                                                  mode_prop[:, :, :, None],
                                                  ir_grid_map,
                                                  ir_grid_points,
                                                  args.num_sampling_points,
                                                  bz_grid)
            _show_scalar(kdos[:, :, :, 0], temperatures, sampling_points, args)

    else:  # Run for tensor
        if args.gv:
            gv_sum2 = f_kappa['gv_by_gv'][:]
            # gv x gv is divied by primitive cell volume.
            unit_conversion = primitive.volume
            mode_prop = gv_sum2.reshape((1,) + gv_sum2.shape) / unit_conversion
        else:
            mode_prop = f_kappa['mode_kappa'][:]

        if args.mfp:
            if 'mean_free_path' in f_kappa:
                mfp = f_kappa['mean_free_path'][:]
                mean_freepath = np.sqrt((mfp ** 2).sum(axis=3))
            else:
                mean_freepath = _get_mfp(f_kappa['gamma'][:],
                                         f_kappa['group_velocity'][:])
            if args.temperature is not None:
                (temperatures,
                 mode_prop,
                 mean_freepath) = _set_T_target(temperatures,
                                                mode_prop,
                                                args.temperature,
                                                mean_freepath=mean_freepath)

            kdos, sampling_points = _run_mfp_dos(mean_freepath,
                                                 mode_prop,
                                                 ir_grid_map,
                                                 ir_grid_points,
                                                 args.num_sampling_points,
                                                 bz_grid)
            _show_tensor(kdos, temperatures, sampling_points, args)
        else:
            if args.temperature is not None and not args.gv:
                temperatures, mode_prop = _set_T_target(temperatures,
                                                        mode_prop,
                                                        args.temperature)
            kdos, sampling_points = _run_prop_dos(frequencies,
                                                  mode_prop,
                                                  ir_grid_map,
                                                  ir_grid_points,
                                                  args.num_sampling_points,
                                                  bz_grid)
            _show_tensor(kdos, temperatures, sampling_points, args)
