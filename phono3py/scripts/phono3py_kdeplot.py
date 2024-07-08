import argparse
import os
import sys

import h5py
import numpy as np
from scipy import stats

epsilon = 1.0e-8


def collect_data(gamma, weights, frequencies, cutoff, max_freq):
    """Collect data for making input of Gaussian-KDE."""
    freqs = []
    mode_prop = []
    for w, freq, g in zip(weights, frequencies, gamma):
        tau = 1.0 / np.where(g > 0, g, -1) / (2 * 2 * np.pi)
        if cutoff:
            tau = np.where(tau < cutoff, tau, -1)

        condition = tau > 0
        _tau = np.extract(condition, tau)
        _freq = np.extract(condition, freq)

        if max_freq is None:
            freqs += list(_freq) * w
            mode_prop += list(_tau) * w
        else:
            freqs += list(np.extract(freq < max_freq, freq)) * w
            mode_prop += list(np.extract(freq < max_freq, tau)) * w

    x = np.array(freqs)
    y = np.array(mode_prop)

    return x, y


def run_KDE(x, y, nbins, x_max=None, y_max=None, density_ratio=0.1):
    """Run Gaussian-KDE by scipy."""
    x_min = 0
    if x_max is None:
        _x_max = np.rint(x.max() * 1.1)
    else:
        _x_max = x_max
    y_min = 0
    if y_max is None:
        _y_max = np.rint(y.max())
    else:
        _y_max = y_max
    values = np.vstack([x.ravel(), y.ravel()])
    kernel = stats.gaussian_kde(values)

    xi, yi = np.mgrid[x_min : _x_max : nbins * 1j, y_min : _y_max : nbins * 1j]
    positions = np.vstack([xi.ravel(), yi.ravel()])
    zi = np.reshape(kernel(positions).T, xi.shape)

    if y_max is None:
        zi_max = np.max(zi)
        indices = []
        for i, r_zi in enumerate((zi.T)[::-1]):
            if indices:
                indices.append(nbins - i - 1)
            elif np.max(r_zi) > zi_max * density_ratio:
                indices = [nbins - i - 1]
        short_nbinds = len(indices)

        ynbins = nbins**2 // short_nbinds
        xi, yi = np.mgrid[x_min : _x_max : nbins * 1j, y_min : _y_max : ynbins * 1j]
        positions = np.vstack([xi.ravel(), yi.ravel()])
        zi = np.reshape(kernel(positions).T, xi.shape)
    else:
        short_nbinds = nbins

    return xi, yi, zi, short_nbinds


def plot(
    plt,
    xi,
    yi,
    zi,
    x,
    y,
    short_nbinds,
    nbins,
    y_max=None,
    z_max=None,
    cmap=None,
    aspect=None,
    flip=False,
    no_points=False,
    show_colorbar=True,
    point_size=5,
    title=None,
):
    """Plot lifetime distribution."""
    xmax = np.max(x)
    ymax = np.max(y)
    x_cut = []
    y_cut = []
    threshold = ymax / nbins * short_nbinds / nbins * (nbins - 1)
    for _x, _y in zip(x, y):
        if epsilon < _y and _y < threshold and epsilon < _x and _x < xmax - epsilon:
            x_cut.append(_x)
            y_cut.append(_y)

    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_tick_params(which="both", direction="out")
    ax.yaxis.set_tick_params(which="both", direction="out")

    if flip:
        # Start Flip
        plt.pcolormesh(
            yi[:, :nbins], xi[:, :nbins], zi[:, :nbins], vmax=z_max, cmap=cmap
        )
        if show_colorbar:
            plt.colorbar()
        if not no_points:
            plt.scatter(y_cut, x_cut, s=point_size, c="k", marker=".", linewidth=0)
        plt.ylim(ymin=0, ymax=xi.max())
        if y_max is None:
            plt.xlim(xmin=0, xmax=(np.max(y_cut) + epsilon))
        else:
            plt.xlim(xmin=0, xmax=(y_max + epsilon))
        plt.xlabel("Lifetime (ps)", fontsize=18)
        plt.ylabel("Phonon frequency (THz)", fontsize=18)
        # End Flip
    else:
        plt.pcolormesh(
            xi[:, :nbins], yi[:, :nbins], zi[:, :nbins], vmax=z_max, cmap=cmap
        )
        if show_colorbar:
            plt.colorbar()
        if not no_points:
            plt.scatter(x_cut, y_cut, s=point_size, c="k", marker=".", linewidth=0)
        plt.xlim(xmin=0, xmax=xi.max())
        if y_max is None:
            plt.ylim(ymin=0, ymax=(np.max(y_cut) + epsilon))
        else:
            plt.ylim(ymin=0, ymax=(y_max + epsilon))
        if title:
            plt.title(title, fontsize=20)
        plt.xlabel("Phonon frequency (THz)", fontsize=18)
        plt.ylabel("Lifetime (ps)", fontsize=18)

    if aspect is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * aspect
        ax.set_aspect(ax_aspect)

    if title:
        plt.title(title, fontsize=20)

    return fig


def get_options():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot property density with gaussian KDE"
    )
    parser.add_argument(
        "--aspect", type=float, default=None, help="The ration, height/width of canvas"
    )
    parser.add_argument("--cmap", dest="cmap", default=None, help="Matplotlib cmap")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help=(
            "Property (y-axis) below this value is included in "
            "data before running Gaussian-KDE"
        ),
    )
    parser.add_argument(
        "--dr",
        "--density-ratio",
        dest="density_ratio",
        type=float,
        default=0.1,
        help=(
            "Minimum density ratio with respect to maximum "
            "density used to determine drawing region"
        ),
    )
    parser.add_argument("--flip", action="store_true", help="Flip x and y.")
    parser.add_argument(
        "--fmax", type=float, default=None, help="Max frequency to plot"
    )
    parser.add_argument(
        "--hc",
        "--hide-colorbar",
        dest="hide_colorbar",
        action="store_true",
        help="Do not show colorbar",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help=(
            "Number of bins in which data are assigned, "
            "i.e., determining resolution of plot"
        ),
    )
    parser.add_argument(
        "--no-points", dest="no_points", action="store_true", help="Do not show points"
    )
    parser.add_argument("--nu", action="store_true", help="Plot N and U.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        default=None,
        help="Output filename, e.g., to PDF",
    )
    parser.add_argument(
        "--point-size",
        dest="point_size",
        type=float,
        default=5,
        help="Point size (default=5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        dest="temperature",
        help="Temperature to output data at",
    )
    parser.add_argument("--title", dest="title", default=None, help="Plot title")
    parser.add_argument(
        "--xmax", type=float, default=None, help="Set maximum x of draw area"
    )
    parser.add_argument(
        "--ymax", type=float, default=None, help="Set maximum y of draw area"
    )
    parser.add_argument("--zmax", type=float, default=None, help="Set maximum indisity")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()
    return args


def main(args):
    """Run phono3py-kdeplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "serif"

    #
    # Initial setting
    #
    if os.path.isfile(args.filenames[0]):
        f = h5py.File(args.filenames[0], "r")
    else:
        print("File %s doens't exist." % args.filenames[0])
        sys.exit(1)

    if args.title:
        title = args.title
    else:
        title = None

    #
    # Set temperature
    #
    temperatures = f["temperature"][:]
    if len(temperatures) > 29:
        t_index = 30
    else:
        t_index = 0
    for i, t in enumerate(temperatures):
        if np.abs(t - args.temperature) < epsilon:
            t_index = i
            break

    print(
        "Temperature at which lifetime density is drawn: %7.3f" % temperatures[t_index]
    )

    #
    # Set data
    #
    weights = f["weight"][:]
    frequencies = f["frequency"][:]
    symbols = [
        "",
    ]

    gammas = [
        f["gamma"][t_index],
    ]
    if args.nu:
        if "gamma_N" in f:
            gammas.append(f["gamma_N"][t_index])
            symbols.append("N")
        if "gamma_U" in f:
            gammas.append(f["gamma_U"][t_index])
            symbols.append("U")

    #
    # Run
    #
    for gamma, s in zip(gammas, symbols):
        x, y = collect_data(gamma, weights, frequencies, args.cutoff, args.fmax)
        xi, yi, zi, short_nbinds = run_KDE(
            x,
            y,
            args.nbins,
            x_max=args.xmax,
            y_max=args.ymax,
            density_ratio=args.density_ratio,
        )
        fig = plot(
            plt,
            xi,
            yi,
            zi,
            x,
            y,
            short_nbinds,
            args.nbins,
            y_max=args.ymax,
            z_max=args.zmax,
            cmap=args.cmap,
            aspect=args.aspect,
            flip=args.flip,
            no_points=args.no_points,
            show_colorbar=(not args.hide_colorbar),
            point_size=args.point_size,
            title=title,
        )

        if args.output_filename:
            fig.savefig(args.output_filename)
        else:
            if s:
                fig.savefig("lifetime-%s.png" % s)
            else:
                fig.savefig("lifetime.png")
        plt.close(fig)


def run():
    """Run phono3py-kdeplot script."""
    main(get_options())
