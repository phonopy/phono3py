import argparse
import os
import sys

import h5py
import numpy as np

epsilon = 1.0e-8


def get_options():
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="Plot collision matrix eigenvalues")
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        dest="temperature",
        help="Temperature to output data at",
    )
    parser.add_argument(
        "--savetxt",
        dest="savetxt",
        action="store_true",
        default=False,
        help="Write coleigs to text file",
    )
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()
    return args


def get_t_index(temperatures, args):
    """Return temperature index."""
    if len(temperatures) > 29:
        t_index = 30
    else:
        t_index = 0
    for i, t in enumerate(temperatures):
        if np.abs(t - args.temperature) < epsilon:
            t_index = i
            break
    return t_index


def plot_one(ax, coleigs, temperatures, args):
    """Plot collision matrix eigenvalues at one temperature."""
    t_index = get_t_index(temperatures, args)
    coleigs_p = coleigs[t_index].copy()
    coleigs_n = coleigs[t_index].copy()
    coleigs_p[coleigs_p < 0] = np.nan
    coleigs_n[coleigs_n >= 0] = np.nan
    ax.semilogy(coleigs_p, "b.", markersize=1, clip_on=False)
    ax.semilogy(-coleigs_n, "r.", markersize=1, clip_on=False)
    ax.set_xlim(0, len(coleigs_p))


def plot_one_file(ax, args):
    """Plot collision matrix eigenvalues at one temperature from file."""
    filename = args.filenames[0]
    if os.path.isfile(filename):
        with h5py.File(filename, "r") as f:
            coleigs = f["collision_eigenvalues"][:]
            temperatures = f["temperature"][:]
        plot_one(ax, coleigs, temperatures, args)
    else:
        print("File %s doens't exist." % filename)
        sys.exit(1)


def plot_more(ax, coleigs, temperatures, args):
    """Update plot by collision matrix eigenvalues from one file."""
    t_index = get_t_index(temperatures[0], args)
    y = [np.abs(v[t_index]) for v in coleigs]
    ax.semilogy(np.transpose(y), ".", markersize=5)
    ax.set_xlim(0, len(y[0]))


def plot_more_files(ax, args):
    """Plot collision matrix eigenvalues from multiple files."""
    temperatures = []
    coleigs = []
    for filename in args.filenames:
        if os.path.isfile(filename):
            with h5py.File(filename, "r") as f:
                coleigs.append(f["collision_eigenvalues"][:])
                temperatures.append(f["temperature"][:])
        else:
            print("File %s doens't exist." % filename)
            sys.exit(1)
    plot_more(ax, coleigs, temperatures, args)


def plot(args: argparse.Namespace):
    """Plot collision matrix eigenvalues."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")

    if len(args.filenames) == 1:
        plot_one_file(ax, args)
    else:
        plot_more_files(ax, args)

    # for filename in args.filenames:

    plt.show()


def write_dat(args: argparse.Namespace):
    """Write collision matrix eigenvalues to a file in gnuplot style."""
    with open("coleigs.dat", "w") as w:
        for filename in args.filenames:
            with h5py.File(filename) as f:
                print("#", file=w)
                print(f"# Filename {filename}", file=w)
                print("#", file=w)
                print("#             eigenvalue      log10(abs(eigenvalue))", file=w)
                log_vals = np.log10(np.abs(f["collision_eigenvalues"][:]))
                for coleigs, coleigs_log in zip(f["collision_eigenvalues"], log_vals):
                    for i, (val, val_log) in enumerate(zip(coleigs, coleigs_log)):
                        print(f"{i + 1:<10d} {val:15.8e}     {val_log:15.8f}", file=w)
                    print(file=w)
                print(file=w)
                print(file=w)


def main(args: argparse.Namespace):
    """Run phono3py-coleigplot."""
    if args.savetxt:
        write_dat(args)
    else:
        plot(args)


def run():
    """Run phono3py-coleigplot script."""
    main(get_options())
