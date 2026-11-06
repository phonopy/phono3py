# SPDX-License-Identifier: BSD-3-Clause
"""Reduction of values over degenerate sets of bands."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def average_over_degenerate_sets(
    values: NDArray, degenerate_ids: NDArray[np.int64], axis: int
) -> NDArray[np.double]:
    """Replace values by their means over each set of degenerate bands.

    Parameters
    ----------
    values : ndarray
        Values with a band axis.
    degenerate_ids : ndarray
        Smallest band index in the degenerate set of each band, see phonopy's
        get_degenerate_ids. shape=(bands,), dtype='int64'
    axis : int
        Band axis of ``values``.

    Returns
    -------
    ndarray
        Averaged values in the same shape as ``values``. ``values`` itself
        is returned when no bands are degenerate.

    """
    starts, counts = _get_sets(degenerate_ids)
    if len(starts) == len(degenerate_ids):
        return values
    shape = [1] * values.ndim
    shape[axis] = len(counts)
    means = np.add.reduceat(values, starts, axis=axis) / counts.reshape(shape)
    return np.repeat(means, counts, axis=axis)


def minimum_over_degenerate_sets(
    values: NDArray, degenerate_ids: NDArray[np.int64], axis: int
) -> NDArray:
    """Replace values by their minima over each set of degenerate bands.

    For flags of 0 and 1, the minimum is AND.

    Parameters are those of ``average_over_degenerate_sets``.

    """
    starts, counts = _get_sets(degenerate_ids)
    if len(starts) == len(degenerate_ids):
        return values
    return np.repeat(np.minimum.reduceat(values, starts, axis=axis), counts, axis=axis)


def _get_sets(
    degenerate_ids: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return first band indices and sizes of degenerate sets of bands."""
    num_band = len(degenerate_ids)
    starts = np.flatnonzero(degenerate_ids == np.arange(num_band))
    return starts, np.diff(starts, append=num_band)
