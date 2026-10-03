"""Reduction of values over degenerate sets of bands."""

# Copyright (C) 2020 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
