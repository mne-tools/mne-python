# -*- coding: utf-8 -*-
"""Some utility functions for rank estimation."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

# import operator

import numpy as np
from scipy import linalg

# from .io.pick import pick_info, _picks_by_type, _pick_data_channels
# from .defaults import _handle_default
# from .utils import logger
from .utils import _compute_row_norms
# from .utils import _apply_scaling_array, _apply_scaling_cov
# from .utils import _undo_scaling_array, _undo_scaling_cov
# from .proj import setup_proj


def estimate_rank(data, tol='auto', return_singular=False, norm=True):
    """Estimate the rank of data.

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one.

    Parameters
    ----------
    data : array
        Data to estimate the rank of (should be 2-dimensional).
    tol : float | 'auto'
        Tolerance for singular values to consider non-zero in
        calculating the rank. The singular values are calculated
        in this method such that independent data are expected to
        have singular value around one. Can be 'auto' to use the
        same thresholding as ``scipy.linalg.orth``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    norm : bool
        If True, data will be scaled by their estimated row-wise norm.
        Else data are assumed to be scaled. Defaults to True.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    data = data.copy()  # operate on a copy
    if norm is True:
        norms = _compute_row_norms(data)
        data /= norms[:, np.newaxis]
    s = linalg.svd(data, compute_uv=False, overwrite_a=True)
    rank = _estimate_rank_from_s(s, tol)
    if return_singular is True:
        return rank, s
    else:
        return rank


def _estimate_rank_from_s(s, tol='auto'):
    """Estimate the rank of a matrix from its singular values.

    Parameters
    ----------
    s : list of float
        The singular values of the matrix.
    tol : float | 'auto'
        Tolerance for singular values to consider non-zero in calculating the
        rank. Can be 'auto' to use the same thresholding as
        ``scipy.linalg.orth``.

    Returns
    -------
    rank : int
        The estimated rank.
    """
    if isinstance(tol, str):
        if tol != 'auto':
            raise ValueError('tol must be "auto" or float')
        eps = np.finfo(float).eps
        tol = len(s) * np.amax(s) * eps

    tol = float(tol)
    rank = np.sum(s > tol)
    return rank
