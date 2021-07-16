# -*- coding: utf-8 -*-
"""Compute resolution matrix for beamformers."""
# Authors: olaf.hauk@mrc-cbu.cam.ac.uk
#
# License: BSD-3-Clause
import numpy as np

from ..io.pick import pick_channels, pick_info, pick_channels_forward
from ..evoked import EvokedArray
from ..utils import logger, fill_doc
from ._lcmv import apply_lcmv


@fill_doc
def make_lcmv_resolution_matrix(filters, forward, info):
    """Compute resolution matrix for LCMV beamformer.

    Parameters
    ----------
    filters : instance of Beamformer
         Dictionary containing filter weights from LCMV beamformer
         (see mne.beamformer.make_lcmv).
    forward : instance of Forward
        Forward Solution with leadfield matrix.
    %(info_not_none)s Used to compute LCMV filters.

    Returns
    -------
    resmat : array, shape (n_dipoles_lcmv, n_dipoles_fwd)
        Resolution matrix (filter matrix multiplied to leadfield from
        forward solution). Numbers of rows (n_dipoles_lcmv) and columns
        (n_dipoles_fwd) may differ by a factor depending on orientation
        constraints of filter and forward solution, respectively (e.g. factor 3
        for free dipole orientation versus factor 1 for scalar beamformers).
    """
    # don't include bad channels from noise covariance matrix
    bads_filt = filters['noise_cov']['bads']
    ch_names = filters['noise_cov']['names']

    # good channels
    ch_names = [c for c in ch_names if (c not in bads_filt)]

    # adjust channels in forward solution
    forward = pick_channels_forward(forward, ch_names, ordered=True)

    # get leadfield matrix from forward solution
    leadfield = forward['sol']['data']

    # get the filter weights for beamformer as matrix
    filtmat = _get_matrix_from_lcmv(filters, forward, info)

    # compute resolution matrix
    resmat = filtmat.dot(leadfield)

    shape = resmat.shape

    logger.info('Dimensions of LCMV resolution matrix: %d by %d.' % shape)

    return resmat


def _get_matrix_from_lcmv(filters, forward, info, max_ori_out='signed',
                          verbose=None):
    """Get inverse matrix for LCMV beamformer.

    Parameters
    ----------
    filters : instance of Beamformer
        LCMV spatial filter.
    forward : instance of Forward
        The forward solution.
    %(info_not_none)s Used to compute the LCMV filters.
    max_ori_out : str
        As for beamformer.apply_lcmv(). Default 'signed'.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose() and
        Logging documentation for more).

    Returns
    -------
    invmat : array, shape (n_dipoles, n_channels)
        Inverse matrix associated with LCMV beamformer filters.
    """
    # number of channels for identity matrix
    info = pick_info(
        info, pick_channels(info['ch_names'], filters['ch_names']))
    n_chs = len(info['ch_names'])

    # create identity matrix as input for inverse operator
    # set elements to zero for non-selected channels
    id_mat = np.eye(n_chs)

    # convert identity matrix to evoked data type (pretending it's an epochs
    evo_ident = EvokedArray(id_mat, info=info, tmin=0.)

    # apply beamformer to identity matrix
    stc_lcmv = apply_lcmv(evo_ident, filters, max_ori_out='signed',
                          verbose=verbose)

    # turn source estimate into numpsy array
    invmat = stc_lcmv.data

    return invmat
