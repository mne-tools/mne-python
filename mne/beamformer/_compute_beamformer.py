"""Functions shared between different beamformer types."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Roman Goj <roman.goj@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
from scipy import linalg

from ..io.constants import FIFF
from ..io.proj import make_projector
from ..io.pick import (pick_channels_forward, pick_info)
from ..minimum_norm.inverse import _get_vertno
from ..source_space import label_src_vertno_sel
from ..utils import logger, warn, estimate_rank
from ..channels.channels import _contains_ch_type


def _reg_pinv(x, reg, rcond=1e-15):
    """Compute a regularized pseudoinverse of a square array.

    Parameters
    ----------
    x : ndarray, shape (n, n)
        Square array to invert.
    reg : float
        Regularization parameter.
    rcond : float | 'auto'
        Cutoff for small singular values. Singular values smaller (in modulus)
        than `rcond` * largest_singular_value (again, in modulus) are set to
        zero. Use 'auto' to attempt to automatically set a sane value. Defaults
        to 1e-15.
    """
    covrank, s = estimate_rank(x, tol='auto', norm=False, return_singular=True)

    # This adds the regularization without using np.eye
    d = reg * np.trace(x) / len(x)
    x = x.copy()
    x.flat[::x.shape[0] + 1] += d

    if covrank < len(x):
        if reg == 0:
            warn('Covariance matrix is rank-deficient and no regularization '
                 'is done.')

        if rcond == 'auto':
            # Reduce the toleration of the pseudo-inverse to force a solution
            s = linalg.svd(x, compute_uv=False)
            tol = s[covrank - 1:covrank + 1].mean()
            tol = max(
                tol,
                len(x) * linalg.norm(x) * np.finfo(float).eps
            )
            rcond = tol / s.max()

    if rcond == 'auto':
        rcond = 1e-15

    return linalg.pinv(x, rcond=rcond), d


def _eig_inv(x, rank):
    """Compute a pseudoinverse with smallest component set to zero."""
    U, s, V = linalg.svd(x)

    # pseudoinverse is computed by setting eigenvalues not included in
    # signalspace to zero
    s_inv = np.zeros(s.shape)
    s_inv[:rank] = 1. / s[:rank]

    x_inv = np.dot(V.T, s_inv[:, np.newaxis] * U.T)
    return x_inv


def _setup_picks(info, forward, data_cov=None, noise_cov=None):
    """Return good channels common to forward model and covariance matrices."""
    # get a list of all channel names:
    fwd_ch_names = forward['info']['ch_names']

    # handle channels from forward model and info:
    ch_names = _compare_ch_names(info['ch_names'], fwd_ch_names, info['bads'])

    # inform about excluding channels:
    if (data_cov is not None and set(info['bads']) != set(data_cov['bads']) and
            (len(set(ch_names).intersection(data_cov['bads'])) > 0)):
        logger.info('info["bads"] and data_cov["bads"] do not match, '
                    'excluding bad channels from both.')
    if (noise_cov is not None and
            set(info['bads']) != set(noise_cov['bads']) and
            (len(set(ch_names).intersection(noise_cov['bads'])) > 0)):
        logger.info('info["bads"] and noise_cov["bads"] do not match, '
                    'excluding bad channels from both.')

    # handle channels from data cov if data cov is not None
    # Note: data cov is supposed to be None in tf_lcmv
    if data_cov is not None:
        ch_names = _compare_ch_names(ch_names, data_cov.ch_names,
                                     data_cov['bads'])

    # handle channels from noise cov if noise cov available:
    if noise_cov is not None:
        ch_names = _compare_ch_names(ch_names, noise_cov.ch_names,
                                     noise_cov['bads'])

    picks = [info['ch_names'].index(k) for k in ch_names if k in
             info['ch_names']]
    return picks


def _compare_ch_names(names1, names2, bads):
    """Return channel names of common and good channels."""
    ch_names = [ch for ch in names1 if ch not in bads and ch in names2]
    return ch_names


def _check_one_ch_type(info, picks, noise_cov):
    """Check number of sensor types and presence of noise covariance matrix."""
    info_pick = pick_info(info, sel=picks)
    ch_types =\
        [_contains_ch_type(info_pick, tt) for tt in ('mag', 'grad', 'eeg')]
    if sum(ch_types) > 1 and noise_cov is None:
        raise ValueError('Source reconstruction with several sensor types '
                         'requires a noise covariance matrix to be '
                         'able to apply whitening.')


def _pick_channels_spatial_filter(ch_names, filters):
    """Return data channel indices to be used with spatial filter.

    Unlike ``pick_channels``, this respects the order of ch_names.
    """
    sel = []
    # first check for channel discrepancies between filter and data:
    for ch_name in filters['ch_names']:
        if ch_name not in ch_names:
            raise ValueError('The spatial filter was computed with channel %s '
                             'which is not present in the data. You should '
                             'compute a new spatial filter restricted to the '
                             'good data channels.' % ch_name)
    # then compare list of channels and get selection based on data:
    sel = [ii for ii, ch_name in enumerate(ch_names)
           if ch_name in filters['ch_names']]
    return sel


def _check_cov_matrix(data_cov):
    if data_cov is None:
        raise ValueError('Source reconstruction with beamformers requires '
                         'a data covariance matrix.')


def _check_proj_match(info, filters):
    """Check whether SSP projections in data and spatial filter match."""
    proj_data, _, _ = make_projector(info['projs'],
                                     filters['ch_names'])
    if not np.array_equal(proj_data, filters['proj']):
            raise ValueError('The SSP projections present in the data '
                             'do not match the projections used when '
                             'calculating the spatial filter.')


def _prepare_beamformer_input(info, forward, label, picks, pick_ori,
                              fwd_norm=None):
    """Input preparation common for all beamformer functions.

    Check input values, prepare channel list and gain matrix. For documentation
    of parameters, please refer to _apply_lcmv.
    """
    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if pick_ori in ['normal', 'max-power', 'vector']:
        if not is_free_ori:
            raise ValueError(
                'Normal or max-power orientation can only be picked '
                'when a forward operator with free orientation is used.')
    elif pick_ori is not None:
        raise ValueError('pick_ori must be one of "normal", "max-power", '
                         '"vector", or None, got %s' % (pick_ori,))
    if pick_ori == 'normal' and not forward['surf_ori']:
        # XXX eventually this could just call convert_forward_solution
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')
    # Restrict forward solution to selected channels
    info_ch_names = [ch['ch_name'] for ch in info['chs']]
    ch_names = [info_ch_names[k] for k in picks]
    fwd_ch_names = forward['sol']['row_names']
    # Keep channels in forward present in info:
    fwd_ch_names = [ch for ch in fwd_ch_names if ch in info_ch_names]
    forward = pick_channels_forward(forward, fwd_ch_names)
    picks_forward = [fwd_ch_names.index(ch) for ch in ch_names]

    # Get gain matrix (forward operator)
    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, forward['src'])

        if is_free_ori:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        G = forward['sol']['data'][:, src_sel]
    else:
        vertno = _get_vertno(forward['src'])
        G = forward['sol']['data']

    # Apply SSPs
    proj, ncomp, _ = make_projector(info['projs'], fwd_ch_names)

    if info['projs']:
        G = np.dot(proj, G)

    # Pick after applying the projections. This makes a copy of G, so further
    # operations can be safely done in-place.
    G = G[picks_forward]
    proj = proj[np.ix_(picks_forward, picks_forward)]

    # Normalize the leadfield if requested
    if fwd_norm == 'dipole':  # each orientation separately
        G /= np.linalg.norm(G, axis=0)
    elif fwd_norm == 'vertex':  # all three orientations per loc jointly
        depth_prior = np.sum(G ** 2, axis=0)
        if is_free_ori:
            depth_prior = depth_prior.reshape(-1, 3).sum(axis=1)
        # Spherical leadfield can be zero at the center
        depth_prior[depth_prior == 0.] = np.min(
            depth_prior[depth_prior != 0.])
        if is_free_ori:
            depth_prior = np.repeat(depth_prior, 3)
        source_weighting = np.sqrt(1. / depth_prior)
        G *= source_weighting[np.newaxis, :]
    elif fwd_norm is not None:
        raise ValueError('Got invalid value for "fwd_norm". Valid '
                         'values are: "dipole", "vertex" or None.')

    return is_free_ori, ch_names, proj, vertno, G
