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


def _check_one_ch_type(info, picks, noise_cov, method):
    """Check number of sensor types and presence of noise covariance matrix."""
    # XXX : ugly hack to avoid picking subset of info with applied comps
    comps = info['comps']
    info['comps'] = []
    info_pick = pick_info(info, sel=picks)
    info['comps'] = comps
    ch_types =\
        [_contains_ch_type(info_pick, tt) for tt in ('mag', 'grad', 'eeg')]
    if method == 'lcmv' and sum(ch_types) > 1 and noise_cov is None:
        raise ValueError('Source reconstruction with several sensor types '
                         'requires a noise covariance matrix to be '
                         'able to apply whitening.')
    elif method == 'dics' and sum(ch_types) > 1:
        warn('The use of several sensor types with the DICS beamformer is '
             'not heavily tested yet.')


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


def _check_proj_match(info, filters):
    """Check whether SSP projections in data and spatial filter match."""
    proj_data, _, _ = make_projector(info['projs'],
                                     filters['ch_names'])
    if not np.array_equal(proj_data, filters['proj']):
            raise ValueError('The SSP projections present in the data '
                             'do not match the projections used when '
                             'calculating the spatial filter.')


def _check_src_type(filters):
    """Check whether src_type is in filters and set custom warning."""
    if 'src_type' not in filters:
        filters['src_type'] = None
    warn_text = ('The spatial filter does not contain src_type and a robust '
                 'guess of src_type is not possible without src. Consider '
                 'recomputing the filter.')
    return filters, warn_text


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


def _compute_beamformer(method, G, Cm, reg, n_orient, weight_norm,
                        pick_ori, reduce_rank, rank, is_free_ori,
                        inversion=None):
    """Compute a spatial filter (LCMV or DICS)."""
    # Tikhonov regularization using reg parameter d to control for
    # trade-off between spatial resolution and noise sensitivity
    if method == 'lcmv':
        Cm_inv, d = _reg_pinv(Cm.copy(), reg)
    elif method == 'dics':
        Cm_inv, _ = _reg_pinv(Cm, reg, rcond='auto')

    if weight_norm is not None and inversion is not 'single':
        # Compute square of Cm_inv used for weight normalization
        Cm_inv_sq = np.dot(Cm_inv, Cm_inv)

        if weight_norm == 'nai':
            # estimate noise level based on covariance matrix, taking the
            # smallest eigenvalue that is not zero
            noise, _ = linalg.eigh(Cm)
            if rank is not None:
                rank_Cm = rank
            else:
                rank_Cm = estimate_rank(Cm, tol='auto', norm=False,
                                        return_singular=False)
                noise = noise[len(noise) - rank_Cm]

                # use either noise floor or regularization parameter d
                noise = max(noise, d)

    # compute spatial filter
    W = np.dot(G.T, Cm_inv)
    n_sources = G.shape[1] // n_orient

    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        if method == 'lcmv' and np.all(Gk == 0.):
            continue
        Ck = np.dot(Wk, Gk)

        if method == 'dics':
            # Normalize the spatial filters:
            if Wk.ndim == 2 and len(Wk) > 1:
                # Free source orientation
                if inversion == 'single':
                    # Invert for each dipole separately using plain division
                    Wk /= np.diag(Ck)[:, np.newaxis]
                elif inversion == 'matrix':
                    # Invert for all dipoles simultaneously using matrix
                    # inversion.
                    Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
            else:
                # Fixed source orientation
                Wk /= Ck

        # compute scalar beamformer by finding the source orientation
        # which maximizes output source power
        if pick_ori == 'max-power':
            if weight_norm is not None and inversion is not 'single':
                # finding optimal orientation for NAI and unit-noise-gain
                # based on [2]_, Eq. 4.47
                tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))

                if reduce_rank:
                    # use pseudo inverse computation setting smallest component
                    # to zero if the leadfield is not full rank
                    tmp_inv = _eig_inv(tmp, tmp.shape[0] - 1)
                else:
                    # use straight inverse with full rank leadfield
                    try:
                        tmp_inv = linalg.inv(tmp)
                    except np.linalg.linalg.LinAlgError:
                        raise ValueError('Singular matrix detected when '
                                         'estimating spatial filters. '
                                         'Consider reducing the rank of the '
                                         'leadfield by using  '
                                         'reduce_rank=True.')

                power = np.dot(tmp_inv, np.dot(Wk, Gk))

            elif weight_norm is not None and inversion == 'single':
                # First make the filters unit gain, then apply them to the
                # CSD matrix to compute power.
                norm = 1 / np.sqrt(np.sum(Wk ** 2, axis=1))
                Wk_norm = Wk / norm[:, np.newaxis]
                power = Wk_norm.dot(Cm).dot(Wk_norm.T)

            else:
                if method == 'dics':
                    # Compute spectral power by applying the spatial filters to
                    # the CSD matrix.
                    power = Wk.dot(Cm).dot(Wk.T)
                elif method == 'lcmv':
                    # no weight-normalization and max-power is not implemented
                    # yet for lcmv beamformer:
                    raise NotImplementedError('The max-power orientation '
                                              'selection is not yet '
                                              'implemented with weight_norm '
                                              'set to None.')

            # compute the orientation:
            if method == 'lcmv':
                eig_vals, eig_vecs = linalg.eig(power)

                if np.iscomplex(eig_vecs).any():
                    raise ValueError('The eigenspectrum of the leadfield '
                                     'at this voxel is complex. Consider '
                                     'reducing the rank of the leadfield '
                                     'by using reduce_rank=True.')

                idx_max = eig_vals.argmax()
                max_ori = eig_vecs[:, idx_max]
                Wk[:] = np.dot(max_ori, Wk)
                Gk = np.dot(Gk, max_ori)

                # compute spatial filter for NAI or unit-noise-gain
                tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
                denom = np.sqrt(tmp)
                Wk /= denom
                if weight_norm == 'nai':
                    Wk /= np.sqrt(noise)

                is_free_ori = False

            elif method == 'dics':
                # Compute the direction of max power
                u, s, _ = np.linalg.svd(power.real)
                max_ori = u[:, 0]

                Wk[:] = np.dot(max_ori, Wk)

        else:  # do vector beamformer
            if method == 'lcmv':
                # compute the filters:
                if is_free_ori:
                    # Free source orientation
                    Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
                else:
                    # Fixed source orientation
                    Wk /= Ck

                # handle noise normalization with free/normal source
                # orientation:
                if weight_norm == 'nai':
                    raise NotImplementedError('Weight normalization with '
                                              'neural activity index is not '
                                              'implemented yet with free or '
                                              'fixed orientation.')

                elif weight_norm == 'unit-noise-gain':
                    noise_norm = np.sum(Wk ** 2, axis=1)
                    if is_free_ori:
                        noise_norm = np.sum(noise_norm)
                    noise_norm = np.sqrt(noise_norm)
                    if noise_norm == 0.:
                        noise_norm_inv = 0.  # avoid division by 0
                    else:
                        noise_norm_inv = 1. / noise_norm
                    Wk[:] *= noise_norm_inv

    # picking source orientation maximizing output source power
    if pick_ori == 'max-power':
        W = W[0::3]
    elif pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

    if method == 'dics':
        if weight_norm == 'unit-noise-gain':
            # Scale weights so that W @ I @ W.T == I
            if pick_ori is None and n_orient > 1:
                # Compute the norm for each set of 3 dipoles
                W = W.reshape(-1, 3, W.shape[1])
                norm = np.sqrt(np.sum(W ** 2, axis=(1, 2)))
                W /= norm[:, np.newaxis, np.newaxis]
                W = W.reshape(-1, W.shape[2])
            else:
                # Compute the norm for each dipole
                norm = np.sqrt(np.sum(W ** 2, axis=1))
                W /= norm[:, np.newaxis]

    return W, is_free_ori
