"""Compute Linearly constrained minimum variance (LCMV) beamformer."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)
from copy import deepcopy

import numpy as np
from scipy import linalg

from ..io.constants import FIFF
from ..io.proj import make_projector
from ..io.pick import (
    pick_types, pick_channels_forward, pick_channels_cov, pick_info)
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import _get_vertno, combine_xyz, _check_reference
from ..cov import compute_whitener, compute_covariance
from ..source_estimate import _make_stc, SourceEstimate
from ..source_space import label_src_vertno_sel
from ..utils import logger, verbose, warn, estimate_rank
from .. import Epochs
from ..externals import six
from ..channels.channels import _contains_ch_type


def _deprecate_picks(info, picks):
    if picks is not None:
        warn('Specifying picks is deprecated and will be removed in 0.16. '
             'Specifying the selection of channels in info and setting picks '
             'to None will remove this warning.',
             DeprecationWarning)

        info = pick_info(info, picks)
    return info


def _reg_pinv(x, reg):
    """Compute a regularized pseudoinverse of a square array."""
    if reg == 0:
        covrank = estimate_rank(x, tol='auto', norm=False,
                                return_singular=False)
        if covrank < x.shape[0]:
            warn('Covariance matrix is rank-deficient, but no regularization '
                 'is done.')

    # This adds it to the diagonal without using np.eye
    d = reg * np.trace(x) / len(x)
    x.flat[::x.shape[0] + 1] += d
    return linalg.pinv(x), d


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
    for ii, ch_name in enumerate(ch_names):
        if ch_name in filters['ch_names']:
            sel.append(ii)
    return sel


def _check_cov_matrix(data_cov):
    if data_cov is None:
        raise ValueError('Source reconstruction with beamformers requires '
                         'a data covariance matrix.')


@verbose
def make_lcmv(info, forward, data_cov, reg=0.05, noise_cov=None, label=None,
              pick_ori=None, rank=None, weight_norm='unit-noise-gain',
              reduce_rank=False, verbose=None):
    """Compute LCMV spatial filter.

    Parameters
    ----------
    info : dict
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are not used.
    forward : dict
        Forward operator.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    noise_cov : Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.
    label : Label
        Restricts the LCMV solution to a given label.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
        If None, the solution depends on the forward model: if the orientation
        is fixed, a scalar beamformer is computed. If the forward model has
        free orientation, a vector beamformer is computed, combining the output
        for all source orientations.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if 'nai', the Neural Activity Index [1]_ will be computed,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    reduce_rank : bool
        If True, the rank of the leadfield will be reduced by 1 for each
        spatial location. Setting reduce_rank to True is typically necessary
        if you use a single sphere model for MEG.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    filters | dict
        Beamformer weights.

    Notes
    -----
    The original reference is [1]_.

    References
    ----------
    .. [1] Van Veen et al. Localization of brain electrical activity via
           linearly constrained minimum variance spatial filtering.
           Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    picks = _setup_picks(info, forward, data_cov, noise_cov)

    is_free_ori, ch_names, proj, vertno, G = \
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']

    # check number of sensor types present in the data
    _check_one_ch_type(info, picks, noise_cov)

    # apply SSPs
    is_ssp = False
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
        is_ssp = True

    if noise_cov is not None:
        # Handle whitening + data covariance
        whitener, _ = compute_whitener(noise_cov, info, picks, rank=rank)
        # whiten the leadfield
        G = np.dot(whitener, G)
        # whiten  data covariance
        Cm = np.dot(whitener, np.dot(Cm, whitener.T))
    else:
        whitener = None

    # Tikhonov regularization using reg parameter d to control for
    # trade-off between spatial resolution and noise sensitivity
    Cm_inv, d = _reg_pinv(Cm.copy(), reg)

    if weight_norm is not None:
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

        # Compute square of Cm_inv used for weight normalization
        Cm_inv_sq = np.dot(Cm_inv, Cm_inv)

    del Cm

    # leadfield rank and optional rank reduction
    if reduce_rank:
        if not pick_ori == 'max-power':
            raise NotImplementedError('The computation of spatial filters '
                                      'with rank reduction using reduce_rank '
                                      'parameter is only implemented with '
                                      'pick_ori=="max-power".')
        if not isinstance(reduce_rank, bool):
            raise ValueError('reduce_rank has to be True or False '
                             ' (got %s).' % reduce_rank)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        if np.all(Gk == 0.):
            continue
        Ck = np.dot(Wk, Gk)

        # Compute scalar beamformer by finding the source orientation which
        # maximizes output source power
        if pick_ori == 'max-power':
            # weight normalization and orientation selection:
            if weight_norm is not None and pick_ori == 'max-power':
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
                                         'estimating LCMV filters. Consider '
                                         'reducing the rank of the leadfield '
                                         'by using reduce_rank=True.')

                eig_vals, eig_vecs = linalg.eig(np.dot(tmp_inv,
                                                       np.dot(Wk, Gk)))

                if np.iscomplex(eig_vecs).any():
                    raise ValueError('The eigenspectrum of the leadfield at '
                                     'this voxel is complex. Consider '
                                     'reducing the rank of the leadfield by '
                                     'using reduce_rank=True.')

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

            # no weight-normalization and max-power is not implemented yet:
            else:
                raise NotImplementedError('The max-power orientation '
                                          'selection is not yet implemented '
                                          'with weight_norm set to None.')

        else:  # do vector beamformer
            # compute the filters:
            if is_free_ori:
                # Free source orientation
                Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
            else:
                # Fixed source orientation
                Wk /= Ck

            # handle noise normalization with free/normal source orientation:
            if weight_norm == 'nai':
                raise NotImplementedError('Weight normalization with neural '
                                          'activity index is not implemented '
                                          'yet with free or fixed '
                                          'orientation.')

            if weight_norm == 'unit-noise-gain':
                noise_norm = np.sum(Wk ** 2, axis=1)
                if is_free_ori:
                    noise_norm = np.sum(noise_norm)
                noise_norm = np.sqrt(noise_norm)
                if noise_norm == 0.:
                    noise_norm_inv = 0  # avoid division by 0
                else:
                    noise_norm_inv = 1. / noise_norm
                Wk[:] *= noise_norm_inv

    # Pick source orientation maximizing output source power
    if pick_ori == 'max-power':
        W = W[0::3]
    elif pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

    filters = dict(weights=W, data_cov=data_cov, noise_cov=noise_cov,
                   whitener=whitener, weight_norm=weight_norm,
                   pick_ori=pick_ori, ch_names=ch_names, proj=proj,
                   is_ssp=is_ssp, vertices=vertno, is_free_ori=is_free_ori,
                   nsource=forward['nsource'], src=deepcopy(forward['src']))

    return filters


def _subject_from_filter(filters):
    """Get subject id from inverse operator."""
    return filters['src'][0].get('subject_his_id', None)


def _check_proj_match(info, filters):
    """Check whether SSP projections in data and spatial filter match."""
    proj_data, _, _ = make_projector(info['projs'],
                                     filters['ch_names'])
    if not np.array_equal(proj_data, filters['proj']):
            raise ValueError('The SSP projections present in the data '
                             'do not match the projections used when '
                             'calculating the spatial filter.')


def _apply_lcmv(data, filters, info, tmin, max_ori_out):
    """Apply LCMV spatial filter to data for source reconstruction."""
    if max_ori_out == 'abs':
        warn('max_ori_out and the return of absolute values is deprecated and '
             'will be removed in 0.16. Set it to "signed" to remove this '
             'warning, this will return signed time series.',
             DeprecationWarning)

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    W = filters['weights']

    subject = _subject_from_forward(filters)
    for i, M in enumerate(data):
        if len(M) != len(filters['ch_names']):
            raise ValueError('data and picks must have the same length')

        if not return_single:
            logger.info("Processing epoch : %d" % (i + 1))

        if filters['is_ssp']:
            # check whether data and filter projs match
            _check_proj_match(info, filters)
            # apply projection
            M = np.dot(filters['proj'], M)

        if filters['whitener'] is not None:
            M = np.dot(filters['whitener'], M)

        # project to source space using beamformer weights
        if filters['is_free_ori']:
            sol = np.dot(W, M)
            logger.info('combining the current components...')
            sol = combine_xyz(sol)

        else:
            # Linear inverse: do computation here or delayed
            if (M.shape[0] < W.shape[0] and
                    filters['pick_ori'] != 'max-power'):
                sol = (W, M)
            else:
                sol = np.dot(W, M)
            if filters['pick_ori'] == 'max-power' and max_ori_out == 'abs':
                sol = np.abs(sol)

        tstep = 1.0 / info['sfreq']
        yield _make_stc(sol, vertices=filters['vertices'], tmin=tmin,
                        tstep=tstep, subject=subject)

    logger.info('[done]')


def _prepare_beamformer_input(info, forward, label, picks, pick_ori):
    """Input preparation common for all beamformer functions.

    Check input values, prepare channel list and gain matrix. For documentation
    of parameters, please refer to _apply_lcmv.
    """
    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if pick_ori in ['normal', 'max-power'] and not is_free_ori:
        raise ValueError('Normal or max-power orientation can only be picked '
                         'when a forward operator with free orientation is '
                         'used.')
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

    # Pick after applying the projections
    G = G[picks_forward]
    proj = proj[np.ix_(picks_forward, picks_forward)]

    return is_free_ori, ch_names, proj, vertno, G


@verbose
def apply_lcmv(evoked, filters, max_ori_out='abs', verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on evoked data.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert.
    filters : dict
        LCMV spatial filter (beamformer weights).
        Filter weights returned from `make_lcmv`.
    max_ori_out: 'abs' | 'signed'
        Specify in case of pick_ori='max-power'.
        If 'abs', the absolute value of the source space time series will be
        returned,
        if 'signed', the signed source space time series will be returned.
        'abs' is deprecated and will be removed in 0.16. Set max_ori_out to
        'signed' to remove this warning.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses.

    See Also
    --------
    apply_lcmv_raw, apply_lcmv_epochs
    """
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    sel = _pick_channels_spatial_filter(evoked.ch_names, filters)
    data = data[sel]

    stc = _apply_lcmv(data=data, filters=filters, info=info,
                      tmin=tmin, max_ori_out=max_ori_out)

    return six.advance_iterator(stc)


@verbose
def apply_lcmv_epochs(epochs, filters, max_ori_out='abs',
                      return_generator=False, verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on single trial data.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    filters : dict
        LCMV spatial filter (beamformer weights)
        Filter weights returned from `make_lcmv`.
    max_ori_out: 'abs' | 'signed'
        Specify in case of pick_ori='max-power'.
        If 'abs', the absolute value of the source space time series will be
        returned,
        if 'signed', the signed source space time series will be returned.
        'abs' is deprecated and will be removed in 0.16. Set max_ori_out to
        'signed' to remove this warning.
    return_generator : bool
         Return a generator object instead of a list. This allows iterating
         over the stcs without having to keep them all in memory.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).


    Returns
    -------
    stc: list | generator of (SourceEstimate | VolSourceEstimate)
        The source estimates for all epochs.

    See Also
    --------
    apply_lcmv_raw, apply_lcmv
    """
    _check_reference(epochs)

    info = epochs.info
    tmin = epochs.times[0]

    sel = _pick_channels_spatial_filter(epochs.ch_names, filters)
    data = epochs.get_data()[:, sel, :]
    stcs = _apply_lcmv(data=data, filters=filters, info=info,
                       tmin=tmin, max_ori_out=max_ori_out)

    if not return_generator:
        stcs = [s for s in stcs]

    return stcs


@verbose
def apply_lcmv_raw(raw, filters, start=None, stop=None, max_ori_out='abs',
                   verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on raw data.

    NOTE : This implementation has not been heavily tested so please
    report any issue or suggestions.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data to invert.
    filters : dict
        LCMV spatial filter (beamformer weights).
        Filter weights returned from `make_lcmv`.
    start : int
        Index of first time sample (index not time is seconds).
    stop : int
        Index of first time sample not to include (index not time is seconds).
    max_ori_out: 'abs' | 'signed'
        Specify in case of pick_ori='max-power'.
        If 'abs', the absolute value of the source space time series will be
        returned,
        if 'signed', the signed source space time series will be returned.
        'abs' is deprecated and will be removed in 0.16. Set max_ori_out to
        'signed' to remove this warning.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses.

    See Also
    --------
    apply_lcmv_epochs, apply_lcmv
    """
    _check_reference(raw)

    info = raw.info

    sel = _pick_channels_spatial_filter(raw.ch_names, filters)
    data, times = raw[sel, start:stop]
    tmin = times[0]

    stc = _apply_lcmv(data=data, filters=filters, info=info,
                      tmin=tmin, max_ori_out=max_ori_out)

    return six.advance_iterator(stc)


@verbose
def lcmv(evoked, forward, noise_cov=None, data_cov=None, reg=0.05, label=None,
         pick_ori=None, picks=None, rank=None, weight_norm='unit-noise-gain',
         max_ori_out='abs', reduce_rank=False, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on evoked data.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
        If None, the solution depends on the forward model: if the orientation
        is fixed, a scalar beamformer is computed. If the forward model has
        free orientation, a vector beamformer is computed, combining the output
        for all source orientations.
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
        picks is deprecated and will be removed in 0.16, use pick_channels or
        pick_types instead.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if 'nai', the Neural Activity Index [1]_ will be computed,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    max_ori_out: 'abs' | 'signed'
        Specify in case of pick_ori='max-power'.
        If 'abs', the absolute value of the source space time series will be
        returned,
        if 'signed', the signed source space time series will be returned.
        'abs' is deprecated and will be removed in 0.16. Set max_ori_out to
        'signed' to remove this warning.

        .. versionadded:: 0.15.0
    reduce_rank : bool
        If True, the rank of the leadfield will be reduced by 1 for each
        spatial location. Setting reduce_rank to True is typically necessary
        if you use a sphere model for MEG as in this case the actual
        rank is 2 not 3.

        .. versionadded:: 0.15.0
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses.

    See Also
    --------
    make_lcmv, apply_lcmv
    lcmv_raw, lcmv_epochs

    Notes
    -----
    The original reference is [1]_.

    References
    ----------
    .. [1] Van Veen et al. Localization of brain electrical activity via
           linearly constrained minimum variance spatial filtering.
           Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    info = evoked.info
    info = _deprecate_picks(info, picks)

    # check whether data covariance is supplied
    _check_cov_matrix(data_cov)

    # construct spatial filter
    filters = make_lcmv(info=info, forward=forward, data_cov=data_cov,
                        reg=reg, noise_cov=noise_cov, label=label,
                        pick_ori=pick_ori, rank=rank, weight_norm=weight_norm,
                        reduce_rank=reduce_rank)

    # apply spatial filter to evoked data
    stc = apply_lcmv(evoked=evoked, filters=filters, max_ori_out=max_ori_out)

    return stc


@verbose
def lcmv_epochs(epochs, forward, noise_cov, data_cov, reg=0.05, label=None,
                pick_ori=None, return_generator=False, picks=None, rank=None,
                weight_norm='unit-noise-gain', max_ori_out='abs',
                reduce_rank=False, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on single trial data.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
        If None, the solution depends on the forward model: if the orientation
        is fixed, a scalar beamformer is computed. If the forward model has
        free orientation, a vector beamformer is computed, combining the output
        for all source orientations.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
        picks is deprecated and will be removed in 0.16, use pick_channels or
        pick_types instead.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if 'nai', the Neural Activity Index [1]_ will be computed,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    max_ori_out: 'abs' | 'signed'
        Specify in case of pick_ori='max-power'.
        If 'abs', the absolute value of the source space time series will be
        returned,
        if 'signed', the signed source space time series will be returned.
        'abs' is deprecated and will be removed in 0.16. Set max_ori_out to
        'signed' to remove this warning.

        .. versionadded:: 0.15.0
    reduce_rank : bool
        If True, the rank of the leadfield will be reduced by 1 for each
        spatial location. Setting reduce_rank to True is typically necessary
        if you use a sphere model for MEG as in this case the actual
        rank is 2 not 3.

        .. versionadded:: 0.15.0
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc: list | generator of (SourceEstimate | VolSourceEstimate)
        The source estimates for all epochs.

    See Also
    --------
    make_lcmv, apply_lcmv_epochs
    lcmv_raw, lcmv

    Notes
    -----
    The original reference is [1]_.

    References
    ----------
    .. [1] Van Veen et al. Localization of brain electrical activity via
           linearly constrained minimum variance spatial filtering.
           Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    info = epochs.info
    info = _deprecate_picks(info, picks)

    # check whether data covariance is supplied
    _check_cov_matrix(data_cov)

    # construct spatial filter
    filters = make_lcmv(info=info, forward=forward, data_cov=data_cov,
                        reg=reg, noise_cov=noise_cov, label=label,
                        pick_ori=pick_ori, rank=rank, weight_norm=weight_norm,
                        reduce_rank=reduce_rank)

    # apply spatial filter to epochs
    stcs = apply_lcmv_epochs(epochs=epochs, filters=filters,
                             max_ori_out=max_ori_out,
                             return_generator=return_generator)

    return stcs


@verbose
def lcmv_raw(raw, forward, noise_cov, data_cov, reg=0.05, label=None,
             start=None, stop=None, picks=None, pick_ori=None, rank=None,
             weight_norm='unit-noise-gain', max_ori_out='abs',
             reduce_rank=False, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on raw data.
    NOTE : This implementation has not been heavily tested so please
    report any issue or suggestions.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data to invert.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label.
    start : int
        Index of first time sample (index not time is seconds).
    stop : int
        Index of first time sample not to include (index not time is seconds).
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
        picks is deprecated and will be removed in 0.16, use pick_channels or
        pick_types instead.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
        If None, the solution depends on the forward model: if the orientation
        is fixed, a scalar beamformer is computed. If the forward model has
        free orientation, a vector beamformer is computed, combining the output
        for all source orientations.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if 'nai', the Neural Activity Index [1]_ will be computed,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    max_ori_out: 'abs' | 'signed'
        Specify in case of pick_ori='max-power'.
        If 'abs', the absolute value of the source space time series will be
        returned,
        if 'signed', the signed source space time series will be returned.
        'abs' is deprecated and will be removed in 0.16. Set max_ori_out to
        'signed' to remove this warning.
    reduce_rank : bool
        If True, the rank of the leadfield will be reduced by 1 for each
        spatial location. Setting reduce_rank to True is typically necessary
        if you use a sphere model for MEG as in this case the actual
        rank is 2 not 3.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses.

    See Also
    --------
    make_lcmv, apply_lcmv_raw
    lcmv, lcmv_epochs

    Notes
    -----
    The original reference is [1]_.

    References
    ----------
    .. [1] Van Veen et al. Localization of brain electrical activity via
           linearly constrained minimum variance spatial filtering.
           Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    info = raw.info
    info = _deprecate_picks(info, picks)

    # check whether data covariance is supplied
    _check_cov_matrix(data_cov)

    # construct spatial filter
    filters = make_lcmv(info=info, forward=forward, data_cov=data_cov,
                        reg=reg, noise_cov=noise_cov, label=label,
                        pick_ori=pick_ori, rank=rank, weight_norm=weight_norm,
                        reduce_rank=reduce_rank)

    # apply spatial filter to epochs
    stc = apply_lcmv_raw(raw=raw, filters=filters, start=start, stop=stop,
                         max_ori_out=max_ori_out)

    return stc


@verbose
def _lcmv_source_power(info, forward, noise_cov, data_cov, reg=0.05,
                       label=None, picks=None, pick_ori=None, rank=None,
                       weight_norm=None, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer."""
    if weight_norm not in [None, 'unit-noise-gain']:
        raise ValueError('Unrecognized weight normalization option in '
                         'weight_norm, available choices are None and '
                         '"unit-noise-gain", got "%s".' % weight_norm)

    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')

    is_free_ori, ch_names, proj, vertno, G =\
        _prepare_beamformer_input(
            info, forward, label, picks, pick_ori)

    # Handle whitening
    info = pick_info(
        info, [info['ch_names'].index(k) for k in ch_names
               if k in info['ch_names']])

    if noise_cov is not None:
        whitener, _ = compute_whitener(noise_cov, info, picks, rank=rank)

        # whiten the leadfield
        G = np.dot(whitener, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))

    if noise_cov is not None:
        Cm = np.dot(whitener, np.dot(Cm, whitener.T))

    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # This modifies Cm inplace, regularizing it
    Cm_inv, d = _reg_pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    source_power = np.zeros((n_sources, 1))
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        if weight_norm == 'unit-noise-gain':
            # Noise normalization
            noise_norm = np.dot(Wk, Wk.T)
            noise_norm = noise_norm.trace()

        # Calculating source power
        sp_temp = np.dot(np.dot(Wk, Cm), Wk.T)
        if weight_norm == 'unit-noise-gain':
            sp_temp /= max(noise_norm, 1e-40)  # Avoid division by 0

        if pick_ori == 'normal':
            source_power[k, 0] = sp_temp[2, 2]
        else:
            source_power[k, 0] = sp_temp.trace()

    logger.info('[done]')

    subject = _subject_from_forward(forward)
    return SourceEstimate(source_power, vertices=vertno, tmin=1,
                          tstep=1, subject=subject)


@verbose
def tf_lcmv(epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
            freq_bins, subtract_evoked=False, reg=0.05, label=None,
            pick_ori=None, n_jobs=1, picks=None, rank=None,
            weight_norm='unit-noise-gain', raw=None, verbose=None):
    """5D time-frequency beamforming based on LCMV.

    Calculate source power in time-frequency windows using a spatial filter
    based on the Linearly Constrained Minimum Variance (LCMV) beamforming
    approach [1]_. Band-pass filtered epochs are divided into time windows
    from which covariance is computed and used to create a beamformer
    spatial filter.

    .. note:: This implementation has not been heavily tested so please
              report any issues or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs. It is recommended to pass epochs that have
        been constructed with ``preload=False`` (i.e., not preloaded or
        read from disk) so that the parameter ``raw=None`` can be used
        below, as this ensures the correct :class:`mne.io.Raw` instance is
        used for band-pass filtering.
    forward : dict
        Forward operator.
    noise_covs : list of instances of Covariance | None
        Noise covariance for each frequency bin. If provided, whitening will be
        done. Providing noise covariances is mandatory if you mix sensor types,
        e.g., gradiometers with magnetometers or EEG with MEG.
    tmin : float
        Minimum time instant to consider.
    tmax : float
        Maximum time instant to consider.
    tstep : float
        Spacing between consecutive time windows, should be smaller than or
        equal to the shortest time window length.
    win_lengths : list of float
        Time window lengths in seconds. One time window length should be
        provided for each frequency bin.
    freq_bins : list of tuples of float
        Start and end point of frequency bins of interest.
    subtract_evoked : bool
        If True, subtract the averaged evoked response prior to computing the
        tf source grid.
    reg : float
        The regularization for the whitened data covariance.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
        If None, the solution depends on the forward model: if the orientation
        is fixed, a scalar beamformer is computed. If the forward model has
        free orientation, a vector beamformer is computed, combining the output
        for all source orientations.
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
        picks is deprecated and will be removed in 0.16, use pick_channels
        or pick_types instead.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    weight_norm : 'unit-noise-gain' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    raw : instance of Raw | None
        The raw instance used to construct the epochs.
        Must be provided unless epochs are constructed with
        ``preload=False``.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stcs : list of SourceEstimate
        Source power at each time window. One SourceEstimate object is returned
        for each frequency bin.

    References
    ----------
    .. [1] Dalal et al. Five-dimensional neuroimaging: Localization of the
           time-frequency dynamics of cortical activity.
           NeuroImage (2008) vol. 40 (4) pp. 1686-1700
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    info = epochs.info
    info = _deprecate_picks(info, picks)

    _check_reference(epochs)

    if pick_ori not in [None, 'normal']:
        raise ValueError('Unrecognized orientation option in pick_ori, '
                         'available choices are None and normal')
    if noise_covs is not None and len(noise_covs) != len(freq_bins):
        raise ValueError('One noise covariance object expected per frequency '
                         'bin')
    if len(win_lengths) != len(freq_bins):
        raise ValueError('One time window length expected per frequency bin')
    if any(win_length < tstep for win_length in win_lengths):
        raise ValueError('Time step should not be larger than any of the '
                         'window lengths')

    # Extract raw object from the epochs object
    raw = epochs._raw if raw is None else raw
    if raw is None:
        raise ValueError('The provided epochs object does not contain the '
                         'underlying raw object. Please use preload=False '
                         'when constructing the epochs object or pass the '
                         'underlying raw instance to this function')

    if noise_covs is None:
        picks = _setup_picks(info, forward, data_cov=None)
    else:
        picks = _setup_picks(info, forward, data_cov=None,
                             noise_cov=noise_covs[0])
    ch_names = [epochs.ch_names[k] for k in picks]

    # check number of sensor types present in the data
    _check_one_ch_type(info, picks, noise_covs)

    # Use picks from epochs for picking channels in the raw object
    raw_picks = [raw.ch_names.index(c) for c in ch_names]

    # Make sure epochs.events contains only good events:
    epochs.drop_bad()

    # Multiplying by 1e3 to avoid numerical issues, e.g. 0.3 // 0.05 == 5
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))

    # create a list to iterate over if no noise covariances are given
    if noise_covs is None:
        noise_covs = [None] * len(win_lengths)

    sol_final = []
    for (l_freq, h_freq), win_length, noise_cov in \
            zip(freq_bins, win_lengths, noise_covs):
        n_overlap = int((win_length * 1e3) // (tstep * 1e3))

        raw_band = raw.copy()
        raw_band.filter(l_freq, h_freq, picks=raw_picks, method='iir',
                        n_jobs=n_jobs, iir_params=dict(output='ba'))
        raw_band.info['highpass'] = l_freq
        raw_band.info['lowpass'] = h_freq
        epochs_band = Epochs(raw_band, epochs.events, epochs.event_id,
                             tmin=epochs.tmin, tmax=epochs.tmax, baseline=None,
                             picks=raw_picks, proj=epochs.proj, preload=True)
        del raw_band

        if subtract_evoked:
            epochs_band.subtract_evoked()

        sol_single = []
        sol_overlap = []
        for i_time in range(n_time_steps):
            win_tmin = tmin + i_time * tstep
            win_tmax = win_tmin + win_length

            # If in the last step the last time point was not covered in
            # previous steps and will not be covered now, a solution needs to
            # be calculated for an additional time window
            if i_time == n_time_steps - 1 and win_tmax - tstep < tmax and\
               win_tmax >= tmax + (epochs.times[-1] - epochs.times[-2]):
                warn('Adding a time window to cover last time points')
                win_tmin = tmax - win_length
                win_tmax = tmax

            if win_tmax < tmax + (epochs.times[-1] - epochs.times[-2]):
                logger.info('Computing time-frequency LCMV beamformer for '
                            'time window %d to %d ms, in frequency range '
                            '%d to %d Hz' % (win_tmin * 1e3, win_tmax * 1e3,
                                             l_freq, h_freq))

                # Counteracts unsafe floating point arithmetic ensuring all
                # relevant samples will be taken into account when selecting
                # data in time windows
                win_tmin = win_tmin - 1e-10
                win_tmax = win_tmax + 1e-10

                # Calculating data covariance from filtered epochs in current
                # time window
                data_cov = compute_covariance(epochs_band, tmin=win_tmin,
                                              tmax=win_tmax)

                stc = _lcmv_source_power(epochs_band.info, forward, noise_cov,
                                         data_cov, reg=reg, label=label,
                                         pick_ori=pick_ori,
                                         weight_norm=weight_norm,
                                         verbose=verbose)
                sol_single.append(stc.data[:, 0])

            # Average over all time windows that contain the current time
            # point, which is the current time window along with
            # n_overlap - 1 previous ones
            if i_time - n_overlap < 0:
                curr_sol = np.mean(sol_single[0:i_time + 1], axis=0)
            else:
                curr_sol = np.mean(sol_single[i_time - n_overlap + 1:
                                              i_time + 1], axis=0)

            # The final result for the current time point in the current
            # frequency bin
            sol_overlap.append(curr_sol)

        # Gathering solutions for all time points for current frequency bin
        sol_final.append(sol_overlap)

    sol_final = np.array(sol_final)

    # Creating stc objects containing all time points for each frequency bin
    stcs = []
    for i_freq, _ in enumerate(freq_bins):
        stc = SourceEstimate(sol_final[i_freq, :, :].T, vertices=stc.vertices,
                             tmin=tmin, tstep=tstep, subject=stc.subject)
        stcs.append(stc)

    return stcs
