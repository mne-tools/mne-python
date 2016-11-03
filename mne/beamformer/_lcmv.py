"""Compute Linearly constrained minimum variance (LCMV) beamformer."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

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
from ..utils import logger, verbose, warn
from .. import Epochs
from ..externals import six


def _setup_picks(picks, info, forward, noise_cov=None):
    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')

    ok_ch_names = set([c['ch_name'] for c in forward['info']['chs']])
    if noise_cov is not None:
        ok_ch_names.union(set(noise_cov.ch_names))

    if noise_cov is not None and set(info['bads']) != set(noise_cov['bads']):
        logger.info('info["bads"] and noise_cov["bads"] do not match, '
                    'excluding bad channels from both')

    bads = set(info['bads'])
    if noise_cov is not None:
        bads.union(set(noise_cov['bads']))

    ok_ch_names -= bads

    ch_names = [info['chs'][k]['ch_name'] for k in picks]
    ch_names = [c for c in ch_names if c in ok_ch_names]

    picks = [info['ch_names'].index(k) for k in ch_names if k in
             info['ch_names']]
    return picks


@verbose
def _apply_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                label=None, picks=None, pick_ori=None, rank=None,
                verbose=None):
    """LCMV beamformer for evoked data, single epochs, and raw data.

    Parameters
    ----------
    data : array or list / iterable
        Sensor space data. If data.ndim == 2 a single observation is assumed
        and a single stc is returned. If data.ndim == 3 or if data is
        a list / iterable, a list of stc's is returned.
    info : dict
        Measurement info.
    tmin : float
        Time of first sample.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate (or list of thereof)
        Source time courses.
    """
    is_free_ori, ch_names, proj, vertno, G = \
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, info, picks, rank=rank)

    # whiten the leadfield
    G = np.dot(whitener, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
    Cm = np.dot(whitener, np.dot(Cm, whitener.T))

    # Calculating regularized inverse, equivalent to an inverse operation after
    # the following regularization:
    # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        # Find source orientation maximizing output source power
        if pick_ori == 'max-power':
            eig_vals, eig_vecs = linalg.eigh(Ck)

            # Choosing the eigenvector associated with the middle eigenvalue.
            # The middle and not the minimal eigenvalue is used because MEG is
            # insensitive to one (radial) of the three dipole orientations and
            # therefore the smallest eigenvalue reflects mostly noise.
            for i in range(3):
                if i != eig_vals.argmax() and i != eig_vals.argmin():
                    idx_middle = i

            # TODO: The eigenvector associated with the smallest eigenvalue
            # should probably be used when using combined EEG and MEG data
            max_ori = eig_vecs[:, idx_middle]

            Wk[:] = np.dot(max_ori, Wk)
            Ck = np.dot(max_ori, np.dot(Ck, max_ori))
            is_free_ori = False

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

    # Pick source orientation maximizing output source power
    if pick_ori == 'max-power':
        W = W[0::3]

    # Preparing noise normalization
    noise_norm = np.sum(W ** 2, axis=1)
    if is_free_ori:
        noise_norm = np.sum(np.reshape(noise_norm, (-1, 3)), axis=1)
    noise_norm = np.sqrt(noise_norm)

    # Pick source orientation normal to cortical surface
    if pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

    # Applying noise normalization
    if not is_free_ori:
        W /= noise_norm[:, None]

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    subject = _subject_from_forward(forward)
    for i, M in enumerate(data):
        if len(M) != len(picks):
            raise ValueError('data and picks must have the same length')

        if not return_single:
            logger.info("Processing epoch : %d" % (i + 1))

        # SSP and whitening
        if info['projs']:
            M = np.dot(proj, M)
        M = np.dot(whitener, M)

        # project to source space using beamformer weights

        if is_free_ori:
            sol = np.dot(W, M)
            logger.info('combining the current components...')
            sol = combine_xyz(sol)
            sol /= noise_norm[:, None]
        else:
            # Linear inverse: do computation here or delayed
            if M.shape[0] < W.shape[0] and pick_ori != 'max-power':
                sol = (W, M)
            else:
                sol = np.dot(W, M)
            if pick_ori == 'max-power':
                sol = np.abs(sol)

        tstep = 1.0 / info['sfreq']
        yield _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                        subject=subject)

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
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')

    # Restrict forward solution to selected channels
    info_ch_names = [c['ch_name'] for c in info['chs']]
    ch_names = [info_ch_names[k] for k in picks]
    fwd_ch_names = forward['sol']['row_names']
    # Keep channels in forward present in info:
    fwd_ch_names = [c for c in fwd_ch_names if c in info_ch_names]
    forward = pick_channels_forward(forward, fwd_ch_names)
    picks_forward = [fwd_ch_names.index(c) for c in ch_names]

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
def lcmv(evoked, forward, noise_cov, data_cov, reg=0.01, label=None,
         pick_ori=None, picks=None, rank=None, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on evoked data.

    NOTE : This implementation has not been heavily tested so please
    report any issue or suggestions.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert
    forward : dict
        Forward operator
    noise_cov : Covariance
        The noise covariance
    data_cov : Covariance
        The data covariance
    reg : float
        The regularization for the whitened data covariance.
    label : Label
        Restricts the LCMV solution to a given label
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses

    See Also
    --------
    lcmv_raw, lcmv_epochs

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880

    The reference for finding the max-power orientation is:
    Sekihara et al. Asymptotic SNR of scalar and vector minimum-variance
    beamformers for neuromagnetic source reconstruction.
    Biomedical Engineering (2004) vol. 51 (10) pp. 1726--34
    """
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    picks = _setup_picks(picks, info, forward, noise_cov)

    data = data[picks]

    stc = _apply_lcmv(
        data=data, info=info, tmin=tmin, forward=forward, noise_cov=noise_cov,
        data_cov=data_cov, reg=reg, label=label, picks=picks, rank=rank,
        pick_ori=pick_ori)

    return six.advance_iterator(stc)


@verbose
def lcmv_epochs(epochs, forward, noise_cov, data_cov, reg=0.01, label=None,
                pick_ori=None, return_generator=False, picks=None, rank=None,
                verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Compute Linearly Constrained Minimum Variance (LCMV) beamformer
    on single trial data.

    NOTE : This implementation has not been heavily tested so please
    report any issue or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
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
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc: list | generator of (SourceEstimate | VolSourceEstimate)
        The source estimates for all epochs

    See Also
    --------
    lcmv_raw, lcmv

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880

    The reference for finding the max-power orientation is:
    Sekihara et al. Asymptotic SNR of scalar and vector minimum-variance
    beamformers for neuromagnetic source reconstruction.
    Biomedical Engineering (2004) vol. 51 (10) pp. 1726--34
    """
    _check_reference(epochs)

    info = epochs.info
    tmin = epochs.times[0]

    picks = _setup_picks(picks, info, forward, noise_cov)

    data = epochs.get_data()[:, picks, :]
    stcs = _apply_lcmv(
        data=data, info=info, tmin=tmin, forward=forward, noise_cov=noise_cov,
        data_cov=data_cov, reg=reg, label=label, picks=picks, rank=rank,
        pick_ori=pick_ori)

    if not return_generator:
        stcs = [s for s in stcs]

    return stcs


@verbose
def lcmv_raw(raw, forward, noise_cov, data_cov, reg=0.01, label=None,
             start=None, stop=None, picks=None, pick_ori=None, rank=None,
             verbose=None):
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
        The noise covariance.
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
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses

    See Also
    --------
    lcmv, lcmv_epochs

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880

    The reference for finding the max-power orientation is:
    Sekihara et al. Asymptotic SNR of scalar and vector minimum-variance
    beamformers for neuromagnetic source reconstruction.
    Biomedical Engineering (2004) vol. 51 (10) pp. 1726--34
    """
    _check_reference(raw)

    info = raw.info

    picks = _setup_picks(picks, info, forward, noise_cov)

    data, times = raw[picks, start:stop]
    tmin = times[0]

    stc = _apply_lcmv(
        data=data, info=info, tmin=tmin, forward=forward, noise_cov=noise_cov,
        data_cov=data_cov, reg=reg, label=label, picks=picks, rank=rank,
        pick_ori=pick_ori)

    return six.advance_iterator(stc)


@verbose
def _lcmv_source_power(info, forward, noise_cov, data_cov, reg=0.01,
                       label=None, picks=None, pick_ori=None,
                       rank=None, verbose=None):
    """Linearly Constrained Minimum Variance (LCMV) beamformer.

    Calculate source power in a time window based on the provided data
    covariance. Noise covariance is used to whiten the data covariance making
    the output equivalent to the neural activity index as defined by
    Van Veen et al. 1997.

    NOTE : This implementation has not been heavily tested so please
    report any issues or suggestions.

    Parameters
    ----------
    info : dict
        Measurement info, e.g. epochs.info.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label | None
        Restricts the solution to a given label.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate
        Source power with a single time point representing the entire time
        window for which data covariance was calculated.

    Notes
    -----
    The original reference is:
    Van Veen et al. Localization of brain electrical activity via linearly
    constrained minimum variance spatial filtering.
    Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    """
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
    whitener, _ = compute_whitener(noise_cov, info, picks, rank=rank)

    # whiten the leadfield
    G = np.dot(whitener, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
    Cm = np.dot(whitener, np.dot(Cm, whitener.T))

    # Calculating regularized inverse, equivalent to an inverse operation after
    # the following regularization:
    # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

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

        # Noise normalization
        noise_norm = np.dot(Wk, Wk.T)
        noise_norm = noise_norm.trace()

        # Calculating source power
        sp_temp = np.dot(np.dot(Wk, Cm), Wk.T)
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
            freq_bins, subtract_evoked=False, reg=0.01, label=None,
            pick_ori=None, n_jobs=1, picks=None, rank=None, verbose=None):
    """5D time-frequency beamforming based on LCMV.

    Calculate source power in time-frequency windows using a spatial filter
    based on the Linearly Constrained Minimum Variance (LCMV) beamforming
    approach. Band-pass filtered epochs are divided into time windows from
    which covariance is computed and used to create a beamformer spatial
    filter.

    NOTE : This implementation has not been heavily tested so please
    report any issues or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_covs : list of instances of Covariance
        Noise covariance for each frequency bin.
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
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
        is installed properly and CUDA is initialized.
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stcs : list of SourceEstimate
        Source power at each time window. One SourceEstimate object is returned
        for each frequency bin.

    Notes
    -----
    The original reference is:
    Dalal et al. Five-dimensional neuroimaging: Localization of the
    time-frequency dynamics of cortical activity.
    NeuroImage (2008) vol. 40 (4) pp. 1686-1700
    """
    _check_reference(epochs)

    if pick_ori not in [None, 'normal']:
        raise ValueError('Unrecognized orientation option in pick_ori, '
                         'available choices are None and normal')
    if len(noise_covs) != len(freq_bins):
        raise ValueError('One noise covariance object expected per frequency '
                         'bin')
    if len(win_lengths) != len(freq_bins):
        raise ValueError('One time window length expected per frequency bin')
    if any(win_length < tstep for win_length in win_lengths):
        raise ValueError('Time step should not be larger than any of the '
                         'window lengths')

    # Extract raw object from the epochs object
    raw = epochs._raw
    if raw is None:
        raise ValueError('The provided epochs object does not contain the '
                         'underlying raw object. Please use preload=False '
                         'when constructing the epochs object')

    picks = _setup_picks(picks, epochs.info, forward, noise_covs[0])
    ch_names = [epochs.ch_names[k] for k in picks]

    # Use picks from epochs for picking channels in the raw object
    raw_picks = [raw.ch_names.index(c) for c in ch_names]

    # Make sure epochs.events contains only good events:
    epochs.drop_bad()

    # Multiplying by 1e3 to avoid numerical issues, e.g. 0.3 // 0.05 == 5
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))

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
                                         pick_ori=pick_ori, verbose=verbose)
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
