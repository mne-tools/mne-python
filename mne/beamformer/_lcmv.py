"""Compute Linearly constrained minimum variance (LCMV) beamformer."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Roman Goj <roman.goj@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
from scipy import linalg

from ..io.pick import (pick_types, pick_channels_cov, pick_info)
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import combine_xyz, _check_reference
from ..cov import compute_whitener, compute_covariance
from ..source_estimate import _make_stc, SourceEstimate, _get_src_type
from ..utils import logger, verbose, warn, _validate_type, _reg_pinv
from .. import Epochs
from ._compute_beamformer import (
    _setup_picks, _pick_channels_spatial_filter,
    _check_proj_match, _prepare_beamformer_input, _check_one_ch_type,
    _compute_beamformer, _check_src_type, Beamformer, _check_rank)


@verbose
def make_lcmv(info, forward, data_cov, reg=0.05, noise_cov=None, label=None,
              pick_ori=None, rank='full', weight_norm='unit-noise-gain',
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
    pick_ori : None | 'normal' | 'max-power' | 'vector'
        For forward solutions with fixed orientation, None (default) must be
        used and a scalar beamformer is computed. For free-orientation forward
        solutions, a vector beamformer is computed and:

            None
                Pools the orientations by taking the norm.
            'normal'
                Keeps only the radial component.
            'max-power'
                Selects orientations that maximize output source power at
                each location.
            'vector'
                Keeps the currents for each direction separate

    rank : int | None | 'full'
        This controls the effective rank of the covariance matrix when
        computing the inverse. The rank can be set explicitly by specifying an
        integer value. If ``None``, the rank will be automatically estimated.
        Since applying regularization will always make the covariance matrix
        full rank, the rank is estimated before regularization in this case. If
        'full', the rank will be estimated after regularization and hence
        will mean using the full rank, unless ``reg=0`` is used.
        The default in ``'full'``.
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if 'nai', the Neural Activity Index [1]_ will be computed,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    reduce_rank : bool
        If True, the rank of the leadfield will be reduced by 1 for each
        spatial location. Setting reduce_rank to True is typically necessary
        if you use a single sphere model for MEG.
    %(verbose)s

    Returns
    -------
    filters : instance of Beamformer
        Dictionary containing filter weights from LCMV beamformer.
        Contains the following keys:

            'weights' : array
                The filter weights of the beamformer.
            'data_cov' : instance of Covariance
                The data covariance matrix used to compute the beamformer.
            'noise_cov' : instance of Covariance | None
                The noise covariance matrix used to compute the beamformer.
            'whitener' : None | array
                Whitening matrix, provided if whitening was applied to the
                covariance matrix and leadfield during computation of the
                beamformer weights.
            'weight_norm' : 'unit-noise-gain'| 'nai' | None
                Type of weight normalization used to compute the filter
                weights.
            'pick_ori' : None | 'normal'
                Orientation selection used in filter computation.
            'ch_names' : list
                Channels used to compute the beamformer.
            'proj' : array
                Projections used to compute the beamformer.
            'is_ssp' : bool
                If True, projections were applied prior to filter computation.
            'vertices' : list
                Vertices for which the filter weights were computed.
            'is_free_ori' : bool
                If True, the filter was computed with free source orientation.
            'src_type' : str
                Type of source space.

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
    rank = _check_rank(rank)

    is_free_ori, ch_names, proj, vertno, G, nn = \
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    if 'estimator' in data_cov:
        del data_cov['estimator']

    # check number of sensor types present in the data
    _check_one_ch_type(info, picks, noise_cov, 'lcmv')

    # apply SSPs
    is_ssp = False
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
        is_ssp = True

    if noise_cov is not None:
        # Handle whitening + data covariance
        whitener_rank = None if rank == 'full' else rank
        whitener, _, rank = compute_whitener(
            noise_cov, info, picks, rank=whitener_rank, return_rank=True)
        # whiten the leadfield
        G = np.dot(whitener, G)
        # whiten data covariance
        Cm = np.dot(whitener, np.dot(Cm, whitener.T))
        noise_cov = noise_cov.copy()
        if 'estimator' in noise_cov:
            del noise_cov['estimator']
    else:
        whitener = None
        rank = G.shape[0]

    # leadfield rank and optional rank reduction
    if reduce_rank:
        if not pick_ori == 'max-power':
            raise NotImplementedError('The computation of spatial filters '
                                      'with rank reduction using reduce_rank '
                                      'parameter is only implemented with '
                                      'pick_ori=="max-power".')
        _validate_type(reduce_rank, bool, "reduce_rank", "a boolean")

    # compute spatial filter
    n_orient = 3 if is_free_ori else 1
    W = _compute_beamformer(G, Cm, reg, n_orient, weight_norm,
                            pick_ori, reduce_rank, rank,
                            inversion='matrix', nn=nn)

    # get src type to store with filters for _make_stc
    src_type = _get_src_type(forward['src'], vertno)

    # get subject to store with filters
    subject_from = _subject_from_forward(forward)

    # Is the computed beamformer a scalar or vector beamformer?
    is_free_ori = is_free_ori if pick_ori in [None, 'vector'] else False

    filters = Beamformer(
        kind='LCMV', weights=W, data_cov=data_cov, noise_cov=noise_cov,
        whitener=whitener, weight_norm=weight_norm, pick_ori=pick_ori,
        ch_names=ch_names, proj=proj, is_ssp=is_ssp, vertices=vertno,
        is_free_ori=is_free_ori, nsource=forward['nsource'], src_type=src_type,
        source_nn=forward['source_nn'].copy(), subject=subject_from, rank=rank)

    return filters


def _apply_lcmv(data, filters, info, tmin, max_ori_out):
    """Apply LCMV spatial filter to data for source reconstruction."""
    if max_ori_out != 'signed':
        raise ValueError('max_ori_out must be "signed", got %s'
                         % (max_ori_out,))

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    W = filters['weights']

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
        vector = False
        if filters['is_free_ori']:
            sol = np.dot(W, M)
            if filters['pick_ori'] == 'vector':
                vector = True
            else:
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

        # compatibility with 0.16, add src_type as None if not present:
        filters, warn_text = _check_src_type(filters)

        yield _make_stc(sol, vertices=filters['vertices'], tmin=tmin,
                        tstep=tstep, subject=filters['subject'],
                        vector=vector, source_nn=filters['source_nn'],
                        src_type=filters['src_type'], warn_text=warn_text)

    logger.info('[done]')


@verbose
def apply_lcmv(evoked, filters, max_ori_out='signed', verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on evoked data.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert.
    filters : instance of Beamformer
        LCMV spatial filter (beamformer weights).
        Filter weights returned from :func:`make_lcmv`.
    max_ori_out: 'signed'
        Specify in case of pick_ori='max-power'.
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate | VectorSourceEstimate
        Source time courses.

    See Also
    --------
    make_lcmv, apply_lcmv_raw, apply_lcmv_epochs
    """
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    sel = _pick_channels_spatial_filter(evoked.ch_names, filters)
    data = data[sel]

    stc = _apply_lcmv(data=data, filters=filters, info=info,
                      tmin=tmin, max_ori_out=max_ori_out)

    return next(stc)


@verbose
def apply_lcmv_epochs(epochs, filters, max_ori_out='signed',
                      return_generator=False, verbose=None):
    """Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Apply Linearly Constrained Minimum Variance (LCMV) beamformer weights
    on single trial data.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    filters : instance of Beamformer
        LCMV spatial filter (beamformer weights)
        Filter weights returned from :func:`make_lcmv`.
    max_ori_out: 'signed'
        Specify in case of pick_ori='max-power'.
    return_generator : bool
         Return a generator object instead of a list. This allows iterating
         over the stcs without having to keep them all in memory.
    %(verbose)s


    Returns
    -------
    stc: list | generator of (SourceEstimate | VolSourceEstimate)
        The source estimates for all epochs.

    See Also
    --------
    make_lcmv, apply_lcmv_raw, apply_lcmv
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
def apply_lcmv_raw(raw, filters, start=None, stop=None, max_ori_out='signed',
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
    filters : instance of Beamformer
        LCMV spatial filter (beamformer weights).
        Filter weights returned from :func:`make_lcmv`.
    start : int
        Index of first time sample (index not time is seconds).
    stop : int
        Index of first time sample not to include (index not time is seconds).
    max_ori_out: 'signed'
        Specify in case of pick_ori='max-power'.
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses.

    See Also
    --------
    make_lcmv, apply_lcmv_epochs, apply_lcmv
    """
    _check_reference(raw)

    info = raw.info

    sel = _pick_channels_spatial_filter(raw.ch_names, filters)
    data, times = raw[sel, start:stop]
    tmin = times[0]

    stc = _apply_lcmv(data=data, filters=filters, info=info,
                      tmin=tmin, max_ori_out=max_ori_out)

    return next(stc)


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

    is_free_ori, ch_names, proj, vertno, G, _ =\
        _prepare_beamformer_input(
            info, forward, label, picks, pick_ori)

    # Handle whitening
    picks = [info['ch_names'].index(k) for k in ch_names
             if k in info['ch_names']]
    info = pick_info(info, picks)
    del picks  # everything should be picked now

    # XXX this could maybe use pca=True to avoid needing to use
    # _reg_pinv(..., rank=rank) later
    if noise_cov is not None:
        whitener_rank = None if rank == 'full' else rank
        whitener, _ = compute_whitener(
            noise_cov, info, rank=whitener_rank)

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
    Cm_inv, d, _ = _reg_pinv(Cm, reg, rank=rank)

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
            pick_ori=None, n_jobs=1, rank='full',
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
    freq_bins : list of tuple of float
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
        Number of jobs to run in parallel.
        Can be 'cuda' if ``cupy`` is installed properly.
    rank : int | None | 'full'
        This controls the effective rank of the covariance matrix when
        computing the inverse. The rank can be set explicitly by specifying an
        integer value. If ``None``, the rank will be automatically estimated.
        Since applying regularization will always make the covariance matrix
        full rank, the rank is estimated before regularization in this case. If
        'full', the rank will be estimated after regularization and hence
        will mean using the full rank, unless ``reg=0`` is used.
        The default is ``'full'``.
    weight_norm : 'unit-noise-gain' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    raw : instance of Raw | None
        The raw instance used to construct the epochs.
        Must be provided unless epochs are constructed with
        ``preload=False``.
    %(verbose)s

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
    _check_reference(epochs)
    rank = _check_rank(rank)

    if pick_ori not in [None, 'normal']:
        raise ValueError('pick_ori must be one of "normal" and None, '
                         'got %s' % (pick_ori,))
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
        picks = _setup_picks(epochs.info, forward, data_cov=None)
    else:
        picks = _setup_picks(epochs.info, forward, data_cov=None,
                             noise_cov=noise_covs[0])
    ch_names = [epochs.ch_names[k] for k in picks]

    # check number of sensor types present in the data
    _check_one_ch_type(epochs.info, picks, noise_covs, 'lcmv')

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
                                         pick_ori=pick_ori, rank=rank,
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
