"""Dynamic Imaging of Coherent Sources (DICS)."""

# Authors: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from scipy import linalg

from ..utils import logger, verbose, warn
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import combine_xyz, _check_reference
from ..source_estimate import _make_stc
from ..time_frequency import CrossSpectralDensity, csd_epochs
from ._lcmv import _prepare_beamformer_input, _setup_picks
from ..externals import six


@verbose
def _apply_dics(data, info, tmin, forward, noise_csd, data_csd, reg,
                label=None, picks=None, pick_ori=None, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Calculate the DICS spatial filter based on a given cross-spectral
    density object and return estimates of source activity based on given data.

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
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses
    """
    is_free_ori, _, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    Cm = data_csd.data

    # Calculating regularized inverse, equivalent to an inverse operation after
    # regularization: Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient

    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        # TODO: max-power is not implemented yet, however DICS does employ
        # orientation picking when one eigen value is much larger than the
        # other

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        # Noise normalization
        noise_norm = np.dot(np.dot(Wk.conj(), noise_csd.data), Wk.T)
        noise_norm = np.abs(noise_norm).trace()
        Wk /= np.sqrt(noise_norm)

    # Pick source orientation normal to cortical surface
    if pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

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

        # Apply SSPs
        if info['projs']:
            M = np.dot(proj, M)

        # project to source space using beamformer weights
        if is_free_ori:
            sol = np.dot(W, M)
            logger.info('combining the current components...')
            sol = combine_xyz(sol)
        else:
            # Linear inverse: do not delay compuation due to non-linear abs
            sol = np.dot(W, M)

        tstep = 1.0 / info['sfreq']
        if np.iscomplexobj(sol):
            sol = np.abs(sol)  # XXX : STC cannot contain (yet?) complex values
        yield _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                        subject=subject)

    logger.info('[done]')


@verbose
def dics(evoked, forward, noise_csd, data_csd, reg=0.01, label=None,
         pick_ori=None, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Compute a Dynamic Imaging of Coherent Sources (DICS) beamformer
    on evoked data and return estimates of source time courses.

    NOTE : Fixed orientation forward operators will result in complex time
    courses in which case absolute values will be  returned. Therefore the
    orientation will no longer be fixed.

    NOTE : This implementation has not been heavily tested so please
    report any issues or suggestions.

    Parameters
    ----------
    evoked : Evoked
        Evoked data.
    forward : dict
        Forward operator.
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses

    See Also
    --------
    dics_epochs

    Notes
    -----
    The original reference is:
    Gross et al. Dynamic imaging of coherent sources: Studying neural
    interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """
    _check_reference(evoked)
    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    picks = _setup_picks(picks=None, info=info, forward=forward)
    data = data[picks]

    stc = _apply_dics(data, info, tmin, forward, noise_csd, data_csd, reg=reg,
                      label=label, pick_ori=pick_ori, picks=picks)
    return six.advance_iterator(stc)


@verbose
def dics_epochs(epochs, forward, noise_csd, data_csd, reg=0.01, label=None,
                pick_ori=None, return_generator=False, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Compute a Dynamic Imaging of Coherent Sources (DICS) beamformer
    on single trial data and return estimates of source time courses.

    NOTE : Fixed orientation forward operators will result in complex time
    courses in which case absolute values will be  returned. Therefore the
    orientation will no longer be fixed.

    NOTE : This implementation has not been heavily tested so please
    report any issues or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc: list | generator of SourceEstimate | VolSourceEstimate
        The source estimates for all epochs

    See Also
    --------
    dics

    Notes
    -----
    The original reference is:
    Gross et al. Dynamic imaging of coherent sources: Studying neural
    interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """
    _check_reference(epochs)

    info = epochs.info
    tmin = epochs.times[0]

    picks = _setup_picks(picks=None, info=info, forward=forward)
    data = epochs.get_data()[:, picks, :]

    stcs = _apply_dics(data, info, tmin, forward, noise_csd, data_csd, reg=reg,
                       label=label, pick_ori=pick_ori, picks=picks)

    if not return_generator:
        stcs = list(stcs)

    return stcs


@verbose
def dics_source_power(info, forward, noise_csds, data_csds, reg=0.01,
                      label=None, pick_ori=None, verbose=None):
    """Dynamic Imaging of Coherent Sources (DICS).

    Calculate source power in time and frequency windows specified in the
    calculation of the data cross-spectral density matrix or matrices. Source
    power is normalized by noise power.

    NOTE : This implementation has not been heavily tested so please
    report any issues or suggestions.

    Parameters
    ----------
    info : dict
        Measurement info, e.g. epochs.info.
    forward : dict
        Forward operator.
    noise_csds : instance or list of instances of CrossSpectralDensity
        The noise cross-spectral density matrix for a single frequency or a
        list of matrices for multiple frequencies.
    data_csds : instance or list of instances of CrossSpectralDensity
        The data cross-spectral density matrix for a single frequency or a list
        of matrices for multiple frequencies.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source power with frequency instead of time.

    Notes
    -----
    The original reference is:
    Gross et al. Dynamic imaging of coherent sources: Studying neural
    interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    """
    if isinstance(data_csds, CrossSpectralDensity):
        data_csds = [data_csds]

    if isinstance(noise_csds, CrossSpectralDensity):
        noise_csds = [noise_csds]

    def csd_shapes(x):
        return tuple(c.data.shape for c in x)

    if (csd_shapes(data_csds) != csd_shapes(noise_csds) or
       any(len(set(csd_shapes(c))) > 1 for c in [data_csds, noise_csds])):
        raise ValueError('One noise CSD matrix should be provided for each '
                         'data CSD matrix and vice versa. All CSD matrices '
                         'should have identical shape.')

    frequencies = []
    for data_csd, noise_csd in zip(data_csds, noise_csds):
        if not np.allclose(data_csd.frequencies, noise_csd.frequencies):
            raise ValueError('Data and noise CSDs should be calculated at '
                             'identical frequencies')

        # If CSD is summed over multiple frequencies, take the average
        # frequency
        if(len(data_csd.frequencies) > 1):
            frequencies.append(np.mean(data_csd.frequencies))
        else:
            frequencies.append(data_csd.frequencies[0])
    fmin = frequencies[0]

    if len(frequencies) > 2:
        fstep = []
        for i in range(len(frequencies) - 1):
            fstep.append(frequencies[i + 1] - frequencies[i])
        if not np.allclose(fstep, np.mean(fstep), 1e-5):
            warn('Uneven frequency spacing in CSD object, frequencies in the '
                 'resulting stc file will be inaccurate.')
        fstep = fstep[0]
    elif len(frequencies) > 1:
        fstep = frequencies[1] - frequencies[0]
    else:
        fstep = 1  # dummy value

    picks = _setup_picks(picks=None, info=info, forward=forward)

    is_free_ori, _, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks=picks,
                                  pick_ori=pick_ori)

    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    source_power = np.zeros((n_sources, len(data_csds)))
    n_csds = len(data_csds)

    logger.info('Computing DICS source power...')
    for i, (data_csd, noise_csd) in enumerate(zip(data_csds, noise_csds)):
        if n_csds > 1:
            logger.info('    computing DICS spatial filter %d out of %d' %
                        (i + 1, n_csds))

        Cm = data_csd.data

        # Calculating regularized inverse, equivalent to an inverse operation
        # after the following regularization:
        # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
        Cm_inv = linalg.pinv(Cm, reg)

        # Compute spatial filters
        W = np.dot(G.T, Cm_inv)
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
            noise_norm = np.dot(np.dot(Wk.conj(), noise_csd.data), Wk.T)
            noise_norm = np.abs(noise_norm).trace()

            # Calculating source power
            sp_temp = np.dot(np.dot(Wk.conj(), data_csd.data), Wk.T)
            sp_temp /= max(noise_norm, 1e-40)  # Avoid division by 0

            if pick_ori == 'normal':
                source_power[k, i] = np.abs(sp_temp)[2, 2]
            else:
                source_power[k, i] = np.abs(sp_temp).trace()

    logger.info('[done]')

    subject = _subject_from_forward(forward)
    return _make_stc(source_power, vertices=vertno, tmin=fmin / 1000.,
                     tstep=fstep / 1000., subject=subject)


@verbose
def tf_dics(epochs, forward, noise_csds, tmin, tmax, tstep, win_lengths,
            freq_bins, subtract_evoked=False, mode='fourier', n_ffts=None,
            mt_bandwidths=None, mt_adaptive=False, mt_low_bias=True, reg=0.01,
            label=None, pick_ori=None, verbose=None):
    """5D time-frequency beamforming based on DICS.

    Calculate source power in time-frequency windows using a spatial filter
    based on the Dynamic Imaging of Coherent Sources (DICS) beamforming
    approach. For each time window and frequency bin combination cross-spectral
    density (CSD) is computed and used to create a beamformer spatial filter
    with noise CSD used for normalization.

    NOTE : This implementation has not been heavily tested so please
    report any issues or suggestions.

    Parameters
    ----------
    epochs : Epochs
        Single trial epochs.
    forward : dict
        Forward operator.
    noise_csds : list of instances of CrossSpectralDensity
        Noise cross-spectral density for each frequency bin.
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
    mode : str
        Spectrum estimation mode can be either: 'multitaper' or 'fourier'.
    n_ffts : list | None
        FFT lengths to use for each frequency bin.
    mt_bandwidths : list of float
        The bandwidths of the multitaper windowing function in Hz. Only used in
        'multitaper' mode. One value should be provided for each frequency bin.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into CSD. Only used
        in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stcs : list of SourceEstimate | VolSourceEstimate
        Source power at each time window. One SourceEstimate object is returned
        for each frequency bin.

    Notes
    -----
    The original reference is:
    Dalal et al. Five-dimensional neuroimaging: Localization of the
    time-frequency dynamics of cortical activity.
    NeuroImage (2008) vol. 40 (4) pp. 1686-1700

    NOTE : Dalal et al. used a synthetic aperture magnetometry beamformer (SAM)
    in each time-frequency window instead of DICS.
    """
    _check_reference(epochs)

    if pick_ori not in [None, 'normal']:
        raise ValueError('Unrecognized orientation option in pick_ori, '
                         'available choices are None and normal')
    if len(noise_csds) != len(freq_bins):
        raise ValueError('One noise CSD object expected per frequency bin')
    if len(win_lengths) != len(freq_bins):
        raise ValueError('One time window length expected per frequency bin')
    if any(win_length < tstep for win_length in win_lengths):
        raise ValueError('Time step should not be larger than any of the '
                         'window lengths')
    if n_ffts is not None and len(n_ffts) != len(freq_bins):
        raise ValueError('When specifying number of FFT samples, one value '
                         'must be provided per frequency bin')
    if mt_bandwidths is not None and len(mt_bandwidths) != len(freq_bins):
        raise ValueError('When using multitaper mode and specifying '
                         'multitaper transform bandwidth, one value must be '
                         'provided per frequency bin')

    if n_ffts is None:
        n_ffts = [None] * len(freq_bins)
    if mt_bandwidths is None:
        mt_bandwidths = [None] * len(freq_bins)

    # Multiplying by 1e3 to avoid numerical issues, e.g. 0.3 // 0.05 == 5
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))

    # Subtract evoked response
    if subtract_evoked:
        epochs.subtract_evoked()

    sol_final = []
    for freq_bin, win_length, noise_csd, n_fft, mt_bandwidth in\
            zip(freq_bins, win_lengths, noise_csds, n_ffts, mt_bandwidths):
        n_overlap = int((win_length * 1e3) // (tstep * 1e3))

        # Scale noise CSD to allow data and noise CSDs to have different length
        noise_csd = deepcopy(noise_csd)
        noise_csd.data /= noise_csd.n_fft

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
                logger.info('Computing time-frequency DICS beamformer for '
                            'time window %d to %d ms, in frequency range '
                            '%d to %d Hz' % (win_tmin * 1e3, win_tmax * 1e3,
                                             freq_bin[0], freq_bin[1]))

                # Counteracts unsafe floating point arithmetic ensuring all
                # relevant samples will be taken into account when selecting
                # data in time windows
                win_tmin = win_tmin - 1e-10
                win_tmax = win_tmax + 1e-10

                # Calculating data CSD in current time window
                data_csd = csd_epochs(epochs, mode=mode,
                                      fmin=freq_bin[0],
                                      fmax=freq_bin[1], fsum=True,
                                      tmin=win_tmin, tmax=win_tmax,
                                      n_fft=n_fft,
                                      mt_bandwidth=mt_bandwidth,
                                      mt_low_bias=mt_low_bias)

                # Scale data CSD to allow data and noise CSDs to have different
                # length
                data_csd.data /= data_csd.n_fft

                stc = dics_source_power(epochs.info, forward, noise_csd,
                                        data_csd, reg=reg, label=label,
                                        pick_ori=pick_ori)
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
        stc = _make_stc(sol_final[i_freq, :, :].T, vertices=stc.vertices,
                        tmin=tmin, tstep=tstep, subject=stc.subject)
        stcs.append(stc)

    return stcs
